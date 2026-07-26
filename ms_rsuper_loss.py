"""MS-RSuper composite report-supervised loss.

Implements the four constraint families described in
Ge, Huang, Liu (arXiv:2602.20994), §2.2 - §2.4:

    L_report = sum_k L_exist^(k)                    [substructure presence / absence]
             + w_size  * L_size                     [largest-lesion size, MAE]
             + w_count * L_count                    [one-sided minimal multiplicity]
             + w_prior * L_prior                    [intra- vs extra-axial cohort prior]

The model outputs three substructure probability maps (after sigmoid):
    P_ET  : Enhancing Tumor      <- constrained by T1c findings
    P_ED  : Edema                <- constrained by FLAIR findings
    P_TC  : Tumor Core           <- constrained by T1/T2 findings
    P_WT  := P_ET + P_ED + P_TC  <- constrained by global findings

The cue dictionary expected per sample (see report_extraction/schema.py):

    cue = {
        "cohort":      "MEN" | "MET" | None,
        "d_max_mm":    float | None,                 # longest diameter of largest lesion
        "n_qual":      int   | None,                 # minimal multiplicity (>=1)
        "substruct":   {                             # per-substructure cues
            "ET":  {"state": "present"|"absent"|None, "lambda": float},
            "ED":  {"state": ..., "lambda": ...},
            "TC":  {"state": ..., "lambda": ...},
        },
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cc3d  # connected-components-3d
except ImportError:  # pragma: no cover
    cc3d = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _soft_volume(prob: torch.Tensor) -> torch.Tensor:
    """Differentiable volume estimate = sum of probabilities (in voxels)."""
    return prob.flatten(1).sum(dim=1)  # [B]


def _connected_components(binary_mask: torch.Tensor) -> List[torch.Tensor]:
    """Per-batch list of integer label tensors; computed without gradient.

    Uses cc3d (26-connectivity) when available, else a torch fallback.
    Each returned tensor has the same spatial shape as one sample; voxel value
    is the component id (0 = background).
    """
    out = []
    binary_np = binary_mask.detach().cpu().numpy().astype("uint8")
    for b in range(binary_np.shape[0]):
        if cc3d is None:
            # crude fallback: treat whole foreground as a single component
            labels = (binary_np[b] > 0).astype("int64")
        else:
            # cc3d returns uint16/uint32; torch indexing on CPU rejects uint16.
            # Cast to int64 for portable torch ops downstream.
            labels = cc3d.connected_components(binary_np[b], connectivity=26
                                               ).astype("int64")
        out.append(torch.from_numpy(labels).to(binary_mask.device))
    return out


def _largest_component_soft_volume(prob: torch.Tensor, threshold: float = 0.5
                                   ) -> torch.Tensor:
    """For each batch element, return the soft volume of the largest CC.

    Components are formed on the thresholded mask (no gradient through that),
    but the returned volume sums `prob` over the component mask, so gradient
    still flows into the prediction.
    """
    binary = (prob > threshold).float()
    labels = _connected_components(binary)  # list of [*spatial]
    vols = []
    for b, lab in enumerate(labels):
        unique = torch.unique(lab)
        unique = unique[unique != 0]
        if len(unique) == 0:
            vols.append(torch.zeros((), device=prob.device, dtype=prob.dtype))
            continue
        comp_vols_soft = []
        for cid in unique.tolist():
            mask = (lab == cid).to(prob.dtype)
            comp_vols_soft.append((prob[b] * mask).sum())
        comp_vols_soft = torch.stack(comp_vols_soft)
        vols.append(comp_vols_soft.max())
    return torch.stack(vols)  # [B]


def _component_count(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Integer component count per batch element (no gradient)."""
    binary = (prob > threshold).float()
    labels = _connected_components(binary)
    counts = []
    for lab in labels:
        u = torch.unique(lab)
        counts.append(torch.tensor(int((u != 0).sum().item()),
                                   device=prob.device, dtype=prob.dtype))
    return torch.stack(counts)  # [B]


# ---------------------------------------------------------------------------
# individual loss terms
# ---------------------------------------------------------------------------

def _lam(sub_cue: Dict) -> float:
    """Read the certainty weight from a substructure cue.

    The report parser emits the key as ``lambda_`` (trailing underscore,
    because ``lambda`` is a Python reserved word). Accept both spellings so
    older cue files keep working; default to 1.0 when absent.
    """
    return float(sub_cue.get("lambda_", sub_cue.get("lambda", 1.0)))


def existence_loss(prob_k: torch.Tensor,
                   state: Optional[str],
                   lam: float = 1.0,
                   min_voxels: float = 1.0) -> torch.Tensor:
    """L_exist^(k), paper eq. (1).

    Args:
        prob_k:  [B, *spatial] sigmoid probability of substructure k.
        state:   "present" | "absent" | None.
        lam:     certainty weight in [0, 1].
        min_voxels: threshold below which `present` is penalised.
    """
    if state is None or lam <= 0.0:
        return prob_k.new_zeros(())
    # Normalise volumes to a fraction of the patch so the term is O(1) and
    # comparable to L_seg (raw voxel counts ~1e6 explode training). See D-17.
    norm = float(prob_k[0].numel())
    V_k = _soft_volume(prob_k) / norm  # [B], fraction in [0, 1]
    if state == "present":
        loss = torch.clamp(min_voxels / norm - V_k, min=0.0)
    elif state == "absent":
        loss = V_k
    else:
        return prob_k.new_zeros(())
    return lam * loss.mean()


def size_loss(prob_wt: torch.Tensor,
              d_max_frac: Optional[float],
              threshold: float = 0.5) -> torch.Tensor:
    """L_size = | d_max_frac - (largest-CC soft volume / numel) |  (MAE).

    `d_max_frac` is the report's largest-lesion volume as a *fraction of the
    whole-volume voxel count* (computed in the dataloader; resolution-independent
    so it survives the report-sample resampling, D-16). The predicted side is the
    largest connected component's soft volume divided by numel — also a fraction.
    Both in [0, 1] -> O(1) and comparable to L_seg (D-17).
    """
    if d_max_frac is None:
        return prob_wt.new_zeros(())
    norm = float(prob_wt[0].numel())
    max_frac = _largest_component_soft_volume(prob_wt, threshold=threshold) / norm
    target = prob_wt.new_full((max_frac.shape[0],), float(d_max_frac))
    return F.l1_loss(max_frac, target)


def count_loss(prob_wt: torch.Tensor,
               n_qual: Optional[int],
               threshold: float = 0.5) -> torch.Tensor:
    """L_count = max(0, N_qual - |C_pred|).

    Counting is non-differentiable, so this loss is a *scalar penalty* that
    flags samples where the model is under-predicting lesions. We additionally
    add a soft term: when the count is short, penalise (N_qual - count) *
    (1 - max_subthreshold_prob) to provide an actual gradient — encouraging
    the most-confident-yet-sub-threshold blob to cross 0.5.
    """
    if n_qual is None or n_qual <= 0:
        return prob_wt.new_zeros(())
    cpred = _component_count(prob_wt, threshold=threshold)  # [B], detached value
    deficit = torch.clamp(prob_wt.new_full(cpred.shape, float(n_qual)) - cpred, min=0.0)
    # soft gradient term: push the brightest sub-threshold voxel up
    sub = torch.where(prob_wt < threshold, prob_wt,
                      prob_wt.new_zeros(1).expand_as(prob_wt))
    brightest = sub.flatten(1).max(dim=1).values  # [B]
    soft_term = (1.0 - brightest) * deficit
    return soft_term.mean()


def rsuper_volume_loss(prob_wt: torch.Tensor,
                       d_max_frac: Optional[float]) -> torch.Tensor:
    """Naive R-Super [4] symmetric TOTAL-volume loss: |V_WT - target|.

    Unlike MS-RSuper's one-sided L_size (which targets the LARGEST component),
    this sums the WHOLE WT volume and matches it symmetrically to the report's
    size cue — treating the reported (largest-lesion) size AS IF it were the
    total tumor volume. This is exactly the assumption the paper shows is
    "confused by partial reports" (it penalises the model for predicting MORE
    than the single reported lesion). Faithful reproduction of the R-Super
    baseline the paper compares against.
    """
    if d_max_frac is None:
        return prob_wt.new_zeros(())
    norm = float(prob_wt[0].numel())
    v_pred = prob_wt.flatten(1).sum(dim=1) / norm   # TOTAL WT volume fraction
    target = prob_wt.new_full((v_pred.shape[0],), float(d_max_frac))
    return F.l1_loss(v_pred, target)                # SYMMETRIC (both directions)


def rsuper_count_loss(prob_wt: torch.Tensor,
                      n_qual: Optional[int],
                      threshold: float = 0.5) -> torch.Tensor:
    """Naive R-Super symmetric count loss: penalise |count - N| in BOTH
    directions (vs MS-RSuper's one-sided max(0, N-count)).

    Under-count -> push the brightest sub-threshold blob up (more components).
    Over-count  -> push down the mean confidence of the super-threshold region
                   (fewer components). Symmetric, so it punishes the model for
                   predicting MORE lesions than the (possibly partial) report.
    """
    if n_qual is None or n_qual <= 0:
        return prob_wt.new_zeros(())
    cpred = _component_count(prob_wt, threshold=threshold)            # detached
    target = prob_wt.new_full(cpred.shape, float(n_qual))
    under = torch.clamp(target - cpred, min=0.0)
    over  = torch.clamp(cpred - target, min=0.0)
    # soft up-push for under-count
    sub = torch.where(prob_wt < threshold, prob_wt,
                      prob_wt.new_zeros(1).expand_as(prob_wt))
    brightest = sub.flatten(1).max(dim=1).values                     # [B]
    # soft down-push for over-count: mean prob over super-threshold voxels
    supmask = (prob_wt >= threshold).float()
    sup_mean = (prob_wt * supmask).flatten(1).sum(dim=1) \
               / (supmask.flatten(1).sum(dim=1) + 1e-6)              # [B]
    loss = (1.0 - brightest) * under + sup_mean * over
    return loss.mean()


def prior_loss(prob_wt: torch.Tensor,
               cohort: Optional[str],
               m_parench: Optional[torch.Tensor],
               m_dural: Optional[torch.Tensor]) -> torch.Tensor:
    """L_prior — penalise anatomically wrong predictions.

    For MEN (extra-axial), penalise P_WT inside the parenchyma.
    For MET (intra-axial), penalise P_WT inside the dural shell.
    """
    if cohort is None:
        return prob_wt.new_zeros(())
    # Normalise the in-region probability mass to a patch fraction (D-17).
    norm = float(prob_wt[0].numel())
    if cohort == "MEN" and m_parench is not None:
        return (prob_wt * m_parench).flatten(1).sum(dim=1).mean() / norm
    if cohort == "MET" and m_dural is not None:
        return (prob_wt * m_dural).flatten(1).sum(dim=1).mean() / norm
    return prob_wt.new_zeros(())


# ---------------------------------------------------------------------------
# composite L_report
# ---------------------------------------------------------------------------

@dataclass
class MSRSuperWeights:
    w_size: float = 1.0
    w_count: float = 0.5
    w_prior: float = 0.2
    threshold: float = 0.5
    min_voxels_present: float = 1.0
    # D-22: minimum certainty weight (λ) required to activate L_exist for the
    # ED channel. Vague cues ("possible edema", λ=0.50) are skipped to prevent
    # spurious ED false-positives in MEN (diagnosed: B has 2× more ED FP than
    # A). Set to 0.0 to disable filtering (original paper behaviour).
    exist_ed_lambda_min: float = 0.85
    # "ms_rsuper" = our full 4-family loss. "rsuper" = naive R-Super [4]
    # baseline: symmetric total-WT volume + symmetric count only (no exist /
    # prior / substructure alignment). Used to reproduce the paper's R-Super row.
    loss_mode: str = "ms_rsuper"


class MSRSuperLoss(nn.Module):
    """Composite report-supervised loss for MS-RSuper.

    Model output convention: a Tensor of shape [B, 3, D, H, W] in logit space,
    channels ordered (ET, ED, TC). The loss applies sigmoid internally.
    """

    SUBSTRUCT_INDEX = {"ET": 0, "ED": 1, "TC": 2}

    def __init__(self, weights: Optional[MSRSuperWeights] = None):
        super().__init__()
        self.w = weights or MSRSuperWeights()

    def forward(self,
                logits: torch.Tensor,
                cues: Sequence[Dict],
                priors: Optional[Dict[str, torch.Tensor]] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, 3, *spatial], channel order (ET, ED, TC).
            cues:   list of per-sample cue dicts (len == B).
            priors: optional dict with keys "parench" and "dural", each
                    [B, *spatial] binary masks aligned to logits.

        Returns:
            dict with the individual scalar losses plus "total".
        """
        assert logits.shape[1] == 3, "expected (ET, ED, TC) channels"
        probs = torch.sigmoid(logits)
        p_et, p_ed, p_tc = probs[:, 0], probs[:, 1], probs[:, 2]
        p_wt = torch.clamp(p_et + p_ed + p_tc, max=1.0)

        device = logits.device
        zero = lambda: torch.zeros((), device=device)

        # ---- naive R-Super [4] baseline branch -------------------------------
        if self.w.loss_mode == "rsuper":
            l_vol_total = zero()
            l_cnt_total = zero()
            for b, cue in enumerate(cues):
                l_vol_total = l_vol_total + rsuper_volume_loss(
                    p_wt[b:b+1], cue.get("d_max_frac"))
                l_cnt_total = l_cnt_total + rsuper_count_loss(
                    p_wt[b:b+1], cue.get("n_qual"), self.w.threshold)
            B = max(len(cues), 1)
            l_vol = l_vol_total / B
            l_cnt = l_cnt_total / B
            total = self.w.w_size * l_vol + self.w.w_count * l_cnt
            return {
                "L_exist":  zero(),
                "L_size":   l_vol,     # symmetric total-volume (R-Super)
                "L_count":  l_cnt,     # symmetric count (R-Super)
                "L_prior":  zero(),
                "L_report": total,
            }
        # ---- MS-RSuper full loss (default) -----------------------------------

        l_exist_total = zero()
        l_size_total  = zero()
        l_count_total = zero()
        l_prior_total = zero()

        # per-sample because cue contents differ
        for b, cue in enumerate(cues):
            sub = cue.get("substruct", {}) or {}
            # ET (T1c)
            et = sub.get("ET") or {}
            l_exist_total = l_exist_total + existence_loss(
                p_et[b:b+1], et.get("state"), _lam(et),
                self.w.min_voxels_present)
            # ED (FLAIR) — D-22: skip low-confidence "present" cues to avoid
            # spurious false-positives in cohorts where ED labels are sparse.
            ed = sub.get("ED") or {}
            ed_lam = _lam(ed)
            ed_state = ed.get("state")
            # If the report says "present" but certainty is below the threshold,
            # treat it as null (no constraint) rather than forcing ED prediction.
            if (ed_state == "present"
                    and ed_lam < self.w.exist_ed_lambda_min):
                ed_state = None
            l_exist_total = l_exist_total + existence_loss(
                p_ed[b:b+1], ed_state, ed_lam,
                self.w.min_voxels_present)
            # TC (T1/T2)
            tc = sub.get("TC") or {}
            l_exist_total = l_exist_total + existence_loss(
                p_tc[b:b+1], tc.get("state"), _lam(tc),
                self.w.min_voxels_present)

            # global
            l_size_total = l_size_total + size_loss(
                p_wt[b:b+1], cue.get("d_max_frac"), self.w.threshold)
            l_count_total = l_count_total + count_loss(
                p_wt[b:b+1], cue.get("n_qual"), self.w.threshold)

            # prior
            mp = priors["parench"][b:b+1] if priors and "parench" in priors else None
            md = priors["dural"][b:b+1]   if priors and "dural"   in priors else None
            l_prior_total = l_prior_total + prior_loss(
                p_wt[b:b+1], cue.get("cohort"), mp, md)

        B = max(len(cues), 1)
        l_exist  = l_exist_total / B
        l_size   = l_size_total  / B
        l_count  = l_count_total / B
        l_prior  = l_prior_total / B

        total = (l_exist
                 + self.w.w_size  * l_size
                 + self.w.w_count * l_count
                 + self.w.w_prior * l_prior)

        return {
            "L_exist":  l_exist,
            "L_size":   l_size,
            "L_count":  l_count,
            "L_prior":  l_prior,
            "L_report": total,
        }
