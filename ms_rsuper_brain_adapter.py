#!/usr/bin/env python3
"""MS-RSuper on BraTS2020: cue extraction + our-model adapter.

The loss math is NOT reimplemented here. The included `ms_rsuper_loss.py` is
byte-identical to the file in the MS-R-Super reproduction repository
(https://github.com/jwkl0990-glitch/MS-R-Super, ms_rsuper_train/losses/), which
cites Ge, Huang, Liu (arXiv:2602.20994). This module only

  (a) builds the per-sample cue dicts their loss expects, from our BraTS2020
      report text, and
  (b) converts our 4-class softmax output into the (ET, ED, TC, WT) probability
      maps their loss operates on,

then calls their term functions directly, so their formulas, weights and
reductions are used unchanged.

CHANNEL MAPPING.  Our brain model is a 4-class softmax with
CLASS_NAMES = {1: NCR, 2: ED, 3: ET} (see scripts/train_film_seg_textdir.py):

    p_ET = probs[:, 3]
    p_ED = probs[:, 2]
    p_TC = probs[:, 1] + probs[:, 3]        # BraTS tumour core = NCR + ET
    p_WT = 1 - probs[:, 0]                  # same definition the trainer uses

Their `MSRSuperLoss.forward` assumes three INDEPENDENT sigmoid channels; ours
are mutually exclusive softmax classes.  We therefore bypass `forward` and call
`existence_loss` / `size_loss` / `count_loss` / `prior_loss` with probabilities
we construct, which keeps every term's math verbatim while respecting our
architecture.  This is the one deliberate deviation and it is unavoidable
without retraining a region-based nnU-Net.

WHAT THIS DATASET CAN AND CANNOT DRIVE (measured over all 368 reports):

  L_exist  BARELY DRIVEABLE. Reports state edema (368/368), necrosis (364/368)
           and enhancing regions (339/368). The 29 omitted enhancing mentions
           drive an absence penalty only under the unresolved assumption that
           omission from this enumerated template means absent. Under an
           explicit-negation-only interpretation, the absence branch has no cues.
  L_size   DRIVEN, with a documented semantic caveat: their d_max_frac is the
           LARGEST COMPONENT's volume fraction, whereas our size cue is a
           WHOLE-TUMOUR volume quartile (thresholds.json). We pass the WT-derived
           fraction. Their naive-R-Super branch (loss_mode="rsuper") matches our
           cue semantics more exactly.
  L_count  WEAKLY DRIVEN. All 368 compact reports use the singular construction
           "A ... tumor", which supports the minimum-count cue n_qual=1. This
           enforces at least one predicted component but supplies no multifocal
           count. Multiple named lobes do not imply multiple lesions, and
           "fragmented" is a shape descriptor, not an exact component count.
  L_prior  STRUCTURALLY INERT.  Defined only for MEN (parenchymal mask) and MET
           (dural mask). BraTS2020 is glioma-only, so cohort is None and their
           prior_loss returns 0 *by their own code*, not by our omission.

Report this honestly: on BraTS2020 an MS-RSuper comparison exercises the size
family and, conditionally on the omission interpretation above, a limited part
of the existence family.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ms_rsuper_loss import (  # included byte-identical from the cited upstream repo
    MSRSuperWeights,
    count_loss,
    existence_loss,
    prior_loss,
    size_loss,
)

# Our BraTS2020 reports are templated and unhedged ("The tumor contains both
# necrosis and enhancing regions"), so certainty is 1.0 throughout. Their
# exist_ed_lambda_min=0.85 filter therefore never suppresses an ED cue here.
CERTAINTY = 1.0


def extract_brain_cues(text: str, size_target_voxels: Optional[float],
                       numel: float) -> Dict:
    """Build one MS-RSuper cue dict from a BraTS2020 templated report.

    `size_target_voxels` is the trainer's existing report-derived whole-tumour
    volume target (quartile midpoint); we convert it to their fraction convention.
    """
    t = (text or "").lower()
    has_enhancing = "enhancing" in t
    has_necrosis = "necrosis" in t
    has_edema = "edema" in t

    def state(flag: bool) -> Optional[str]:
        # Operational assumption for this reproduction: omission from the
        # enumerated component sentence means "absent". Whether MS-RSuper
        # intends omission or only explicit negation is unresolved; see §4.1
        # of MS_RSUPER_REPRODUCTION_NOTES.md. Under the latter interpretation,
        # false flags must map to None instead.
        return "present" if flag else "absent"

    return {
        "cohort": None,              # BraTS2020 is glioma-only -> L_prior == 0
        # Every compact report explicitly describes at least one tumour.
        # This is a minimum count, not an assertion of one connected component.
        "n_qual": 1,
        "d_max_frac": (float(size_target_voxels) / numel
                       if size_target_voxels else None),
        "substruct": {
            "ET": {"state": state(has_enhancing), "lambda_": CERTAINTY},
            "ED": {"state": state(has_edema), "lambda_": CERTAINTY},
            # tumour core = necrotic core + enhancing tumour
            "TC": {"state": state(has_necrosis or has_enhancing),
                   "lambda_": CERTAINTY},
        },
    }


class MSRSuperBrainLoss(nn.Module):
    """Apply the vendored MS-RSuper terms to our 4-class softmax brain model."""

    def __init__(self, weights: Optional[MSRSuperWeights] = None):
        super().__init__()
        self.w = weights or MSRSuperWeights()

    def forward(self, logits: torch.Tensor, cues: List[Dict],
                priors: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        """logits: [B, 4, D, H, W] softmax logits over (bg, NCR, ED, ET)."""
        if logits.shape[1] != 4:
            raise RuntimeError(
                f"expected 4-class brain logits (bg, NCR, ED, ET), got {logits.shape[1]}")
        probs = torch.softmax(logits, dim=1)
        p_ncr, p_ed, p_et = probs[:, 1], probs[:, 2], probs[:, 3]
        p_tc = torch.clamp(p_ncr + p_et, max=1.0)
        p_wt = 1.0 - probs[:, 0]

        device = logits.device
        zero = lambda: torch.zeros((), device=device)
        l_exist = zero(); l_size = zero(); l_count = zero(); l_prior = zero()

        for b, cue in enumerate(cues):
            sub = cue.get("substruct", {}) or {}

            et = sub.get("ET") or {}
            l_exist = l_exist + existence_loss(
                p_et[b:b + 1], et.get("state"),
                float(et.get("lambda_", 1.0)), self.w.min_voxels_present)

            ed = sub.get("ED") or {}
            ed_lam = float(ed.get("lambda_", 1.0))
            ed_state = ed.get("state")
            # their D-22 rule, applied verbatim
            if ed_state == "present" and ed_lam < self.w.exist_ed_lambda_min:
                ed_state = None
            l_exist = l_exist + existence_loss(
                p_ed[b:b + 1], ed_state, ed_lam, self.w.min_voxels_present)

            tc = sub.get("TC") or {}
            l_exist = l_exist + existence_loss(
                p_tc[b:b + 1], tc.get("state"),
                float(tc.get("lambda_", 1.0)), self.w.min_voxels_present)

            l_size = l_size + size_loss(
                p_wt[b:b + 1], cue.get("d_max_frac"), self.w.threshold)
            l_count = l_count + count_loss(
                p_wt[b:b + 1], cue.get("n_qual"), self.w.threshold)

            mp = priors["parench"][b:b + 1] if priors and "parench" in priors else None
            md = priors["dural"][b:b + 1] if priors and "dural" in priors else None
            l_prior = l_prior + prior_loss(p_wt[b:b + 1], cue.get("cohort"), mp, md)

        B = max(len(cues), 1)
        l_exist, l_size = l_exist / B, l_size / B
        l_count, l_prior = l_count / B, l_prior / B
        total = (l_exist
                 + self.w.w_size * l_size
                 + self.w.w_count * l_count
                 + self.w.w_prior * l_prior)
        return {"L_exist": l_exist, "L_size": l_size, "L_count": l_count,
                "L_prior": l_prior, "L_report": total}
