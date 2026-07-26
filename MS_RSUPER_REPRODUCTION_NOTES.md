# MS-RSuper on our BraTS2020 setup — reproduction notes

**Purpose.** We attempted to run MS-RSuper as a baseline. This documents exactly how our
brain data, split and report cues are constructed, and **states plainly which of the four
MS-RSuper loss terms our dataset cannot drive**. Written for review by the MS-RSuper
authors; corrections welcome.

Prepared 2026-07-26. All numbers below were measured on our copy of the data, not
estimated.

---

## 1. What we compared against

Loss implementation vendored **byte-identical** from the reproduction repository
`github.com/jwkl0990-glitch/MS-R-Super`, file `ms_rsuper_train/losses/ms_rsuper_loss.py`
(sha256 `4ee4846e9d44c92d03953805cf5e4292bea74b535cb22b701d5520b757d4efee`), which cites
Ge, Huang, Liu (arXiv:2602.20994). We did not modify the loss. Defaults confirmed as
shipped: `w_size=1.0`, `w_count=0.5`, `w_prior=0.2`, `threshold=0.5`,
`min_voxels_present=1.0`, `exist_ed_lambda_min=0.85`.

**Note:** that repository is itself a third-party reproduction, not the authors' code. If
an official implementation exists we would rather compare against it.
The upstream repository has no `LICENSE`/`COPYING` file and this source file has no
license header. It is included here for private reproduction review with its original
bytes, URL, commit (`7421fbc2a028127461e67a436a813224eb985839`) and SHA preserved.

---

## 2. Dataset

**Source.** BraTS 2020 training release, `MICCAI_BraTS2020_TrainingData`, **369 cases**
on disk. 368 of them carry our generated report text (one case lacks it).

Per case we use the four standard modalities plus `{case}_seg.nii`. Labels follow the
BraTS convention, remapped internally to `{1: NCR, 2: ED, 3: ET}` (our
`CLASS_NAMES`), with a 4-class softmax head (`0 = background`).

**This is a single-cohort glioma dataset.** We do not have BraTS-MEN or BraTS-MET.

**Split.** 26 labeled training cases (`--labeled-cases 26`), the remainder used
unlabeled/report-supervised, and a **frozen 56-exam test set**. Three seeds, 42/1/2.
Checkpoints are validation-selected; the test set is touched once per arm.

> **Limitation in this write-up:** the split-construction function
> (`create_brats3d_dataloaders` in `brats3d_dataset.py`) is not present in the code
> snapshot we archived for the paper, so we cannot paste its exact roster logic here. The
> resulting test roster is pinned by SHA in every evaluation record, and we can supply
> the case-ID list on request. We flag this rather than reconstruct it from memory.

**Evaluation.** Strict one-to-one lesion matching (maximum-cardinality Hungarian,
any-overlap admissibility, 3D 6-connectivity). Brain lesion metrics apply a **≥50-voxel
component filter to both prediction and ground truth**, because BraTS whole-tumour GT
fragments into ~17 components/exam of which 61% are exactly one voxel; the filter removes
92% of components for 0.04% of tumour volume. Exam-union Dice is computed **unfiltered**.

---

## 3. How our report cues are produced — please read this part

**Our "reports" are synthetic and derived from the ground-truth mask.** They are not
radiology reports. `scripts/correct_text_v6.py` reads `{case}_seg.nii` and
`{case}_flair.nii`, computes mask properties (per-class voxel counts, laterality, lobe
occupancy, a size quartile, a shape class), and emits a 2–3 sentence template:

```
sent1: "A {size} {shape} tumor is located in the {laterality} {lobes}."
sent2: "There is {severity} edema surrounding the tumor core."
sent3: one of
       "The tumor contains both necrosis and enhancing regions."
       "The tumor contains necrosis."
       "The tumor contains an enhancing region."
```

Consequences that matter for your loss:

- Presence statements are **oracle-derived**: "necrosis" appears iff `ncr_voxels > 0`,
  "enhancing" iff `et_voxels > 0`. Cue extraction is therefore noiseless — an
  optimistic setting for any report-supervised method, ours included.
- **Sentence 2 is emitted unconditionally**, so edema is stated as present in every case.
- There is no diameter in millimetres, no lesion count, and no cohort label.

Measured cue counts over all 368 reports:

| substructure | present | absent |
|---|---:|---:|
| ET | 339 | 29 |
| ED | **368** | **0** |
| TC | **368** | **0** |

---

## 4. Which of your loss terms we can and cannot drive

| term | status on our data | reason |
|---|---|---|
| `L_exist` | **barely driveable** | see below |
| `L_size` | driveable, with a semantic caveat | we have a whole-tumour volume quartile, not a largest-lesion diameter |
| `L_count` | **not driveable** | reports contain no lesion count |
| `L_prior` | **not driveable** | glioma-only cohort |

### 4.1 `L_exist` — the term we understand to be the main contribution

We were told the useful new term is `L_exist`. On our data it is close to inert, for two
independent reasons.

**(a) The `present` branch effectively never fires.** With `min_voxels=1.0` and a
128³ patch, `min_voxels/norm = 4.768e-7`, so `clamp(min_voxels/norm - V_k, min=0)` is
non-zero only when the predicted soft volume falls below **one voxel in 2,097,152**.
Measured:

| predicted tumour fraction | `L_exist(present)` | `L_exist(absent)` |
|---:|---:|---:|
| 5e-2 | **0** | 5.0e-2 |
| 1e-2 | **0** | 1.0e-2 |
| 1e-3 | 9.5e-4 | 1.0e-3 |

**(b) The `absent` branch has almost nothing to act on**, because our template states
edema and tumour core as present in 100% of cases. The only absence signal is ET, in
**29/368 cases (7.9%)**.

Furthermore, that 7.9% depends on a generous reading. Our template enumerates contents,
so we scored "no mention of enhancing" as `state="absent"`. Under the conservative
reading — no mention → `state=None`, which is what we would expect an LLM parser to emit
— **`L_exist` is identically zero on all 368 cases**.

**Question for the authors:** is `absent` intended only for explicit negation ("no
enhancement"), or also for omission from an enumerated findings list? Our conclusion
about testability flips on this.

### 4.2 `L_size`

Driveable but not identical to yours. Your `d_max_frac` is the **largest connected
component's** volume fraction, parsed from a reported diameter. We have a **whole-tumour**
volume quartile (thresholds `[5000, 15000, 40000]` voxels, midpoint as target), converted
to a fraction. We pass that to `size_loss` unchanged. Your naive-R-Super branch
(`loss_mode="rsuper"`, symmetric total-volume) actually matches our cue semantics more
closely than `L_size` does.

### 4.3 `L_count`

Not driveable. Our reports never state multiplicity. "fragmented" (93/368) is a shape
descriptor produced by `classify_shape`, not a lesion count. We pass `n_qual=None`, and
your `count_loss` returns zero — verified numerically.

### 4.4 `L_prior`

Not driveable. Your prior penalises `p_wt` inside a parenchymal mask for MEN and inside a
dural mask for MET. BraTS2020 is glioma-only, so `cohort=None` and your `prior_loss`
returns zero **by your own code path**, not by an omission on our side. We also have no
MNI152 compartment masks in this pipeline.

---

## 5. Adaptation we had to make, and one bug we hit

**Architecture.** Your loss expects `[B, 3, D, H, W]` logits as (ET, ED, TC) with
independent sigmoids. Our brain model is a 4-class softmax. Rather than reimplement your
math, we call your term functions directly with probabilities built from our softmax:

```
p_ET = probs[:, 3]
p_ED = probs[:, 2]
p_TC = clamp(probs[:, 1] + probs[:, 3], max=1)     # NCR + ET
p_WT = 1 - probs[:, 0]
```

This is our only deliberate deviation from your implementation. If you consider it
invalid, we would rather hear that than publish it.

**Dependency bug worth flagging.** `cc3d` is an optional import in your file, and the
fallback treats *all foreground as a single component*. With `cc3d` absent, `size_loss`
silently degenerates from largest-component volume to total volume — i.e. it becomes your
naive R-Super baseline — and `count_loss` always sees one component. We hit this and
installed `connected-components-3d 4.0.0`. A hard failure or a warning might save others
the same silent degradation.

---

## 6. What we conclude, and what we are not claiming

On BraTS2020 with GT-derived templated reports, an MS-RSuper comparison exercises
**`L_size` plus an absence penalty active on 7.9% of cases in one substructure**, with
`L_count` and `L_prior` structurally zero. We therefore do **not** believe a number
produced on this dataset would say anything meaningful about MS-RSuper, and we are not
reporting one as such.

A prior arm in our tables was labelled "ms-R-Super-inspired". That arm is **not** your
method: it is R-Super's Volume+Ball loss applied across our four BraTS classes, at
`report_weight=0.1`, with no existence, count or prior term. We are correcting the
wording so it cannot be read as an MS-RSuper comparison.

**If you think MS-RSuper should be testable here, we would be glad to be corrected** —
in particular on the `absent`-versus-`None` question in §4.1, and on whether a
whole-tumour size cue is an acceptable substitute for `d_max`.

---

## 7. Files

| file | contents |
|---|---|
| `ms_rsuper_loss.py` | included upstream loss, byte-identical, sha256 `4ee4846e…`; source and license status documented above |
| `ms_rsuper_brain_adapter.py` | cue extraction + softmax→(ET,ED,TC,WT) adapter |
| `test_ms_rsuper_brain.py` | validation: weights, cue counts over all 368 reports, proof that `L_count`/`L_prior` are exactly zero, gradient flow, channel-mapping sanity |
| `scripts/correct_text_v6.py` | our report generator (GT-derived templates) |

Attribution note: our repository also vendors `rsuper_losses.py` (2,090 lines) from
R-Super without a license header. We are adding attribution before release.
