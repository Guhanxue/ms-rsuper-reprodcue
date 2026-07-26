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
on disk. 368 of them carry TextBraTS-derived templated text (one case lacks it).

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

## 3. Provenance and content of the report cues — please read this part

**The report corpus comes from TextBraTS; it was not authored by our study.**
TextBraTS extends BraTS2020 with textual annotations. Its published creation pipeline
used GPT-4o to draft preliminary reports from videos of FLAIR and ground-truth slices,
followed by automated quality control and review by two radiologists, with disagreements
resolved by a third. The TextBraTS paper also evaluates standardized templated
representations of those annotations:

- paper: https://papers.miccai.org/miccai-2025/paper/2164_paper.pdf
- dataset: https://huggingface.co/datasets/Jupitern52/TextBraTS

Our experiments use a compact, 2–3 sentence TextBraTS-derived templated representation:

```
sent1: "A {size} {shape} tumor is located in the {laterality} {lobes}."
sent2: "There is {severity} edema surrounding the tumor core."
sent3: one of
       "The tumor contains both necrosis and enhancing regions."
       "The tumor contains necrosis."
       "The tumor contains an enhancing region."
```

**Unresolved preprocessing provenance.** These compact files do not match the current
public TextBraTS narrative files byte-for-byte (verified for cases 001--003). An earlier
project note attributed the compact rendering to `scripts/correct_text_v6.py`, but that
script is absent from the archived paper repository and could not be located. We
therefore cannot currently distinguish whether this exact compact representation was
supplied as a TextBraTS template variant or rendered locally from TextBraTS/BraTS-derived
attributes. We do not claim that our study created the underlying reports.

Consequences that matter for your loss:

- In the staged compact corpus, the cue audit finds 339 enhancing mentions, 368 edema
  mentions, and 368 tumour-core mentions. Without the missing preprocessing script, we
  should not claim an exact mask-to-text generation rule or perfect cue accuracy as
  verified facts.
- **Sentence 2 is emitted unconditionally**, so edema is stated as present in every case.
- There is no diameter in millimetres, explicit multifocal count, or cohort label.
  However, all 368 reports use the singular construction "A ... tumor", which safely
  establishes a minimum count of one (`n_qual=1`). It does **not** establish that the
  tumour is one continuous connected component: a single tumour may cross several
  lobes, and "fragmented" describes morphology rather than verified multiplicity.

Measured cue counts over all 368 reports:

| substructure | present | absent |
|---|---:|---:|
| ET | 339 | 29 |
| ED | **368** | **0** |
| TC | **368** | **0** |

### 3.1 How we implement the R-Super baseline

Our reported Brain **R-Super loss** row is a within-study implementation of the
report-derived loss principle, not a reproduction of R-Super's complete published
system. The executable contract is:

| item | our implementation |
|---|---|
| model | plain four-class 3D U-Net (`bg`, NCR, ED, ET); no FiLM or report input in the network |
| dense supervision | Dice--cross-entropy on random crops from the same 26 labeled cases |
| report supervision | 231 TextBraTS report cases, evaluated as padded full exams rather than random crops |
| prediction constrained | whole-tumour probability, `p_WT = 1 - p_background` |
| report size cue | compact size word mapped through the TextBraTS whole-tumour quartile thresholds; the bin midpoint is the target volume |
| spatial support | the report-derived full-exam envelope supplied as R-Super's allowed segment mask |
| report loss | R-Super Volume+Ball helper, weighted by `0.1` |
| optimization | 90 epochs, report batch size 2, seeds 42/1/2 |
| inference | image only; the report branch and loss are absent |

Mechanically, the size-bin midpoint is converted to a sphere-equivalent diameter.
R-Super's spherical/Gaussian convolution locates a candidate centre inside the report
envelope, and its Volume+Ball helper constructs the report-derived target and applies
the volume and pseudo-mask penalties to `p_WT`. The reference dense `conv3d` locator
would require kernels as large as approximately \(91^3\) voxels in this setting. We
replace only that detached locator computation with numerically equivalent
centrosymmetric FFT linear convolution followed by the identical ``same'' crop; the
loss definition and gradients after target construction are unchanged.

The matched labeled-only control uses the same model, split seed, optimizer, number of
epochs, full training roster, and evaluation, but sets `report_weight=0`; this makes the
report-dependent forward unreachable. The final Brain comparison therefore isolates
adding the R-Super report loss within this training recipe.

Two distinctions prevent overclaiming:

1. This row does **not** reproduce R-Super's Merlin initialization, larger segmentation
   model, or original training scale. It tests the loss mechanism on our common backbone
   and label budget.
2. It does **not** use `MSRSuperWeights(loss_mode="rsuper")` from the included
   third-party MS-RSuper reproduction. That convenience branch implements symmetric
   total-volume and count losses. Our reported R-Super row instead calls the original
   R-Super Volume+Ball helpers staged by
   `experiments/strict_lesion_eval/train_rsuper_nofilm.py` with `--rsuper-ball`.
   The trainer's older adapted volume+laterality fallback is also bypassed.

This adaptation has a task-specific limitation: a compact spherical target is an
imperfect model of an irregular or disconnected glioma, while the TextBraTS size word
is a whole-tumour volume bin rather than a measured lesion diameter. Multiple named
lobes are used to form a spatial envelope; they are not converted into multiple balls
or an exact lesion count.

---

## 4. Which of your loss terms we can and cannot drive

| term | status on our data | reason |
|---|---|---|
| `L_exist` | **barely driveable** | see below |
| `L_size` | driveable, with a semantic caveat | we have a whole-tumour volume quartile, not a largest-lesion diameter |
| `L_count` | **weakly driveable** | every report supports only the lower bound `n_qual=1`; no multifocal count is available |
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

Weakly driveable as a minimum-count constraint. All 368 compact reports begin with the
singular construction "A ... tumor", while none contains explicit multiple-lesion,
multifocal, or plural-lesion language. We therefore pass `n_qual=1`. This is valid for
the one-sided MS-RSuper objective: it requires at least one predicted component without
penalising additional components.

We do **not** infer one lesion per named lobe or assume one continuous tumour. One
contiguous tumour can cross several lobes, and "fragmented" (93/368) is a morphology
descriptor rather than a verified count. Consequently, this cue penalises only an empty
thresholded prediction. It provides no supervision about whether the case contains one
or multiple components and is largely redundant with foreground-existence supervision.
The validation test confirms that the term is positive with a non-zero gradient when no
component is predicted and zero once at least one component is present.

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

On BraTS2020 with these TextBraTS-derived compact templates, an MS-RSuper comparison
exercises **`L_size`, a minimum-one-component constraint, and an absence penalty active
on 7.9% of cases in one substructure under the omission-as-absence interpretation**.
`L_count` contains no multifocal information, and `L_prior` remains structurally zero.
We therefore do **not** believe a number produced on this dataset would exercise the
full MS-RSuper method, and we are not reporting one as such.

The reported R-Super baseline is documented in §3.1 and is **not** MS-RSuper. A separate
historical arm was labelled "ms-R-Super-inspired"; that arm is also not the authors'
method and must not be used as an MS-RSuper result. We keep these names and mechanisms
separate so neither can be read as a full MS-RSuper comparison.

**If you think MS-RSuper should be testable here, we would be glad to be corrected** —
in particular on the `absent`-versus-`None` question in §4.1, and on whether a
whole-tumour size cue is an acceptable substitute for `d_max`.

---

## 7. Files

| file | contents |
|---|---|
| `ms_rsuper_loss.py` | included upstream loss, byte-identical, sha256 `4ee4846e…`; source and license status documented above |
| `ms_rsuper_brain_adapter.py` | cue extraction + softmax→(ET,ED,TC,WT) adapter |
| `test_ms_rsuper_brain.py` | validation: weights, cue counts over all 368 reports, minimum-count liveness, proof that `L_prior` is exactly zero, gradient flow, channel-mapping sanity |
| `scripts/correct_text_v6.py` | previously cited compact-template preprocessing script; absent from the archived repository, so its role is not independently verifiable |

Attribution note: the training snapshot stages `scripts/rsuper_losses.py` from the
R-Super codebase without a license header. That file is not part of this small share
repository; its source attribution must remain attached wherever the full training
artifact is released.
