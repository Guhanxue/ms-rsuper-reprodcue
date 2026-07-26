# MS-RSuper reproduction notes (UCSF / CLASP)

Notes prepared for review by the MS-RSuper authors.

- `MS_RSUPER_REPRODUCTION_NOTES.md` — how we pull BraTS2020, how the split is built,
  the provenance and content of the TextBraTS-derived report cues, the exact within-study
  R-Super baseline contract, and **which MS-RSuper loss terms our dataset can and cannot
  drive**, with measurements.
- `ms_rsuper_brain_adapter.py` — cue extraction + adapter from our 4-class softmax to
  the (ET, ED, TC, WT) probabilities the loss expects.
- `ms_rsuper_loss.py` — the exact third-party MS-RSuper reproduction loss used by
  the adapter. This is distinct from the original R-Super Volume+Ball helper used
  by our R-Super baseline (§3.1 of the notes).
- `test_ms_rsuper_brain.py` — validation over all 368 reports; verifies the
  minimum-one-component count cue, proves `L_prior` is exactly zero, and checks
  gradient flow and channel mapping.

The included loss is `ms_rsuper_train/losses/ms_rsuper_loss.py` from
https://github.com/jwkl0990-glitch/MS-R-Super at upstream commit
`7421fbc2a028127461e67a436a813224eb985839`. It is byte-identical, with SHA-256
`4ee4846e9d44c92d03953805cf5e4292bea74b535cb22b701d5520b757d4efee`.
The upstream repository provides no `LICENSE`/`COPYING` file and the source file
has no license header; it is included here solely to make the reviewed private
reproduction self-contained and remains attributable to its upstream authors.

The report root defaults to the UCSF staging path and can be overridden with
`MS_RSUPER_TEXT_ROOT`:

```bash
MS_RSUPER_TEXT_ROOT=/path/to/brain_data/text \
python test_ms_rsuper_brain.py
```

Open questions for the authors are in §4.1 and §4.2 of the notes. In particular,
the current cue extractor operationalizes omission from the enumerated template as
`absent`. That is an explicit experimental assumption awaiting the authors' answer,
not a settled interpretation of MS-RSuper.

The brain text is attributed to
[TextBraTS](https://papers.miccai.org/miccai-2025/paper/2164_paper.pdf), not to our
study. Section 3 records the remaining uncertainty about the exact compact-template
preprocessing step.
