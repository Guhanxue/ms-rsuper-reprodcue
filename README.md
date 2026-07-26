# MS-RSuper reproduction notes (UCSF / CLASP)

Notes prepared for review by the MS-RSuper authors.

- `MS_RSUPER_REPRODUCTION_NOTES.md` — how we pull BraTS2020, how the split is built,
  how our report cues are produced, and **which of the four loss terms our dataset
  cannot drive**, with measurements.
- `ms_rsuper_brain_adapter.py` — cue extraction + adapter from our 4-class softmax to
  the (ET, ED, TC, WT) probabilities the loss expects.
- `test_ms_rsuper_brain.py` — validation over all 368 reports; proves `L_count` and
  `L_prior` are exactly zero on this dataset.

The loss itself is **not** copied here. We used
`ms_rsuper_train/losses/ms_rsuper_loss.py` from
https://github.com/jwkl0990-glitch/MS-R-Super byte-identical,
sha256 `4ee4846e9d44c92d03953805cf5e4292bea74b535cb22b701d5520b757d4efee`.

To run the validation, obtain that upstream file separately, verify its SHA-256,
and expose its parent directory on `PYTHONPATH`; do not copy it into this repository.
The report root defaults to the UCSF staging path and can be overridden with
`MS_RSUPER_TEXT_ROOT`:

```bash
PYTHONPATH=/path/to/upstream/losses:$PYTHONPATH \
MS_RSUPER_TEXT_ROOT=/path/to/brain_data/text \
python test_ms_rsuper_brain.py
```

Open questions for the authors are in §4.1 and §4.2 of the notes. In particular,
the current cue extractor operationalizes omission from the enumerated template as
`absent`. That is an explicit experimental assumption awaiting the authors' answer,
not a settled interpretation of MS-RSuper.
