# Push receipt and handover

Repository: `github.com/Guhanxue/ms-rsuper-reprodcue` (private).
Requested by Hanxue and pushed on 2026-07-26 so the MS-R-Super authors can review
our BraTS reproduction notes.

## Push status and credential facts

The initial commit was pushed successfully to `main` over HTTPS. Verified facts:

- `gh` CLI is not installed
- no `GITHUB_TOKEN` or `GH_TOKEN` is exposed in the environment
- Git's system configuration provides the `osxkeychain` credential helper; HTTPS
  authentication succeeded without exposing a credential
- global Git identity is configured as `Hanxue Gu <andy@sobek.ai>`
- `ssh -T git@github.com` returns `Permission denied (publickey)`, so the local key
  is not authorized for GitHub
- initial pushed commit: `62366d3bd71d3614b918e3ecf9c88ae4786ad8b8`

## What to push

The implementation and documentation from
`text_supervised_paper/ms_rsuper_share/` in the OneDrive project tree:

| file | contents |
|---|---|
| `README.md` | short index, points at the upstream loss by URL and sha |
| `MS_RSUPER_REPRODUCTION_NOTES.md` | the deliverable: our BraTS2020 setup, TextBraTS report provenance, split, exact R-Super baseline contract, MS-RSuper cue extraction, and which loss terms our data cannot drive |
| `ms_rsuper_loss.py` | exact third-party MS-RSuper reproduction loss used by the adapter; distinct from our original R-Super Volume+Ball baseline; SHA and attribution preserved |
| `ms_rsuper_brain_adapter.py` | cue extraction plus adapter from our 4-class softmax to the (ET, ED, TC, WT) probabilities their loss expects |
| `test_ms_rsuper_brain.py` | validation over all 368 reports, including the minimum-one-component count cue and proof that the anatomical prior is exactly zero here |
| `.gitignore` | excludes Python artifacts |
| `PUSH_HANDOVER_FOR_CODEX.md` | this push receipt and provenance boundary |

## Upstream loss provenance

Hanxue explicitly requested on 2026-07-26 that the exact loss be included so this
private review repository is self-contained. It is the upstream reproduction authors'
file, not our implementation. Preserve its bytes and attribution:

```
https://github.com/jwkl0990-glitch/MS-R-Super
upstream commit 7421fbc2a028127461e67a436a813224eb985839
ms_rsuper_train/losses/ms_rsuper_loss.py
sha256 4ee4846e9d44c92d03953805cf5e4292bea74b535cb22b701d5520b757d4efee
```

The upstream repository contains no `LICENSE`/`COPYING` file and the loss has no
license header. Do not remove the provenance statement or present this file as ours.

## Checks completed

- The cited upstream loss was cloned independently and its SHA-256 matched
  `4ee4846e9d44c92d03953805cf5e4292bea74b535cb22b701d5520b757d4efee`.
- The adapter test passed against that exact upstream file and all 368 reports:
  ET 339 present/29 omitted, ED 368/0, TC 368/0; all reports supplied
  `n_qual=1`; the minimum-count loss was live for an empty prediction;
  `L_prior` was exactly zero; size loss, gradients, and the absent-ET control passed.
- Secret scan over all tracked files: clean. No tokens, keys, passwords or credentials.
- Only internal path referenced is `/scratch/user/hagu`, a cluster path, which is
  harmless to share.
- No patient data, no case identifiers, no image files.
- `ms_rsuper_loss.py` is tracked byte-identical to the cited upstream SHA.

## Commands used

```bash
cd "<project>/text_supervised_paper/ms_rsuper_share"
git init -b codex/ms-rsuper-brats-share
git add .gitignore README.md MS_RSUPER_REPRODUCTION_NOTES.md \
  PUSH_HANDOVER_FOR_CODEX.md ms_rsuper_brain_adapter.py test_ms_rsuper_brain.py
git commit -m "Document R-Super and MS-RSuper BraTS reproduction limits"
git remote add origin https://github.com/Guhanxue/ms-rsuper-reprodcue.git
git push -u origin HEAD:main
```

## One unresolved interpretation

The notes ask the MS-R-Super authors two direct questions in section 4. Our conclusion
about how much of `L_exist` is active depends on the first answer: if omission from the
enumerated findings list means `absent`, 29/368 cases (7.9%) activate the ET-absence
branch; if only explicit negation means `absent`, none do. Do not present either branch
as settled until the authors reply.
