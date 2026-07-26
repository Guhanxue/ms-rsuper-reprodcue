# Push handover for Codex

Task: push this folder to `github.com/Guhanxue/ms-rsuper-reprodcue` (private).
Requested by Hanxue 2026-07-26 so the MS-R-Super authors can review our reproduction
notes.

## Why I could not push it

No GitHub credentials are reachable from my environment. Verified, not assumed:

- `gh` CLI is not installed
- no `GITHUB_TOKEN` or `GH_TOKEN` in the environment, and no git credential helper
- no global git user.name or user.email configured
- `ssh -T git@github.com` returns `Permission denied (publickey)`, so the local key
  `~/.ssh/id_ed25519.pub` is not registered on that account

## What to push

Everything in this folder, which is
`text_supervised_paper/ms_rsuper_share/` in the OneDrive project tree.

| file | contents |
|---|---|
| `README.md` | short index, points at the upstream loss by URL and sha |
| `MS_RSUPER_REPRODUCTION_NOTES.md` | the deliverable: our BraTS2020 pull, split, cue generation, and which loss terms our data cannot drive |
| `ms_rsuper_brain_adapter.py` | cue extraction plus adapter from our 4-class softmax to the (ET, ED, TC, WT) probabilities their loss expects |
| `test_ms_rsuper_brain.py` | validation over all 368 reports, including proof that count and prior are exactly zero here |

## What NOT to push, deliberately

Do not add `ms_rsuper_loss.py`. It is the MS-R-Super authors' file, vendored byte
identical, and it carries no license header. Copying it into Hanxue's repository is an
attribution problem we do not need, and the authors already have their own code. The
notes reference it by URL and sha256 instead:

```
https://github.com/jwkl0990-glitch/MS-R-Super
ms_rsuper_train/losses/ms_rsuper_loss.py
sha256 4ee4846e9d44c92d03953805cf5e4292bea74b535cb22b701d5520b757d4efee
```

If a reviewer asks for it, point them at that URL rather than committing a copy.

## Checks already done

- Secret scan over all four files: clean. No tokens, keys, passwords or credentials.
- Only internal path referenced is `/scratch/user/hagu`, a cluster path, which is
  harmless to share.
- No patient data, no case identifiers, no image files.

## Commands

```bash
cd "<project>/text_supervised_paper/ms_rsuper_share"
git init -b main
git add -A
git commit -m "MS-RSuper reproduction notes: BraTS2020 setup, split, and non-reproducible loss terms"
git remote add origin git@github.com:Guhanxue/ms-rsuper-reprodcue.git
git push -u origin main
```

If SSH is not set up on the pushing machine either, switch the remote to HTTPS and
authenticate with a personal access token:

```bash
git remote set-url origin https://github.com/Guhanxue/ms-rsuper-reprodcue.git
```

If the repository already has commits, use `git pull --rebase origin main` before
pushing rather than forcing.

## One thing worth flagging to Hanxue after the push

The notes ask the MS-R-Super authors two direct questions in section 4. Our conclusion
that their method is not testable on BraTS2020 depends on the answer to the first one, so
the note should not be presented to co-authors as settled until they reply.
