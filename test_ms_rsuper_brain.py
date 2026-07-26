#!/usr/bin/env python3
"""Validate the MS-RSuper brain adapter against the vendored loss and real reports."""
from __future__ import annotations

import glob
import os
import sys

import torch

from ms_rsuper_brain_adapter import MSRSuperBrainLoss, extract_brain_cues
from ms_rsuper_loss import MSRSuperWeights

TEXT_ROOT = os.environ.get(
    "MS_RSUPER_TEXT_ROOT",
    "/scratch/user/hagu/clasp_clean/final_clasp_20260721/brain_data/text",
)
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL += 1


print("=== 1. defaults match the published MS-RSuper weights ===")
w = MSRSuperWeights()
check("w_size == 1.0", w.w_size == 1.0, str(w.w_size))
check("w_count == 0.5", w.w_count == 0.5, str(w.w_count))
check("w_prior == 0.2", w.w_prior == 0.2, str(w.w_prior))
check("threshold == 0.5", w.threshold == 0.5, str(w.threshold))
check("exist_ed_lambda_min == 0.85", w.exist_ed_lambda_min == 0.85, str(w.exist_ed_lambda_min))

print("\n=== 2. cue extraction over ALL real reports ===")
files = sorted(glob.glob(os.path.join(TEXT_ROOT, "*", "*_flair_text.txt")))
print(f"  reports found: {len(files)}")
if not files:
    raise RuntimeError(
        f"no reports found under {TEXT_ROOT!r}; set MS_RSUPER_TEXT_ROOT to "
        "the BraTS report tree"
    )
states = {"ET": {"present": 0, "absent": 0}, "ED": {"present": 0, "absent": 0},
          "TC": {"present": 0, "absent": 0}}
n_min_one_cues = n_cohort_cues = 0
for f in files:
    cue = extract_brain_cues(open(f).read(), 20000.0, 128 ** 3)
    for k in states:
        states[k][cue["substruct"][k]["state"]] += 1
    n_min_one_cues += cue["n_qual"] == 1
    n_cohort_cues += cue["cohort"] is not None
for k, v in states.items():
    print(f"    {k}: present={v['present']}  absent={v['absent']}")
check("ED present in every report", states["ED"]["present"] == len(files))
check("ET has BOTH present and absent cues", states["ET"]["present"] > 0 and states["ET"]["absent"] > 0,
      f"{states['ET']['present']}/{states['ET']['absent']}")
check("every report yields minimum count n_qual=1",
      n_min_one_cues == len(files), f"{n_min_one_cues} cues")
check("no report yields a cohort cue", n_cohort_cues == 0, f"{n_cohort_cues} cues")

print("\n=== 3. minimum count is satisfied by ordinary foreground; prior is zero ===")
torch.manual_seed(0)
logits = torch.randn(2, 4, 16, 24, 24, requires_grad=True)
cues = [extract_brain_cues(open(files[i]).read(), 20000.0, 16 * 24 * 24) for i in (0, 1)]
loss = MSRSuperBrainLoss()
out = loss(logits, cues)
for k, v in out.items():
    print(f"    {k} = {float(v):.8f}")
check("L_count == 0 when >=1 component is predicted", float(out["L_count"]) == 0.0)
check("L_prior == 0 exactly", float(out["L_prior"]) == 0.0)
# L_exist is legitimately 0 here: with min_voxels=1.0 the "present" term only
# fires when predicted soft volume drops below ~1 voxel of the whole patch, and
# random logits predict far more than that. Term liveness is proven in test 5.
check("L_exist == 0 for satisfied present-cues", float(out["L_exist"]) == 0.0)
check("L_size > 0 (term is live)", float(out["L_size"]) > 0.0)

print("\n=== 4. gradient flows into the prediction ===")
out["L_report"].backward()
g = logits.grad
check("gradient is finite", bool(torch.isfinite(g).all()))
check("gradient is non-zero", float(g.abs().sum()) > 0, f"{float(g.abs().sum()):.4f}")

print("\n=== 5. channel mapping sanity: absent-ET cue penalises ET mass ===")
lg = torch.full((1, 4, 8, 8, 8), -6.0)
lg[:, 3] = 6.0            # force nearly all mass into the ET channel
cue_absent = [{"cohort": None, "n_qual": None, "d_max_frac": None,
               "substruct": {"ET": {"state": "absent", "lambda_": 1.0},
                             "ED": {"state": None, "lambda_": 1.0},
                             "TC": {"state": None, "lambda_": 1.0}}}]
cue_present = [{"cohort": None, "n_qual": None, "d_max_frac": None,
                "substruct": {"ET": {"state": "present", "lambda_": 1.0},
                              "ED": {"state": None, "lambda_": 1.0},
                              "TC": {"state": None, "lambda_": 1.0}}}]
la = float(loss(lg, cue_absent)["L_exist"])
lp = float(loss(lg, cue_present)["L_exist"])
print(f"    ET-absent cue -> L_exist={la:.6f}   ET-present cue -> L_exist={lp:.6f}")
check("absent cue penalises a saturated ET prediction", la > lp)
check("present cue is ~0 when ET is saturated", lp < 1e-6)

print("\n=== 6. minimum-count cue is live when no component is predicted ===")
lg_empty = torch.full((1, 4, 8, 8, 8), -6.0, requires_grad=True)
with torch.no_grad():
    lg_empty[:, 0] = 6.0       # force background; no WT component at threshold 0.5
cue_min_one = [{"cohort": None, "n_qual": 1, "d_max_frac": None,
                "substruct": {"ET": {"state": None, "lambda_": 1.0},
                              "ED": {"state": None, "lambda_": 1.0},
                              "TC": {"state": None, "lambda_": 1.0}}}]
out_empty = loss(lg_empty, cue_min_one)
lc = out_empty["L_count"]
print(f"    empty prediction -> L_count={float(lc):.6f}")
check("minimum-count cue penalises zero predicted components", float(lc) > 0.0)
lc.backward()
check("minimum-count penalty has non-zero gradient",
      float(lg_empty.grad.abs().sum()) > 0.0,
      f"{float(lg_empty.grad.abs().sum()):.6f}")

print(f"\n{'ALL_MS_RSUPER_ADAPTER_TESTS_PASS' if FAIL == 0 else f'{FAIL} CHECKS FAILED'}")
sys.exit(1 if FAIL else 0)
