# 08 — Remediation Plan

## Selected branch

**Branch:** `postprocessing_filter` → **Option A: Relax the filter threshold.**

The filter rejecting real cells is Gate 1's `max_area=350`. Raise the threshold to cover the actual cell-size distribution in this FOV type (GRIN 2P, GCaMP6s, cortical pyramidal).

Rationale for threshold choice: rejected cells span 377–503 px (diameter ~22–25 px, ~16–18 μm at typical 2P-GRIN pixel sizes — within the normal range of cortical pyramidal soma). Moving `max_area` to **600** covers all observed cells with a small margin, while still rejecting obvious multi-cell merges (area > 600 = diameter > 27 px ≈ 20 μm, which would be unusually large for a single soma).

Not changed: `min_area=80`, solidity/eccentricity/contrast thresholds, DoG conjunctive rule. These are not contributing to the regression and the data does not justify changing them in this commit.

## Exact edits

### Edit 1 — `roigbiv/pipeline/types.py`

Line 170:
```
- max_area: int = 350
+ max_area: int = 600
```

That is the entire code change for the regression fix.

### Edit 2 — `configs/pipeline.yaml` (if it overrides `max_area`)

Verify this file does not hard-code `max_area: 350`. If it does, update to `600` to match. If not, no change needed.

## Expected outcome

- Stage 1 on the test FOV: 101 detected → **all ~96–101 pass Gate 1 area check** (none rejected for `area_high`). A handful may still fail other criteria (solidity, eccentricity) — unlikely per the observed data where only `area_high` fired.
- The specific 17 rejects + 5 flags identified in 05 are recovered.
- No new false positives introduced (threshold is relaxed in one direction only).

## Validation plan

1. **Fast validation (sub-second):** reuse `mean_M.tif`, `vcorr_S.tif`, `dog_map.tif` from existing `test_output/cli_verify/.../summary/`. Re-run Stage 1 + Gate 1 only, using the fixed config. Confirm reject count drops from 17 to ~0 and accept count rises.
2. **Full-pipeline validation:** user can re-run `python -m roigbiv.cli run test_raw/...` end-to-end to confirm downstream stages (subtraction, Stage 2/3/4, merging) also behave correctly with the larger accept set.
3. **Visual check:** render `mean_M` overlay with the new gate outcomes and compare against `2D_bypass_gate1/raw_candidates_overlay.png`. All previously-red outlines should now be green.

## Rollback plan

Single-line revert: `git revert <fix-commit>` or edit `max_area` back to 350. No data, model, or downstream artifact is altered by this change.

## Follow-up work (separate branch / Trello card, not this fix)

- **Auto-diameter recall opportunity.** Experiment 2.B(d) showed `diameter=None` finds 30 more cells beyond the 101-candidate set. Worth evaluating as a separate opt-in flag (`cfg.diameter: int | None = 12`, with None meaning auto).
- **Training data audit.** run014 was AP@0.5=0.68; run015_epoch_0280 is 0.61. Overall AP@0.5:0.95 improved but AP@0.5 dropped. If the HITL-curated ground truth grows, re-evaluate whether an earlier run015 epoch has better AP@0.5 for production.
- **Spec update.** The spec (`docs/roi-pipeline-specification.md` §6) hard-codes `max_area=350`. After the fix, update the spec to reflect the new value and the rationale.
