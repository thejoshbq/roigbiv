# 03 — Data Inventory

## Raw test data

- `test_raw/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc.tif` — 4.3 GB, motion-corrected (`_mc` suffix)
- Shape (from pipeline_log.json): 8624 × 505 × 493
- Acquisition: GCaMP6s, PrL-NAc projection target, fs=30 Hz, tau=1.0
- Only one FOV is present — all diagnostic experiments use this FOV as the test case.

## Training data (external)

- Per `scripts/compare_models.py` defaults: `/mnt/external/ROIGBIV-DATA/cellpose_ready/{annotated,masks}/`
- Not checked into the project tree. Whether this mount is currently available is unverified; will check if Experiment 2.E is reached.

## Ground-truth ROI exports

- `RoiSet.zip` / `*.roi` files — searched under `~/Otis-Lab`; none found for the test FOV per user ("No — use visual inspection only").
- `data/raw/`, `data/annotated/`, `data/masks/`, etc. — directories not populated locally (config references them but training pulls from the external drive).

## Pipeline outputs available for comparison

Two prior runs of the new pipeline exist:

### Run 1: `test_output/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002/`
- Pipeline executed 2026-04-15 15:57 UTC, fs=7.5 (incorrect)
- Stage 1: 101 detected → 79 accepted / 5 flagged / 17 rejected
- Final: 109 ROIs

### Run 2: `test_output/cli_verify/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002/`
- Pipeline executed 2026-04-15 21:21 UTC, fs=30 (correct)
- **Stage 1: 101 detected → 79 accepted / 5 flagged / 17 rejected** (identical to Run 1)
- Final: 138 ROIs (more in downstream stages due to fs correction)

**Cellpose Stage 1 output is identical across both runs**, as expected — Stage 1 doesn't use fs. Downstream Stage 2/3/4 differed. Our regression is Stage 1 scoped.

### Reusable artifacts under `test_output/cli_verify/.../`

- `summary/mean_M.tif` — raw morphology image (505×493, float32) — what Cellpose receives as channel 1 input
- `summary/vcorr_S.tif` — 8-neighbor temporal correlation map (channel 2 input)
- `summary/mean_S.tif`, `max_S.tif`, `std_S.tif`, `dog_map.tif`, `mean_L.tif`
- `stage1/stage1_probs.tif` — continuous cellprob map (raw Cellpose output)
- `stage1/stage1_masks.tif` — labeled mask (accept + flag only, NOT rejects)
- `stage1/stage1_report.json` — full per-ROI feature values and gate outcomes for all 101 candidates

The Foundation artifacts (L+S, summary) are the expensive compute step (minutes on this FOV). Diagnostic experiments will **reuse** these and only re-run Stage 1.
