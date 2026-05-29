# 09 — Validation Results

## Fast validation (Stage 1 + Gate 1 only, reusing saved summary artifacts)

Command: `python diagnostics/regression/experiments/4_validate_fix/run.py`
Config: `cfg.max_area = 600` (was 350); all other defaults unchanged; denoise still on.

| Count | Prior (max_area=350) | Post-fix (max_area=600) | Δ |
|---|---:|---:|---:|
| Detected (Cellpose) | 101 | 101 | 0 |
| Accepted | 79 | **101** | **+22** |
| Flagged | 5 | 0 | −5 |
| Rejected | 17 | **0** | **−17** |
| `area_high` rejects | 17 | **0** | fixed |
| Other reject reasons | 0 | 0 | unchanged |

Previously-missed cells (17 rejects + 5 flags = 22 total, all spanning 351–503 px) are **all now accepted**. No new rejections on any other criterion. Zero rejects total — consistent with the earlier Phase 2 finding that no other gate criterion was actually firing on this FOV.

## Visual check

`diagnostics/regression/experiments/4_validate_fix/post_fix_overlay.png`: every visible neuron now carries a green accept outline, including all previously-red rejected cells (large pyramidal neurons in the left-center of the FOV). No red or yellow outlines remain.

## No new regressions (scope caveat)

- Only one test FOV is available (`T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1`). Validation is limited to this FOV. The `/mnt/external/ROIGBIV-DATA/` training-set FOVs were not re-evaluated — if the drive is mounted, user can run `python scripts/compare_models.py` for a broader sanity check before deploying.
- The change is a one-sided relaxation (larger cells allowed; smaller cells unchanged). By construction, it cannot remove any ROI that the prior pipeline accepted. It can only *add* previously-rejected large ROIs to the accept set.
- Downstream stages (subtraction, Stage 2/3/4, merging) will now subtract 22 additional ROIs before Stage 2. This may slightly change Stage 2–4 counts but in the expected direction: less residual signal for the fallback stages to chase, which is the pipeline's intended cascade behavior.

## End-to-end re-run (pending user execution)

Not executed here due to run time (the L+S SVD step on an 8624-frame stack takes several minutes). User can validate end-to-end with:

```bash
conda activate roigbiv
python -m roigbiv.cli run test_raw/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc.tif \
  --output-dir test_output/fix_verify --fs 30
```

Compare the resulting `test_output/fix_verify/stage1/stage1_report.json` against the prior `test_output/cli_verify/.../stage1/stage1_report.json`. Accept count should be 101, flag/reject 0 (matches the fast validation).
