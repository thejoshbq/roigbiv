# 06 — Diagnostic Results

## Experiment 2.A — Stage 1 rejection autopsy

Parsed `stage1_report.json` from `test_output/cli_verify/.../stage1/`. Evidence:

- 17/17 rejects cite **only** `area_high:+N` (N = 27..153 px over `max_area=350`). No reject fails any other criterion.
- 5/5 flags cite **only** `area_high:+N` (N = 1..13 px).
- Rejected-ROI mean `cellpose_prob` = **1.31** vs accepted-ROI mean = 0.71. Cellpose is *more* confident in the rejected cells than in the average accepted cell.
- Rejected-ROI median area = 421 px (max 503 px); rejected-ROI solidity ≥ 0.92 and contrast > 1.0 for most (these are well-shaped, high-contrast cells).
- Conclusion: Gate 1's `max_area=350` is the bottleneck.

## Experiment 2.D — Bypass Gate 1 visualization

Re-ran `run_cellpose_detection()` on saved `mean_M.tif` + `vcorr_S.tif` with exact production config. Reproduced **101 candidates exactly** matching `stage1_report.json` (79 accept / 5 flag / 17 reject after Gate 1).

Rendered `raw_candidates_overlay.png` and `rejects_flags_annotated.png`. Visual result: the 17 red-outlined reject ROIs and 5 yellow-outlined flag ROIs are clearly **the brightest, most obvious pyramidal neurons in the FOV** — the ones the user would immediately miss. Several have strong nuclear shadow and prominent soma-surround contrast. None look like fused blobs or artifacts.

## Experiment 2.B — Stock cyto3 on identical input

Ran stock `cyto3` (no fine-tuning) in four configurations on the same `mean_M`, `vcorr_S`:

| Config | Thresholds | Dual | Diameter | Candidates | Matches FT (IoU≥0.3) | Of 17 FT rejects, cyto3 also finds | Of 5 FT flags, cyto3 also finds | cyto3-only |
|---|---|---|---|---:|---:|---:|---:|---:|
| a: single permissive | cp=−2, flow=0.6 | no | 12 | **96** | 90 | **16/17** | **5/5** | 6 |
| b: dual permissive | cp=−2, flow=0.6 | yes | 12 | 25 | 22 | 7/17 | 1/5 | 3 |
| c: single default | cp=0, flow=0.4 | no | 12 | 81 | 79 | 15/17 | 5/5 | 2 |
| d: single auto-diameter | cp=−2, flow=0.6 | no | None (auto) | 118 | 88 | **17/17** | **5/5** | 30 |

Visual overlay (a): cyan stock-cyto3 outlines sit exactly on top of the red FT-reject outlines and yellow FT-flag outlines. Both models agree these are real cells.

Key conclusions:

- **Stock cyto3 recovers 16–17 of the 17 FT-rejected cells and all 5 flags.** This is the user's observed regression exactly: cells cyto3 catches are being hidden by the pipeline.
- **The fine-tuned model is not the bottleneck** — it also detects these 22 cells as raw candidates. Gate 1, not Cellpose, is removing them.
- **Dual-channel `[mean_M, vcorr_S]` is catastrophic for stock cyto3** (25 detections vs 96 single-channel). The fine-tuned model was trained on this exact input and handles it fine (101 candidates), so this is not a regression driver in production — only a caveat for baseline comparison.
- **Auto-diameter finds 30 additional cells beyond the 101-candidate set.** Suggests a separate recall improvement opportunity: the fixed `diameter=12` may be too narrow for this FOV's true cell-size distribution (range ~11–25 px). Out of scope for the immediate regression fix; flag as future work.

## Experiments 2.C, 2.E, 2.F

Not executed. Rationale:

- **2.C (denoise OFF):** skipped. Gate 1 is clearly the cause; denoise does not explain area_high rejections. If recall is still insufficient after the fix, this can be revisited.
- **2.E (epoch sweep):** skipped. Model is not the bottleneck per 2.B.
- **2.F (threshold sweep):** skipped. Threshold change alone doesn't address the Gate 1 area cutoff.
