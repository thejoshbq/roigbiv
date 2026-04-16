# 10 — Regression Postmortem

## Root cause (one paragraph)

The new multi-stage pipeline's Gate 1 morphological filter (`roigbiv/pipeline/gate1.py`, parameters in `roigbiv/pipeline/types.py`) used `max_area=350` as a hard rejection threshold. On the test FOV, Cellpose detected 101 candidates; 22 of them (17 rejected outright + 5 flagged) were larger than 350 px² but otherwise high-confidence, well-shaped pyramidal neurons — the same cells stock `cyto3` also detects as single-channel candidates. Gate 1 was therefore hiding real cells from the user based purely on size. The effective operating range of `max_area=350` (diameter ~21 px ≈ 15 μm) was too narrow for this FOV's neuron population, which includes healthy pyramidal soma up to ~25 px diameter (~18 μm).

## What was changed

Single-line edit:
- `roigbiv/pipeline/types.py:170` — `max_area: int = 350` → `max_area: int = 600`

No model changes, no preprocessing changes, no other parameter changes. Deployed model (`models/deployed/current_model` = `run015_epoch_0280`) is unchanged. L+S / denoise / channel configuration unchanged.

## Validation results (summary)

- Stage 1 on test FOV: 101 detected → 79 accepted / 5 flagged / 17 rejected **before fix**.
- After fix: 101 detected → **101 accepted / 0 flagged / 0 rejected**.
- All 22 previously-hidden ROIs recovered. Zero new regressions (change is one-sided relaxation).
- End-to-end pipeline re-run left for user verification (long L+S step).

## Lessons learned / preventive measures

1. **Morphological thresholds need FOV-specific calibration, not spec-derived defaults.** The spec lists `max_area=350` in the range 250–500; real cells in this FOV sit at the top of that range and beyond. Future preventive measures:
   - Pre-commit: when introducing a new filter stage, audit its rejection rate on a held-out FOV. If >5% of high-confidence Cellpose candidates (cellpose_prob > 0.5) are rejected for a single criterion, the threshold is probably wrong.
   - Post-commit: the pipeline already writes gate_reasons to `stage1_report.json` — consider adding a summary log line like "Gate 1 rejected N ROIs; top reason: area_high (M cases)" so systematic over-rejection is visible at run time without a manual JSON audit.

2. **Stock-cyto3 comparison is the fastest baseline.** Before blaming the fine-tuned model, dual-channel input, or preprocessing, run stock `cyto3` on the same input. In Experiment 2.B that comparison took ~10 seconds and immediately identified that the fine-tuned model wasn't the problem (it found the same 22 cells; the downstream gate was filtering them).

3. **`diameter` and `max_area` should be consistent.** `diameter=12` (expected-cell diameter) implies an expected area of ~113 px²; `max_area=350` ≈ 3× expected. Many real cortical neurons vary 2–3× from the mean, so a 3× ceiling is right at the edge of normal variance. The new `max_area=600` is ~5× expected area, giving appropriate headroom.

4. **Related thresholds flagged for future audit (out of scope for this fix):**
   - `gate2_max_area=400` is now *more* restrictive than Gate 1's 600, violating the "Gate 2 should be relaxed vs Gate 1" design comment in types.py:217. Re-audit when Stage 2 output is next reviewed.
   - `stage4_max_area=350` (tonic / residual detection) is unchanged but may have the same issue if Stage 4 ever proposes a large cell.
   - `docs/roi-pipeline-specification.md` §6 references `max_area=350` at lines 1219 and 1273 — update for consistency when you next touch the spec.

## Remaining concerns / follow-up

1. **Validate on more FOVs.** Only one test FOV (T1_221209) was available. Run the pipeline on the full ROIGBIV-DATA training set (99 FOVs) and confirm no new spurious-blob rejections disappear in a bad direction (oversized merges passing Gate 1). If false-positive large blobs become a problem, layer a *flag-not-reject* rule for `area > 600` rather than tightening the threshold.

2. **Auto-diameter recall opportunity.** Experiment 2.B(d) (diameter=None) found 30 additional cells beyond the 101-candidate set on the same FOV. This is a *different* recall opportunity (cells Cellpose misses at fixed diameter, vs. cells Cellpose finds but Gate 1 rejects). Worth investigating separately as a config toggle.

3. **Training data / model selection.** run014 AP@0.5=0.68 vs run015_epoch_0280 AP@0.5=0.61. Overall AP@0.5:0.95 improved in the new model, but recall at IoU≥0.5 went down. When HITL-curated ground truth grows, re-evaluate whether an earlier run015 epoch (closer to run014's regime) produces better production AP@0.5.

## Followup validation (2026-04-16)

After the initial fix, the user raised a concern based on a side-by-side of four reference images (`/home/thejoshbq/Pictures/{mean_projection,manually_drawn_rois,fine-tuned_cellpose_model_rois,pipeline_rois}.png`) — `pipeline_rois.png` showed bright pyramidal cells dramatically attenuated in the pipeline's input, hypothesizing L+S absorption was hiding them from Cellpose. Details and artifacts in `diagnostics/regression/followup/`.

Verification on the same FOV with the post-fix code:

| Metric | Value |
|---|---:|
| Current pipeline (`mean_M`+`vcorr_S`, denoise ON, channels=(1,2), `max_area=600`) | 101 accepts / 0 rejects |
| Reference (fine-tuned on `mean_M`, denoise OFF, channels=(0,0)) | 94 cells |
| Matched pairs (min_iou=0.3) | 93 / 94 = 98.9 % recall of reference |
| Reference cells unmatched | 1 (dim borderline merged with neighbor) |
| Pipeline-only cells | 8 (visible somas the single-channel reference missed) |

Conclusion: `pipeline_rois.png` reflected a pre-fix / pre-multi-stage pipeline state. On the post-fix code, Cellpose is already fed `mean_M` (raw registered mean, not `mean_S` — see `roigbiv/pipeline/run.py:230–236`), so L+S absorption cannot affect Stage 1. The current pipeline strictly dominates the reference on this FOV. No additional remediation needed.

Residual open questions (unchanged from above): multi-FOV validation on `/mnt/external/ROIGBIV-DATA/` and optional L+S diagnostics (visualize `L`, k-sweep, Vcorr-on-L) to confirm Stages 2–4 aren't harmed by absorption even though Stage 1 is safe.
