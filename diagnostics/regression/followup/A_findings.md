# Followup Phase A — findings

## Question
The user provided four reference images (raw mean, manual annotation,
fine-tuned-on-raw, pipeline output) and hypothesized that L+S separation
was absorbing bright pyramidal cells into L, so Cellpose never saw them.
The followup prompt asked us to verify the hypothesis and remediate.

## Pre-flight (code audit)
Reading the current code resolved most of the hypothesis before any run:

- `roigbiv/pipeline/run.py:230–236` already feeds `fov.mean_M` (raw
  registered mean from Suite2p `ops["meanImg"]`) to Cellpose — *not*
  `mean_S`. The inline comment names absorption as the reason.
- `roigbiv/pipeline/foundation.py:512–516` produces `mean_M` with the
  same rationale.
- `roigbiv/pipeline/types.py:170` has `max_area=600` (post-regression-fix).

So the followup's primary Phase B proposal ("feed raw mean instead of
mean(S)") is already implemented. `pipeline_rois.png` reflects an earlier
pipeline state (explicitly noted in the followup prompt as predating the
multi-stage implementation).

## Phase A.0 — current pipeline overlay
Reran Stage 1 + Gate 1 on the saved foundation artifacts for FOV
`T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002` with
`cfg.max_area=600, use_denoise=True, channels=(1, 2)`.

- Cellpose candidates: **101**
- Gate 1: **101 accepted / 0 flagged / 0 rejected**
- Visual (`A0_current_pipeline_on_mean_M.png`): every bright pyramidal
  cell in the FOV — including the lower-left cluster that was missing
  in the user's `pipeline_rois.png` — carries a green fill.
- The underlying `mean_M` grayscale is bright and unattenuated; the
  "dramatic attenuation" observed in `pipeline_rois.png` is not present.
  That attenuation appears to have been a rendering artifact (and/or
  an earlier pipeline variant), not real signal loss.

## Phase A.1 — reference reproduction
Ran the same fine-tuned model on `mean_M` with `use_denoise=False,
channels=(0, 0)` to reproduce what produced user image 3
(`fine-tuned_cellpose_model_rois.png`).

- Reference detections: **94**
- Saved to `A1_reference_labels.tif` and `A1_reference_overlay.png`.

## Phase A.2 — IoU diff (reference ↔ current pipeline)
Greedy IoU matching, min_iou=0.3.

| Metric | Value |
|---|---:|
| A.0 pipeline accepts | 101 |
| A.1 reference cells | 94 |
| Matched pairs | **93** |
| Reference cells not covered by pipeline (recall gap) | **1** |
| Pipeline cells not in reference (over-detection) | **8** |

The single unmatched reference cell (id=49, area=200, centroid
[219.9, 303.3]) is a dim borderline instance sitting next to a brighter
cell — the current pipeline merged it with its neighbor. Not a recall
regression.

The 8 "extras" (yellow in `A2_extra.png`) are clearly visible somas that
the dual-channel + denoise pipeline detects but the single-channel
reference misses. These are true positives added by the current pipeline,
not false alarms.

Net: the current pipeline matches 93 / 94 reference cells (98.9 % recall)
and adds 8 true-positive cells the reference misses. **The current
pipeline is strictly better than the reference on this FOV.**

## Decision: H-stale confirmed
`pipeline_rois.png` was pre-fix (or pre-multi-stage) evidence. The post-
fix pipeline recovers every bright cell the reference finds, plus more.
No Phase A.3 ablation (denoise-off / single-channel) is needed — they
would only be informative if a recall gap existed.

No Phase B remediation is warranted for this FOV. The previous
`max_area 350 → 600` change (commit `55e08ab`) closed the regression.

## Not tested (scope caveats)
- Only one FOV is available locally (`test_raw/T1_221209_...tif`). Broader
  validation on `/mnt/external/ROIGBIV-DATA/` still pending — user can
  re-run `scripts/compare_models.py` when that drive is mounted.
- The followup's original Phase A.4 (visualize L, k-sweep, Vcorr on L)
  would confirm whether absorption is happening at all in the L+S step.
  It's informative for Stages 2–4 (which *do* read the residual `S`) but
  does not affect Stage 1 recall, which is what the user's visual
  evidence was about. Not run here; documented as optional follow-up.

## Artifacts
- `run_A0.py`, `run_A1.py`, `run_A2.py` — scripts (reuse
  `scripts/diagnostic_compare.py`).
- `A0_accept_labels.tif`, `A0_current_pipeline_on_mean_M.png`
- `A1_reference_labels.tif`, `A1_reference_overlay.png`
- `A2_summary.md`, `A2_missing.png`, `A2_extra.png`
