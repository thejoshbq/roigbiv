# Phase 2 — PMD Denoising A/B Report

**Branch:** `feat/0-phase2-pmd` · **Engagement:** recall-refinement, Phase 2 item 3 (gate deliverable).
**One variable:** `cfg.use_pmd_denoise` (off vs on). Identical config otherwise; full view swap (D1-b).
**Benchmark:** 13 held-out FOVs (`experiments/harness/heldout_fovs.txt`), scored by the repo's
stratified harness (`roigbiv/eval/harness.py`) against anatomical Cellpose GT masks, IoU ≥ 0.3.
**Raw results:** `experiments/runs/phase2_pmd/ab_results.json` (+ `ab_progress.log`). 26/26 runs, 0 errors.

## Verdict — KEEP OFF BY DEFAULT (no flip)

PMD as configured **fails the engagement's recall-first acceptance standard**. It buys a small recall
gain with a large precision collapse, gains nothing on the tonic/sparse cells it was meant to recover,
and reduces recall in one FOV. The flag stays OFF (default unchanged); no human approval is sought to flip.

## Aggregate (summed over 13 FOVs)

| Metric | PMD off | PMD on | Δ |
|---|---|---|---|
| Recall (TP/(TP+FN)) | **0.624** | **0.663** | **+0.039** |
| Precision (TP/(TP+FP)) | **0.804** | **0.608** | **−0.196** |
| TP | 337 | 358 | +21 |
| FP | **82** | **231** | **+149 (×2.82)** |
| FN | 203 | 182 | −21 |
| Total runtime | 64.1 min | 68.8 min | +7% |

**Acceptance margins (stated explicitly, per the engagement):** I set the acceptable post-review
FP-burden increase at **+15% aggregate** and required **no recall regression in any stratum/FOV**.
Observed: FP **+182%** (a >12× overshoot of the margin) and recall **regressed in 1 FOV**. Both bars fail.

## Stratified by activity type (summed TP / FP)

| Type | TP off | TP on | ΔTP | FP off | FP on | ΔFP |
|---|---|---|---|---|---|---|
| phasic | 265 | 286 | **+21** | 56 | 194 | **+138** |
| ambiguous | 71 | 71 | 0 | 26 | 37 | +11 |
| sparse | 1 | 1 | **0** | 0 | 0 | 0 |
| tonic | 0 | 0 | **0** | 0 | 0 | 0 |
| silent | 0 | 0 | **0** | 0 | 0 | 0 |

- The entire recall gain is **phasic** TP (+21); the new false positives are **also overwhelmingly
  phasic** (+138). PMD smooths the residual, which makes Stage 3's template/MAD detector fire far more —
  capturing a few more real phasic cells but mostly spurious ones.
- **Zero movement on tonic / sparse / silent** — the high-baseline and tonic cells this phase targeted.
  Two compounding reasons: (1) the anatomical Cellpose GT under-represents tonic/silent (the harness
  flags these strata `lower_bound`; tonic/silent GT counts are ~0 here), so the benchmark **cannot
  measure** the sparse/tonic SNR benefit PMD is designed for; (2) on what the benchmark *can* see,
  PMD's effect manifests as phasic over-detection, not tonic recovery.

## Per-FOV

- Recall: **UP in 7**, DOWN in 1 (`EXT-D2-FOV2_BEH`: 0.963→0.926, −1 TP), flat in 5.
- FP increased in **12/13** FOVs (often 2–4×; e.g. `EXT-D10-FOV2_BEH`: 13→50, `EXT-D9-FOV1_BEH`: 4→22).

## Cost / memory / contract

- **Compute:** PMD adds ~modest time (+7% aggregate; per-FOV PMD pass ≈ a few seconds — 21 bands at
  ~0.3–1.4 s each on the cu130/sm_120 GPU).
- **Memory:** bounded by design to ~one row-band (`T × pmd_patch_size × W` float32); **no OOM across
  all 13 FOVs** on the real GPU. Disk: a transient ~2.3 GB denoised memmap per run, reclaimed after
  scoring.
- **Reconstruction contract:** preserved byte-for-byte — the engine code is unmodified; PMD reuses the
  existing `_dense` read path; `test_residual_view.py` stays green; the dense base propagates through
  `with_source` (verified by `test_pmd.py`).

## Recommendation

1. **Do not flip the default.** Ship PMD OFF. The +0.039 recall is not worth precision 0.80→0.61 and a
   ~2.8× FP/HITL-review burden, and it regresses one FOV.
2. The benchmark is **blind to PMD's intended target** (tonic/sparse) because the anatomical GT lacks
   those labels. A fair evaluation of the tonic-recovery hypothesis needs tonic/sparse ground truth —
   which is exactly what **Phase 5** (tonic classification + accumulated HITL labels) would supply.
   Re-evaluating PMD against tonic GT is a natural future experiment, not a Phase-2 default change.
3. If pursued later, the current knobs (`pmd_max_rank=30`, `pmd_rank_margin=0`) are permissive. Candidate
   one-variable follow-ups (each separately benchmarked): higher `pmd_rank_margin` (keep fewer
   components → less over-smoothing), or pairing PMD with a stricter Gate 3 to absorb the extra phasic
   crossings. Not adopted now.

## Reproduce

```
python experiments/phase2_pmd/run_ab.py        # 13 FOVs × {off,on}; resumable; writes ab_results.json
```
Per-arm pipeline runs land in `experiments/runs/phase2_pmd/{stem}_{pmdoff|pmdon}/` (large intermediates
auto-reclaimed; `merged_masks.tif` + `roi_metadata.json` kept for re-scoring).
