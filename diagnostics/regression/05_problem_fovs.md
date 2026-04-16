# 05 — Problem FOVs and Suspect ROIs

## Test FOV

`T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002` (one FOV available, 505×493 pixels, 8624 frames).

## Stage 1 outcome (from `cli_verify` run, fs=30, run015_epoch_0280)

| Outcome | Count | Criterion |
|---|---:|---|
| Accept | 79 | All morphological filters passed |
| Flag | 5 | Exactly one filter within margin (all 5 flags fail `area_high` by 1–13 px) |
| Reject | 17 | At least one filter violated hard (all 17 rejects fail `area_high` by 27–153 px) |
| **Total Cellpose detections** | **101** | before gating |

## Area histogram (all 101 candidates)

| Area bin (px) | Count |
|---|---:|
| 80–150 | 2 |
| 150–250 | 29 |
| 250–330 | 40 |
| 330–350 | 8 |
| **350–400** (rejects start) | **11** |
| 400–500 | 10 |
| 500–1000 | 1 |

The `max_area=350` threshold cuts through a continuous distribution with no natural discontinuity. **22 of 101 (22%) Cellpose detections fall above it.**

## Rejected ROIs (all 17)

Sorted by area; every one rejected **only** for `area_high`. Cellpose probabilities and contrasts are high — these are confident, real-looking detections.

| label_id | area | solidity | ecc | contrast | cp_prob | ns_score | reasons |
|---:|---:|---:|---:|---:|---:|---:|---|
| 86 | 377 | 0.95 | 0.67 | 1.91 | 1.66 | −35.7 | area_high:+27 |
| 95 | 415 | 0.97 | 0.63 | 0.31 | −0.22 | −3.5 | area_high:+65 |
| 94 | 387 | 0.97 | 0.64 | 0.53 | 0.95 | −7.2 | area_high:+37 |
| 62 | 388 | 0.97 | 0.29 | 1.28 | 2.03 | −38.3 | area_high:+38 |
| 46 | 388 | 0.92 | 0.71 | 1.13 | 0.71 | −33.3 | area_high:+38 |
| 66 | 395 | 0.94 | 0.70 | 3.84 | 2.05 | −99.6 | area_high:+45 |
| 57 | 399 | 0.96 | 0.65 | 4.85 | **3.20** | −163.2 | area_high:+49 |
| 93 | 484 | 0.97 | 0.56 | 0.56 | 0.39 | −6.7 | area_high:+134 |
| 99 | 426 | 0.97 | 0.62 | 0.64 | 0.89 | −6.6 | area_high:+76 |
| 40 | 421 | 0.96 | 0.69 | 1.59 | 1.77 | −51.3 | area_high:+71 |
| 42 | 431 | 0.97 | 0.61 | 1.27 | 0.59 | −39.1 | area_high:+81 |
| 2 | 442 | 0.93 | 0.86 | 1.20 | 0.16 | −27.3 | area_high:+92 |
| 37 | 445 | 0.95 | 0.88 | 1.52 | 1.05 | −44.1 | area_high:+95 |
| 71 | 448 | 0.95 | 0.65 | 1.86 | 1.81 | −44.0 | area_high:+98 |
| 68 | 476 | 0.94 | 0.67 | 3.81 | **2.36** | −115.5 | area_high:+126 |
| 63 | 483 | 0.95 | 0.54 | 1.85 | 0.94 | −50.2 | area_high:+133 |
| 59 | 503 | 0.97 | 0.68 | 2.20 | 1.94 | −67.7 | area_high:+153 |

Comparison of `cellpose_prob` statistics:
- Rejects: min=−0.22, max=3.20, mean=**1.31**
- Accepts: min=−0.87, max=2.47, mean=0.71

Rejects are on average more confident than accepts. This is the opposite of what a reasonable filter should do.

## Flagged ROIs (all 5)

| label_id | area | solidity | ecc | contrast | cp_prob | reasons |
|---:|---:|---:|---:|---:|---:|---|
| 50 | 351 | 0.94 | 0.32 | 0.72 | 0.64 | area_high:+1 |
| 82 | 353 | 0.95 | 0.39 | 1.28 | 0.69 | area_high:+3 |
| 49 | 356 | 0.94 | 0.67 | 2.21 | 2.20 | area_high:+6 |
| 45 | 358 | 0.94 | 0.71 | 1.02 | 0.89 | area_high:+8 |
| 30 | 363 | 0.92 | 0.65 | 1.69 | 1.86 | area_high:+13 |

All 5 flags are 1–13 px over `max_area=350`. These also look like real cells.

## Interpretation

The user reported "missed ~5 distinct cells that stock cyto3 would catch." The 17 rejected ROIs are almost certainly a superset of those 5 — Cellpose *is* finding them; Gate 1 is hiding them from the user. The "additional edge cases" likely include some of the 5 flagged ROIs (if the UI treats flag as not-displayed) and a handful of borderline 330–349 px accepts near the threshold.

Suspect pool for subsequent experiments: label IDs 2, 30, 37, 40, 42, 45, 46, 49, 50, 57, 59, 62, 63, 66, 68, 71, 82, 86, 93, 94, 95, 99.

The next experiment (2.D) will render these on the mean_M image to confirm visually that they are real cells, not fused blobs or artifacts.
