# ROI Detection Comparison: roigbiv(ridge) vs roigbiv(robust) vs CNMF-E

Held-out set: T1 PrL-NAc-G6 val split (seed=42, val_frac=0.15, 13 FOVs).  
FOV reported: `T1_230202_PrL-NAc-G6-6F_HI-D2_FOV2_pre-000_mc` (smallest val FOV).  
GT format: manually annotated uint16 label TIFFs (Cellpose training masks).  
IoU matching threshold: 0.3 (greedy one-to-one, `roigbiv.eval.match`).  
Activity-type stratification: derived from roigbiv(ridge) reference run; tonic/silent
recall is a **lower bound** — FN ROIs have no assigned type (Blindspot 13).  
GPU: RTX 5080 sm_120 — **GPU-enabled** under `torch 2.12.0+cu130`
(`cuda_compute_capable()=True`). The runs below were CPU-only because the GPU was
VRAM-starved by the local-Qwen MCP model at the time, not because sm_120 is
unsupported (see DISCOVERY.md §Hardware, RESOLVED).  

---

## 1. Detection Metrics (per activity type)

*Recall is the headline column. Tonic and silent rows: lower bounds (see note §4).*

### roigbiv — ridge solver (spec §5.1, ridge normal equations)

| Stratum     | Recall | Precision | F1    | TP | FP | FN |
|-------------|--------|-----------|-------|----|----|----|
| phasic      | 1.000  | 0.870     | 0.930 | 20 |  3 |  0 |
| sparse      | 1.000  | 1.000     | 1.000 |  1 |  0 |  0 |
| ambiguous   | 1.000  | 0.688     | 0.815 | 22 | 10 |  0 |
| tonic*      |  —     |  —        |  —   |  0 |  0 |  0 |
| silent*     |  —     |  —        |  —   |  0 |  0 |  0 |
| unknown†    | 0.000  |  —        |  —   |  0 |  0 |  9 |
| **Overall** | **0.827** | **0.768** | **0.796** | **43** | **13** | **9** |

*GT counts: 52 ROIs. Predicted: 56.*

### roigbiv — robust solver (one-sided Huber IRLS, kappa=0.5, max_iter=5)

*Activity-type stratification uses the same ridge reference labels.*

| Stratum     | Recall | Precision | F1    | TP | FP | FN |
|-------------|--------|-----------|-------|----|----|----|
| phasic      | 1.000  | 0.870     | 0.930 | 20 |  3 |  0 |
| sparse      | 1.000  | 1.000     | 1.000 |  1 |  0 |  0 |
| ambiguous   | 1.000  | 0.727     | 0.842 | 22 |  8 |  0 |
| tonic*      |  —     |  —        |  —   |  0 |  0 |  0 |
| silent*     |  —     |  —        |  —   |  0 |  0 |  0 |
| unknown†    | 0.000  |  —        |  —   |  0 |  1 |  9 |
| **Overall** | **0.827** | **0.782** | **0.804** | **43** | **12** | **9** |

*GT counts: 52 ROIs. Predicted: 55.*

### CNMF-E (CaImAn 1.13.1; gSig=6, fs=7.5, tau_d=1.0, min_corr=0.8, min_pnr=10.0)

*Activity-type stratification not available — no roi_metadata.json from CNMF-E.*  
*See note §6 regarding 0-TP result.*

| Metric    | Value |
|-----------|-------|
| Recall    | 0.000 |
| Precision | 0.000 |
| F1        | 0.000 |
| TP        |     0 |
| FP        |     6 |
| FN        |    52 |

*GT counts: 52 ROIs. Predicted: 6.*

---

## 2. Section 5.2 Artifact Metrics (roigbiv variants only)

*ring_candidates: ROIs with std_ratio > 3.0 (Blindspot 1 over-subtraction signature).*  
*halo_candidates: ROIs with std_ratio < 0.3 (under-subtraction).*

### ridge

| Stage | n_ROIs | pass_rate | std_ratio_p90 | ring_cands | halo_cands | mean_anticorr |
|-------|--------|-----------|---------------|------------|------------|---------------|
| S1    |   55   |   0.800   |     1.901     |      0     |      0     |    -0.111     |
| S3    |    1   |   1.000   |     1.043     |      0     |      0     |    -0.154     |

### robust

| Stage | n_ROIs | pass_rate | std_ratio_p90 | ring_cands | halo_cands | mean_anticorr |
|-------|--------|-----------|---------------|------------|------------|---------------|
| S1    |   55   |   0.327   |     1.906     |      0     |      0     |     0.016     |
| S3    |    1   |   0.000   |     1.049     |      0     |      0     |     0.255     |

*Robust S1 pass_rate regression (0.327 vs 0.800) driven by mean_ratio failures, not anticorr.
The unclamped IRLS traces include negative values; the mean_ratio criterion flags the resulting
over-subtracted residuals. No ring or halo candidates in either solver.*

---

## 3. Per-Stage New-ROI Counts

### ridge

| Stage | New ROIs | Cumulative |
|-------|----------|------------|
| S1    |       55 |         55 |
| S2    |        0 |         55 |
| S3    |        2 |         57 |
| S4    |        0 |         57 |

*Merged to 56 after cross-stage IoU dedup.*  
*No Stage N > Stage N-1 inversion detected (Blindspot 2).*

### robust

| Stage | New ROIs | Cumulative |
|-------|----------|------------|
| S1    |       55 |         55 |
| S2    |        0 |         55 |
| S3    |        1 |         56 |
| S4    |        0 |         56 |

*Final merged mask: 55 ROIs (one fewer Stage 3 detection vs ridge).*

---

## 4. Isolation Test Results

Synthetic scene: two overlapping disc sources (r=8px, ~30% overlap), ghost ring
contaminant at 50% amplitude of source 1 (experiment: `experiments/harness/test_robust_isolation.py`).

| Solver | Source-1 MSE | Source-1 corr | MSE ratio (robust/ridge) |
|--------|-------------|--------------|--------------------------|
| Ridge  |   0.000874  |    0.9999    |           —              |
| Robust |   0.000573  |    0.9999    |         0.655            |

Robust solver reduces ghost-induced bias by 34.5% in synthetic case. Source-2 MSE also
improves (0.013018 ridge → 0.001954 robust). The synthetic case isolates the IRLS benefit
cleanly because both sources have positive traces, enabling reliable sigma estimation from
the negative residual pool.

---

## 5. Environment Notes

- All pipeline runs here were CPU-only due to transient VRAM contention with the
  local-Qwen MCP model, **not** a missing PyTorch build — RTX 5080 sm_120 is supported
  under `torch 2.12.0+cu130` (see DISCOVERY.md §Hardware, RESOLVED). Results are correct;
  solver difference isolated.
- CNMF-E: isolated `caiman` conda env (CaImAn 1.13.1, NumPy 2.2.6). Not installed
  into roigbiv env (version isolation approach per Phase 3 plan).
- `use_cnn=False` set in initial `CNMFParams` dict (CaImAn 1.13 limitation: `change_params`
  does not propagate to `evaluate_components`).

---

## 6. Notes and Lower Bounds

† **unknown stratum**: FN ROIs (missed by pipeline) have no activity-type label
  because GT masks carry no type annotations. Per-stratum recall for tonic and silent
  would be lower bounds even if those types were present in the GT — FN ROIs of those
  types would be invisible to stratified metrics. See Blindspot 13.

* **tonic / silent rows marked with asterisk** are lower bounds if non-zero.
  Manual annotation under-represents tonic and silent cells (Blindspot 13); any
  recall reported against this GT is a floor, not an estimate of true sensitivity.

**CNMF-E 0-TP note**: CNMF-E found 6 components; none matched GT at IoU≥0.3. This is
consistent with known domain limitations: CNMF-E uses a ring background model validated
on 1-photon GRIN lens data. On 2-photon data, the `corr_pnr` initialisation with
min_corr=0.8/min_pnr=10.0 is highly selective and the ring background model does not
match the structured neuropil background of 2P FOVs. The spatial footprints returned by
CNMF-E may also differ in extent from the Cellpose cell-body GT masks, reducing IoU.
This single-FOV comparison does not generalise — CNMF-E performance on 2P data is
parameter-sensitive and may improve with domain-adapted gSig/min_corr/min_pnr.

**Robust solver sigma estimation**: on real pipeline data, joint ridge warm-start
residuals are contaminated by inter-ROI coupling artifacts (neg_frac≈0.73,
sigma_neg≈262), providing no clean noise floor for kappa_abs calibration. The synthetic
isolation test works because both sources have positive traces and the coupling artifacts
are absent. The IRLS failure mode converts to mean_ratio failures (not anticorr), so
the NNLS anticorr fallback does not fire, leaving those ROIs uncorrected.

---

## 7. Recommendations

### (a) Promote robust solver to default?

**No.** Per Phase 2 decision gate (promote only with ≥0.05 F1 gain on any stratum):
- Overall F1 gain: 0.804 − 0.796 = **0.008** (below 0.05 threshold)
- Recall: identical (0.827) — no recovery of missed ROIs
- Subtraction QC regresses: S1 pass_rate 0.800 → 0.327; S3 pass_rate 1.000 → 0.000

The IRLS approach is sound in synthetic isolation but fails on real pipeline data due to
unreliable sigma estimation from the joint ridge warm-start residuals. Sigma inflates when
negatively-biased ROIs create large positive eps that dominate the negative pool via
over-prediction coupling. Recommended path if revisiting: per-ROI NNLS warm-start
(replacing joint ridge) to obtain coupling-free residuals before IRLS iteration.

**Default remains `subtract_solver = "ridge"`.**

### (b) roigbiv vs CNMF-E on held-out dmPFC data

roigbiv (ridge) substantially outperforms CNMF-E with default parameters on this 2P
PrL-NAc FOV: recall 0.827 vs 0.000, F1 0.796 vs 0.000. The CNMF-E result is likely a
calibration failure (ring background model, 1P-tuned thresholds) rather than an
upper-bound comparison. A fair comparison would require domain-adapted CNMF-E parameters
(lower min_corr/min_pnr, 2P-appropriate background model). Not recommended as a primary
comparison in the methods paper; cite as an out-of-box baseline with appropriate caveats.

---

## References

- Spec §5 (Source Subtraction Engine), §5.1 (trace estimation), §5.2 (validation)
- Blindspot 1 (ring artifacts), 2 (cascade), 5 (overlapping cells),
  7 (z-plane ghosts), 13 (phasic confirmation bias in manual annotation)
- EXTRACT: Inan et al. 2021 (M-estimation for microendoscopy; see docs/novelty_analysis.md §3.5)
