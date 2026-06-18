# Phase 5a — neuropil-relative baseline-elevation QC feature

**Engagement:** recall-refinement (gated, one-variable, recall-first).
**Branch:** `feat/0-phase5-tonic` (cut from Phase-4 config head `81a6669`).
**Scope of 5a:** add a QC feature only. NO decision-logic change — that is 5b,
gated separately (`variable_sequence`: 5a feature → gate → 5b accept tier).

## What the feature is

`neuropil_baseline_elevation` quantifies how far an ROI's own stable baseline
F0 sits **above its annular-neuropil** stable baseline F0. Tonic (high-DC)
somata sit persistently above local background; phasic cells do not. It is a
more direct, composition-robust signal than the current population-median
"high mean / low variance" tonic fallback (`classify.py:66`), which is
confounded by FOV composition.

Definition (per ROI):

```
F0_roi  = stable_baseline(roi.trace,            window=120 s, pct=10)
F0_neu  = stable_baseline(roi.features.F_neuropil, window=120 s, pct=10)
neuropil_baseline_elevation = (F0_roi − F0_neu) / max(|F0_neu|, 1e-6)
```

`stable_baseline` = **median of a sliding low-percentile filter** over a wide
window (`tonic_baseline_window_s`, `baseline_percentile` from config). Design
choices, all required by the directive:
- Computed on **raw** ROI and neuropil fluorescence (`roi.trace`,
  `F_neuropil`) — NOT residual or dF/F — so the DC offset that *defines* tonic
  activity is preserved.
- **Wide** window, not a short rolling ΔF/F0 baseline (which would subtract out
  exactly the offset we want to measure).
- Collapsed to a scalar via the **median** of the per-frame baseline, so slow
  drift (bleaching) doesn't bias it.
- Low percentile (10th) baseline ⇒ **sparse bright transients do not inflate
  it** (verified by test).

Also logged for audit/HITL: `roi_baseline_f0`, `neuropil_baseline_f0`.

## Implementation (diff isolated to feature + its input)

| File | Change |
|---|---|
| `traces.py` | Store the raw neuropil trace as `roi.features['F_neuropil']` (the docstring already promised this; the code dropped it). ndarray feature, dropped from JSON by `_jsonable_features` like `trace_bandpass`. |
| `qc_features.py` | New `_stable_baseline()` + `compute_neuropil_baseline_elevation()`; called in the `compute_all_features` per-ROI loop after `compute_temporal_features`. New `percentile_filter` import. |
| `tests/test_qc_neuropil_elevation.py` | 6 unit tests. |

No change to `classify.py`, any gate, `run.py`, or any config default. No new
config field (reuses `tonic_baseline_window_s`, `baseline_percentile`, `fs`).

## Verification

**Unit (6/6 pass, + 8 traces_io regression):**
- elevated soma (DC +50%) → elevation > 0.2;
- background-level cell → |elevation| < 0.1;
- **sparse bright transients on a flat baseline → |elevation| < 0.1** (the key
  discriminator: spikes must not masquerade as elevation);
- missing `F_neuropil` → 0.0, no crash;
- feature survives `to_serializable()`, `F_neuropil` array dropped;
- short-trace (< 1 window) → whole-trace percentile fallback.

**End-to-end (1 real FOV, settled default = `vcorr_max_fused` + denoise ON):**
`experiments/phase5_tonic/verify_5a.py` on
`T1_230202_…_HI-D2_FOV2_pre-000_mc` →
- 44/44 ROIs carry `neuropil_baseline_elevation` in `roi_metadata.json`;
- `F_neuropil` array correctly absent from JSON;
- elevation min/median/max = **−0.335 / 1.075 / 3.197** (positive median =
  somata generally brighter than surround; spread shows real discrimination).
- (This FOV had 0 tonic-classified ROIs, so no tonic-stratum elevation summary;
  the feature still computes for all activity types, which is the 5a goal.)

## Gate (5a)

**Exit criteria met:** feature present and logged; computes on real data; no
decision logic touched; diff isolated. **STOP — awaiting go-ahead for 5b**
(tonic accept tier for `source_stage ∈ {1,2}` only; Gate-4 / Stage-4 path
untouched). 5b will need a threshold on this feature plus an A/B against the
held-out set.
