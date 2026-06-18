# Phase 5b — Tonic accept tier: gate-aware A/B → INERT on real data (keep OFF)

**Engagement:** recall-refinement (gated, one-variable, recall-first).
**Branch:** `feat/0-phase5-tonic` (impl `5119039`, off 5a `03de4ae`).
**Driver/scorer:** `experiments/phase5_tonic/run_5b.py` + `summarize_5b.py`.
**Raw:** `experiments/runs/phase5_5b/roi_records.json` (13/13 FOVs, 0 errors).

## Verdict

The accept tier is **correct, safe, and OFF by default — but completely inert on
the held-out data.** There are **zero tonic ROIs of any kind** across all 13 FOVs,
so the tier has nothing to promote. The controlling fact is upstream of the tier:
**the current activity classifier never emits a `tonic` label on this dataset.**
This is a `discovery_before_code` STOP — the tier's precondition (anatomical
`activity_type=="tonic"` ROIs) does not exist, and making it non-inert would
require redefining "tonic," which cannot be validated without tonic ground truth.

## What the run showed (433 ROIs, 13 FOVs)

| | count |
|---|---|
| total ROIs | 433 |
| `phasic` | 332 |
| `ambiguous` | 100 |
| `sparse` | 1 |
| **`tonic`** | **0** |
| `silent` | 0 |

| source_stage | count | tonic |
|---|---|---|
| 1 (Cellpose) | 375 | 0 |
| 2 (Suite2p) | 35 | 0 |
| 3 (template) | 23 | 0 |
| **4 (tonic search)** | **0** | — |

Stage 4 — the *dedicated* tonic detector — produced **no candidates** on any FOV.
The classifier's tonic branch produced **no labels** on any FOV. Promotion
population for the tier: **0**.

## Why the classifier emits no tonic — and why elevation doesn't rescue it

The 5a `neuropil_baseline_elevation` feature *does* fire: **267/433 (62%) of
anatomical ROIs sit ≥0.5 above their neuropil.** So the elevation signal is alive —
but it is **not tonic-specific**: bright *phasic* cells are elevated too (197 of
those 267 are phasic).

The only place a misclassified tonic could hide is the **69 anatomical
`ambiguous` ROIs with elevation ≥0.5**. Their temporal features:

| feature | min | median | max |
|---|---|---|---|
| skew | −0.49 | 0.22 | 0.49 |
| n_transients | 5 | 109 | 424 |
| **bp_std / noise_floor** | 0.21 | **0.69** | **0.76** |

All 69 have **low skew** (≤0.49 → they *would* pass `tonic_skew_ok`) and lots of
activity. But the current tonic rule also requires
`bp_std > tonic_bp_std_factor(2.0) × noise_floor` — and **every one of them maxes
out at bp_std/noise = 0.76**, far below 2.0. So **0/69 are tonic-eligible.**

**Root cause:** the two signals measure *different phenomena*.
- The **classifier's** `tonic` criterion keys on **slow bandpass-oscillation
  power** (`bp_std`). These bright, steadily-active cells don't oscillate in that
  band → they land in `ambiguous`.
- The **5a elevation** feature keys on **steady DC offset above the surround**.
  These cells have that in spades.

Neither, alone, is a validated tonic detector, and there is **no tonic ground
truth** to calibrate one (anatomical Cellpose GT carries no activity labels; HITL
corrections are geometry-only). This is the same wall flagged in Phases 2–5: the
benchmark cannot see tonic recall.

## What this means for the engagement's tonic goal

On this held-out set, "tonic neurons" **as currently defined are absent**, and the
elevation-based alternative **cannot be validated** without tonic labels. The
accept tier is a working, OFF-by-default no-op that will activate automatically if
data with classifier-tonic cells (or Stage-4 detections) ever flows through. It
introduces zero risk and zero behavior change shipped.

## Options at the gate (your call — no further code without it)

- **(5b-A) Conclude here. Keep the inert tier OFF; do not redefine tonic without
  GT. (Recommended.)** The tier is correct and ready; the blocker is data
  (no tonic cells + no tonic GT), not code. Banks the engagement's real win
  (Phase-4 fused) and the 5a diagnostic, honestly.
- **(5b-B) Re-scope to fix the *classifier*** so a combined steady-DC + low-skew +
  sustained-activity criterion (using elevation) labels the 69 bright low-skew
  cells tonic. **Not recommended without GT:** it changes classification semantics
  (tonic/silent are skipped by deconvolution; review priority shifts), risks
  demoting genuinely-active cells, and **cannot be A/B-validated** on the current
  benchmark.
- **(5b-C) Generate tonic GT via HITL first** (human-label tonic cells on a few
  FOVs), then revisit 5b properly. The only path that makes the tonic goal
  *measurable*. A data-collection task, outside the code A/B loop.

## Gate

**STOP. Tier stays OFF (no flip).** Recommend **5b-A**. Awaiting your decision
before any classifier change (5b-B) or closing the engagement.
