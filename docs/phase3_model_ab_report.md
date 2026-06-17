# Phase 3 — Stage-1 model A/B: Cellpose-SAM vs deployed CP3

**Engagement:** recall-refinement (gated, one-variable, recall-first).
**Branch:** `feat/0-phase3-model-ab` (cut from Phase-M head `c973a62`; reuses the
OFF-by-default cpsam sidecar). **Driver:** `experiments/phase3_model/run_ab.py`
(commit `3382a87`). **Raw:** `experiments/runs/phase3_model/ab_results.json`.

## Design — 3-arm decomposition

A naive "cpsam vs deployed CP3" swap bundles **two** variables: model architecture
*and* denoising (cpsam 4.x has no `DenoiseModel`; CP3 runs `denoise_cyto3` on ch1).
To honor `one_variable_per_experiment`, the swap was split into three arms so each
pairwise contrast isolates one variable. All else identical: same source images
(`mean_M` morph + `vcorr_S`), same `diameter`, same `cellprob_threshold` /
`flow_threshold` (the sidecar relays `cfg.*` verbatim, `stage1.py:289-290`).

| arm | backend | denoise | role |
|---|---|---|---|
| `cp3` | cellpose3 (deployed checkpoint) | ON | as-deployed baseline |
| `cp3nd` | cellpose3 (same checkpoint) | OFF | isolates denoise |
| `cpsam` | Cellpose-SAM via Phase-M sidecar | n/a (no 4.x equiv) | isolates architecture (raw input) |

- `cp3` ↔ `cp3nd` → **denoise** contribution (model + input fixed)
- `cp3nd` ↔ `cpsam` → **architecture** delta (both on raw `mean_M`)
- `cp3` ↔ `cpsam` → **as-deployed** bottom line

13 held-out FOVs × 3 arms = **39 full pipeline runs** (Stage-1 subtraction
propagates downstream, so a fair A/B needs full runs), scored vs anatomical GT
with the stratified harness. **39/39 OK, 0 errors**, ~3.2 h wall.

## Results (micro-averaged, pooled TP/FP/FN across 13 FOVs)

| arm | recall | precision | TP | FP | FN |
|---|---|---|---|---|---|
| `cp3` (deployed) | 0.624 | 0.804 | 337 | 82 | 203 |
| `cp3nd` (no denoise) | **0.637** | **0.823** | 344 | 74 | 196 |
| `cpsam` (stock SAM) | 0.376 | 0.744 | 203 | **70** | 337 |

**Pairwise recall:**
- denoise (`cp3`→`cp3nd`): **+0.013** (precision +0.019, FP −8 / −10%)
- architecture (`cp3nd`→`cpsam`): **−0.261**
- as-deployed (`cp3`→`cpsam`): **−0.248**

## Verdict — PRIMARY question (model)

**Stock Cellpose-SAM FAILS the recall-first bar decisively. Keep the deployed CP3
checkpoint as default.** cpsam loses **−0.248 overall recall** as-deployed
(−0.261 pure-architecture) and regresses recall on **all 13/13 FOVs** — it never
wins or ties on a single FOV (range −0.13 to −0.49). Its only "win," a slightly
lower FP count (70 vs 82), is an artifact of detecting far fewer cells overall.

This is the expected zero-shot penalty: the deployed CP3 checkpoint is fine-tuned
on this exact lab's GCaMP/PrL-NAc domain; stock cpsam is a generalist that has
never seen it.

**Fine-tune follow-up NOT triggered.** The phase directive gates the cpsam
fine-tune experiment on cpsam *winning or tying* the stock A/B. It did neither, so
that follow-up is not run as part of Phase 3 (it remains a future option — see
below). cpsam stays available as an **OFF-by-default** backend (`stage1_backend`),
no default change.

## Secondary finding — denoise (different variable, flagged, NOT flipped)

Disabling `denoise_cyto3` (`cp3nd`) modestly **beat** the deployed config on the
same checkpoint: recall +0.013, precision +0.019, FP −8. Per-FOV: recall improved
in 7/13, tied 5/13, **regressed in 1/13** (`230118_..._PRE-000`, −0.026). Because
it touches a *separate* variable (`use_denoise`) from the model A/B, and because it
regresses one FOV (so it does not cleanly pass "no recall reduction in any
stratum"), this is **not** auto-flipped. It is a clean candidate for its own
single-variable gate if you want the small lift. Notable that denoise-off wins even
though the checkpoint was likely fine-tuned with denoise in the loop.

## Caveat — GT cannot stratify recall by activity type

As in Phase 2, the anatomical Cellpose GT carries no activity labels, so **missed
cells (FN) cannot be activity-typed** — they all fall in `unknown`, making
by-stratum *recall* degenerate. Only *detected* ROIs get a type, so `by_type`
informs precision/composition only. The load-bearing metrics for a Stage-1 model
A/B are therefore **overall anatomical recall + FP burden** (reported above).
Tonic/silent strata are essentially empty in this GT (`lower_bound`) — the tonic
target is a Phase-5 concern requiring tonic GT, not addressable here.

What `by_type` does show: cpsam is also *less precise* on the strata it does detect
(phasic precision 0.766 vs cp3nd 0.844; ambiguous 0.647 vs 0.755) — it is not
trading recall for cleaner boundaries.

## Recommendation

1. **Keep CP3 as the default Stage-1 model.** cpsam OFF-by-default backend stays
   (useful for future re-test after fine-tuning); no default flip.
2. **Do not run the cpsam fine-tune now** (gate condition unmet). If a SAM-class
   model is still wanted later, the path is: fine-tune cpsam on the HITL corpus and
   re-A/B vs CP3 as its own experiment.
3. **Optional separate gate:** a `use_denoise=False` single-variable A/B for the
   small recall+precision lift, if desired.
