# Phase 4 — Stage-1 channel-2 content A/B

**Engagement:** recall-refinement (gated, one-variable, recall-first).
**Branch:** `feat/0-phase4-channel-ab` (cut from Phase-3 head). **Driver:**
`experiments/phase4_channel/run_ab.py` (commit `1751cea`). **Raw:**
`experiments/runs/phase4_channel/ab_results.json`.

## Why this is not the phase as literally written (discovery STOP)

Phase 4 as specified enriches the Stage-1 input to a **3-channel** stack
`mean_M + vcorr_S + max_S`. Discovery confirmed (per `discovery_before_code`) that
this is **impossible on the winning model**:

- Phase 3 selected the deployed **CP3** checkpoint (stock cpsam lost by −0.248 recall).
- CP3's deployed checkpoint is **architecturally 2-channel** — first conv weight
  `shape=(32, 2, 3, 3)` → `in_channels = 2` (verified by `torch.load`).
- Cellpose has no `channels=(1,2,3)` convention; the input stack is built 2-wide
  (`stage1.py`), call site passes two images (`run.py:502-503`).
- A literal 3rd channel would require **retraining the protected `models/deployed/`
  checkpoint at `nchan=3`** — out of scope, and would bundle a model retrain with a
  channel change (violates `one_variable_per_experiment`).

**Resolved at gate (Option A):** test channel enrichment by varying the *content* of
channel-2 within CP3's real 2-channel budget. Channel-1 (`mean_M` morphology), the
model, the thresholds, and Gate 1's `vcorr_S` use are all fixed. The single variable
is the Stage-1 detector's second input channel.

## Design — 3 arms, one variable (channel-2 content)

| arm | ch2 content | role |
|---|---|---|
| `vcorr` | `vcorr_S` | current/default behavior — baseline |
| `maxs` | `max_S` | residual peak-intensity (single-firer / sparse cue) |
| `fused` | `norm(vcorr_S) ⊕ norm(max_S)` | per-image min-max, elementwise max (union cue) |

All arms: `stage1_backend=cellpose3`, `use_denoise=True` (deployed config), `mean_M`
channel-1. 13 held-out FOVs × 3 arms = **39 full pipeline runs** (Stage-1 subtraction
propagates downstream, so a fair A/B needs full runs), scored vs anatomical GT with the
stratified harness. **39/39 OK, 0 errors.** The `vcorr` arm reproduces Phase-3's `cp3`
arm exactly (R0.624 / P0.804 / FP82) — a clean reproducibility check.

## Results (micro-averaged, pooled TP/FP/FN across 13 FOVs)

| arm | recall | precision | TP | FP | FN |
|---|---|---|---|---|---|
| `vcorr` (baseline) | 0.624 | 0.804 | 337 | 82 | 203 |
| `maxs` | 0.639 | 0.804 | 345 | 84 | 195 |
| `fused` | **0.641** | **0.805** | **346** | 84 | **194** |

**Pairwise recall (vs baseline):**
- `vcorr`→`maxs`: **+0.015** (precision flat, FP +2)
- `vcorr`→`fused`: **+0.017** (precision +0.001, FP +2)

**Per-FOV regression check (the load-bearing test):**
- `fused`: improves recall on **5/13**, ties **8/13**, **regresses 0/13**. Never below baseline.
- `maxs`: improves **8/13**, ties **3/13**, **regresses 2/13** (`230118_BEH` −0.013,
  `230308_BEH` −0.037).

## Recall-first bar

Stated margin for this phase: **no per-FOV recall regression, and pooled post-review FP
increase ≤ +15%.** Pooled FP went 82→84 (**+2.4%**) for both enrichment arms — well
within margin.

- **`fused` PASSES** — recall +0.017, precision +0.001, FP +2.4%, **0/13 FOV
  regressions**, no stratum loses recall. This is the first phase in the engagement to
  cleanly pass the recall-first bar (Phase 2 failed +182% FP; Phase 3 failed −0.248
  recall).
- **`maxs` FAILS** the no-regression clause — pooled recall improves, but it reduces
  recall on 2/13 FOVs. Not eligible for a default flip.

## Verdict

**`vcorr_max_fused` is a clean, recommended default-flip candidate.** It strictly
dominates the baseline (no FOV regresses) and ties or beats `max_S` everywhere on the
no-regression standard, for a negligible FP/precision cost. Per `no_default_flip`, the
flag stays **OFF by default** until you approve the flip explicitly at this gate.

If you prefer not to flip yet, `stage1_ch2_source` remains a config-selectable
OFF-by-default option (no behavior change shipped).

## Caveat — GT cannot stratify recall by activity type

As in Phases 2 and 3, the anatomical Cellpose GT carries no activity labels, so missed
cells (FN) cannot be activity-typed — they all fall in `unknown`, making by-stratum
*recall* degenerate (tonic/silent strata are empty here = `lower_bound`). The per-FOV
overall recall + FP burden are therefore the load-bearing metrics, and the regression
check is operationalized per-FOV. The recall gains that *are* visible land in the
`phasic` stratum (the only typed stratum that grows: 265→273 TP for `fused`). The
intended single-firer/tonic benefit of `max_S` is largely invisible to this GT and is a
Phase-5 concern (requires tonic GT).

## Gate decision (resolved)

**APPROVED 2026-06-17 — default flipped to `stage1_ch2_source = "vcorr_max_fused"`**
(`types.py:303`). This is the first default change of the engagement; it cleanly meets
the recall-first bar (+0.017 recall, 0/13 FOV regressions, +2.4% FP). `vcorr_S` and
`max_S` remain config-selectable for reproducibility/rollback.

## Recommendation

1. **Flip the default to `stage1_ch2_source = "vcorr_max_fused"`** — only on your
   explicit approval (clean pass: +0.017 recall, 0/13 regressions, +2.4% FP). Otherwise
   keep it OFF and config-selectable. — **DONE (approved, see above).**
2. **Do not adopt `max_S` alone** — it regresses 2/13 FOVs (fails no-regression).
3. Re-evaluate the `max_S`/`fused` benefit against tonic GT in Phase 5, where the
   single-firer target the enrichment was designed for becomes measurable.
