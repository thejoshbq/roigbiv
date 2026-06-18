# Confirmatory denoise A/B — `use_denoise` on the post-Phase-4 default

**Engagement:** recall-refinement (gated, one-variable, recall-first).
**Branch:** `feat/0-denoise-ab` (cut from Phase-4 head). **Driver:**
`experiments/phase_denoise_ab/run_off_arm.py`. **Summary:**
`experiments/phase_denoise_ab/summarize.py`. **Raw:**
`experiments/runs/phase_denoise_ab/off_results.json` (OFF arm) +
`experiments/runs/phase4_channel/ab_results.json` `{stem}|fused` (ON arm, reused).

## Why this gate exists

Phase 3's *secondary* finding was that turning `denoise_cyto3` (channel-1 `mean_M`
restoration) OFF beat the deployed config by recall +0.013 / precision +0.019 /
FP −10%. It was **not** auto-adopted because (a) it touches a separate variable
(`use_denoise`) and (b) it regressed 1/13 FOVs at that time.

That measurement was taken at the **old** Stage-1 channel-2 = `vcorr_S`. Phase 4
then flipped the default channel-2 to `vcorr_max_fused`. `use_denoise` acts on
channel-1 while the Phase-4 flip changed channel-2 — the two interact at the CP3
detector, so per `one_variable_per_experiment` the denoise delta had to be
re-measured on the new default before any flip. This gate does exactly that: the
single variable is `use_denoise`, with `stage1_ch2_source=vcorr_max_fused`,
`stage1_backend=cellpose3`, model, and thresholds all fixed.

denoise-ON + fused already existed as the Phase-4 `fused` arm (no recompute);
only the denoise-OFF + fused arm was run (13 incremental full-pipeline runs).

## Results (micro-averaged, pooled TP/FP/FN across 13 FOVs)

| arm | recall | precision | TP | FP | FN |
|---|---|---|---|---|---|
| denoise-ON (deployed) | 0.641 | 0.805 | 346 | 84 | 194 |
| denoise-OFF (candidate) | **0.641** | **0.818** | 346 | **77** | 194 |

- **Recall Δ (off − on): +0.000.** The Phase-3 +0.013 recall gain did **not**
  transfer to the new default — it was an artifact of the `vcorr_S` baseline.
- Precision +0.013, pooled FP 84 → 77 (**−8.3%**).

**Per-FOV regression check (the load-bearing test):** denoise-OFF improves recall
on 4/13, ties 5/13, and **regresses 4/13**:

| FOV | on | off | Δ |
|---|---|---|---|
| `T1_2221230_…_LOW-D9_FOV1_BEH_PT2-002` | 0.882 | 0.855 | −0.026 |
| `T1_230118_…_EXT-D10_FOV2_PRE-000` | 0.641 | 0.628 | −0.013 |
| `T1_230215_…_LOW-D1_FOV1_BEH-001` | 0.261 | 0.217 | −0.043 |
| `T1_230308_…_EXT-D2_FOV2_BEH-003` | 0.963 | 0.926 | −0.037 |

## Recall-first bar

Stated margin (same as Phase 4): **no per-FOV recall regression, and pooled
post-review FP increase ≤ +15%.**

- no-regression: **FAIL** — 4/13 FOVs lose recall.
- FP ≤ +15%: PASS (−8.3%, a precision *gain*).

## Verdict

**FAIL — keep `use_denoise = True` (default unchanged).** denoise-OFF buys no
recall and a modest precision/FP improvement, but the engagement is recall-first
and denoise-OFF regresses recall on 4 FOVs. The deployed CP3 checkpoint was
fine-tuned with `denoise_cyto3` in-loop, so OFF is off-distribution — consistent
with the per-FOV instability seen here. No flip; `use_denoise` stays
config-selectable (already is).

This closes the denoise question raised as a Phase-3 secondary: it does **not**
hold on the shipped default. If the precision win is ever wanted, it must be
re-justified against the recall-first bar (and would still regress these 4 FOVs).
