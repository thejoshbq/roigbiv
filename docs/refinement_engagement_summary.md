# Recall-Refinement Engagement — Plain-Language Summary

*What we set out to do, every major decision so far, and why — in simple terms.*
Last updated: 2026-06-18 (Phase 5b A/B running).

---

## The goal, in one paragraph

ROI G. Biv finds neurons in two-photon calcium-imaging movies. It works in
**stages**: each stage detects some cells, "subtracts" them from the movie, and
passes the leftovers to the next stage. This engagement has one aim: **catch more
of the bright, always-on ("tonic") and high-baseline neurons that the pipeline
tends to miss, and modernize the first detection stage — without disturbing
anything that already works** (the stage-by-stage spine, the quality gates, the
review/export tools, the cross-session registry).

## The ground rules (why we move slowly)

- **One change at a time.** Every experiment changes exactly one knob, so we know
  what caused any difference.
- **"Recall-first" bar.** A change is only accepted if it **doesn't lose any real
  cells** in any movie *and* doesn't flood the human reviewer with too many new
  false alarms (we set explicit limits each time).
- **Test on a locked-away set.** Every change is A/B-tested on the same **13
  held-out movies** the model never trained on.
- **Off by default.** New behavior ships **turned off**. Flipping a default
  requires a passing A/B *and* your explicit approval.
- **Look before you leap.** If the actual code contradicts the plan, we **stop and
  report** instead of forcing the plan through.

### One caveat that shadows the whole engagement

Our "answer key" (ground truth) only marks **where** cells are — not **what kind**
they are. So it literally **cannot grade tonic-cell recall**. It can grade overall
cell-finding and false-alarm rate, which is what we lean on. The tonic-specific
goal needs a different kind of answer key we don't yet have.

---

## The decisions, phase by phase

### Phase 0 — Look before coding
Read the code first. Two surprises:
- **The first planned change was already done.** Plan said "feed the raw movie
  average to the detector instead of the near-zero background-subtracted one." The
  code already did this. → **Skipped that phase.**
- **The shiny new model needs a software upgrade that would break things.** (Handled
  in Phase M.)

### Phase 2 — Extra movie denoising → ❌ Rejected (kept OFF)
- **Idea:** clean up leftover noise in the movie so faint cells stand out (a method
  called PMD).
- **Result:** found a few more cells (+0.039 recall) but **tripled the false
  alarms (+182%)**, and did **nothing** for the tonic cells it was meant to help.
- **Simple why:** smoothing the movie made the later detector "trigger-happy" — it
  fired on lots of noise that now looked cell-shaped. Bad trade. **Rejected.**

### Phase M — New Cellpose version: use a "sidecar," don't upgrade in place
- **Plan said:** upgrade the cell-detector library (Cellpose 3 → 4) to unlock the
  new "SAM" model.
- **Discovery:** a full upgrade would **break the lab's custom-trained model** (the
  new version can't even load it), break other components, **and make the very
  comparison we wanted impossible** (you need both versions side-by-side to compare
  them).
- **Decision:** keep the main environment untouched; run the new model in a
  separate **sidecar environment** when needed.
- **Simple why:** don't rip out the whole engine just to test a new spark plug.

### Phase 3 — New model vs. current model → ✅ Keep the current one (CP3)
- **Idea:** maybe the big new general-purpose "SAM" model beats our lab's older
  fine-tuned one.
- **Result:** the new model **missed far more cells** (−0.248 recall) and lost on
  **all 13 movies**. Its only "win" — fewer false alarms — was just because it
  found fewer cells overall.
- **Simple why:** a **specialist trained on this exact lab's images** beats a
  **generalist that has never seen them**. Kept the current model.
- **Side note:** we noticed turning *off* the denoise step gave a tiny gain, but it
  hurt one movie, so we flagged it for a separate test rather than adopting it.

### Phase 4 — What to put in the detector's 2nd input → ✅ Adopted "fused" *(the one win so far)*
- **Background:** the detector takes **two input images** — one for cell **shape**,
  one for cell **activity**. We tested three things for the activity image:
  the usual *correlation map*, a *peak-brightness map*, or a **fused** blend of
  both.
- **Result:** the **fused** image found a few more cells (+0.017 recall), **no
  movie lost any cells**, and false alarms rose only **+2.4%** (limit was +15%).
- **This was the first clean pass of the whole engagement** → with your approval,
  this became the **new default**. (The peak-brightness-only option was rejected —
  it hurt 2 movies.)
- **Simple why:** the fused image highlights both steadily-correlated cells *and*
  occasional bright firers, so the detector sees a few it used to miss — at almost
  no cost.

### Denoise re-test (its own gate) → ❌ Keep denoise ON
- The Phase-3 hint ("denoise-off helps a little") was **re-tested on the new fused
  default**. The gain **vanished** and **4 of 13 movies got worse**.
- **Simple why:** the earlier gain was a fluke of the old setup; once we improved
  the input channel, turning denoise off only hurt. Kept denoise **ON**.

### Phase 5a — A new "baseline elevation" measurement → ✅ Added (measurement only)
- **Idea:** to spot tonic (always-on, bright) cells, measure **how much brighter a
  cell's resting level is than the ring of tissue around it**. Tonic cells sit
  *persistently* above their surroundings; ordinary flickering cells don't.
- **How it's computed simply:** take the cell's steady baseline and the
  surrounding-ring's steady baseline (using a low, slow estimate so brief flashes
  don't count), and report the relative difference.
- **Important:** this only **records a number** — it changes no decisions yet. It's
  the raw material for Phase 5b.

### Phase 5b — Let confident tonic cells skip review → ⏳ In progress
- **Idea:** if an anatomically-detected cell looks convincingly tonic (high baseline
  elevation), let it be **auto-accepted** so a human doesn't have to review it.
  (Cells found by the dedicated tonic-search stage are left exactly as-is.)
- **Discovery surprise (two problems):**
  1. **The standard scorer can't see this change.** It scores which cell *locations*
     exist — but this change only re-routes cells *out of the review queue*, it
     doesn't add or remove any. So the usual A/B would show "no change."
  2. **There's no tonic answer key** to check whether auto-accepted cells are
     really cells.
- **What we did:** built a **custom, review-aware scorer** that asks "of the cells
  this rule auto-accepts, how many are real (match the location answer key) vs.
  potentially junk escaping review?" — and reports how much human review it saves.
- **Status:** the 13-movie measurement run is going now. Next: pick a safe
  threshold, write the report, and **stop for your approval** before turning
  anything on.

---

## Scorecard so far

| Phase | Change tested | Outcome | Why |
|---|---|---|---|
| 0 | Discovery | Skipped Phase 1 | Already done in code |
| 2 | PMD movie denoising | ❌ Rejected (OFF) | +182% false alarms, no tonic gain |
| M | Cellpose 4 upgrade | ✅ Sidecar, not upgrade | Full upgrade would break working parts |
| 3 | New SAM model vs CP3 | ✅ Keep CP3 | New model lost on all 13 movies |
| 4 | Detector 2nd-channel content | ✅ **Fused (new default)** | +recall, 0 movies regress, +2.4% FP |
| denoise | Denoise off (re-test) | ❌ Keep ON | Gain vanished, 4 movies got worse |
| 5a | Baseline-elevation measure | ✅ Added (measure only) | Foundation for tonic decisions |
| 5b | Auto-accept tonic cells | ⏳ Running | Needs custom scorer + your approval |

**Net so far:** one real improvement adopted (Phase 4 fused channel), several
plausible-but-wrong ideas correctly rejected before they could do harm, and the
tonic work set up carefully on honest footing. Nothing load-bearing was disturbed.
