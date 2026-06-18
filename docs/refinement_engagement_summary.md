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

### Phase 5b — Let confident tonic cells skip review → ⏹️ Built (OFF), but inert on this data
- **Idea:** if an anatomically-detected cell looks convincingly tonic (high baseline
  elevation), let it be **auto-accepted** so a human doesn't have to review it.
  (Cells found by the dedicated tonic-search stage are left exactly as-is.)
- **First surprise:** the standard scorer **can't even see this change** — it scores
  which cell *locations* exist, but this only re-routes cells *out of the review
  queue*. And there's **no tonic answer key**. So we built a **custom, review-aware
  scorer**.
- **Decisive result:** across all 13 movies (433 cells), there are **zero tonic
  cells of any kind.** The classifier labels **none** tonic, and the dedicated
  tonic-search stage found **nothing**. The rule has nothing to act on.
- **Why:** the classifier's "tonic" test looks for a slow *oscillation* pattern that
  none of these cells have. The new 5a "elevation" measure *does* fire (62% of cells
  sit clearly above their surroundings) — but that's **not the same thing as tonic**
  (bright flickering cells are elevated too). The two measure different phenomena,
  and with no tonic answer key we can't tell which definition is "right."
- **Decision (5b-A):** keep the tier **OFF** as a safe, ready no-op. The real
  blocker is **data** (these movies have ~no tonic cells, and we have no tonic
  ground truth) — not code. Don't redefine "tonic" blindly. **Engagement concluded
  here.**

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
| 5b | Auto-accept tonic cells | ⏹️ Built (OFF), inert | No tonic cells in data; no tonic GT |

## Engagement conclusion (2026-06-18)

**Concluded at Phase 5b (option 5b-A).**

- **One adopted improvement:** Phase 4's **fused** second channel — +0.017 recall,
  0/13 movies regressed, +2.4% false alarms. The only change that cleanly passed
  the recall-first bar; default flipped with approval.
- **Four ideas correctly rejected before they could do harm:** PMD denoising
  (false-alarm explosion), the SAM model swap (lost on every movie), denoise-off
  (gain was a fluke), and tonic auto-accept (nothing to act on).
- **One diagnostic banked:** the 5a baseline-elevation measure is logged on every
  ROI for future use.
- **The honest wall:** the tonic-recall goal can't be advanced or even *measured*
  on the current data — these movies contain ~no tonic cells by the working
  definition, and there's no tonic ground truth to define them differently. The
  next real step is a **data/labeling** task (human-label tonic cells → tonic GT),
  not more code.
- **Nothing load-bearing was touched** — the subtractive spine, gates, provenance,
  HITL, registry, and virtual-residual engine are all unchanged. New behavior that
  wasn't adopted remains OFF-by-default and config-selectable for future re-tests.

**Where the work lives:** the cumulative line (Phase-4 fused default + 5a feature +
5b OFF tier + all reports) is on branch **`feat/0-phase5-tonic`**. Per the
engagement's `git_discipline`, nothing was merged to `main` — merging is your call.
