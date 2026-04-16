# 07 — Root Cause Determination

## Root cause (high confidence)

**Gate 1's `max_area=350` threshold rejects 17 of 101 Cellpose candidates + flags 5 more, all on size alone — and those 22 ROIs are real, high-confidence neurons.** This is the regression.

### Evidence (ordered)

1. **100% of rejects cite only `area_high`.** No other morphological criterion contributes (solidity, eccentricity, contrast, DoG shadow all pass).
2. **Reject `cellpose_prob` mean (1.31) exceeds accept mean (0.71).** The filter is selectively removing Cellpose's most confident detections.
3. **Stock `cyto3` (no fine-tuning, no pipeline) finds 16/17 rejects + 5/5 flags** (config 2.B(a)). With auto-diameter it finds 17/17 + 5/5. Both models independently agree these are cells.
4. **Visual inspection** (`2D_bypass_gate1/raw_candidates_overlay.png`): rejects are the brightest, best-shaped pyramidal neurons in the FOV — the exact cells a user would call out as "missing."
5. **Size distribution of the 101 candidates is continuous** across 350 with no natural discontinuity; `max_area=350` is an arbitrary cutoff.

### What is NOT the cause

- **Fine-tuning regression (H3):** ruled out. Fine-tuned model proposes all 22 cells as raw candidates (same as cyto3). Model is finding them; Gate 1 is removing them.
- **Cellpose3 denoise (H1):** not causal for this symptom. Denoise can't explain `area_high` rejections. (Might matter for other edge cases; not for the reported regression.)
- **Dual-channel `vcorr_S` dilution (H4):** real effect on stock cyto3 (dual cuts stock recall from 96 → 25) but NOT on the fine-tuned model (it was trained on this input and hits 101 candidates). Production-correct as designed.
- **Suite2p gating:** irrelevant — this regression is fully contained in Stage 1 + Gate 1.

## Secondary finding (out of scope for this fix)

Auto-diameter Cellpose finds **30 cells beyond the 101-candidate set** (config 2.B(d), with auto-diameter). The fixed `diameter=12` may under-sample the true cell-size distribution in this FOV. This is a **recall opportunity**, not the reported regression. Capture as future work (Trello card) rather than bundle with this fix.

## Confidence

- Root cause (Gate 1 max_area): **very high**. The evidence is quantitative, reproducible, and visual.
- Remaining ambiguity: none for the reported symptom.
