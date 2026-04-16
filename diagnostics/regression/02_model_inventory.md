# 02 — Model Inventory

## Deployed model

- `models/deployed/current_model` — 26 MB, modified 2026-04-07 17:03
- Source checkpoint: `run015_epoch_0280` (promoted in commit `ff58732`)
- Metrics (per commit message, memory): AP@0.5 = 0.609, AP@0.75 = 0.288, AP@0.9 = 0.078
- Base model: fine-tuned from Cellpose `cyto3`
- Training data: 99 image-mask pairs (85 train / 14 val) — expanded from run014's 80 FOVs

## Checkpoint sweep available (for Experiment 2.E if needed)

30 checkpoints under `models/checkpoints/models/`:
- `run015` (baseline) + `run015_epoch_0010` through `run015_epoch_0290` (every 10 epochs)
- Each 26 MB. All dated 2026-04-07 16:47–16:58.

## Lost / unavailable models

- run014 checkpoints — **deleted** after promotion of run015. Cannot empirically reproduce run014 AP@0.5=0.6805 baseline.
- run012 (original v0.1.0 release) — not present.

## "Latest" vs "old" model determination

- **Latest (regressed):** `models/deployed/current_model` = `run015_epoch_0280`
- **"Old" (baseline):** Two meanings possible:
  1. Stock `cyto3` (generalist, no fine-tuning) — our comparison reference
  2. run014 — unreachable (checkpoints deleted). Memory claims it detected more cells; can't verify.
- For diagnostic experiments, "old" = stock cyto3 and "newer" = run015_epoch_0280.

## Inference parameter defaults in current pipeline

From `roigbiv/pipeline/types.py` via `roigbiv/pipeline/stage1.py:129–137`:
- `diameter=12`
- `channels=[1, 2]` (dual-channel)
- `cellprob_threshold=-2.0` (permissive)
- `flow_threshold=0.6` (permissive)
- `normalize={"tile_norm_blocksize": 128}`
- `use_denoise=True` (applies Cellpose3 `denoise_cyto3` pre-inference)

## Notes

- Existing `scripts/compare_models.py` wraps an AP evaluation flow — reuse for Experiment 2.E rather than reimplement.
- Model path hard-coded in: `roigbiv/cli.py:45` (`_DEFAULT_MODEL`), `app.py:27` (`DEFAULT_MODEL`), `configs/pipeline.yaml:7` (`model_path`) — three places, all pointing at `models/deployed/current_model`.
