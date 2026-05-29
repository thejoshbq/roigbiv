# 04 — Pipeline Code Analysis

## A. Cellpose inference parameter table

### New pipeline (`roigbiv/pipeline/stage1.py:80–168`)

| Parameter | Value | Source |
|-----------|-------|--------|
| model | `pretrained_model=models/deployed/current_model` | `cfg.cellpose_model`, `_resolve_model_path()` |
| diameter | 12 | `cfg.diameter` (types.py default) |
| cellprob_threshold | -2.0 | `cfg.cellprob_threshold` |
| flow_threshold | 0.6 | `cfg.flow_threshold` |
| channels | `[1, 2]` (dual) | `cfg.channels` |
| channel_axis | -1 | hardcoded |
| normalize | `{"tile_norm_blocksize": 128}` | `cfg.tile_norm_blocksize` |
| use_denoise | **True** | `cfg.use_denoise` — applies `DenoiseModel(model_type="denoise_cyto3")` to channel-1 input before inference |
| Input ch 1 | `mean_M` (raw movie mean) | Passed from `run.py:235` |
| Input ch 2 | `vcorr_S` (residual 8-nbhd temporal correlation) | Passed from `run.py:235` |

### Reference "old" path (`scripts/run_inference.py:89–124`)

Same thresholds/diameter, but:
- `--denoise` CLI flag defaults to **False**
- Single-image input (mean projection), not dual-channel
- Uses `models/deployed/current_model` by default (same model as new pipeline)

## B. Preprocessing chain (raw TIFF → Cellpose input)

In `roigbiv/pipeline/foundation.py`:

1. **Motion correction wrapper** (foundation.py:38–96) — Suite2p pass with `do_registration=False` for `*_mc.tif`; produces `data.bin`.
2. **L+S separation** (foundation.py:205–298) — temporal binning to ~5000 frames, GPU `torch.svd_lowrank()` for top-200 components, `L = UkΣkVkᵀ` with `k=cfg.k_background` (30), `S = M − L`. Writes `residual_S.dat` memmap.
3. **Summary images on S** (foundation.py:315–423) — `mean_S`, `max_S`, `std_S`, `vcorr_S`. Note `mean_S ≈ 0` by construction (line 513–515).
4. **DoG nuclear-shadow on mean_M** (foundation.py:430–450, 527–530) — `G(σ=6) − G(σ=2)` applied to raw mean for Gate 1 features.
5. **mean_M reconstructed from `data.bin` or read from ops["meanImg"]** (foundation.py:516–525).

Then Stage 1 (`stage1.py:112–121`):
6. If `cfg.use_denoise` (default True) → `denoise_mean_S(mean_M)` via Cellpose3 `DenoiseModel`, else pass-through.
7. Stack as `[mean_M_input, vcorr_S]` into `(H, W, 2)` and call `model.eval()`.

## C. Postprocessing chain (Cellpose masks → final ROI set)

### Stage 1 → Gate 1 (`roigbiv/pipeline/gate1.py:131–178`)

For each Cellpose-proposed mask, compute area / solidity / eccentricity / nuclear_shadow_score / soma_surround_contrast / cellpose_prob, then apply rules:

- **Hard REJECT** if any:
  - `area ∉ [min_area=80, max_area=350]`
  - `solidity < min_solidity=0.55`
  - `eccentricity > max_eccentricity=0.90`
  - `nuclear_shadow_score < dog_strong_neg_threshold` AND `soma_surround_contrast ≤ min_contrast=0.10`
- **FLAG** if exactly one criterion misses within a margin (area ±20, solidity ±0.05, ecc ±0.03, contrast ±0.03).
- **ACCEPT** otherwise.

### Stage 1 → subtraction → Stage 2/3/4

Accept+flag ROIs subtracted from `S` using `std_S` spatial profiles (`subtraction.py`). Stage 2 uses cached Suite2p `stat.npy`/`iscell.npy` from Foundation; Gate 2 (`gate2.py`) rejects high-correlation duplicates. Stages 3/4 operate on deeper residuals.

**Stage 1 scope is Cellpose + Gate 1. The regression diagnosis is scoped to these two.**

## D. Red flags (ranked)

Evidence from `stage1_report.json` analysis:

1. **🚩 Gate 1 `max_area=350` is killing 22 high-confidence Cellpose detections** — see 05_problem_fovs.md. 17 rejects + 5 flags all cite `area_high:+N` as the *only* gate reason. Rejects have mean cellpose_prob=1.31 (vs accept mean 0.71) — Cellpose is MORE confident about the rejects than the average accept. This is the top hypothesis.
2. **Cellpose3 `use_denoise=True`** — new preprocessing wrapper applied to mean_M. If denoise changes intensity texture, the fine-tuned run015 weights no longer match training-time distribution. Still a candidate, but doesn't explain the area_high rejections.
3. **Model swap run014 → run015** — AP@0.5 dropped 0.68 → 0.61. Could explain *some* missed detections if Cellpose itself doesn't propose the masks. But since rejects show cellpose_prob > 1.0, Cellpose is finding the cells — the model isn't the Stage 1 bottleneck on this FOV.
4. **Dual-channel vcorr_S dilution** — hypothetical, low evidence until 2.B runs.

## E. Suite2p gating

Gate 2 (`gate2.py`) rejects Stage 2 candidates that correlate too highly (|r| ≥ 0.7) or anti-correlate (r ≤ −0.5) with nearby Stage 1 ROIs. Relevant only if the missed cells are silent/dim neurons not proposed by Cellpose — i.e., the fallback stages. Given Cellpose *did* propose the 22 large ROIs that Gate 1 killed, Suite2p gating is **not the primary issue**.
