"""
ROI G. Biv pipeline — Stage 1: Spatial Detection via Cellpose (spec §4).

Detects neurons with clear spatial morphology — the highest-confidence
detection stage. Uses Cellpose3 image restoration on the denoised mean of S,
then runs dual-channel inference (channel 1 = denoised mean, channel 2 = Vcorr).

Dual-channel rationale (spec §4):
  - Mean projection misses dim/tonic neurons (low spatial contrast).
  - Vcorr highlights temporally-coherent activity, complementary to brightness.
  - Cellpose combines both for recall on dim-but-active and bright-but-silent cells.

Parameter defaults (spec §18.2, Plan agent D7):
  diameter=12, cellprob_threshold=-2.0, flow_threshold=0.6, channels=[1,2],
  normalize={'tile_norm_blocksize': 128}
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.device import cuda_compute_capable
from roigbiv.pipeline.types import PipelineConfig


# Cellpose 3.x built-in model names. Spec strings matching one of these are
# passed through to CellposeModel(model_type=...) without filesystem lookup.
_CELLPOSE_BUILTINS = frozenset({
    "cyto", "cyto2", "cyto3", "cpsam",
    "nuclei", "tissuenet", "livecell",
    "yeast_PhC", "yeast_BF",
})


def _resolve_model_path(model_spec: str) -> str:
    """Resolve a Cellpose model spec to either a built-in name or an absolute path.

    Resolution order for non-builtin specs:
      1. As given (absolute paths, or relative paths starting with `./`)
      2. Relative to the current working directory
      3. Relative to the roigbiv package root (so the default model resolves
         regardless of where the CLI was invoked from)

    Raises FileNotFoundError if no candidate exists. We explicitly do NOT fall
    back to stock cyto3: an unresolvable path means the pipeline would
    silently use the wrong model, which is exactly the bug this guards
    against.
    """
    if model_spec in _CELLPOSE_BUILTINS:
        return model_spec

    p = Path(model_spec)
    candidates: list[Path] = []
    if p.is_absolute() or model_spec.startswith("."):
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(Path(__file__).resolve().parents[2] / p)

    for cand in candidates:
        resolved = cand.resolve()
        if resolved.exists():
            return str(resolved)

    tried = ", ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        f"Cellpose model spec {model_spec!r} not resolvable. "
        f"Tried: {tried}. Pass an absolute path, a path relative to cwd or "
        f"the roigbiv package root, or a Cellpose built-in name "
        f"(one of: {sorted(_CELLPOSE_BUILTINS)})."
    )


def denoise_mean_S(mean_S: np.ndarray, gpu: bool = True) -> np.ndarray:
    """Apply Cellpose3 image restoration (denoise_cyto3) to mean_S.

    First call downloads ~30 MB of model weights. Subsequent calls are ~fast.

    Parameters
    ----------
    mean_S : (H, W) float32
    gpu    : bool — pass to DenoiseModel

    Returns
    -------
    (H, W) float32 denoised mean
    """
    from cellpose.denoise import DenoiseModel

    # DenoiseModel expects (H, W) or (H, W, C) input; for single channel we pass (H, W, 1)
    x = mean_S.astype(np.float32)
    if x.ndim == 2:
        x_in = x[:, :, None]
    else:
        x_in = x

    dn = DenoiseModel(model_type="denoise_cyto3", gpu=gpu, nchan=1)
    # DenoiseModel.eval returns the restored image(s)
    out = dn.eval(x_in, channels=None, channel_axis=-1, normalize=True, tile=True)
    # The output should be a (H, W, 1) or (H, W) array; squeeze to (H, W)
    if isinstance(out, list):
        out = out[0]
    out = np.asarray(out).squeeze()
    if out.ndim != 2:
        out = out.reshape(mean_S.shape)
    return out.astype(np.float32)


def run_cellpose_detection(
    mean_S: np.ndarray,
    vcorr_S: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[list[np.ndarray], list[float], np.ndarray, np.ndarray]:
    """Run Cellpose inference on a dual-channel (mean_S, vcorr_S) stack.

    Returns
    -------
    masks_list       : list of (H, W) bool — one binary mask per detected ROI
    probs_list       : list of float — per-ROI cellpose probability (from centroid of cellprob map)
    label_image      : (H, W) uint16 — labeled image (0 = background)
    cellprob_map     : (H, W) float32 — continuous cellpose probability map
    """
    from cellpose.models import CellposeModel

    if cfg.force_cpu:
        gpu = False
    elif not cuda_compute_capable():
        print(
            "  WARNING: CUDA device detected but compute probe failed "
            "(sm/CC mismatch — PyTorch build lacks kernels for this GPU); "
            "falling back to CPU for Cellpose.",
            flush=True,
        )
        gpu = False
    else:
        gpu = True
    model_path = _resolve_model_path(cfg.cellpose_model)

    # Cellpose 3.x silently constructs a default model when given a missing
    # `pretrained_model` path, so we route built-in names through `model_type`
    # explicitly and trust `_resolve_model_path` to have raised on bad paths.
    if model_path in _CELLPOSE_BUILTINS:
        model = CellposeModel(gpu=gpu, model_type=model_path)
    else:
        model = CellposeModel(gpu=gpu, pretrained_model=model_path)
    print(f"  Cellpose model loaded: {model_path}", flush=True)

    # Optionally denoise mean_S
    t0 = time.time()
    if cfg.use_denoise:
        try:
            mean_S_input = denoise_mean_S(mean_S, gpu=gpu)
            print(f"  Cellpose3 denoise in {time.time()-t0:.2f}s", flush=True)
        except Exception as exc:
            print(f"  WARNING: Cellpose3 denoise failed ({exc}); using raw mean_S",
                  flush=True)
            mean_S_input = mean_S.astype(np.float32)
    else:
        mean_S_input = mean_S.astype(np.float32)

    # Stack channels as (H, W, 2) with mean at channel 0, Vcorr at channel 1.
    # Cellpose's channels=[1, 2] means "cyto = channel 1, nucleus = channel 2" (1-indexed).
    H, W = mean_S_input.shape
    x = np.stack([mean_S_input, vcorr_S.astype(np.float32)], axis=-1)  # (H, W, 2)

    t0 = time.time()
    masks, flows, styles = model.eval(
        x,
        diameter=cfg.diameter,
        cellprob_threshold=cfg.cellprob_threshold,
        flow_threshold=cfg.flow_threshold,
        channels=list(cfg.channels),
        channel_axis=-1,
        normalize={"tile_norm_blocksize": cfg.tile_norm_blocksize},
    )
    print(f"  Cellpose inference in {time.time()-t0:.2f}s", flush=True)

    # Ensure label image is uint16 (max 65535 ROIs is plenty)
    label_image = np.asarray(masks, dtype=np.uint16)

    # Cellpose 3.x: flows[2] is the cellprob map (dense float probability)
    # flows tuple structure: (RGB flow, XY flows (dy, dx), cellprob, styles...)
    cellprob_map = None
    if isinstance(flows, (list, tuple)) and len(flows) >= 3:
        cp = np.asarray(flows[2], dtype=np.float32)
        if cp.shape == label_image.shape:
            cellprob_map = cp
    if cellprob_map is None:
        # Fall back to a map where each ROI pixel has a constant prob = 1.0
        cellprob_map = (label_image > 0).astype(np.float32)

    # Split labels into per-ROI boolean masks; extract per-ROI prob from centroid
    masks_list = []
    probs_list = []
    unique_ids = np.unique(label_image)
    unique_ids = unique_ids[unique_ids != 0]
    for lid in unique_ids:
        bmask = (label_image == lid)
        if not bmask.any():
            continue
        # centroid probability: mean of cellprob over the mask
        prob = float(cellprob_map[bmask].mean())
        masks_list.append(bmask)
        probs_list.append(prob)

    return masks_list, probs_list, label_image, cellprob_map
