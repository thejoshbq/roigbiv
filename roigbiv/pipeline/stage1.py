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

from roigbiv.pipeline.device import cuda_unavailable_reason
from roigbiv.pipeline.types import PipelineConfig


def _cuda_fallback_message(reason: str) -> str:
    """Accurate CPU-fallback diagnostic keyed by the device probe reason."""
    if reason == "oom":
        return (
            "GPU present but out of VRAM (another process is holding it); "
            "falling back to CPU for Cellpose. Free the GPU (e.g. unload the "
            "local-Qwen model) or rerun — the pipeline's GPU preflight "
            "(gpuguard) normally does this automatically."
        )
    if reason == "sm_mismatch":
        return (
            "CUDA device detected but the PyTorch build lacks kernels for this "
            "GPU's compute capability (sm/CC mismatch); falling back to CPU for "
            "Cellpose. Install a torch build that targets this GPU."
        )
    if reason == "no_cuda":
        return "No usable CUDA device; running Cellpose on CPU."
    return (
        f"CUDA compute probe failed ({reason}); falling back to CPU for Cellpose."
    )


_BASE_DIR: Path = Path(__file__).resolve().parents[2]

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
        candidates.append(_BASE_DIR / p)

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


def list_available_models() -> list[dict]:
    """Return dbc.Select-compatible options for all available Cellpose models.

    Order: deployed model first, then checkpoints (newest first), then builtins.
    Values are strings accepted by _resolve_model_path().
    """
    options: list[dict] = []

    deployed = _BASE_DIR / "models" / "deployed" / "current_model"
    if deployed.exists():
        options.append({
            "label": "current_model (deployed)",
            "value": "models/deployed/current_model",
        })

    checkpoints_dir = _BASE_DIR / "models" / "checkpoints" / "models"
    if checkpoints_dir.is_dir():
        checkpoint_files = sorted(
            (f for f in checkpoints_dir.iterdir() if f.is_file()),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for cf in checkpoint_files:
            options.append({"label": cf.name, "value": str(cf.relative_to(_BASE_DIR))})

    for name in sorted(_CELLPOSE_BUILTINS):
        options.append({"label": name, "value": name})

    return options


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


def _estimate_diameter_px(
    img: np.ndarray,
    n_peaks: int = 30,
    box_radius: int = 40,
) -> Optional[float]:
    """Robust per-image cell-diameter estimate via DoG peaks + Otsu sizing.

    Returns the median equivalent-diameter (px) across detected somata, or
    None if too few peaks are found to be reliable.
    """
    try:
        from skimage.feature import peak_local_max
        from skimage.filters import difference_of_gaussians, threshold_otsu
        from skimage.measure import label as _label, regionprops
    except ImportError:
        return None

    arr = img.astype(np.float32)
    H, W = arr.shape
    # DoG sigmas span GRIN→prism cell radii (~3 to ~15 px).
    dog = difference_of_gaussians(arr, low_sigma=3.0, high_sigma=15.0)
    peaks = peak_local_max(
        dog, min_distance=20, threshold_rel=0.15,
        num_peaks=n_peaks, exclude_border=box_radius,
    )
    diameters: list[float] = []
    for (y, x) in peaks:
        y0, y1 = max(0, y - box_radius), min(H, y + box_radius)
        x0, x1 = max(0, x - box_radius), min(W, x + box_radius)
        crop = arr[y0:y1, x0:x1]
        if crop.size < 100:
            continue
        try:
            t = threshold_otsu(crop)
        except Exception:
            continue
        labels = _label(crop > t)
        cy, cx = y - y0, x - x0
        target = labels[cy, cx]
        if target == 0:
            continue
        for r in regionprops((labels == target).astype(np.uint8)):
            if 30 <= r.area <= 8000:
                diameters.append(float(r.equivalent_diameter))
            break
    if len(diameters) < 5:
        return None
    return float(np.median(diameters))


def run_cellpose_detection(
    morph_channel: np.ndarray,
    vcorr_S: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[list[np.ndarray], list[float], np.ndarray, np.ndarray]:
    """Run Cellpose inference on a dual-channel (morphology, vcorr_S) stack.

    ``morph_channel`` is the morphological channel. The live pipeline passes
    ``fov.mean_M`` (the raw registered-movie mean), NOT ``mean_S``: under the
    default SVD background ``mean_S ≈ 0`` (per-pixel brightness is absorbed into
    L), so ``mean_M`` carries the contrast Cellpose's training expects. See
    ``run_pipeline`` (roigbiv/pipeline/run.py).

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
    else:
        _reason = cuda_unavailable_reason()
        if _reason is None:
            gpu = True
        else:
            print(f"  WARNING: {_cuda_fallback_message(_reason)}", flush=True)
            gpu = False
    model_path = _resolve_model_path(cfg.cellpose_model)

    # Cellpose 3.x silently constructs a default model when given a missing
    # `pretrained_model` path, so we route built-in names through `model_type`
    # explicitly and trust `_resolve_model_path` to have raised on bad paths.
    if model_path in _CELLPOSE_BUILTINS:
        model = CellposeModel(gpu=gpu, model_type=model_path)
    else:
        model = CellposeModel(gpu=gpu, pretrained_model=model_path)
    print(f"  Cellpose model loaded: {model_path}", flush=True)

    # Optionally denoise the morphological channel
    t0 = time.time()
    if cfg.use_denoise:
        try:
            morph_input = denoise_mean_S(morph_channel, gpu=gpu)
            print(f"  Cellpose3 denoise in {time.time()-t0:.2f}s", flush=True)
        except Exception as exc:
            print(f"  WARNING: Cellpose3 denoise failed ({exc}); using raw morphology channel",
                  flush=True)
            morph_input = morph_channel.astype(np.float32)
    else:
        morph_input = morph_channel.astype(np.float32)

    # Build the Cellpose input. Two modes, selected by cfg.channels:
    #   dual-channel (1, 2): morphology at ch0, Vcorr at ch1 — the GRIN default;
    #     Cellpose reads Vcorr as a "nucleus" stain to recover dim-but-active cells.
    #   single-channel (0, 0): morphology only. On dim/diffuse PRISM FOVs the Vcorr
    #     nucleus channel actively SUPPRESSES segmentation (Phase-A isolation: cyto3
    #     0→13 by dropping it), so PRISM/generic profiles run single-channel.
    H, W = morph_input.shape
    single_channel = tuple(cfg.channels)[1] == 0
    if single_channel:
        x = morph_input.astype(np.float32)                                # (H, W)
        channel_axis = None
    else:
        x = np.stack([morph_input, vcorr_S.astype(np.float32)], axis=-1)  # (H, W, 2)
        channel_axis = -1

    # Diameter auto-calibration: peak-detection + Otsu sizing on the mean
    # channel produces a robust per-FOV cell-scale estimate. Replaces
    # cfg.diameter for the main inference call below. Cellpose 3.x's bundled
    # SizeModel is only attached to `Cellpose(...)` wrapper instances; our
    # custom-trained `CellposeModel` doesn't carry one, so we estimate from
    # the image directly.
    effective_diameter = cfg.diameter
    if cfg.diameter_auto:
        t_cal = time.time()
        d_est = _estimate_diameter_px(morph_input)
        if d_est is not None and d_est > 4.0:
            effective_diameter = int(round(d_est))
            print(
                f"  diameter_auto: image estimate {effective_diameter}px "
                f"(overriding cfg.diameter={cfg.diameter}) "
                f"[calibration {time.time()-t_cal:.2f}s]",
                flush=True,
            )
        else:
            print(
                f"  WARNING: diameter_auto image estimate failed "
                f"(d_est={d_est}); keeping cfg.diameter={cfg.diameter}.",
                flush=True,
            )

    t0 = time.time()
    masks, flows, styles = model.eval(
        x,
        diameter=effective_diameter,
        cellprob_threshold=cfg.cellprob_threshold,
        flow_threshold=cfg.flow_threshold,
        channels=list(cfg.channels),
        channel_axis=channel_axis,
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
