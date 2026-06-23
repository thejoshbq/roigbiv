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

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.device import cuda_compute_capable
from roigbiv.pipeline.types import PipelineConfig


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
    None if too few peaks are found to be reliable. Delegates to
    ``optics.measure_soma_scale`` — the single source of truth for soma sizing,
    shared with the post-foundation scale-derivation path.
    """
    from roigbiv.pipeline.optics import measure_soma_scale
    scale = measure_soma_scale(img, n_peaks=n_peaks, box_radius=box_radius)
    return scale.diameter_med if scale.ok else None


def _effective_diameter(morph_input: np.ndarray, cfg: PipelineConfig) -> int:
    """Resolve the diameter for inference: cfg.diameter, or a per-image estimate
    when ``cfg.diameter_auto`` and the estimator succeeds. Backend-agnostic."""
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
    return effective_diameter


def _split_labels(
    label_image: np.ndarray, cellprob_map: np.ndarray
) -> tuple[list[np.ndarray], list[float], np.ndarray, np.ndarray]:
    """Split a label image into per-ROI boolean masks + per-ROI mean cellprob.
    Backend-agnostic: identical for the cellpose3 and cpsam paths."""
    masks_list: list[np.ndarray] = []
    probs_list: list[float] = []
    unique_ids = np.unique(label_image)
    unique_ids = unique_ids[unique_ids != 0]
    for lid in unique_ids:
        bmask = (label_image == lid)
        if not bmask.any():
            continue
        probs_list.append(float(cellprob_map[bmask].mean()))
        masks_list.append(bmask)
    return masks_list, probs_list, label_image, cellprob_map


def _resolve_stage1_ch2(
    vcorr_S: np.ndarray, max_S: Optional[np.ndarray], cfg: PipelineConfig
) -> tuple[np.ndarray, str]:
    """Resolve Cellpose's channel-2 content per ``cfg.stage1_ch2_source``.

    Channel-1 (morphology = mean_M) is fixed by the caller; this picks ch2 only
    (one variable). Default ``"vcorr_S"`` reproduces the historical behavior
    byte-for-byte. Falls back to vcorr_S (with a warning) when ``max_S`` is
    requested but unavailable (e.g. scout-mode foundation produces no max_S).
    Backend-agnostic: used by both the cellpose3 and cpsam paths.
    """
    src = getattr(cfg, "stage1_ch2_source", "vcorr_S")
    v = vcorr_S.astype(np.float32)
    if src == "vcorr_S":
        return v, "vcorr_S"
    if max_S is None:
        print(
            f"  WARNING: stage1_ch2_source={src!r} but max_S unavailable; "
            "falling back to vcorr_S",
            flush=True,
        )
        return v, "vcorr_S(fallback)"
    m = max_S.astype(np.float32)
    if src == "max_S":
        return m, "max_S"
    if src == "vcorr_max_fused":
        def _nz(a: np.ndarray) -> np.ndarray:
            lo, hi = float(a.min()), float(a.max())
            return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        # Normalize each to [0,1] BEFORE combining so neither raw scale dominates;
        # elementwise max = union of "looks correlated" OR "has a bright peak".
        fused = np.maximum(_nz(v), _nz(m)).astype(np.float32)
        return fused, "vcorr_max_fused"
    raise ValueError(
        f"unknown stage1_ch2_source {src!r} "
        "(expected 'vcorr_S', 'max_S', or 'vcorr_max_fused')"
    )


def _resolve_cpsam_python(cfg: PipelineConfig) -> str:
    """Locate the cp-sam (cellpose 4.x) env interpreter.

    Order: cfg.cpsam_sidecar_python → $ROIGBIV_CPSAM_PYTHON → sibling `cp-sam`
    conda env of the running interpreter. Raises if none exists.
    """
    candidates: list[str] = []
    spec = getattr(cfg, "cpsam_sidecar_python", "") or ""
    if spec:
        candidates.append(spec)
    env = os.environ.get("ROIGBIV_CPSAM_PYTHON")
    if env:
        candidates.append(env)
    # sys.prefix == <conda>/envs/roigbiv → sibling env <conda>/envs/cp-sam
    candidates.append(str(Path(sys.prefix).parent / "cp-sam" / "bin" / "python"))
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        "cpsam sidecar interpreter not found. Set cfg.cpsam_sidecar_python or "
        f"$ROIGBIV_CPSAM_PYTHON. Tried: {', '.join(candidates)}"
    )


def _run_cpsam_sidecar(
    x: np.ndarray, diameter: float, cfg: PipelineConfig, gpu: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Run Cellpose-SAM out-of-process in the cp-sam env; return
    ``(label_image uint16, cellprob_map float32)``. The denoised/in-process
    cellpose path is unaffected."""
    py = _resolve_cpsam_python(cfg)
    runner = _BASE_DIR / "scripts" / "cpsam_sidecar.py"
    if not runner.exists():
        raise FileNotFoundError(f"cpsam sidecar runner missing: {runner}")

    with tempfile.TemporaryDirectory(prefix="cpsam_") as td:
        td_p = Path(td)
        in_p = td_p / "input.npy"
        lab_p = td_p / "labels.npy"
        cp_p = td_p / "cellprob.npy"
        man_p = td_p / "manifest.json"
        np.save(str(in_p), x.astype(np.float32))
        man_p.write_text(json.dumps({
            "input": str(in_p),
            "labels_out": str(lab_p),
            "cellprob_out": str(cp_p),
            "diameter": float(diameter),
            "cellprob_threshold": float(cfg.cellprob_threshold),
            "flow_threshold": float(cfg.flow_threshold),
            "gpu": bool(gpu),
            "channel_axis": -1 if x.ndim == 3 else None,
        }))
        print(f"  cpsam sidecar: {py} {runner}", flush=True)
        proc = subprocess.run(
            [py, str(runner), str(man_p)],
            capture_output=True, text=True,
        )
        if proc.stdout.strip():
            for line in proc.stdout.strip().splitlines():
                print(f"  [cpsam] {line}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"cpsam sidecar failed (rc={proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )
        label_image = np.load(str(lab_p)).astype(np.uint16)
        cellprob_map = np.load(str(cp_p)).astype(np.float32)
    return label_image, cellprob_map


def run_cellpose_detection(
    mean_S: np.ndarray,
    vcorr_S: np.ndarray,
    cfg: PipelineConfig,
    *,
    max_S: Optional[np.ndarray] = None,
) -> tuple[list[np.ndarray], list[float], np.ndarray, np.ndarray]:
    """Run Cellpose inference on a dual-channel (morph, ch2) stack.

    ``max_S`` is optional and only consulted when ``cfg.stage1_ch2_source``
    selects it (Phase 4 channel-2 A/B); the default ``"vcorr_S"`` ignores it and
    reproduces the historical 2-channel (morph, vcorr_S) behavior exactly.

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

    # Channel-2 content selection (Phase 4 A/B; default vcorr_S = unchanged).
    ch2, ch2_label = _resolve_stage1_ch2(vcorr_S, max_S, cfg)
    if ch2_label != "vcorr_S":
        print(f"  Stage-1 channel-2 source: {ch2_label}", flush=True)

    backend = getattr(cfg, "stage1_backend", "cellpose3")
    if backend == "cpsam_sidecar":
        # cpsam is channel-invariant + noise-robust: no denoise, 2-channel
        # stack ([morph, ch2]); the channels=(1,2) role convention is inert.
        morph = mean_S.astype(np.float32)
        x = np.stack([morph, ch2], axis=-1)   # (H, W, 2)
        eff = _effective_diameter(morph, cfg)
        t0 = time.time()
        label_image, cellprob_map = _run_cpsam_sidecar(x, eff, cfg, gpu)
        print(f"  cpsam inference in {time.time()-t0:.2f}s", flush=True)
        return _split_labels(label_image, cellprob_map)
    if backend != "cellpose3":
        raise ValueError(
            f"unknown stage1_backend {backend!r} "
            "(expected 'cellpose3' or 'cpsam_sidecar')"
        )

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

    # Stack channels as (H, W, 2) with morph at channel 0, ch2 at channel 1.
    # Cellpose's channels=[1, 2] means "cyto = channel 1, nucleus = channel 2" (1-indexed).
    # ch2 defaults to vcorr_S (Phase 4 may substitute max_S / fused).
    H, W = mean_S_input.shape
    x = np.stack([mean_S_input, ch2], axis=-1)  # (H, W, 2)

    # Diameter auto-calibration: peak-detection + Otsu sizing on the mean
    # channel produces a robust per-FOV cell-scale estimate. Replaces
    # cfg.diameter for the main inference call below. Cellpose 3.x's bundled
    # SizeModel is only attached to `Cellpose(...)` wrapper instances; our
    # custom-trained `CellposeModel` doesn't carry one, so we estimate from
    # the image directly.
    effective_diameter = _effective_diameter(mean_S_input, cfg)

    t0 = time.time()
    masks, flows, styles = model.eval(
        x,
        diameter=effective_diameter,
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
    return _split_labels(label_image, cellprob_map)
