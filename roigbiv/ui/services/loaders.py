"""Unified data loading for the UI.

:class:`FOVBundle` is the Viewer/Review page's view of one pipeline output
directory: mean projection + ROIs (with pipeline + user corrections applied)
+ per-ROI polygon contours + optional cross-session global_cell_id map.

:class:`CrossSessionBundle` groups several FOVBundles sharing a single
``fov_id`` and a ``(session_id, local_label_id) → global_cell_id`` table, so
the viewer can color ROIs consistently across days.

Both bundles are cache-friendly — computed once per output_dir and stored in
:class:`roigbiv.ui.services.app_state.AppState`.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.corrections import (
    apply_corrections,
    load_corrections,
)
from roigbiv.pipeline.loaders import load_fov_from_output_dir
from roigbiv.pipeline.types import ROI


@dataclass
class ROIRender:
    """Viewer-ready per-ROI geometry.

    ``contours`` is a list of ``(y[], x[])`` tuples — one per topologically-
    distinct ring in the mask. Most ROIs have one ring; the list handles
    holes / disconnected components if they ever appear.
    """

    label_id: int
    source_stage: int
    gate_outcome: str
    activity_type: Optional[str]
    area: int
    centroid_yx: tuple[float, float]
    contours: list[tuple[list[float], list[float]]]
    global_cell_id: Optional[str] = None
    is_user: bool = False
    features: dict = field(default_factory=dict)


@dataclass
class FOVBundle:
    """One session / FOV output directory, decoded for the UI."""

    output_dir: Path
    stem: str
    mean_M: Optional[np.ndarray]
    shape: tuple[int, int]
    rois: list[ROIRender]
    registry: Optional[dict]
    session_id: Optional[str]
    fov_id: Optional[str]

    def roi_by_label(self, label_id: int) -> Optional[ROIRender]:
        for r in self.rois:
            if r.label_id == label_id:
                return r
        return None


@dataclass
class SessionRef:
    session_id: str
    session_date: Optional[date]
    output_dir: Path
    fov_posterior: Optional[float]


@dataclass
class CrossSessionBundle:
    fov_id: str
    animal_id: Optional[str]
    region: Optional[str]
    sessions: list[SessionRef]
    bundles: dict[str, FOVBundle]     # keyed by session_id


# ── FOV bundle ─────────────────────────────────────────────────────────────


def load_fov_bundle(output_dir: Path) -> FOVBundle:
    """Load a :class:`FOVBundle`, replaying any HITL corrections."""
    output_dir = Path(output_dir)
    fov, _review_queue = load_fov_from_output_dir(output_dir)
    shape_hw = _hw_shape(fov)

    ops = load_corrections(output_dir)
    rois = apply_corrections(fov.rois, ops, shape_hw) if ops else fov.rois

    registry = _maybe_json(output_dir / "registry_match.json")
    gcid_by_label = _gcid_by_label_from_registry(registry)

    rendered = [
        _render_roi(roi, gcid_by_label.get(int(roi.label_id)))
        for roi in rois
    ]

    return FOVBundle(
        output_dir=output_dir,
        stem=output_dir.name,
        mean_M=fov.mean_M,
        shape=shape_hw,
        rois=rendered,
        registry=registry,
        session_id=(registry or {}).get("session_id"),
        fov_id=(registry or {}).get("fov_id"),
    )


def _hw_shape(fov) -> tuple[int, int]:
    if fov.mean_M is not None:
        H, W = int(fov.mean_M.shape[0]), int(fov.mean_M.shape[1])
        return (H, W)
    if isinstance(fov.shape, tuple) and len(fov.shape) >= 3:
        return (int(fov.shape[1]), int(fov.shape[2]))
    raise ValueError("cannot infer FOV shape from loaded FOVData")


def render_roi(roi: ROI, gcid: Optional[str] = None) -> ROIRender:
    """Public renderer — used by both :func:`load_fov_bundle` and the Review page."""
    return _render_roi(roi, gcid)


def _render_roi(roi: ROI, gcid: Optional[str]) -> ROIRender:
    from skimage.measure import find_contours

    mask = roi.mask
    centroid = _centroid_yx(mask)
    contours: list[tuple[list[float], list[float]]] = []
    if mask is not None and mask.any():
        for ring in find_contours(mask.astype(float), 0.5):
            # find_contours returns (row, col) — keep as (y, x) for Plotly.
            ys = ring[:, 0].tolist()
            xs = ring[:, 1].tolist()
            contours.append((ys, xs))

    features = {}
    # Keep only JSON-native scalars — big arrays stay out of the UI payload.
    for k, v in (roi.features or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            features[k] = v

    return ROIRender(
        label_id=int(roi.label_id),
        source_stage=int(roi.source_stage),
        gate_outcome=str(roi.gate_outcome),
        activity_type=roi.activity_type,
        area=int(roi.area),
        centroid_yx=centroid,
        contours=contours,
        global_cell_id=gcid,
        is_user=bool(features.pop("user_added", False)),
        features=features,
    )


def _centroid_yx(mask: Optional[np.ndarray]) -> tuple[float, float]:
    if mask is None or not mask.any():
        return (0.0, 0.0)
    ys, xs = np.where(mask)
    return (float(ys.mean()), float(xs.mean()))


def _maybe_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _gcid_by_label_from_registry(registry: Optional[dict]) -> dict[int, str]:
    if not registry:
        return {}
    out: dict[int, str] = {}
    for entry in registry.get("cell_assignments", []):
        try:
            lid = int(entry.get("local_label_id"))
            gid = entry.get("global_cell_id")
        except (TypeError, ValueError):
            continue
        if gid:
            out[lid] = str(gid)
    return out


# ── Motion-correction preview discovery ────────────────────────────────────


def list_motion_corrected_fovs(workspace) -> list[tuple[str, str]]:
    """Previewable motion-corrected FOVs in a workspace, for the MC preview.

    Two sources, merged and de-duplicated by *output* stem
    (``tif.stem.replace("_mc", "")`` — the same key Foundation uses for its
    output directory name, so the two sources collide correctly):

    * **Processed FOVs** — Foundation has written its temporal mean
      (``{output_root}/{stem}/summary/mean_M.tif``). Surfaces a FOV as soon as
      motion correction is done, including foundation-only dry runs and FOVs
      still mid-pipeline. Value ``f"summary:{output_dir}"``.
    * **Pre-corrected inputs** — already-motion-corrected stacks sitting in the
      workspace (``*_mc.tif`` or content-tagged) that have *not* been run yet,
      so they have no summary. Drawn from ``workspace.tifs`` (already deduped +
      output-excluded by ``resolve_workspace``) so we never re-run
      ``discover_tifs`` here. Value ``f"input:{tif_path}"`` — rendered via an
      on-demand sampled temporal mean (:func:`mc_input_mean`).

    A processed summary always wins over a pre-corrected input for the same
    stem (the registered ``mean_M`` is the authoritative projection). The value
    string is self-describing (``summary:`` / ``input:`` prefix) because the
    render callback only receives the dropdown value, not the option's kind.

    Filesystem-based (not the registry) on purpose: the registry only lists
    fully-completed, *registered* FOVs.

    Returns ``[(label, value), ...]`` sorted by output stem. ``workspace`` is
    any object exposing ``output_root`` + ``tifs``; ``None`` or missing
    attributes degrade gracefully to whatever source is available.
    """
    # stem -> (label, value); insertion of summary entries first lets the input
    # pass skip any stem already covered by a (winning) summary.
    by_stem: dict[str, tuple[str, str]] = {}

    output_root = getattr(workspace, "output_root", None)
    if output_root is not None:
        output_root = Path(output_root)
        if output_root.exists():
            for mean_path in output_root.glob("*/summary/mean_M.tif"):
                out_dir = mean_path.parent.parent
                by_stem[out_dir.name] = (out_dir.name, f"summary:{out_dir}")

    for tif in getattr(workspace, "tifs", ()) or ():
        tif = Path(tif)
        out_stem = tif.stem.replace("_mc", "")
        if out_stem in by_stem:
            continue  # processed summary wins
        if not _is_precorrected_input(tif):
            continue
        by_stem[out_stem] = (f"{out_stem} (input)", f"input:{tif}")

    return [by_stem[stem] for stem in sorted(by_stem)]


def _is_precorrected_input(tif: Path) -> bool:
    """True if *tif* is an already-motion-corrected stack we can preview.

    Suffix-first: the free ``_mc`` filename check covers the common case without
    opening the file; only an inconclusive suffix falls through to the
    header-reading content check (TIFF Software tag), keeping per-tick cost low.
    """
    if tif.stem.endswith("_mc"):
        return True
    from roigbiv.io import detect_motion_corrected

    try:
        return bool(detect_motion_corrected(tif)[0])
    except Exception:  # noqa: BLE001 — unreadable input is simply not previewable
        return False


# ── Centroid-discovery overlay (Suite2p) ────────────────────────────────────

_DEFAULT_CENTROID_RADIUS = 8  # px — matches PipelineConfig.roi_stamp_radius default


def load_centroids(
    output_dir: Path,
    shape: tuple[int, int],
    radius: int = _DEFAULT_CENTROID_RADIUS,
) -> list["ROIRender"]:
    """Load ``output_dir/centroids.json`` (written by
    :func:`roigbiv.pipeline.centroids.run_centroid_discovery`) as
    :class:`ROIRender` objects, ready for :func:`build_roi_figure`.

    Each centroid is rendered as a fixed-radius disk (``radius``, default
    matching ``PipelineConfig.roi_stamp_radius``) — the same canonical-stamp
    convention pipeline ROIs use (ADR-0003) — rather than the detector's own
    footprint. ``source_stage=2`` is only a palette slot (these render in the
    standalone MC preview, not alongside cascade ROIs), not a claim about which
    detector found them; ``gate_outcome="accept"`` since this step has no gate
    to flag/reject against. Returns ``[]`` if no ``centroids.json`` exists yet.
    """
    from skimage.measure import find_contours

    from roigbiv.pipeline.roi_stamp import disk_mask

    path = Path(output_dir) / "centroids.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return []

    H, W = int(shape[0]), int(shape[1])
    out: list[ROIRender] = []
    for c in payload.get("centroids", []):
        mask = disk_mask(float(c["y"]), float(c["x"]), radius, H, W)
        contours: list[tuple[list[float], list[float]]] = []
        if mask.any():
            for ring in find_contours(mask.astype(float), 0.5):
                contours.append((ring[:, 0].tolist(), ring[:, 1].tolist()))
        out.append(ROIRender(
            label_id=int(c["label_id"]),
            source_stage=2,
            gate_outcome="accept",
            activity_type=None,
            area=int(c.get("npix", 0)),
            centroid_yx=(float(c["y"]), float(c["x"])),
            contours=contours,
            features={"cellpose_prob": float(c.get("cellpose_prob", 0.0))},
        ))
    return out


# ── Motion-correction input mean projection (on-demand, cached) ─────────────

_MC_PREVIEW_FRAMES = 64  # evenly-sampled frames for the input-stack temporal mean
_mc_mean_cache: dict[tuple, np.ndarray] = {}
_mc_mean_lock = threading.Lock()


def mc_input_mean(tif_path) -> Optional[np.ndarray]:
    """Temporal mean of a pre-corrected input stack, for the MC preview.

    Computes a mean projection over up to ``_MC_PREVIEW_FRAMES`` evenly-spaced
    frames (memory- and I/O-bounded: a raw ``_mc.tif`` may be many GB, unlike
    the tiny precomputed ``mean_M.tif``). Sampling is sufficient to reveal
    residual-motion blur/ghosting for a registration-quality preview.

    Cached by ``(resolved path, mtime_ns, size)`` so re-selecting a FOV is
    instant and a regenerated file invalidates. Thread-safe (Dash runs callbacks
    under multi-threaded Flask); the heavy read happens *outside* the lock so
    concurrent renders don't serialize. Returns ``None`` if unreadable.
    """
    import tifffile

    path = Path(tif_path)
    try:
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None

    cached = _mc_mean_cache.get(key)
    if cached is not None:
        return cached

    mean = _read_sampled_mean(path)
    if mean is None:
        return None
    with _mc_mean_lock:
        return _mc_mean_cache.setdefault(key, mean)


def _read_sampled_mean(path: Path) -> Optional[np.ndarray]:
    """Mean over ≤``_MC_PREVIEW_FRAMES`` evenly-spaced frames of a TIFF stack.

    Reads *only* the sampled pages (``TiffFile.asarray(key=...)``) so peak I/O
    and memory scale with the sample count, not the (possibly multi-GB) stack.
    Falls back to a full read + slice if keyed access isn't supported for the
    stack's layout. Returns ``None`` on any read error.
    """
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
            if len(shape) < 3:  # single frame — the mean is the frame itself
                return np.asarray(tf.asarray(), dtype=np.float32)
            n = int(shape[0])
            if n <= _MC_PREVIEW_FRAMES:
                idx = list(range(n))
            else:
                idx = np.linspace(0, n - 1, _MC_PREVIEW_FRAMES).round().astype(int)
                idx = sorted(set(int(i) for i in idx))
            try:
                sample = tf.asarray(key=idx, series=0)
            except Exception:  # noqa: BLE001 — layout doesn't support keyed read
                full = tf.asarray(series=0)
                flat = full.reshape(-1, *full.shape[-2:])
                sample = flat[idx]
    except Exception:  # noqa: BLE001 — unreadable / non-TIFF input
        return None

    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim == 2:  # a single sampled page
        return sample
    return sample.mean(axis=0)


# ── Cross-session bundle ───────────────────────────────────────────────────


def load_cross_session_bundle(fov_id: str, cfg=None) -> CrossSessionBundle:
    """Build a :class:`CrossSessionBundle` for every session tied to ``fov_id``.

    Pass ``cfg`` (a :class:`roigbiv.registry.config.RegistryConfig`) to read
    from a specific registry rather than the process-level env default.
    Sessions with missing output directories are silently dropped.
    """
    from roigbiv.registry import build_store

    store = build_store(cfg=cfg)
    store.ensure_schema()

    fov = store.get_fov(fov_id)
    _all_rows = sorted(
        store.list_sessions(fov_id), key=lambda s: s.session_date or date.min,
    )
    # Deduplicate by output_dir, keeping the row with the highest created_at.
    # Multiple rows can exist when the pipeline or backfill registers the same
    # output_dir more than once; treat the newest as authoritative (consistent
    # with store.get_session_by_output_dir).
    _best: dict[str, object] = {}
    for _row in _all_rows:
        _ex = _best.get(_row.output_dir)
        if _ex is None or (
            _row.created_at is not None
            and (_ex.created_at is None or _row.created_at > _ex.created_at)
        ):
            _best[_row.output_dir] = _row
    sessions_rows = sorted(_best.values(), key=lambda s: s.session_date or date.min)

    session_refs: list[SessionRef] = []
    bundles: dict[str, FOVBundle] = {}
    for row in sessions_rows:
        out_dir = Path(row.output_dir)
        if not out_dir.exists():
            continue
        try:
            bundle = load_fov_bundle(out_dir)
        except Exception:  # noqa: BLE001
            continue
        # Replace registry-derived gcids with the authoritative DB observations,
        # in case the on-disk registry_match.json is stale after a rematch.
        gcids_by_label = {
            int(obs.local_label_id): obs.global_cell_id
            for obs in store.list_observations_for_session(row.session_id)
        }
        for rr in bundle.rois:
            rr.global_cell_id = gcids_by_label.get(rr.label_id, rr.global_cell_id)

        session_refs.append(SessionRef(
            session_id=row.session_id,
            session_date=row.session_date,
            output_dir=out_dir,
            fov_posterior=row.fov_posterior,
        ))
        bundles[row.session_id] = bundle

    return CrossSessionBundle(
        fov_id=fov_id,
        animal_id=getattr(fov, "animal_id", None),
        region=getattr(fov, "region", None),
        sessions=session_refs,
        bundles=bundles,
    )
