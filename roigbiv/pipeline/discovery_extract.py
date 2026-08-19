"""Discovery-triggered trace extraction — the ``merged_masks.tif`` entry point.

Extracts mean (plus any requested extra statistics) directly from a FOV's
``merged_masks.tif`` — the one label image the registry actually
fingerprints/matches against (``roigbiv/registry/orchestrator.py``), produced
either by a full pipeline cascade or by
:func:`roigbiv.pipeline.workspace.run_tracking` stamping Discovery's saved
centroids. Unlike :mod:`roigbiv.pipeline.reextract`, which rebuilds ROIs from
HITL ``corrected_masks.tif``/``corrected_metadata.json`` and inherits
identifiers from an existing primary ``traces/`` sidecar, this entry point has
neither — it may be the FIRST extraction ever run for this FOV. ROI fields a
bare label image cannot supply fall back to the same sentinel values
``reextract.py`` already uses for HITL-added ROIs with no metadata
(``source_stage=99``, ``confidence="moderate"``, ``gate_outcome="accept"``).
``fs``/``Ly``/``Lx`` are read from Suite2p's own ``ops.npy`` (written by
Motion Correction, before Discovery or Tracking can produce a
``merged_masks.tif``) rather than a traces sidecar, since one may not exist
yet.

If ``registry_match.json`` exists for this FOV, it is passed straight through
as ``registry_report`` to ``build_sidecar`` — that function already reads
``cell_assignments`` generically off any dict shaped like
``register_or_match``'s return value (exactly what this file persists), so
unlike ``reextract.py`` no synthetic report needs to be forged here.

Writes to the primary ``traces/`` location when this FOV has none yet;
otherwise a revision-scoped ``traces/discovery-{hash12}/`` sibling, hashed the
same way as HITL corrections revisions, for idempotency. Never mutates an
existing primary bundle produced by an unrelated cascade run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from roigbiv.pipeline.overlap_correction import (
    correct_overlapping_traces,
    find_overlap_groups,
)
from roigbiv.pipeline.traces import extract_all_traces_full
from roigbiv.pipeline.traces_io import compute_corrections_rev, write_traces_bundle
from roigbiv.pipeline.types import ROI, PipelineConfig

log = logging.getLogger(__name__)


@dataclass
class _DiscoveryExtractFOV:
    """Thin FOVData-like shim — see ``reextract.py``'s ``_ReextractFOV`` for
    why this stays a small duplicate rather than a shared module: neither
    extraction path has detection-stage state to resurrect, and a shared shim
    would be a speculative abstraction over two five-field dataclasses.
    """

    data_bin_path: Path
    shape: tuple[int, int, int]
    output_dir: Path
    std_S: Optional[np.ndarray] = None
    rois: list = field(default_factory=list)


def extract_from_merged_masks(
    fov_output_dir: Path,
    *,
    cfg: Optional[PipelineConfig] = None,
    stats: tuple[str, ...] = (),
    skip_overlap_correction: bool = False,
) -> Path:
    """Extract mean (+ any requested extra statistics) from this FOV's
    current ``merged_masks.tif``.

    Parameters
    ----------
    fov_output_dir
        The FOV's pipeline output directory (must contain ``merged_masks.tif``
        and a Suite2p ``plane0/{ops.npy,data.bin}`` pair, at either
        ``suite2p/plane0/`` or ``{stem}/suite2p/plane0/`` — see
        ``resolve_suite2p``).
    cfg
        Optional override. Defaults to a ``PipelineConfig`` built from
        ``suite2p/plane0/ops.npy``'s ``fs`` (roigbiv-default neuropil params).
    stats
        Extra statistics beyond mean, e.g. ``("median", "mode")``. Mean is
        always extracted. Empty by default — mean-only.
    skip_overlap_correction
        If True, skip overlap correction even when overlap groups are
        present — mirrors ``reextract.py``'s flag. Overlap correction only
        ever applies to the mean trace; see the ``roigbiv.pipeline.traces``
        module docstring for why.

    Returns
    -------
    Path to the written bundle directory. Idempotent: if a matching
    ``discovery-{hash}/`` sidecar already exists, returns it without
    re-reading ``data.bin``.

    Raises
    ------
    FileNotFoundError
        If ``merged_masks.tif`` or the Suite2p ``ops.npy``/``data.bin`` pair
        is missing.
    """
    fov_output_dir = Path(fov_output_dir)

    masks_path = fov_output_dir / "merged_masks.tif"
    if not masks_path.exists():
        raise FileNotFoundError(
            f"no merged_masks.tif at {masks_path}; run centroid discovery, "
            "save boundaries, and run Tracking first."
        )

    data_bin_path, fov_shape, ops_fs = resolve_suite2p(fov_output_dir)

    if cfg is None:
        cfg = PipelineConfig(output_dir=fov_output_dir, no_viewer=True,
                              fs=ops_fs, frame_averaging=1)

    rois = _load_rois_from_label_image(masks_path)
    if not rois:
        raise ValueError(f"merged_masks.tif at {masks_path} has no labeled ROIs.")
    rois.sort(key=lambda r: int(r.label_id))

    corrections_rev = compute_corrections_rev(rois)
    primary_sidecar_path = fov_output_dir / "traces" / "traces_meta.json"
    is_primary = not primary_sidecar_path.exists()
    target_subdir = "traces" if is_primary else f"traces/discovery-{corrections_rev}"

    if not is_primary:
        target_sidecar = fov_output_dir / target_subdir / "traces_meta.json"
        if target_sidecar.exists():
            try:
                existing = json.loads(target_sidecar.read_text())
                if existing.get("corrections_rev") == corrections_rev:
                    log.info("discovery_extract: %s already up to date, skipping",
                             target_sidecar.parent)
                    return fov_output_dir / target_subdir
            except (OSError, json.JSONDecodeError):
                pass  # fall through and regenerate

    std_S = _maybe_load_std_S(fov_output_dir)
    fov_shim = _DiscoveryExtractFOV(
        data_bin_path=data_bin_path, shape=fov_shape,
        output_dir=fov_output_dir, std_S=std_S, rois=rois,
    )

    full = extract_all_traces_full(fov_shim, rois, cfg, stats=stats)
    F_raw, F_neu, F_corrected = full["mean"]

    if not skip_overlap_correction and std_S is not None:
        groups = find_overlap_groups(rois)
        if groups:
            F_corrected = correct_overlapping_traces(
                fov_shim, rois, groups, F_corrected, cfg,
            )
            log.info("discovery_extract: overlap correction applied to %d "
                     "ROI(s) across %d group(s)",
                     sum(len(g) for g in groups), len(groups))
    elif not skip_overlap_correction and std_S is None:
        log.warning("discovery_extract: std_S.tif missing; skipping overlap correction")

    registry_report = _load_registry_report(fov_output_dir)
    extra_stats = {name: full[name] for name in stats if name != "mean"}

    bundle_dir = write_traces_bundle(
        rois, F_raw, F_neu, F_corrected, fov_output_dir, cfg,
        source="discovery",
        registry_report=registry_report,
        corrections_rev=None if is_primary else corrections_rev,
        data_bin_path=data_bin_path,
        fov_shape=fov_shape,
        subdir=target_subdir,
        extra_stats=extra_stats or None,
    )
    log.info("discovery_extract: wrote %s", bundle_dir)
    return bundle_dir


# ── helpers ────────────────────────────────────────────────────────────────


def resolve_suite2p(fov_output_dir: Path) -> tuple[Path, tuple[int, int, int], float]:
    """``(data_bin_path, (T, Ly, Lx), fs)`` from Suite2p's own ``ops.npy``.

    Motion Correction writes this before Discovery or Tracking can produce a
    ``merged_masks.tif``, so it's a source of truth that doesn't depend on a
    traces sidecar existing yet (unlike ``reextract.py``'s provenance-based
    resolution, which requires one).

    Two layouts exist on disk (see ``resume.py::_suite2p_plane_dir``, the
    other place this ambiguity is handled): ``run_suite2p_fov`` itself always
    writes to ``{fov_output_dir}/{stem}/suite2p/plane0/`` (an extra
    ``{stem}``-named subdirectory), while some FOVs have since been
    flattened to ``{fov_output_dir}/suite2p/plane0/`` directly. Try both,
    preferring whichever actually has ``data.bin`` on disk.
    """
    stem = fov_output_dir.name
    candidates = (
        fov_output_dir / stem / "suite2p" / "plane0",
        fov_output_dir / "suite2p" / "plane0",
    )
    plane0 = next((c for c in candidates if (c / "data.bin").exists()), candidates[0])
    ops_path = plane0 / "ops.npy"
    data_bin_path = plane0 / "data.bin"
    if not ops_path.exists() or not data_bin_path.exists():
        raise FileNotFoundError(
            f"Suite2p ops.npy/data.bin not found under {plane0}; "
            "run Motion Correction first."
        )
    ops = np.load(ops_path, allow_pickle=True).item()
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    fs = float(ops.get("fs", 30.0))
    bytes_per_frame = Ly * Lx * 2  # int16
    T = data_bin_path.stat().st_size // bytes_per_frame
    return data_bin_path, (T, Ly, Lx), fs


def _load_rois_from_label_image(masks_path: Path) -> list[ROI]:
    """Rebuild ``ROI`` objects from ``merged_masks.tif`` (uint16 label image).

    No per-ROI morphology/confidence/gate metadata exists for a bare label
    image — fields it can't supply fall back to the same sentinel values
    ``reextract.py::_load_corrected_rois`` uses for HITL-added ROIs.
    """
    label_img = tifffile.imread(str(masks_path))
    rois: list[ROI] = []
    for label in np.unique(label_img):
        label_id = int(label)
        if label_id == 0:
            continue
        mask = label_img == label_id
        if not mask.any():
            continue
        rois.append(ROI(
            mask=mask.astype(bool),
            label_id=label_id,
            source_stage=99,
            confidence="moderate",
            gate_outcome="accept",
            area=int(mask.sum()),
            solidity=0.0,
            eccentricity=0.0,
            nuclear_shadow_score=0.0,
            soma_surround_contrast=0.0,
        ))
    return rois


def _maybe_load_std_S(fov_output_dir: Path) -> Optional[np.ndarray]:
    """Load ``summary/std_S.tif`` if it exists (needed for overlap correction)."""
    path = fov_output_dir / "summary" / "std_S.tif"
    if not path.exists():
        return None
    try:
        return np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    except Exception:  # noqa: BLE001
        log.warning("discovery_extract: failed to read %s", path)
        return None


def _load_registry_report(fov_output_dir: Path) -> Optional[dict]:
    """This FOV's persisted ``registry_match.json``, or ``None`` if it hasn't
    gone through Tracking yet — rows just won't carry ``global_cell_id``."""
    path = fov_output_dir / "registry_match.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
