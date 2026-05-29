"""HITL-aware trace re-extraction.

Re-runs trace extraction against the corrected ROI set (``corrected_masks.tif``
+ ``corrected_metadata.json``), writes a revision-scoped sibling bundle at
``{fov_output_dir}/traces/corrections-{hash12}/``, and never mutates the
primary ``traces/traces.npy``.

Inherits identifiers from the primary sidecar — ``session_id`` / ``fov_id`` /
``registry_decision`` copy over, and each ROI that survived corrections
with its original ``label_id`` keeps its ``global_cell_id``. Fresh labels
minted by add / merge / split get no ``global_cell_id`` (re-extract never
writes to the registry DB).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from roigbiv.pipeline.corrections import (
    corrected_masks_path,
    corrected_metadata_path,
    corrections_log_path,
)
from roigbiv.pipeline.overlap_correction import (
    correct_overlapping_traces,
    find_overlap_groups,
)
from roigbiv.pipeline.traces import extract_all_traces
from roigbiv.pipeline.traces_io import (
    compute_corrections_rev,
    write_traces_bundle,
)
from roigbiv.pipeline.types import ROI, PipelineConfig

log = logging.getLogger(__name__)


@dataclass
class _ReextractFOV:
    """Thin FOVData-like shim with just the fields extract_all_traces +
    correct_overlapping_traces need. Avoids resurrecting residual paths or
    detection state that re-extract has no source of truth for.
    """

    data_bin_path: Path
    shape: tuple[int, int, int]
    output_dir: Path
    std_S: Optional[np.ndarray] = None
    rois: list = field(default_factory=list)


def reextract_from_corrections(
    fov_output_dir: Path,
    *,
    cfg: Optional[PipelineConfig] = None,
    skip_overlap_correction: bool = False,
) -> Path:
    """Re-extract traces from the current HITL corrections.

    Parameters
    ----------
    fov_output_dir
        The FOV's pipeline output directory (contains ``traces/``,
        ``corrections/``, ``summary/``).
    cfg
        Optional override. Defaults to a ``PipelineConfig`` reconstructed
        from the primary sidecar's ``fs`` / ``frame_averaging`` (plus
        roigbiv-default neuropil params). Pass an explicit cfg to customize.
    skip_overlap_correction
        If True, skip the overlap-correction step even when overlap groups
        are present. Useful for tests and when std_S.tif is missing.

    Returns
    -------
    Path to the written ``traces/corrections-{rev}/`` directory. Idempotent:
    if the target already exists with a matching ``corrections_rev`` sidecar,
    returns the existing path without re-reading ``data.bin``.
    """
    fov_output_dir = Path(fov_output_dir)

    primary_sidecar_path = fov_output_dir / "traces" / "traces_meta.json"
    if not primary_sidecar_path.exists():
        raise FileNotFoundError(
            f"primary traces sidecar missing at {primary_sidecar_path}; "
            "run the pipeline first."
        )
    primary_sidecar = json.loads(primary_sidecar_path.read_text())

    # Reconstruct a cfg if the caller didn't supply one.
    if cfg is None:
        cfg = _cfg_from_sidecar(primary_sidecar)

    # Load corrected ROIs.
    rois = _load_corrected_rois(fov_output_dir)
    if not rois:
        raise ValueError(
            f"no corrected ROIs found under {fov_output_dir / 'corrections'}."
        )
    rois.sort(key=lambda r: int(r.label_id))

    corrections_rev = compute_corrections_rev(rois)
    target_subdir = f"traces/corrections-{corrections_rev}"
    target_sidecar = fov_output_dir / target_subdir / "traces_meta.json"
    if target_sidecar.exists():
        try:
            existing = json.loads(target_sidecar.read_text())
            if existing.get("corrections_rev") == corrections_rev:
                log.info("reextract: %s already up to date, skipping",
                         target_sidecar.parent)
                return fov_output_dir / target_subdir
        except (OSError, json.JSONDecodeError):
            # Fall through and regenerate.
            pass

    # Resolve data.bin + shape + std_S.
    data_bin_path, fov_shape = _resolve_data_bin(primary_sidecar, fov_output_dir)
    std_S = _maybe_load_std_S(fov_output_dir)

    fov_shim = _ReextractFOV(
        data_bin_path=data_bin_path,
        shape=fov_shape,
        output_dir=fov_output_dir,
        std_S=std_S,
        rois=rois,
    )

    F_raw, F_neu, F_corrected = extract_all_traces(fov_shim, rois, cfg)

    # Mirror run.py's overlap-correction pass so corrections-rev traces are
    # consistent with the pipeline's treatment (skip only if requested or
    # std_S is unavailable — overlap correction requires it).
    if not skip_overlap_correction and std_S is not None:
        groups = find_overlap_groups(rois)
        if groups:
            F_corrected = correct_overlapping_traces(
                fov_shim, rois, groups, F_corrected, cfg,
            )
            log.info("reextract: overlap correction applied to %d ROI(s) "
                     "across %d group(s)",
                     sum(len(g) for g in groups), len(groups))
    elif not skip_overlap_correction and std_S is None:
        log.warning("reextract: std_S.tif missing; skipping overlap correction")

    # Inherit registry identifiers + per-label global_cell_id from primary.
    inherited_report = _synthetic_report_from_primary(primary_sidecar, rois)

    bundle_dir = write_traces_bundle(
        rois,
        F_raw,
        F_neu,
        F_corrected,
        fov_output_dir,
        cfg,
        source="corrections",
        registry_report=inherited_report,
        corrections_rev=corrections_rev,
        data_bin_path=data_bin_path,
        fov_shape=fov_shape,
        corrections_log=corrections_log_path(fov_output_dir),
        subdir=target_subdir,
    )
    log.info("reextract: wrote %s", bundle_dir)
    return bundle_dir


# ── helpers ────────────────────────────────────────────────────────────────


def _cfg_from_sidecar(primary_sidecar: dict) -> PipelineConfig:
    """Build a minimal PipelineConfig from the primary sidecar."""
    neuropil = primary_sidecar.get("neuropil") or {}
    return PipelineConfig(
        fs=float(primary_sidecar.get("fs", 30.0)),
        frame_averaging=int(primary_sidecar.get("frame_averaging", 1)),
        neuropil_coeff=float(neuropil.get("alpha", 0.7)),
        neuropil_inner_buffer=int(neuropil.get("inner_buffer", 2)),
        neuropil_outer_radius=int(neuropil.get("outer_radius", 15)),
    )


def _load_corrected_rois(fov_output_dir: Path) -> list[ROI]:
    """Rebuild ROI objects from ``corrected_masks.tif`` + ``corrected_metadata.json``.

    The metadata JSON is a list of :meth:`ROI.to_serializable` dicts (no mask
    pixels — those come from the label image). Entries missing from the
    label image are skipped.
    """
    masks_path = corrected_masks_path(fov_output_dir)
    meta_path = corrected_metadata_path(fov_output_dir)
    if not masks_path.exists() or not meta_path.exists():
        return []

    label_img = tifffile.imread(str(masks_path))
    meta = json.loads(meta_path.read_text())

    rois: list[ROI] = []
    for entry in meta:
        try:
            label_id = int(entry["label_id"])
        except (KeyError, TypeError, ValueError):
            continue
        mask = label_img == label_id
        if not mask.any():
            continue
        rois.append(ROI(
            mask=mask.astype(bool),
            label_id=label_id,
            source_stage=int(entry.get("source_stage", 99)),
            confidence=str(entry.get("confidence", "moderate")),
            gate_outcome=str(entry.get("gate_outcome", "accept")),
            area=int(entry.get("area", int(mask.sum()))),
            solidity=float(entry.get("solidity", 0.0)),
            eccentricity=float(entry.get("eccentricity", 0.0)),
            nuclear_shadow_score=float(entry.get("nuclear_shadow_score", 0.0)),
            soma_surround_contrast=float(entry.get("soma_surround_contrast", 0.0)),
            activity_type=entry.get("activity_type"),
        ))
    return rois


def _resolve_data_bin(
    primary_sidecar: dict,
    fov_output_dir: Path,
) -> tuple[Path, tuple[int, int, int]]:
    """Pick a usable ``data.bin`` path from the primary sidecar's provenance.

    Prefer a workspace-relative path (portable across machines), then the
    recorded absolute path. Raises with a clear error if neither exists.
    """
    prov = primary_sidecar.get("provenance") or {}
    fov_shape = prov.get("fov_shape")
    if not fov_shape or len(fov_shape) != 3:
        raise ValueError(
            "primary sidecar has no fov_shape; cannot reconstruct data.bin geometry."
        )
    shape_tuple = tuple(int(x) for x in fov_shape)

    candidates: list[Path] = []
    rel = prov.get("data_bin_path")
    if rel:
        # Walk up from the FOV output dir to find a workspace root that
        # contains the relative path (typically ...<workspace>/output/<stem>).
        for parent in [fov_output_dir, *fov_output_dir.parents]:
            candidates.append(parent / rel)
    absolute = prov.get("data_bin_path_abs")
    if absolute:
        candidates.append(Path(absolute))

    for c in candidates:
        if c.exists():
            return c.resolve(), shape_tuple

    raise FileNotFoundError(
        f"data.bin not found. Tried: {[str(c) for c in candidates]}. "
        "Record a correct provenance.data_bin_path in traces/traces_meta.json "
        "or pass an explicit cfg with a resolvable path."
    )


def _maybe_load_std_S(fov_output_dir: Path) -> Optional[np.ndarray]:
    """Load ``summary/std_S.tif`` if it exists (needed for overlap correction)."""
    path = fov_output_dir / "summary" / "std_S.tif"
    if not path.exists():
        return None
    try:
        return np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    except Exception:  # noqa: BLE001
        log.warning("reextract: failed to read %s", path)
        return None


def _synthetic_report_from_primary(
    primary_sidecar: dict,
    corrected_rois: list[ROI],
) -> Optional[dict]:
    """Forge a registry-report-shaped dict so `build_sidecar` can fill in
    ``session_id`` / ``fov_id`` / per-row ``global_cell_id`` the same way it
    does on the primary pipeline path.

    Propagates ``global_cell_id`` only for labels present in both the
    primary sidecar and the current corrected ROI set.
    """
    session_id = primary_sidecar.get("session_id")
    fov_id = primary_sidecar.get("fov_id")
    decision = primary_sidecar.get("registry_decision")
    if not fov_id and not session_id:
        return None

    primary_rows = primary_sidecar.get("rois") or []
    gid_by_label: dict[int, str] = {}
    for row in primary_rows:
        gid = row.get("global_cell_id")
        if gid is None:
            continue
        try:
            gid_by_label[int(row["local_label_id"])] = str(gid)
        except (KeyError, TypeError, ValueError):
            continue

    corrected_labels = {int(r.label_id) for r in corrected_rois}
    cell_assignments = [
        {"local_label_id": lid, "global_cell_id": gid, "match_kind": "inherited"}
        for lid, gid in gid_by_label.items()
        if lid in corrected_labels
    ]
    return {
        "decision": decision,
        "session_id": session_id,
        "fov_id": fov_id,
        "cell_assignments": cell_assignments,
    }
