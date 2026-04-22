"""Per-ROI trace persistence with stable identifiers for pynapse handoff.

Writes a self-describing ``traces/`` bundle alongside each FOV's pipeline
outputs. Primary consumer is `pynapse.SignalRecording`, which reads the
``.npy`` array and identifies neurons by row index only — the row-to-ID
mapping lives entirely in ``traces_meta.json``.

Layout::

    {fov_output_dir}/traces/
        traces.npy           # F_corrected, float32 (n_rois, n_frames) — PRIMARY
        traces_raw.npy       # F_raw
        traces_neuropil.npy  # F_neu (always written; neuropil.present flags content)
        traces_meta.json

When HITL corrections are applied, `reextract_from_corrections` writes a
sibling ``corrections-{hash12}/`` subdir with the same four files. The
primary ``traces.npy`` is never mutated.

The sidecar is **byte-deterministic** across reruns of the same inputs and
registry state: ``sort_keys=True``, no wall-clock fields, ROI order locked
by ``label_id``, corrections_rev computed from the replayed ROI set (not
the JSONL bytes, which change under undo/redo).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.types import ROI, PipelineConfig

SCHEMA_VERSION = 1


# ── corrections_rev ────────────────────────────────────────────────────────


def compute_corrections_rev(rois_corrected: list[ROI]) -> str:
    """Return a 12-char SHA256 over the replayed ROI set.

    Canonical projection per ROI: ``label_id | sha256(mask.bytes) | activity_type``.
    Stable under undo/redo (hashing the JSONL directly is not — it carries
    per-op uuids and wall-clock ts, and `write_corrections` overwrites the
    log on undo).
    """
    h = hashlib.sha256()
    for r in sorted(rois_corrected, key=lambda x: int(x.label_id)):
        h.update(str(int(r.label_id)).encode())
        h.update(b"|")
        if r.mask is not None:
            h.update(hashlib.sha256(np.ascontiguousarray(r.mask).tobytes()).digest())
        h.update(b"|")
        h.update((r.activity_type or "").encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


# ── sidecar construction ───────────────────────────────────────────────────


def build_sidecar(
    rois_sorted: list[ROI],
    F_corrected: np.ndarray,
    cfg: PipelineConfig,
    *,
    source: str,
    registry_report: Optional[dict] = None,
    corrections_rev: Optional[str] = None,
    data_bin_path: Optional[Path] = None,
    fov_shape: Optional[tuple[int, int, int]] = None,
    corrections_log: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
) -> dict:
    """Build the ``traces_meta.json`` payload.

    Parameters
    ----------
    rois_sorted
        ROI list already sorted by ``label_id`` (same ordering used for rows).
    F_corrected
        The primary trace matrix; only its shape/dtype are read here.
    cfg
        Carries ``fs`` (effective Hz in roigbiv convention) and ``frame_averaging``.
    source
        ``"pipeline"`` | ``"corrections"``.
    registry_report
        The dict returned by ``register_or_match``, or ``None`` when the
        pipeline ran without registry or the decision was ``"review"``.
    corrections_rev
        12-char hash from `compute_corrections_rev`; ``None`` for pipeline source.
    data_bin_path, workspace_root
        Provenance — so re-extract can find ``data.bin`` on future runs.
        If ``workspace_root`` is given and ``data_bin_path`` is inside it,
        we record a relative path. Absolute path is always recorded.
    """
    n_rois, n_frames = int(F_corrected.shape[0]), int(F_corrected.shape[1])

    # Build per-row identifier entries
    cell_map: dict[int, str] = {}
    if registry_report is not None:
        for ca in registry_report.get("cell_assignments") or []:
            try:
                cell_map[int(ca["local_label_id"])] = ca["global_cell_id"]
            except (KeyError, TypeError, ValueError):
                continue

    rois_payload: list[dict] = []
    for i, roi in enumerate(rois_sorted):
        entry: dict = {
            "row_index": int(i),
            "local_label_id": int(roi.label_id),
            "source_stage": int(roi.source_stage),
            "gate_outcome": str(roi.gate_outcome),
            "confidence": str(roi.confidence),
        }
        if roi.activity_type is not None:
            entry["activity_type"] = roi.activity_type
        gid = cell_map.get(int(roi.label_id))
        if gid is not None:
            entry["global_cell_id"] = str(gid)
        rois_payload.append(entry)

    # Top-level registry block
    if registry_report is None:
        session_id: Optional[str] = None
        fov_id: Optional[str] = None
        registry_decision: Optional[str] = None
    else:
        registry_decision = registry_report.get("decision")
        session_id = registry_report.get("session_id")
        fov_id = registry_report.get("fov_id")

    # Neuropil presence (derived upstream; default to True unless caller
    # explicitly flagged empty via a stored features dict — handled in writer).
    neuropil = {
        "alpha": float(cfg.neuropil_coeff),
        "inner_buffer": int(cfg.neuropil_inner_buffer),
        "outer_radius": int(cfg.neuropil_outer_radius),
    }

    provenance: dict = {}
    if data_bin_path is not None:
        abs_bin = Path(data_bin_path).resolve()
        provenance["data_bin_path_abs"] = str(abs_bin)
        if workspace_root is not None:
            try:
                rel = abs_bin.relative_to(Path(workspace_root).resolve())
                provenance["data_bin_path"] = str(rel)
            except ValueError:
                provenance["data_bin_path"] = None
    if fov_shape is not None:
        provenance["fov_shape"] = [int(x) for x in fov_shape]
    if corrections_log is not None:
        provenance["corrections_log"] = str(corrections_log)

    fs = float(cfg.fs)
    frame_averaging = int(cfg.frame_averaging)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "fs": fs,
        "frame_averaging": frame_averaging,
        "effective_fps": fs,
        "n_rois": n_rois,
        "n_frames": n_frames,
        "shape": [n_rois, n_frames],
        "dtype": "float32",
        "session_id": session_id,
        "fov_id": fov_id,
        "registry_decision": registry_decision,
        "corrections_rev": corrections_rev,
        "neuropil": neuropil,
        "provenance": provenance,
        "rois": rois_payload,
        "files": {
            "primary": "traces.npy",
            "raw": "traces_raw.npy",
            "neuropil": "traces_neuropil.npy",
        },
    }


# ── bundle writer ──────────────────────────────────────────────────────────


def write_traces_bundle(
    rois_sorted: list[ROI],
    F_raw: np.ndarray,
    F_neu: np.ndarray,
    F_corrected: np.ndarray,
    output_dir: Path,
    cfg: PipelineConfig,
    *,
    source: str,
    registry_report: Optional[dict] = None,
    corrections_rev: Optional[str] = None,
    data_bin_path: Optional[Path] = None,
    fov_shape: Optional[tuple[int, int, int]] = None,
    corrections_log: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
    subdir: str = "traces",
) -> Path:
    """Write ``traces.npy`` / ``traces_raw.npy`` / ``traces_neuropil.npy`` /
    ``traces_meta.json`` into ``{output_dir}/{subdir}/``.

    Returns the bundle directory path. Idempotent — overwrites existing files.
    """
    bundle_dir = Path(output_dir) / subdir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    F_corrected = np.ascontiguousarray(F_corrected, dtype=np.float32)
    F_raw = np.ascontiguousarray(F_raw, dtype=np.float32)
    F_neu = np.ascontiguousarray(F_neu, dtype=np.float32)

    np.save(str(bundle_dir / "traces.npy"), F_corrected)
    np.save(str(bundle_dir / "traces_raw.npy"), F_raw)
    np.save(str(bundle_dir / "traces_neuropil.npy"), F_neu)

    sidecar = build_sidecar(
        rois_sorted,
        F_corrected,
        cfg,
        source=source,
        registry_report=registry_report,
        corrections_rev=corrections_rev,
        data_bin_path=data_bin_path,
        fov_shape=fov_shape,
        corrections_log=corrections_log,
        workspace_root=workspace_root,
    )
    # Neuropil presence: all-zero output signals an empty-annulus degenerate run.
    sidecar["neuropil"]["present"] = bool(F_neu.size and np.any(F_neu != 0.0))

    (bundle_dir / "traces_meta.json").write_text(
        json.dumps(sidecar, sort_keys=True, indent=2) + "\n"
    )
    return bundle_dir


# ── single-shot finalize for run.py / workspace.py ─────────────────────────


def finalize_fov_bundle(
    rois_sorted: list[ROI],
    F_raw: np.ndarray,
    F_neu: np.ndarray,
    F_corrected: np.ndarray,
    output_dir: Path,
    cfg: PipelineConfig,
    *,
    registry_report: Optional[dict] = None,
    data_bin_path: Optional[Path] = None,
    fov_shape: Optional[tuple[int, int, int]] = None,
    workspace_root: Optional[Path] = None,
) -> Path:
    """Write the primary ``traces/`` bundle after the pipeline (and registry,
    if it ran) have finished. Called once per FOV, from both classic and
    workspace paths.
    """
    return write_traces_bundle(
        rois_sorted,
        F_raw,
        F_neu,
        F_corrected,
        output_dir,
        cfg,
        source="pipeline",
        registry_report=registry_report,
        corrections_rev=None,
        data_bin_path=data_bin_path,
        fov_shape=fov_shape,
        workspace_root=workspace_root,
        subdir="traces",
    )
