"""Additive human-in-the-loop ROI corrections.

Pipeline outputs (``merged_masks.tif``, ``roi_metadata.json``, the per-stage
TIFFs and reports) are *frozen* — corrections never overwrite them. Instead,
each user action is appended to ``corrections/corrections.jsonl`` as a single
:class:`CorrectionOp`. ``materialize`` replays that log against the frozen ROI
list and writes ``corrections/corrected_masks.tif`` +
``corrections/corrected_metadata.json``.

This means:

* Every correction is auditable (``corrections.jsonl`` is the source of truth).
* Reverting is just deleting the JSONL entry / file and re-materializing.
* Re-registering against the registry is an explicit user action that reads
  the *corrected* artifacts; the pipeline outputs themselves are never
  rewritten.

Operation types
---------------
``add``      add a brand-new ROI from a polygon
``delete``   remove an existing ROI by ``label_id``
``merge``    delete a set of ROIs and add a single replacement polygon
``split``    delete one ROI and add several replacement polygons
``edit``     replace an existing ROI's mask with a new polygon
``relabel``  change an existing ROI's ``activity_type``

Polygons are ``[[y, x], ...]`` lists of pixel coordinates (matching the row
``[y, x]`` ordering used everywhere else in this codebase).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile

from roigbiv.pipeline.types import ROI

USER_STAGE_SENTINEL: int = 99
"""``source_stage`` value assigned to user-added ROIs.

Distinct from the pipeline's 1-4 to keep color palettes / filters cleanly
separable in the viewer.
"""


# ── Operation model ────────────────────────────────────────────────────────


@dataclass
class CorrectionOp:
    """One human-in-the-loop ROI correction.

    Stored as a single JSON line (one per op). Fields are deliberately a
    discriminated-union of every supported operation rather than a class
    hierarchy — keeps replay logic flat and JSON round-tripping trivial.
    """

    op: str                        # "add" | "delete" | "merge" | "split" | "edit" | "relabel"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # add / edit / merge / split
    polygon: Optional[list[list[float]]] = None
    polygons: Optional[list[list[list[float]]]] = None

    # delete / merge / split / edit / relabel
    label_id: Optional[int] = None
    label_ids: Optional[list[int]] = None

    activity_type: Optional[str] = None
    notes: str = ""

    @classmethod
    def add(cls, polygon: list[list[float]], *,
            activity_type: Optional[str] = None, notes: str = "") -> "CorrectionOp":
        return cls(op="add", polygon=polygon,
                   activity_type=activity_type, notes=notes)

    @classmethod
    def delete(cls, label_id: int, notes: str = "") -> "CorrectionOp":
        return cls(op="delete", label_id=int(label_id), notes=notes)

    @classmethod
    def merge(cls, label_ids: list[int],
              polygon: Optional[list[list[float]]] = None,
              notes: str = "") -> "CorrectionOp":
        return cls(op="merge", label_ids=[int(lid) for lid in label_ids],
                   polygon=polygon, notes=notes)

    @classmethod
    def split(cls, label_id: int,
              polygons: list[list[list[float]]], notes: str = "") -> "CorrectionOp":
        return cls(op="split", label_id=int(label_id),
                   polygons=polygons, notes=notes)

    @classmethod
    def edit(cls, label_id: int, polygon: list[list[float]],
             notes: str = "") -> "CorrectionOp":
        return cls(op="edit", label_id=int(label_id),
                   polygon=polygon, notes=notes)

    @classmethod
    def relabel(cls, label_id: int, activity_type: str,
                notes: str = "") -> "CorrectionOp":
        return cls(op="relabel", label_id=int(label_id),
                   activity_type=activity_type, notes=notes)

    def to_jsonable(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if v is not None and v != ""}
        out["op"] = self.op
        out["id"] = self.id
        out["ts"] = self.ts
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "CorrectionOp":
        return cls(
            op=d["op"],
            id=d.get("id", str(uuid.uuid4())),
            ts=d.get("ts", datetime.now(timezone.utc).isoformat()),
            polygon=d.get("polygon"),
            polygons=d.get("polygons"),
            label_id=d.get("label_id"),
            label_ids=d.get("label_ids"),
            activity_type=d.get("activity_type"),
            notes=d.get("notes", ""),
        )


# ── On-disk layout helpers ─────────────────────────────────────────────────


def corrections_dir(output_dir: Path) -> Path:
    """``{output_dir}/corrections/`` — created on demand."""
    d = Path(output_dir) / "corrections"
    d.mkdir(parents=True, exist_ok=True)
    return d


def corrections_log_path(output_dir: Path) -> Path:
    return corrections_dir(output_dir) / "corrections.jsonl"


def corrected_masks_path(output_dir: Path) -> Path:
    return corrections_dir(output_dir) / "corrected_masks.tif"


def corrected_metadata_path(output_dir: Path) -> Path:
    return corrections_dir(output_dir) / "corrected_metadata.json"


def append_correction(output_dir: Path, op: CorrectionOp) -> None:
    """Append one op to the JSONL log. Creates the file if missing."""
    log_path = corrections_log_path(output_dir)
    with log_path.open("a") as f:
        f.write(json.dumps(op.to_jsonable()) + "\n")


def load_corrections(output_dir: Path) -> list[CorrectionOp]:
    """Read all ops from the JSONL log (empty list if no log exists)."""
    log_path = corrections_log_path(output_dir)
    if not log_path.exists():
        return []
    ops: list[CorrectionOp] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ops.append(CorrectionOp.from_dict(json.loads(line)))
    return ops


def write_corrections(output_dir: Path, ops: list[CorrectionOp]) -> None:
    """Replace the entire log with ``ops`` (used for undo: drop the tail)."""
    log_path = corrections_log_path(output_dir)
    if not ops:
        if log_path.exists():
            log_path.unlink()
        return
    with log_path.open("w") as f:
        for op in ops:
            f.write(json.dumps(op.to_jsonable()) + "\n")


# ── Replay ─────────────────────────────────────────────────────────────────


def apply_corrections(
    rois: list[ROI],
    ops: list[CorrectionOp],
    shape: tuple[int, int],
) -> list[ROI]:
    """Replay ``ops`` against ``rois`` and return the corrected ROI list.

    Pure function: never mutates ``rois`` or any of its members. ``shape`` is
    ``(H, W)`` of the FOV — needed to rasterize polygon-defined ROIs.

    Replay semantics:
      * ``add``     append a fresh ROI rasterized from the polygon
      * ``delete``  drop the ROI with that label_id
      * ``edit``    rebuild the masking polygon for that label_id
      * ``relabel`` overwrite ``activity_type``
      * ``merge``   delete every label in ``label_ids``; add one new ROI
                    (polygon = explicit polygon, else union of removed masks)
      * ``split``   delete the label; add one new ROI per polygon

    Unknown / malformed ops are skipped with no error — replay must always
    produce a valid result.
    """
    H, W = int(shape[0]), int(shape[1])

    # Work on shallow clones so we never touch the inputs.
    current: dict[int, ROI] = {int(r.label_id): _clone_roi(r) for r in rois}
    next_label_id = (max(current.keys()) + 1) if current else 1

    for op in ops:
        try:
            if op.op == "add":
                if not op.polygon:
                    continue
                mask = _polygon_to_mask(op.polygon, H, W)
                if not mask.any():
                    continue
                roi = _build_user_roi(mask, next_label_id,
                                      activity_type=op.activity_type)
                current[next_label_id] = roi
                next_label_id += 1

            elif op.op == "delete":
                current.pop(int(op.label_id), None)

            elif op.op == "edit":
                if op.label_id is None or not op.polygon:
                    continue
                roi = current.get(int(op.label_id))
                if roi is None:
                    continue
                mask = _polygon_to_mask(op.polygon, H, W)
                if not mask.any():
                    continue
                roi.mask = mask
                roi.area = int(mask.sum())

            elif op.op == "relabel":
                if op.label_id is None or not op.activity_type:
                    continue
                roi = current.get(int(op.label_id))
                if roi is None:
                    continue
                roi.activity_type = op.activity_type

            elif op.op == "merge":
                if not op.label_ids:
                    continue
                removed_masks = []
                for lid in op.label_ids:
                    r = current.pop(int(lid), None)
                    if r is not None and r.mask is not None:
                        removed_masks.append(r.mask)
                if op.polygon:
                    mask = _polygon_to_mask(op.polygon, H, W)
                else:
                    if not removed_masks:
                        continue
                    mask = np.zeros((H, W), dtype=bool)
                    for m in removed_masks:
                        mask |= m
                if not mask.any():
                    continue
                roi = _build_user_roi(mask, next_label_id,
                                      activity_type=op.activity_type)
                current[next_label_id] = roi
                next_label_id += 1

            elif op.op == "split":
                if op.label_id is None or not op.polygons:
                    continue
                current.pop(int(op.label_id), None)
                for poly in op.polygons:
                    mask = _polygon_to_mask(poly, H, W)
                    if not mask.any():
                        continue
                    roi = _build_user_roi(mask, next_label_id,
                                          activity_type=op.activity_type)
                    current[next_label_id] = roi
                    next_label_id += 1

            # silently ignore unknown ops — log forward-compat
        except Exception:  # noqa: BLE001
            # Don't let a single bad op nuke the whole replay.
            continue

    return [current[k] for k in sorted(current.keys())]


def materialize(
    rois_corrected: list[ROI],
    output_dir: Path,
    shape: tuple[int, int],
) -> tuple[Path, Path]:
    """Write ``corrected_masks.tif`` + ``corrected_metadata.json``.

    Returns the two paths written.
    """
    H, W = int(shape[0]), int(shape[1])
    mask_img = np.zeros((H, W), dtype=np.uint16)
    meta: list[dict[str, Any]] = []
    for roi in rois_corrected:
        if roi.mask is None or not roi.mask.any():
            continue
        mask_img[roi.mask] = int(roi.label_id)
        meta.append(roi.to_serializable())

    masks_path = corrected_masks_path(output_dir)
    meta_path = corrected_metadata_path(output_dir)
    tifffile.imwrite(str(masks_path), mask_img)
    meta_path.write_text(json.dumps(meta, indent=2))
    return masks_path, meta_path


# ── helpers ────────────────────────────────────────────────────────────────


def _polygon_to_mask(polygon: list[list[float]], H: int, W: int) -> np.ndarray:
    """Rasterize a ``[[y, x], ...]`` polygon to a ``(H, W)`` boolean mask."""
    from skimage.draw import polygon2mask

    if not polygon:
        return np.zeros((H, W), dtype=bool)
    pts = np.asarray(polygon, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((H, W), dtype=bool)
    return polygon2mask((H, W), pts).astype(bool)


def _build_user_roi(
    mask: np.ndarray,
    label_id: int,
    *,
    activity_type: Optional[str],
) -> ROI:
    """Construct a fresh ROI for a user-added correction."""
    from skimage.measure import regionprops

    area = int(mask.sum())
    solidity = 0.0
    eccentricity = 0.0
    if area > 0:
        try:
            props = regionprops(mask.astype(np.uint8))
            if props:
                p = props[0]
                solidity = float(getattr(p, "solidity", 0.0))
                eccentricity = float(getattr(p, "eccentricity", 0.0))
        except Exception:  # noqa: BLE001
            pass

    return ROI(
        mask=mask,
        label_id=int(label_id),
        source_stage=USER_STAGE_SENTINEL,
        confidence="moderate",
        gate_outcome="accept",
        area=area,
        solidity=solidity,
        eccentricity=eccentricity,
        activity_type=activity_type,
        gate_reasons=["user_correction"],
        features={"user_added": True},
    )


def _clone_roi(roi: ROI) -> ROI:
    """Shallow clone with an independent mask + features dict + gate_reasons list."""
    return ROI(
        mask=None if roi.mask is None else roi.mask.copy(),
        label_id=int(roi.label_id),
        source_stage=int(roi.source_stage),
        confidence=str(roi.confidence),
        gate_outcome=str(roi.gate_outcome),
        area=int(roi.area),
        solidity=float(roi.solidity),
        eccentricity=float(roi.eccentricity),
        nuclear_shadow_score=float(roi.nuclear_shadow_score),
        soma_surround_contrast=float(roi.soma_surround_contrast),
        cellpose_prob=roi.cellpose_prob,
        iscell_prob=roi.iscell_prob,
        event_count=roi.event_count,
        corr_contrast=roi.corr_contrast,
        trace=None if roi.trace is None else roi.trace.copy(),
        trace_corrected=(None if roi.trace_corrected is None
                        else roi.trace_corrected.copy()),
        activity_type=roi.activity_type,
        features=dict(roi.features),
        gate_reasons=list(roi.gate_reasons),
    )
