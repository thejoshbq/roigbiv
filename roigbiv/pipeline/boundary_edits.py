"""Additive human-in-the-loop boundary edits — hand-drawn overrides of a
label's seeded shape.

Mirrors :mod:`roigbiv.pipeline.centroid_edits` exactly in pattern, applied to
the second geometry track instead of the first (see
``docs/adr/0005-seeded-boundaries-parallel-geometry-track.md``). Pipeline
boundary output (``boundaries.tif`` / ``boundaries.json``, drawn by
:func:`roigbiv.pipeline.boundaries.seeded_labels`) is never overwritten by an
edit. Instead, each user action is appended to ``corrections/boundaries.jsonl``
as a single :class:`BoundaryOp`. :func:`apply_boundary_ops` replays that log
against the current auto-computed label image and returns the effective one.

Operation types
---------------
``draw``    replace a label's entire shape with a hand-drawn polygon
``delete``  cancel that label's manual override — it reverts to whatever the
            *current* auto computation produces for it, not to a prior manual
            state (there is no prior state to go back to; "delete" means "stop
            overriding")

**Precedence rule (the whole point of this module):** a label with an active
``draw`` op keeps its manual shape regardless of what ``seeded_labels`` would
currently compute for it — across a ``capture_px``/``min_area`` retune, across
other cells' centroid edits, across anything. A label with no active ``draw``
op always reflects the current auto computation. This is why replay takes the
*freshly computed* auto label image as input on every call rather than
diffing against a stored prior state: the auto pixels are the baseline, manual
ops are a pure overlay on top of it.

**Warning semantics differ from corrections.py:** that module silently skips
malformed ops for forward-compat. This module returns a human-readable warning
string for each skipped op, because silently discarding a researcher's manual
edit is exactly the failure this whole feature exists to prevent.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BoundaryOp:
    """One human-in-the-loop boundary edit.

    Stored as a single JSON line (one per op), same convention as
    :class:`~roigbiv.pipeline.centroid_edits.CentroidOp`.
    """

    op: str                         # "draw" | "delete"
    label: int                      # the cell this op targets — required
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # draw only — a closed ring as [[y, x], [y, x], ...], >= 3 points.
    ring: Optional[list] = None

    notes: str = ""

    @classmethod
    def draw(cls, label: int, ring: list, notes: str = "") -> "BoundaryOp":
        return cls(op="draw", label=int(label),
                   ring=[[float(y), float(x)] for y, x in ring], notes=notes)

    @classmethod
    def delete(cls, label: int, notes: str = "") -> "BoundaryOp":
        return cls(op="delete", label=int(label), notes=notes)

    def to_jsonable(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if v is not None and v != ""}
        out["op"] = self.op
        out["label"] = self.label
        out["id"] = self.id
        out["ts"] = self.ts
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "BoundaryOp":
        return cls(
            op=d["op"],
            label=int(d["label"]),
            id=d.get("id", str(uuid.uuid4())),
            ts=d.get("ts", datetime.now(timezone.utc).isoformat()),
            ring=d.get("ring"),
            notes=d.get("notes", ""),
        )


# ── On-disk layout helpers ─────────────────────────────────────────────────


def _corrections_dir(output_dir: Path) -> Path:
    """``{output_dir}/corrections/`` — created on demand."""
    d = Path(output_dir) / "corrections"
    d.mkdir(parents=True, exist_ok=True)
    return d


def boundary_log_path(output_dir: Path) -> Path:
    """Path to the boundary edits JSONL log."""
    return _corrections_dir(output_dir) / "boundaries.jsonl"


def append_boundary_op(output_dir: Path, op: BoundaryOp) -> None:
    """Append one op to the JSONL log. Creates the file if missing."""
    log_path = boundary_log_path(output_dir)
    with log_path.open("a") as f:
        f.write(json.dumps(op.to_jsonable()) + "\n")


def load_boundary_ops(output_dir: Path) -> list[BoundaryOp]:
    """Read all ops from the JSONL log (empty list if no log exists)."""
    log_path = boundary_log_path(output_dir)
    if not log_path.exists():
        return []
    ops: list[BoundaryOp] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ops.append(BoundaryOp.from_dict(json.loads(line)))
    return ops


def write_boundary_ops(output_dir: Path, ops: list[BoundaryOp]) -> None:
    """Replace the entire log with ``ops`` (used for undo: drop the tail)."""
    log_path = boundary_log_path(output_dir)
    if not ops:
        if log_path.exists():
            log_path.unlink()
        return
    with log_path.open("w") as f:
        for op in ops:
            f.write(json.dumps(op.to_jsonable()) + "\n")


# ── Replay ─────────────────────────────────────────────────────────────────


def active_manual_labels(ops: list[BoundaryOp]) -> set[int]:
    """Which labels currently carry an undeleted ``draw`` op.

    A cheap replay for callers that want to validate a gesture before writing
    it — e.g. refusing a ``delete`` for a label with no active manual
    boundary — without duplicating :func:`apply_boundary_ops`'s full replay.
    """
    active: set[int] = set()
    for op in ops:
        if op.op == "draw" and op.ring and len(op.ring) >= 3:
            active.add(op.label)
        elif op.op == "delete":
            active.discard(op.label)
    return active


def _rasterize_ring(ring: list, height: int, width: int) -> np.ndarray:
    """A ``[[y, x], ...]`` polygon as a ``(height, width)`` boolean mask."""
    from skimage.draw import polygon2mask

    if not ring or len(ring) < 3:
        return np.zeros((height, width), dtype=bool)
    pts = np.asarray(ring, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((height, width), dtype=bool)
    return polygon2mask((height, width), pts).astype(bool)


def apply_boundary_ops(
    labels: np.ndarray,
    origins: dict,
    ops: list[BoundaryOp],
) -> tuple[np.ndarray, dict, list[str]]:
    """Replay ``ops`` over auto-computed ``labels``/``origins``.

    Pure function: never mutates its inputs (``labels`` is copied before any
    write). Returns ``(labels, origins, warnings)``:

    * a ``draw`` op for a label clears every pixel that label currently holds
      — whether from the auto computation or a prior ``draw`` for the same
      label — and rasterizes the new polygon into its place, marked
      :data:`~roigbiv.pipeline.seeded_masks.ORIGIN_MANUAL`.
    * a ``delete`` op cancels the active ``draw`` for a label; since ``labels``
      already carries that label's *current* auto shape (the caller always
      recomputes it fresh — see module docstring), doing nothing further is
      exactly "revert to automatic".
    * overlapping manual polygons resolve last-drawn-wins, in chronological op
      order — mirrors ``seeded_labels``'s own note that a later label
      overwrites an earlier one where two regions touch.

    Malformed or conflicting ops are skipped with a warning message, same
    convention as :func:`roigbiv.pipeline.centroid_edits.apply_centroid_ops`.
    """
    from roigbiv.pipeline.seeded_masks import ORIGIN_MANUAL

    warnings: list[str] = []
    active: dict[int, BoundaryOp] = {}
    order: list[int] = []   # chronological order of the *last* op touching a label

    for op in ops:
        try:
            if op.op == "draw":
                if not op.ring or len(op.ring) < 3:
                    warnings.append(
                        f"draw: label {op.label} needs at least 3 ring points "
                        f"— skipped")
                    continue
                if op.label in active:
                    order.remove(op.label)
                order.append(op.label)
                active[op.label] = op

            elif op.op == "delete":
                if op.label not in active:
                    warnings.append(
                        f"delete: label {op.label} has no active manual "
                        f"boundary — skipped")
                    continue
                del active[op.label]
                order.remove(op.label)

            else:
                warnings.append(
                    f"unknown: label {op.label} has unknown op {op.op!r} — skipped")

        except Exception:  # noqa: BLE001 — one bad op must not nuke the replay
            warnings.append(f"{op.op}: label {op.label} raised exception — skipped")

    if not order:
        return labels, origins, warnings

    height, width = labels.shape[:2]
    out = labels.copy()
    out_origins = dict(origins)

    for label in order:
        op = active[label]
        out[out == label] = 0
        mask = _rasterize_ring(op.ring, height, width)
        if not mask.any():
            warnings.append(
                f"draw: label {label} polygon rasterized to zero pixels — "
                f"reverted to auto")
            continue
        out[mask] = label
        out_origins[label] = ORIGIN_MANUAL

    return out, out_origins, warnings


def layer_boundary_ops(result, output_dir: Path):
    """Load and replay this FOV's boundary edits over a :class:`SeededMasks`.

    Mutates and returns ``result`` — the same in-place convention
    :func:`roigbiv.pipeline.boundaries.compute_boundaries` already uses for
    ``result.warnings``. Every caller that draws this FOV's boundaries from
    scratch (the persisted path through ``compute_boundaries``, and the
    live-tuning cache in ``roigbiv.ui.services.boundary_preview.preview``)
    routes through this, so a manual override never depends on which entry
    point drew the auto shape underneath it.
    """
    ops = load_boundary_ops(output_dir)
    if not ops:
        return result

    new_labels, new_origins, op_warnings = apply_boundary_ops(
        result.labels, result.origins, ops)
    result.labels = new_labels
    result.origins = new_origins
    result.areas = {
        label: int(np.count_nonzero(new_labels == label))
        for label in sorted(new_origins)
    }
    result.warnings = list(result.warnings) + [
        f"boundary edit: {w}" for w in op_warnings]
    return result
