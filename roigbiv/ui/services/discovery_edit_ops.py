"""Discovery page centroid edits — pure, registry-free, one FOV at a time.

The Discovery page operates on raw per-FOV output (``centroids.json`` +
optionally a flow cache), not the cross-session registry — a FOV need not have
gone through tracking to have centroids worth correcting. This module mirrors
the op-writing pattern in :mod:`roigbiv.ui.services.cell_edit_ops`
(``_do_add`` / ``_do_delete`` / ``_do_move``) but drops everything that pattern
carries only for cross-session state: no ``TrackedFOV``, no match-op linking,
no registry replay, no selection. A Discovery edit is exactly one
:class:`~roigbiv.pipeline.centroid_edits.CentroidOp` appended to
``{output_dir}/corrections/centroids.jsonl``, and the response is a fresh read
of the effective centroid set via
:func:`roigbiv.pipeline.centroid_masks.load_effective_centroids`.

Nothing here may import ``dash`` — the same reasoning as ``cell_edit_ops``:
the Flask endpoint that calls :func:`apply_gesture`
(:mod:`roigbiv.ui.routes.discovery_api`) cannot use a callback's return shape,
and this module's own tests want to call it directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

GESTURE_KINDS = ("add", "delete", "move", "undo")


@dataclass
class Gesture:
    """One described pointer action against this FOV's centroids."""

    kind: str
    label: Optional[int] = None
    # Full-resolution image pixels, exactly as clicked/dropped — never snapped
    # to an existing centroid. See cell_edit_ops.Gesture for why that matters.
    y: Optional[float] = None
    x: Optional[float] = None

    @classmethod
    def from_payload(cls, payload: dict) -> "Gesture":
        if not isinstance(payload, dict):
            raise ValueError("gesture must be an object")
        kind = payload.get("kind")
        if kind not in GESTURE_KINDS:
            raise ValueError(f"unknown gesture kind: {kind!r}")

        gesture = cls(
            kind=kind,
            label=_opt_int(payload.get("label")),
            y=_opt_float(payload.get("y")),
            x=_opt_float(payload.get("x")),
        )
        if kind in ("delete", "move") and gesture.label is None:
            raise ValueError(f"{kind} requires a label")
        if kind in ("add", "move") and (gesture.y is None or gesture.x is None):
            raise ValueError(f"{kind} requires y and x")
        return gesture


@dataclass
class GestureResult:
    """What the gesture did, and the fresh centroid set to repaint from.

    ``centroids`` is ``None`` when nothing was written — the caller then
    leaves the sheet exactly as the browser already has it, same convention as
    ``cell_edit_ops.GestureResult.fov``.
    """

    ok: bool
    message: str
    centroids: Optional[list[tuple[int, float, float]]] = None
    warnings: list = field(default_factory=list)
    status: int = 200


def _opt_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected an integer, got {value!r}") from None


def _opt_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected a number, got {value!r}") from None


def _refused(message: str, *, status: int = 200) -> GestureResult:
    return GestureResult(ok=False, message=message, status=status)


def _effective(output_dir: Path):
    from roigbiv.pipeline.centroid_masks import load_effective_centroids

    return load_effective_centroids(output_dir)


def _as_list(centroids: dict) -> list[tuple[int, float, float]]:
    return [(label, y, x) for label, (y, x) in sorted(centroids.items())]


def apply_gesture(output_dir: Path, gesture: Gesture) -> GestureResult:
    """Write the op *gesture* implies, then return the fresh centroid set.

    *output_dir* must already exist with a ``centroids.json`` — a Discovery
    edit corrects detector output, so there has to be some to correct.
    """
    from roigbiv.pipeline.centroid_edits import (
        CentroidOp,
        append_centroid_op,
        load_centroid_ops,
        next_label,
        write_centroid_ops,
    )

    output_dir = Path(output_dir)
    if not (output_dir / "centroids.json").exists():
        return _refused(
            "this FOV has no centroids yet — run centroid discovery first",
            status=409)

    if gesture.kind == "undo":
        ops = load_centroid_ops(output_dir)
        if not ops:
            return _refused("nothing to undo")
        write_centroid_ops(output_dir, ops[:-1])
        effective, warnings = _effective(output_dir)
        return GestureResult(ok=True, message="undid the last centroid edit",
                             centroids=_as_list(effective), warnings=warnings)

    if gesture.kind == "add":
        effective, _warnings = _effective(output_dir)
        label = next_label(effective, load_centroid_ops(output_dir))
        append_centroid_op(output_dir, CentroidOp.add(label, gesture.y, gesture.x))
        effective, warnings = _effective(output_dir)
        return GestureResult(ok=True, message="added a centroid",
                             centroids=_as_list(effective), warnings=warnings)

    effective, _warnings = _effective(output_dir)
    if gesture.label not in effective:
        return _refused(
            f"no centroid with label {gesture.label} in this FOV", status=400)

    if gesture.kind == "delete":
        append_centroid_op(output_dir, CentroidOp.delete(gesture.label))
        verb = "deleted a centroid"
    else:  # move
        append_centroid_op(
            output_dir, CentroidOp.move(gesture.label, gesture.y, gesture.x))
        verb = "moved a centroid"

    effective, warnings = _effective(output_dir)
    return GestureResult(ok=True, message=verb, centroids=_as_list(effective),
                         warnings=warnings)
