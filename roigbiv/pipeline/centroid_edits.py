"""Additive human-in-the-loop centroid edits.

Pipeline centroid discovery output (``centroids.json``) is *frozen* — edits
never overwrite it. Instead, each user action is appended to
``corrections/centroids.jsonl`` as a single :class:`CentroidOp`. ``apply_centroid_ops``
replays that log against the frozen centroid dict and returns the effective
centroid set.

This means:

* Every centroid edit is auditable (``centroids.jsonl`` is the source of truth).
* Reverting is just deleting the JSONL entry and re-applying.
* Re-registering against the registry is an explicit user action that reads
  the *edited* centroids; the pipeline outputs themselves are never rewritten.

Operation types
---------------
``add``     add a brand-new soma centroid at (y, x) with an explicit label
``delete``  remove an existing centroid by ``label``
``move``    relocate an existing centroid to a new (y, x)

**Critical: labels are never reused.** When ``add`` carries a label in the op,
that label is literal text in the log — replay does *not* derive a new label
from ``max(current)+1``. Why? Because a second correction log (cross-session cell
links) references centroids as ``(session_stem, label)`` pairs. If replay derived
a new label as ``max(current)+1``, then the sequence ``add A (label 9) → delete 9
→ add B`` would hand B the label 9, and a link op naming label 9 would silently
point at a different physical cell. So the label must be explicit. Use
``next_label`` to compute the next safe label *before* writing an ``add`` op.

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


@dataclass
class CentroidOp:
    """One human-in-the-loop centroid edit.

    Stored as a single JSON line (one per op). Fields are a discriminated union
    of every supported operation (add / delete / move) rather than a class
    hierarchy — keeps replay logic flat and JSON round-tripping trivial.
    """

    op: str                        # "add" | "delete" | "move"
    label: int                     # required for all three
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # add / move
    y: Optional[float] = None
    x: Optional[float] = None

    notes: str = ""

    @classmethod
    def add(cls, label: int, y: float, x: float, notes: str = "") -> "CentroidOp":
        return cls(op="add", label=int(label), y=float(y), x=float(x), notes=notes)

    @classmethod
    def delete(cls, label: int, notes: str = "") -> "CentroidOp":
        return cls(op="delete", label=int(label), notes=notes)

    @classmethod
    def move(cls, label: int, y: float, x: float, notes: str = "") -> "CentroidOp":
        return cls(op="move", label=int(label), y=float(y), x=float(x), notes=notes)

    def to_jsonable(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if v is not None and v != ""}
        out["op"] = self.op
        out["label"] = self.label
        out["id"] = self.id
        out["ts"] = self.ts
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "CentroidOp":
        return cls(
            op=d["op"],
            label=int(d["label"]),
            id=d.get("id", str(uuid.uuid4())),
            ts=d.get("ts", datetime.now(timezone.utc).isoformat()),
            y=d.get("y"),
            x=d.get("x"),
            notes=d.get("notes", ""),
        )


# ── On-disk layout helpers ─────────────────────────────────────────────────


def _corrections_dir(output_dir: Path) -> Path:
    """``{output_dir}/corrections/`` — created on demand."""
    d = Path(output_dir) / "corrections"
    d.mkdir(parents=True, exist_ok=True)
    return d


def centroid_log_path(output_dir: Path) -> Path:
    """Path to the centroid edits JSONL log."""
    return _corrections_dir(output_dir) / "centroids.jsonl"


def append_centroid_op(output_dir: Path, op: CentroidOp) -> None:
    """Append one op to the JSONL log. Creates the file if missing."""
    log_path = centroid_log_path(output_dir)
    with log_path.open("a") as f:
        f.write(json.dumps(op.to_jsonable()) + "\n")


def load_centroid_ops(output_dir: Path) -> list[CentroidOp]:
    """Read all ops from the JSONL log (empty list if no log exists)."""
    log_path = centroid_log_path(output_dir)
    if not log_path.exists():
        return []
    ops: list[CentroidOp] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ops.append(CentroidOp.from_dict(json.loads(line)))
    return ops


def write_centroid_ops(output_dir: Path, ops: list[CentroidOp]) -> None:
    """Replace the entire log with ``ops`` (used for undo: drop the tail)."""
    log_path = centroid_log_path(output_dir)
    if not ops:
        if log_path.exists():
            log_path.unlink()
        return
    with log_path.open("w") as f:
        for op in ops:
            f.write(json.dumps(op.to_jsonable()) + "\n")


# ── Replay ─────────────────────────────────────────────────────────────────


def next_label(
    base: dict[int, tuple[float, float]],
    ops: list[CentroidOp],
) -> int:
    """Compute the next safe label (never reused, even after delete).

    Scans both ``base`` and the ops list to find the high-water mark of all
    labels ever seen (including deleted ones), then returns one past it.
    Returns 1 when both are empty.

    This ensures that a label handed to an ``add`` op is never handed out
    again, preventing mismatches in external correlation logs that reference
    centroids by (session_stem, label).
    """
    all_labels = set(base.keys()) | {op.label for op in ops}
    return (max(all_labels) + 1) if all_labels else 1


def apply_centroid_ops(
    base: dict[int, tuple[float, float]],
    ops: list[CentroidOp],
) -> tuple[dict[int, tuple[float, float]], list[str]]:
    """Replay ``ops`` against ``base`` centroids and return the edited set.

    Pure function: never mutates ``base``. Returns a tuple of:
    * ``dict[int, tuple[float, float]]``: label → (y, x) of effective centroids
    * ``list[str]``: warning messages for skipped ops (human-readable)

    Replay semantics:
      * ``add``     insert a new centroid with its explicit label
      * ``delete``  remove the centroid at that label
      * ``move``    update the (y, x) of the centroid at that label

    Malformed or conflicting ops are skipped with a warning message returned
    in the list. Every warning includes the op type and label for tracing.
    """
    # Work on a shallow copy so we never touch the input.
    current: dict[int, tuple[float, float]] = dict(base)
    warnings: list[str] = []

    for op in ops:
        try:
            if op.op == "add":
                if op.y is None or op.x is None:
                    warnings.append(
                        f"add: label {op.label} missing y or x — skipped"
                    )
                    continue
                if op.label in current:
                    warnings.append(
                        f"add: label {op.label} already present — skipped"
                    )
                    continue
                current[op.label] = (float(op.y), float(op.x))

            elif op.op == "delete":
                if op.label not in current:
                    warnings.append(
                        f"delete: label {op.label} is not present — skipped"
                    )
                    continue
                current.pop(op.label)

            elif op.op == "move":
                if op.y is None or op.x is None:
                    warnings.append(
                        f"move: label {op.label} missing y or x — skipped"
                    )
                    continue
                if op.label not in current:
                    warnings.append(
                        f"move: label {op.label} is not present — skipped"
                    )
                    continue
                current[op.label] = (float(op.y), float(op.x))

            else:
                warnings.append(
                    f"unknown: label {op.label} has unknown op '{op.op}' — skipped"
                )

        except Exception:  # noqa: BLE001
            # Don't let a single bad op nuke the whole replay.
            warnings.append(f"{op.op}: label {op.label} raised exception — skipped")

    return current, warnings
