"""Gate-time mask sidecar: every candidate's mask as its gate saw it, rejects included.

After gating, ``roi_stamp.canonicalize`` replaces each accept/flag mask with a fixed-radius disk, so the
detected shape (a Cellpose outline, a Suite2p footprint, a connected component) is gone from everything the
run saves. ``save_gate_masks`` is called just before that, once per stage, and writes
``stage{N}/stage{N}_gate_masks.npz``: label ids, CSR offsets and flat pixel indices, plus the frame shape. It
reads the ROIs and never changes them. It is a diagnostic record: any failure prints a warning, leaves no
sidecar for that stage, and the run goes on. Nothing in the pipeline reads it back.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def gate_masks_path(output_dir: Path, stage: int) -> Path:
    return Path(output_dir) / f"stage{stage}" / f"stage{stage}_gate_masks.npz"


def save_gate_masks(output_dir: Path, stage: int, rois: Sequence, shape: tuple | None = None) -> Path | None:
    """Write the stage's gate-time masks; returns the path, or None when the write failed.

    Any earlier sidecar is removed first, so a failed write leaves none rather than a stale one beside the new
    report. Every error is caught and printed: this record must never stop a run.
    """
    path = gate_masks_path(output_dir, stage)
    tmp = path.with_name(path.stem + ".tmp.npz")
    try:
        path.unlink(missing_ok=True)
        if shape is None:
            shape = rois[0].mask.shape
        bad = [int(r.label_id) for r in rois if np.shape(r.mask) != tuple(shape)]
        if bad:
            raise ValueError(f"labels {bad[:5]} have masks of another shape than {tuple(shape)}")
        parts = [np.flatnonzero(r.mask).astype(np.int64) for r in rois]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tmp,
            label_id=np.array([int(r.label_id) for r in rois], dtype=np.int64),
            offsets=np.cumsum([0] + [p.size for p in parts]).astype(np.int64),
            flat_idx=np.concatenate(parts) if parts else np.zeros(0, np.int64),
            shape=np.array(shape, dtype=np.int64),
        )
        os.replace(tmp, path)
        return path
    except Exception as e:  # noqa: BLE001  (a diagnostic record; the run goes on)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"  WARNING: stage {stage} gate masks not saved ({type(e).__name__}: {e}); the run continues", flush=True)
        return None


def load_gate_masks(path: Path) -> tuple[tuple, dict]:
    """(shape, {label_id: sorted flat pixel indices})."""
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    masks = {int(lab): z["flat_idx"][off[i] : off[i + 1]] for i, lab in enumerate(z["label_id"])}
    return tuple(int(v) for v in z["shape"]), masks
