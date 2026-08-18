"""FOV trace export to pandas HDF5.

Produces a self-contained ``.h5`` file where column names *are* neuron
identifiers — no external index dictionary required. Registered neurons
use their ``global_cell_id`` (UUID, persistent across sessions) as the
column name; unregistered neurons use ``lcl:<local_label_id>`` so the two
name spaces never collide.

::

    s = pd.HDFStore("session.h5", "r")
    dff = s["/dff"]   # (n_frames, n_rois)  index = time (s)
    f   = s["/f"]     # neuropil-corrected F, same shape
    meta = s["/meta"] # per-neuron attributes, indexed by neuron_id

    # merge two sessions — shared global_cell_ids align automatically
    merged = pd.concat([dff1, dff2], axis=1)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import warnings

import numpy as np
import pandas as pd  # requires pytables (pip install tables) for HDFStore

from roigbiv.ui.services.trace_viz import _select_bundle_dir

_KIND_FILES: dict[str, str] = {
    "f":        "traces.npy",
    "raw":      "traces_raw.npy",
    "neuropil": "traces_neuropil.npy",
    # Present only in bundles a Discovery-triggered extraction wrote with
    # extra_stats (roigbiv.pipeline.discovery_extract) — corrected only for
    # now, raw/neuropil variants exist on disk but aren't exported as their
    # own kinds yet. _load_kind already returns None silently when a bundle
    # predates these, so requesting them against an older bundle is a no-op.
    "median":   "traces_median.npy",
    "mode":     "traces_mode.npy",
}


def _neuron_id(roi: dict) -> str:
    gcid = roi.get("global_cell_id")
    if gcid:
        return str(gcid)
    return f"lcl:{roi['local_label_id']}"


def _load_kind(bundle_dir: Path, kind: str) -> Optional[np.ndarray]:
    if kind == "dff":
        local = bundle_dir / "dFF.npy"
        if local.exists():
            return np.load(local, mmap_mode="r")
        if bundle_dir.name == "traces":
            parent = bundle_dir.parent / "dFF.npy"
            if parent.exists():
                return np.load(parent, mmap_mode="r")
        return None
    fname = _KIND_FILES.get(kind)
    if fname is None:
        raise ValueError(f"Unknown kind {kind!r}; valid: dff, f, raw, neuropil")
    path = bundle_dir / fname
    return np.load(path, mmap_mode="r") if path.exists() else None


def export_fov_traces(
    output_dir: Path,
    out_path: Path,
    *,
    kinds: tuple[str, ...] = ("dff", "f"),
) -> Path:
    """Write an HDF5 file containing trace DataFrames for an FOV session.

    Parameters
    ----------
    output_dir:
        Pipeline output directory for the session (contains ``traces/``).
    out_path:
        Destination ``.h5`` file. Created or overwritten.
    kinds:
        Signal types to include. Each becomes a separate HDF5 key.

    Returns
    -------
    Path
        ``out_path``, for chaining.

    Raises
    ------
    FileNotFoundError
        When no ``traces/`` bundle exists in ``output_dir``.
    """
    output_dir = Path(output_dir)
    out_path = Path(out_path)

    bundle_dir, _source = _select_bundle_dir(output_dir)
    if bundle_dir is None:
        raise FileNotFoundError(
            f"No traces/ bundle found in {output_dir}. "
            "Run the pipeline first."
        )

    sidecar = json.loads((bundle_dir / "traces_meta.json").read_text())
    rois: list[dict] = sorted(
        sidecar.get("rois") or [],
        key=lambda r: int(r.get("row_index", 0)),
    )
    fs: float = float(sidecar.get("fs") or 0.0)
    n_frames: int = int(sidecar.get("n_frames") or 0)

    neuron_ids = [_neuron_id(r) for r in rois]
    time_index = pd.Index(
        np.arange(n_frames, dtype=np.float64) / fs if fs else np.arange(n_frames),
        name="time_s",
    )

    meta_df = pd.DataFrame(
        [
            {
                "local_label_id": int(r.get("local_label_id", -1)),
                "global_cell_id": r.get("global_cell_id"),
                "source_stage": int(r.get("source_stage", 0)),
                "gate_outcome": r.get("gate_outcome", ""),
                "confidence": r.get("confidence", ""),
                "activity_type": r.get("activity_type"),
                "session_id": sidecar.get("session_id"),
                "fov_id": sidecar.get("fov_id"),
                "fs": fs,
                "n_frames": n_frames,
            }
            for r in rois
        ],
        index=pd.Index(neuron_ids, name="neuron_id"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # UUID column names trigger NaturalNameWarning from pytables — harmless,
    # since we always access via store[key], never via attribute syntax.
    with warnings.catch_warnings(), \
         pd.HDFStore(str(out_path), mode="w", complevel=4, complib="blosc") as store:
        warnings.simplefilter("ignore")
        for kind in kinds:
            matrix = _load_kind(bundle_dir, kind)
            if matrix is None:
                continue
            arr = np.asarray(matrix, dtype=np.float32)
            # shape is (n_rois, n_frames); transpose → (n_frames, n_rois)
            n = min(len(time_index), arr.shape[1])
            df = pd.DataFrame(
                arr[:, :n].T,
                index=time_index[:n],
                columns=neuron_ids,
            )
            store.put(kind, df, format="table", data_columns=True)
        store.put("meta", meta_df, format="table")

    return out_path


def export_fov_traces_to_tempfile(
    output_dir: Path,
    *,
    kinds: tuple[str, ...] = ("dff", "f"),
    suffix: str = "_traces.h5",
) -> Path:
    """Export to a named temp file; caller is responsible for cleanup."""
    stem = Path(output_dir).name
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{stem}", suffix=suffix, delete=False,
    )
    tmp.close()
    return export_fov_traces(output_dir, Path(tmp.name), kinds=kinds)
