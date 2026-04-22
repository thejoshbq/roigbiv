"""Footprint-derived FOV fingerprint (v3).

Identity for a FOV is established by the **spatial footprints** from the
pipeline's unified mask (``merged_masks.tif``), not by the mean projection.
The fingerprint hash is a deterministic sha256 over:

    * frame shape (H, W),
    * the sorted per-ROI tuple ``(label_id, centroid_y, centroid_x, area)``
      with integer-pixel centroids.

The mean projection is persisted alongside the fingerprint only as context
(used by the adapter's alignment step when the FOV is re-clustered later).

Versions
--------
This module emits ``FINGERPRINT_VERSION = 3``. Legacy v1 / v2 deserialisers
are kept as shims so Streamlit can still render old FOVs in the registry tab;
they are not produced by new code and are marked ``# legacy-read-only``.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np

from roigbiv.registry.roicat_adapter import (
    centroids_from_merged_masks,
    footprints_from_merged_masks,
)

FINGERPRINT_VERSION = 3


@dataclass
class Fingerprint:
    """The v3 fingerprint bundle — hash + blobs + structured per-ROI arrays.

    The blobs are opaque ``.npy`` byte strings suitable for a
    :class:`roigbiv.registry.blob.BlobStore`. The structured arrays are
    convenient in-memory views (same data, numpy shape).
    """

    fingerprint_hash: str
    fingerprint_version: int
    merged_masks_blob: bytes
    mean_m_blob: bytes
    centroids_blob: bytes
    label_ids: np.ndarray  # (N,) int64
    centroids: np.ndarray  # (N, 2) int64 — (y, x) integer-pixel
    areas: np.ndarray      # (N,) int64 — pixel counts per ROI


def compute_fingerprint(
    merged_masks: np.ndarray, mean_m: np.ndarray
) -> Fingerprint:
    """Compute a v3 fingerprint from a unified mask + mean projection.

    Parameters
    ----------
    merged_masks : np.ndarray
        ``(H, W)`` integer label image. Each pixel value is the ``label_id``
        of the ROI it belongs to; 0 is background.
    mean_m : np.ndarray
        ``(H, W)`` mean projection of the motion-corrected movie. Persisted
        as context; **not** part of the fingerprint hash.
    """
    merged_masks = np.asarray(merged_masks, dtype=np.uint16)
    mean_m = np.asarray(mean_m, dtype=np.float32)
    if merged_masks.shape != mean_m.shape:
        raise ValueError(
            f"merged_masks shape {merged_masks.shape} != mean_m shape {mean_m.shape}"
        )

    _, label_ids = footprints_from_merged_masks(merged_masks)
    centroids = centroids_from_merged_masks(merged_masks)
    if label_ids.size > 0:
        areas = np.array(
            [int((merged_masks == lbl).sum()) for lbl in label_ids.tolist()],
            dtype=np.int64,
        )
    else:
        areas = np.zeros((0,), dtype=np.int64)

    hasher = hashlib.sha256()
    hasher.update(b"roigbiv-v3;shape=")
    hasher.update(
        np.array([merged_masks.shape[0], merged_masks.shape[1]], dtype=np.int64).tobytes()
    )
    hasher.update(b";rois=")
    if label_ids.size > 0:
        order = np.argsort(label_ids, kind="stable")
        canonical = np.stack(
            [
                label_ids[order].astype(np.int64),
                centroids[order, 0].astype(np.int64),
                centroids[order, 1].astype(np.int64),
                areas[order].astype(np.int64),
            ],
            axis=1,
        )
        hasher.update(canonical.tobytes())

    return Fingerprint(
        fingerprint_hash=hasher.hexdigest(),
        fingerprint_version=FINGERPRINT_VERSION,
        merged_masks_blob=_serialize_array(merged_masks),
        mean_m_blob=_serialize_array(mean_m),
        centroids_blob=_serialize_array(centroids),
        label_ids=label_ids,
        centroids=centroids,
        areas=areas,
    )


# ── Blob helpers (read + write) ────────────────────────────────────────────


def deserialize_merged_masks(blob: bytes) -> np.ndarray:
    """Load a ``.npy``-encoded merged_masks blob."""
    return np.load(io.BytesIO(blob), allow_pickle=False)


def deserialize_mean_m(blob: bytes) -> np.ndarray:
    """Load a ``.npy``-encoded mean projection blob."""
    return np.load(io.BytesIO(blob), allow_pickle=False)


def deserialize_centroids(blob: bytes) -> np.ndarray:
    """Load a ``.npy``-encoded ``(N, 2)`` centroid blob."""
    return np.load(io.BytesIO(blob), allow_pickle=False)


def serialize_array(arr: np.ndarray) -> bytes:
    """Public helper for serialising a numpy array to ``.npy`` bytes."""
    return _serialize_array(arr)


def _serialize_array(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


# ── Legacy (v1 / v2) read-only shims ───────────────────────────────────────
# These are kept so Streamlit / CLI code inspecting FOV rows minted before the
# v3 migration keeps rendering. New v3 code does not emit these formats.


def deserialize_cells(blob: bytes) -> list:  # legacy-read-only
    """Load a legacy v1/v2 cell structured array and return a list of dicts."""
    arr = np.load(io.BytesIO(blob), allow_pickle=False)
    return [
        {name: (int(row[name]) if np.issubdtype(arr.dtype[name], np.integer)
                else float(row[name]))
         for name in arr.dtype.names}
        for row in arr
    ]


def deserialize_embeddings(blob: bytes) -> np.ndarray:  # legacy-read-only
    """Load a legacy v2 embedding array."""
    return np.load(io.BytesIO(blob), allow_pickle=False).astype(np.float32)
