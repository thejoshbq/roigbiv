"""Pure fingerprint computation.

Takes numpy arrays in and produces:
  * fingerprint_hash (deterministic sha256 of canonical payload)
  * serialized blobs (bytes) for the BlobStore

No DB, no filesystem. The orchestrator wires blob storage around this.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np

FINGERPRINT_SHAPE = (64, 64)


@dataclass
class CellFeature:
    local_label_id: int
    centroid_y: float
    centroid_x: float
    area: int
    solidity: float
    eccentricity: float
    nuclear_shadow_score: float = 0.0
    soma_surround_contrast: float = 0.0


@dataclass
class Fingerprint:
    fingerprint_hash: str
    mean_m_blob: bytes
    centroid_blob: bytes
    mean_m_downsampled: np.ndarray
    centroids: list[CellFeature]


def downsample(image: np.ndarray, shape: tuple[int, int] = FINGERPRINT_SHAPE) -> np.ndarray:
    """Deterministic downsample via block-mean. Pads to multiple of target first."""
    image = np.asarray(image, dtype=np.float32)
    H, W = image.shape
    th, tw = shape
    bh = max(1, H // th)
    bw = max(1, W // tw)
    trim_h = bh * th
    trim_w = bw * tw
    img = image[:trim_h, :trim_w]
    img = img.reshape(th, bh, tw, bw).mean(axis=(1, 3))
    return img.astype(np.float32)


def compute_fingerprint(
    mean_m: np.ndarray,
    cells: Sequence[CellFeature],
) -> Fingerprint:
    """Compute the canonical fingerprint for a FOV.

    Hash inputs:
      * downsampled mean_M quantized to 4 decimals (stability under minor
        float rounding across machines)
      * centroids sorted by (y, x, area) and quantized

    Blobs:
      * mean_m_blob: full-resolution mean_M as .npy bytes (for phase correlation)
      * centroid_blob: all CellFeatures as a structured array .npy bytes
    """
    downsampled = downsample(mean_m)

    hasher = hashlib.sha256()
    hasher.update(b"mean_m:")
    hasher.update(np.round(downsampled, 4).tobytes())

    sorted_cells = sorted(
        cells,
        key=lambda c: (round(c.centroid_y, 1), round(c.centroid_x, 1), c.area),
    )
    hasher.update(b"centroids:")
    hasher.update(
        np.array(
            [(
                round(c.centroid_y, 1),
                round(c.centroid_x, 1),
                int(c.area),
            ) for c in sorted_cells],
            dtype=[("y", np.float32), ("x", np.float32), ("a", np.int32)],
        ).tobytes()
    )

    mean_m_blob = _serialize_array(np.asarray(mean_m, dtype=np.float32))
    centroid_blob = _serialize_cells(list(cells))

    return Fingerprint(
        fingerprint_hash=hasher.hexdigest(),
        mean_m_blob=mean_m_blob,
        centroid_blob=centroid_blob,
        mean_m_downsampled=downsampled,
        centroids=list(cells),
    )


def deserialize_mean_m(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


def deserialize_cells(blob: bytes) -> list[CellFeature]:
    arr = np.load(io.BytesIO(blob), allow_pickle=False)
    return [
        CellFeature(
            local_label_id=int(r["local_label_id"]),
            centroid_y=float(r["centroid_y"]),
            centroid_x=float(r["centroid_x"]),
            area=int(r["area"]),
            solidity=float(r["solidity"]),
            eccentricity=float(r["eccentricity"]),
            nuclear_shadow_score=float(r["nuclear_shadow_score"]),
            soma_surround_contrast=float(r["soma_surround_contrast"]),
        )
        for r in arr
    ]


def _serialize_array(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _serialize_cells(cells: list[CellFeature]) -> bytes:
    dtype = [
        ("local_label_id", np.int32),
        ("centroid_y", np.float32),
        ("centroid_x", np.float32),
        ("area", np.int32),
        ("solidity", np.float32),
        ("eccentricity", np.float32),
        ("nuclear_shadow_score", np.float32),
        ("soma_surround_contrast", np.float32),
    ]
    arr = np.array(
        [(
            c.local_label_id,
            c.centroid_y,
            c.centroid_x,
            c.area,
            c.solidity,
            c.eccentricity,
            c.nuclear_shadow_score,
            c.soma_surround_contrast,
        ) for c in cells],
        dtype=dtype,
    )
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def cells_from_masks(
    merged_masks: np.ndarray,
    roi_metadata: list[dict],
) -> list[CellFeature]:
    """Extract CellFeatures from a uint16 label image + roi_metadata.json rows.

    The merged_masks.tif label value for each cell equals its `label_id`; this
    function recomputes centroids from the mask (rather than trusting a field
    that may not exist in roi_metadata) and pulls morphology fields from the
    metadata by label_id.
    """
    from scipy import ndimage as ndi

    meta_by_label = {int(m["label_id"]): m for m in roi_metadata}
    labels = np.unique(merged_masks)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return []

    centroids = ndi.center_of_mass(merged_masks > 0, merged_masks, labels)
    out: list[CellFeature] = []
    for lab, (cy, cx) in zip(labels.tolist(), centroids):
        meta = meta_by_label.get(int(lab), {})
        out.append(CellFeature(
            local_label_id=int(lab),
            centroid_y=float(cy),
            centroid_x=float(cx),
            area=int(meta.get("area", int((merged_masks == lab).sum()))),
            solidity=float(meta.get("solidity", 0.0)),
            eccentricity=float(meta.get("eccentricity", 0.0)),
            nuclear_shadow_score=float(meta.get("nuclear_shadow_score", 0.0)),
            soma_surround_contrast=float(meta.get("soma_surround_contrast", 0.0)),
        ))
    return out


def cells_from_rois(rois) -> list[CellFeature]:
    """Build CellFeatures from live ROI objects (in-memory FOVData path)."""
    from scipy import ndimage as ndi

    out: list[CellFeature] = []
    for roi in rois:
        if getattr(roi, "gate_outcome", None) == "reject":
            continue
        mask = roi.mask
        if mask is None or not mask.any():
            continue
        cy, cx = ndi.center_of_mass(mask)
        out.append(CellFeature(
            local_label_id=int(roi.label_id),
            centroid_y=float(cy),
            centroid_x=float(cx),
            area=int(roi.area),
            solidity=float(roi.solidity),
            eccentricity=float(roi.eccentricity),
            nuclear_shadow_score=float(roi.nuclear_shadow_score),
            soma_surround_contrast=float(roi.soma_surround_contrast),
        ))
    return out
