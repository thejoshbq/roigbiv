"""Load and validate a benchmark manifest; load a FOV's annotation points.

The benchmark root holds ``manifest.json`` and the files it names. The root is
the ``root`` argument, else the ``ROIGBIV_BENCH_ROOT`` environment variable.
"""
# contract stub: U1 replaces the function bodies below; signatures are frozen.
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from bench.schema import FOV, AnnotationPoints, Benchmark

ENV_ROOT = "ROIGBIV_BENCH_ROOT"
MANIFEST_NAME = "manifest.json"

PathLike = Union[str, Path]


class ManifestError(ValueError):
    """The manifest, the acquisition file or the root is missing or invalid."""


def resolve_root(root: Optional[PathLike] = None) -> Path:
    """Return the benchmark root.

    ``root`` wins when given; otherwise ``os.environ[ENV_ROOT]``. Raise
    :class:`ManifestError` when neither is set, or when the directory has no
    ``manifest.json``.
    """
    return Path()


def load_benchmark(
    root: Optional[PathLike] = None,
    acquisition: Optional[PathLike] = None,
    verify: bool = False,
) -> Benchmark:
    """Load ``<root>/manifest.json`` into a :class:`Benchmark`.

    Rules (each failure raises :class:`ManifestError` naming the field, e.g.
    ``"fovs[2].split"``):

    1. Top-level keys ``benchmark``, ``version`` and ``fovs`` are required;
       ``Benchmark.name`` comes from ``benchmark``.
    2. Each FOV needs ``id, animal, split_group, split, tier, lens, region,
       cell_type, image_shape, images, annotations, movie, fs_hz, averaging,
       notes``. Extra keys are ignored. Duplicate ``id`` values are an error.
    3. ``split`` in ``schema.SPLITS``, ``tier`` in ``schema.TIERS``.
       ``image_shape`` becomes a ``tuple[int, int]``. ``notes`` becomes a tuple.
    4. ``images`` is a dict with at least one key; each value gives
       ``ImageRef(path, sha256, derived=bool(value.get("derived", False)))``.
    5. ``annotations`` is a non-empty list; each gives
       ``AnnotationRef(path, sha256, rater, n_rois)``.
    6. ``movie`` is ``null`` or a dict with ``kind`` in ``schema.MOVIE_KINDS``;
       it gives ``MovieRef(kind, path, n_frames=value["frames_read"],
       sha256=value.get("sha256"))``.
    7. Every stored path must be relative and must not contain a ``..`` part.
    8. ``acquisition`` (optional) is a JSON file
       ``{"manifest_sha256": ..., "movies": {fov_id: {"fs_hz": .., "averaging": ..}}}``.
       If ``manifest_sha256`` is present it must equal the sha256 of the
       manifest file's bytes. Every listed ``fov_id`` must exist. A listed
       value fills ``fs_hz``/``averaging`` where the manifest has ``null``; a
       non-null manifest value that differs from a non-null acquisition value
       is an error. ``null`` in the acquisition file leaves the field as is.
    9. ``verify=True`` checks the sha256 of every image, every annotation and
       every ``tif`` movie against the file under the root; a mismatch or a
       missing file is an error. ``frame_dir`` movies are not hashed here.
    """
    return Benchmark(root=Path(), name="", version="", fovs=())


def abs_path(bench: Benchmark, rel: str) -> Path:
    """``bench.root / rel``."""
    return Path()


def select(
    bench: Benchmark,
    *,
    split: Optional[str] = None,
    tier: Optional[str] = None,
    lens: Optional[str] = None,
) -> tuple[FOV, ...]:
    """FOVs matching every filter that is not ``None``, in manifest order."""
    return ()


def load_annotation(bench: Benchmark, fov: FOV, index: int = 0) -> AnnotationPoints:
    """Parse ``fov.annotations[index]`` into centroids and areas.

    Reads the RoiSet with ``roifile.roiread`` and walks the ROIs in file order:

    - POINT ROIs emit one centroid per coordinate, ``(y, x) = (coord[1],
      coord[0])``, with area ``nan``.
    - Other ROIs with 3 or more coordinates are rasterized with
      ``skimage.draw.polygon(coords[:, 1], coords[:, 0], shape=fov.image_shape)``;
      the centroid is ``scipy.ndimage.center_of_mass`` of that ROI's own mask
      and the area is its pixel count. ROIs with fewer than 3 coordinates or an
      empty raster are skipped.

    Centroids then equal ``centroid_bakeoff.ground_truth.imagej_roiset_to_centroids``
    for the same file. ``rater`` and ``path`` come from the ``AnnotationRef``.
    """
    return AnnotationPoints(
        centroids=np.zeros((0, 2), dtype=np.float32),
        areas=np.zeros(0, dtype=np.float64),
    )
