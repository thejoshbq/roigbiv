"""Load and validate a benchmark manifest; load a FOV's annotation points.

The benchmark root holds ``manifest.json`` and the files it names. The root is
the ``root`` argument, else the ``ROIGBIV_BENCH_ROOT`` environment variable.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Union

import numpy as np
import roifile
from scipy.ndimage import center_of_mass
from skimage.draw import polygon as draw_polygon

from bench.schema import (
    FOV, MOVIE_KINDS, SPLITS, TIERS, AnnotationPoints, AnnotationRef, Benchmark, ImageRef, MovieRef,
)

ENV_ROOT = "ROIGBIV_BENCH_ROOT"
MANIFEST_NAME = "manifest.json"

PathLike = Union[str, Path]

_FOV_KEYS = (
    "id", "animal", "split_group", "split", "tier", "lens", "region", "cell_type",
    "image_shape", "images", "annotations", "movie", "fs_hz", "averaging", "notes",
)


class ManifestError(ValueError):
    """The manifest, the acquisition file or the root is missing or invalid."""


def resolve_root(root: Optional[PathLike] = None) -> Path:
    """Return the benchmark root.

    ``root`` wins when given; otherwise ``os.environ[ENV_ROOT]``. Raise
    :class:`ManifestError` when neither is set, or when the directory has no
    ``manifest.json``.
    """
    if root is None:
        env = os.environ.get(ENV_ROOT)
        if not env:
            raise ManifestError(f"no benchmark root: pass root or set {ENV_ROOT}")
        root = env
    path = Path(root)
    if not (path / MANIFEST_NAME).is_file():
        raise ManifestError(f"{path} has no {MANIFEST_NAME}")
    return path


def _check_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field}: expected a non-empty path string")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts:
        raise ManifestError(f"{field}: path must be relative and stay inside the root: {value!r}")
    return value


def _require(obj: dict, key: str, field: str) -> Any:
    if key not in obj:
        raise ManifestError(f"{field}.{key} missing")
    return obj[key]


def _parse_fov(i: int, raw: Any) -> FOV:
    field = f"fovs[{i}]"
    if not isinstance(raw, dict):
        raise ManifestError(f"{field}: expected an object")
    for key in _FOV_KEYS:
        _require(raw, key, field)

    split = raw["split"]
    if split not in SPLITS:
        raise ManifestError(f"{field}.split: {split!r} not in {SPLITS}")
    tier = raw["tier"]
    if tier not in TIERS:
        raise ManifestError(f"{field}.tier: {tier!r} not in {TIERS}")

    shape = raw["image_shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise ManifestError(f"{field}.image_shape: expected [height, width]")
    image_shape = (int(shape[0]), int(shape[1]))

    images_raw = raw["images"]
    if not isinstance(images_raw, dict) or not images_raw:
        raise ManifestError(f"{field}.images: expected a non-empty object")
    images = {}
    for name, value in images_raw.items():
        ifield = f"{field}.images.{name}"
        if not isinstance(value, dict):
            raise ManifestError(f"{ifield}: expected an object")
        images[name] = ImageRef(
            path=_check_path(_require(value, "path", ifield), f"{ifield}.path"),
            sha256=_require(value, "sha256", ifield),
            derived=bool(value.get("derived", False)),
        )

    anns_raw = raw["annotations"]
    if not isinstance(anns_raw, list) or not anns_raw:
        raise ManifestError(f"{field}.annotations: expected a non-empty list")
    annotations = []
    for j, value in enumerate(anns_raw):
        afield = f"{field}.annotations[{j}]"
        if not isinstance(value, dict):
            raise ManifestError(f"{afield}: expected an object")
        annotations.append(AnnotationRef(
            path=_check_path(_require(value, "path", afield), f"{afield}.path"),
            sha256=_require(value, "sha256", afield),
            rater=_require(value, "rater", afield),
            n_rois=_require(value, "n_rois", afield),
        ))

    movie = None
    movie_raw = raw["movie"]
    if movie_raw is not None:
        mfield = f"{field}.movie"
        if not isinstance(movie_raw, dict):
            raise ManifestError(f"{mfield}: expected null or an object")
        kind = _require(movie_raw, "kind", mfield)
        if kind not in MOVIE_KINDS:
            raise ManifestError(f"{mfield}.kind: {kind!r} not in {MOVIE_KINDS}")
        movie = MovieRef(
            kind=kind,
            path=_check_path(_require(movie_raw, "path", mfield), f"{mfield}.path"),
            n_frames=_require(movie_raw, "frames_read", mfield),
            sha256=movie_raw.get("sha256"),
        )

    notes = raw["notes"]
    if not isinstance(notes, list):
        raise ManifestError(f"{field}.notes: expected a list")

    return FOV(
        id=raw["id"],
        animal=raw["animal"],
        split_group=raw["split_group"],
        split=split,
        tier=tier,
        lens=raw["lens"],
        region=raw["region"],
        cell_type=raw["cell_type"],
        image_shape=image_shape,
        images=images,
        annotations=tuple(annotations),
        movie=movie,
        fs_hz=raw["fs_hz"],
        averaging=raw["averaging"],
        notes=tuple(notes),
    )


def _apply_acquisition(fovs: list[FOV], acq: Any, manifest_sha: str) -> list[FOV]:
    if not isinstance(acq, dict):
        raise ManifestError("acquisition: expected an object")
    if "manifest_sha256" in acq and acq["manifest_sha256"] != manifest_sha:
        raise ManifestError("acquisition.manifest_sha256 does not match manifest.json")
    movies = acq.get("movies", {})
    if not isinstance(movies, dict):
        raise ManifestError("acquisition.movies: expected an object")

    by_id = {f.id: i for i, f in enumerate(fovs)}
    for fov_id in movies:
        if fov_id not in by_id:
            raise ManifestError(f"acquisition.movies.{fov_id}: no such FOV")

    out = list(fovs)
    for fov_id, values in movies.items():
        i = by_id[fov_id]
        fov = out[i]
        fs_hz, averaging = fov.fs_hz, fov.averaging
        for name, current in (("fs_hz", fov.fs_hz), ("averaging", fov.averaging)):
            new = values.get(name)
            if new is None:
                continue
            if current is None:
                if name == "fs_hz":
                    fs_hz = new
                else:
                    averaging = new
            elif current != new:
                raise ManifestError(f"fovs[{i}].{name}: manifest {current!r} conflicts with acquisition {new!r}")
        out[i] = dataclasses.replace(fov, fs_hz=fs_hz, averaging=averaging)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_file(root: Path, rel: str, expected: Optional[str], field: str) -> None:
    path = root / rel
    if not path.is_file():
        raise ManifestError(f"{field}: missing file {rel}")
    if expected is None or _sha256(path) != expected:
        raise ManifestError(f"{field}: sha256 mismatch for {rel}")


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
    base = resolve_root(root)
    manifest_path = base / MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    try:
        data = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{manifest_path}: expected a JSON object")
    for key in ("benchmark", "version", "fovs"):
        _require(data, key, "manifest")
    if not isinstance(data["fovs"], list):
        raise ManifestError("fovs: expected a list")

    fovs = [_parse_fov(i, raw) for i, raw in enumerate(data["fovs"])]
    seen: set[str] = set()
    for i, fov in enumerate(fovs):
        if fov.id in seen:
            raise ManifestError(f"fovs[{i}].id: duplicate {fov.id!r}")
        seen.add(fov.id)

    if acquisition is not None:
        acq = json.loads(Path(acquisition).read_text())
        fovs = _apply_acquisition(fovs, acq, hashlib.sha256(manifest_bytes).hexdigest())

    if verify:
        for i, fov in enumerate(fovs):
            for name, ref in fov.images.items():
                _verify_file(base, ref.path, ref.sha256, f"fovs[{i}].images.{name}")
            for j, ann in enumerate(fov.annotations):
                _verify_file(base, ann.path, ann.sha256, f"fovs[{i}].annotations[{j}]")
            if fov.movie is not None and fov.movie.kind == "tif":
                _verify_file(base, fov.movie.path, fov.movie.sha256, f"fovs[{i}].movie")

    return Benchmark(root=base, name=data["benchmark"], version=data["version"], fovs=tuple(fovs))


def abs_path(bench: Benchmark, rel: str) -> Path:
    """``bench.root / rel``."""
    return bench.root / rel


def select(
    bench: Benchmark,
    *,
    split: Optional[str] = None,
    tier: Optional[str] = None,
    lens: Optional[str] = None,
) -> tuple[FOV, ...]:
    """FOVs matching every filter that is not ``None``, in manifest order."""
    return tuple(
        f for f in bench.fovs
        if (split is None or f.split == split)
        and (tier is None or f.tier == tier)
        and (lens is None or f.lens == lens)
    )


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
    ref = fov.annotations[index]
    rois = roifile.roiread(str(abs_path(bench, ref.path)))
    if not isinstance(rois, list):
        rois = [rois]

    centroids: list[tuple[float, float]] = []
    areas: list[float] = []
    for roi in rois:
        coords = roi.coordinates()
        if coords is None or len(coords) == 0:
            continue
        if roi.roitype == roifile.ROI_TYPE.POINT:
            for x, y in coords:
                centroids.append((float(y), float(x)))
                areas.append(float("nan"))
            continue
        if len(coords) < 3:
            continue
        rr, cc = draw_polygon(coords[:, 1], coords[:, 0], shape=fov.image_shape)
        if rr.size == 0:
            continue
        mask = np.zeros(fov.image_shape, dtype=bool)
        mask[rr, cc] = True
        cy, cx = center_of_mass(mask)
        centroids.append((float(cy), float(cx)))
        areas.append(float(rr.size))

    return AnnotationPoints(
        centroids=np.asarray(centroids, dtype=np.float32).reshape(-1, 2),
        areas=np.asarray(areas, dtype=np.float64),
        rater=ref.rater,
        path=ref.path,
    )
