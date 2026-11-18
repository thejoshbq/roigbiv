"""Shared types for ``scripts/bench``.

Field names follow the benchmark's ``manifest.json`` (v0, frozen 2026-11-12)
so a loaded manifest round-trips without renaming. Paths stay relative to the
benchmark root; ``bench.manifest.abs_path`` joins them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SPLITS = ("dev", "test")
TIERS = ("core", "flagged")
MOVIE_KINDS = ("tif", "frame_dir")


@dataclass(frozen=True)
class ImageRef:
    """A summary image (``std`` or ``avg``) stored in the benchmark."""

    path: str
    sha256: str
    derived: bool = False


@dataclass(frozen=True)
class AnnotationRef:
    """One hand-drawn ImageJ RoiSet for a FOV."""

    path: str
    sha256: str
    rater: Optional[str]
    n_rois: int


@dataclass(frozen=True)
class MovieRef:
    """A movie: one multi-page TIFF (``tif``) or a folder of one-frame TIFFs (``frame_dir``).

    ``sha256`` is the file hash for ``tif`` and ``None`` for ``frame_dir``.
    """

    kind: str
    path: str
    n_frames: int
    sha256: Optional[str] = None


@dataclass(frozen=True)
class FOV:
    """One FOV-session entry of the manifest."""

    id: str
    animal: str
    split_group: str
    split: str
    tier: str
    lens: str
    region: str
    cell_type: str
    image_shape: tuple[int, int]
    images: dict[str, ImageRef]
    annotations: tuple[AnnotationRef, ...]
    movie: Optional[MovieRef]
    fs_hz: Optional[float]
    averaging: Optional[int]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Benchmark:
    """A loaded benchmark: its root directory and every FOV in manifest order."""

    root: Path
    name: str
    version: str
    fovs: tuple[FOV, ...]


@dataclass(frozen=True)
class AnnotationPoints:
    """Centroids and areas parsed from one RoiSet.

    ``centroids`` is ``(N, 2)`` float32 ``(y, x)``. ``areas`` is ``(N,)``
    float64 pixel counts, ``nan`` for POINT ROIs (a point has no area).
    """

    centroids: np.ndarray
    areas: np.ndarray
    rater: Optional[str] = None
    path: str = ""


@dataclass(frozen=True)
class FovScore:
    """Centroid match counts for one FOV against one annotation."""

    fov_id: str
    lens: str
    n_gt: int
    n_pred: int
    n_tp: int
    max_distance: float

    @property
    def precision(self) -> Optional[float]:
        return self.n_tp / self.n_pred if self.n_pred else None

    @property
    def recall(self) -> Optional[float]:
        return self.n_tp / self.n_gt if self.n_gt else None

    @property
    def f1(self) -> Optional[float]:
        denom = self.n_gt + self.n_pred
        return 2 * self.n_tp / denom if denom else None


@dataclass(frozen=True)
class GroupScore:
    """Micro-averaged counts over a group of FOVs, plus the mean per-FOV recall."""

    group: str
    n_fovs: int
    n_gt: int
    n_pred: int
    n_tp: int
    macro_recall: Optional[float]

    @property
    def precision(self) -> Optional[float]:
        return self.n_tp / self.n_pred if self.n_pred else None

    @property
    def recall(self) -> Optional[float]:
        return self.n_tp / self.n_gt if self.n_gt else None


@dataclass(frozen=True)
class RaterPair:
    """Agreement between two raters' centroids on one FOV (rater_a < rater_b)."""

    fov_id: str
    rater_a: str
    rater_b: str
    n_a: int
    n_b: int
    n_tp: int
    max_distance: float

    @property
    def f1(self) -> Optional[float]:
        denom = self.n_a + self.n_b
        return 2 * self.n_tp / denom if denom else None


@dataclass(frozen=True)
class SplitAudit:
    """Counts per split and the dev/test leaks, over the FOVs audited."""

    fovs_per_split: dict[str, int]
    animals_per_split: dict[str, int]
    sessions_per_animal: dict[str, int]
    animal_leaks: tuple[str, ...]
    group_leaks: tuple[str, ...]
    flagged_in_test: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not (self.animal_leaks or self.group_leaks or self.flagged_in_test)


@dataclass(frozen=True)
class G0Result:
    """The G0 size gate: FOVs with an image and a RoiSet, animals, movies."""

    n_fovs: int
    n_animals: int
    n_movies: int
    min_fovs: int
    min_animals: int
    min_movies: int

    @property
    def fovs_ok(self) -> bool:
        return self.n_fovs >= self.min_fovs

    @property
    def animals_ok(self) -> bool:
        return self.n_animals >= self.min_animals

    @property
    def movies_ok(self) -> bool:
        return self.n_movies >= self.min_movies

    @property
    def passed(self) -> bool:
        return self.fovs_ok and self.animals_ok and self.movies_ok


@dataclass(frozen=True)
class ActivityImages:
    """Activity-summary images from one movie, each ``(H, W)`` float32."""

    activity_max: np.ndarray
    temporal_sd: np.ndarray
    n_frames: int
    n_bins: int
    n_blocks: int


@dataclass(frozen=True)
class UnionScore:
    """Candidate-pool coverage of one FOV's ground truth across proposers.

    ``n_candidates``, ``covered`` and ``unique`` are keyed by source name.
    ``covered[s]`` counts GT cells source ``s`` matches on its own;
    ``unique[s]`` counts the ones no other source matches.
    """

    fov_id: str
    n_gt: int
    n_candidates: dict[str, int]
    pool_n_tp: int
    covered: dict[str, int]
    unique: dict[str, int]
    max_distance: float

    @property
    def n_candidates_total(self) -> int:
        return sum(self.n_candidates.values())

    @property
    def pool_recall(self) -> Optional[float]:
        return self.pool_n_tp / self.n_gt if self.n_gt else None

    @property
    def pool_precision(self) -> Optional[float]:
        total = self.n_candidates_total
        return self.pool_n_tp / total if total else None


@dataclass(frozen=True)
class DecideResult:
    """A scores-only decider fit on dev FOVs and scored on test FOVs."""

    accept_threshold: float
    model_trained: bool
    n_train_rows: int
    per_fov: tuple[FovScore, ...]
    dev_threshold_sweep: list[dict] = field(default_factory=list)
