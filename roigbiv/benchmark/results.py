"""Aggregate benchmark-run result schema (issue #32).

Defines the on-disk contract that the future benchmark runner (#28) and
object-level matcher (#30) should write to: one `benchmark_run.json` file per
pipeline-mode invocation, serialized from `BenchmarkRun.to_dict()`.

`BenchmarkRun` is the top-level, one-per-pipeline-mode-invocation envelope;
`BenchmarkRunResult` is one FOV's metrics within that run. Composes the
existing metrics.py dataclasses and schema.py's stratification fields —
does not redefine them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from roigbiv.benchmark.metrics import (
    DetectionMetrics,
    HitlMetrics,
    RuntimeMetrics,
    TraceMetrics,
    TrackingMetrics,
)

# Stage -> detector name, per the fixed sequential-pipeline architecture.
# Used for the detector/stage breakdown table without depending on the
# unmerged CandidateROI.source_detector field (#42/#43). Once CandidateROI
# lands in main, prefer reading source_detector directly instead of this map.
STAGE_DETECTOR_MAP: dict[int, str] = {
    1: "cellpose",
    2: "suite2p",
    3: "template_sweep",
    4: "tonic_search",
}


@dataclass
class DetectorStageCount:
    """Count of ROIs attributed to one (stage, detector) pair for one FOV."""

    source_stage: int
    source_detector: str
    count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DetectorStageCount":
        return cls(**payload)


@dataclass
class BenchmarkRunResult:
    """One FOV's benchmark result within a single pipeline-mode run."""

    dataset_id: str
    fov_id: str
    quality_tier: str
    lens_type: str
    detection: DetectionMetrics = field(default_factory=DetectionMetrics.empty)
    tracking: TrackingMetrics = field(default_factory=TrackingMetrics.empty)
    runtime: RuntimeMetrics = field(default_factory=RuntimeMetrics.empty)
    hitl: HitlMetrics = field(default_factory=HitlMetrics.empty)
    trace: TraceMetrics = field(default_factory=TraceMetrics.empty)
    detector_stage_counts: Optional[list[DetectorStageCount]] = None
    notes: Optional[str] = None

    @property
    def has_ground_truth(self) -> bool:
        """True iff detection metrics were actually computed (metrics.py's
        `.empty()` convention is how a missing-GT FOV is represented)."""
        return self.detection != DetectionMetrics.empty()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BenchmarkRunResult":
        dsc = payload.get("detector_stage_counts")
        return cls(
            dataset_id=payload["dataset_id"],
            fov_id=payload["fov_id"],
            quality_tier=payload["quality_tier"],
            lens_type=payload["lens_type"],
            detection=DetectionMetrics.from_dict(payload.get("detection") or {}),
            tracking=TrackingMetrics.from_dict(payload.get("tracking") or {}),
            runtime=RuntimeMetrics.from_dict(payload.get("runtime") or {}),
            hitl=HitlMetrics.from_dict(payload.get("hitl") or {}),
            trace=TraceMetrics.from_dict(payload.get("trace") or {}),
            detector_stage_counts=(
                None if dsc is None else [DetectorStageCount.from_dict(c) for c in dsc]
            ),
            notes=payload.get("notes"),
        )


@dataclass
class BenchmarkRun:
    """Top-level envelope: one pipeline-mode's full benchmark invocation."""

    pipeline_mode: str
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    config_hash: Optional[str] = None
    created_at: str = ""
    manifest_path: Optional[str] = None
    results: list[BenchmarkRunResult] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BenchmarkRun":
        return cls(
            schema_version=payload.get("schema_version", 1),
            pipeline_mode=payload["pipeline_mode"],
            git_commit=payload.get("git_commit"),
            git_dirty=payload.get("git_dirty"),
            config_hash=payload.get("config_hash"),
            created_at=payload.get("created_at", ""),
            manifest_path=payload.get("manifest_path"),
            results=[BenchmarkRunResult.from_dict(r) for r in payload.get("results", [])],
        )
