"""Serializable benchmark metric data models.

Defines DetectionMetrics, TrackingMetrics, RuntimeMetrics, HitlMetrics, and
TraceMetrics — plain dataclasses that standardize how benchmark results are
represented and persisted as JSON. Pure data model: metric computation logic
lives in the matcher (roadmap item A6), not here.

Distinct from roigbiv.eval, which is an earlier, ad-hoc dict-based scoring
harness (stratified_metrics, iou_match) for the sequential pipeline's own
QC loop. This package is the formalized data model for the separate
roigbiv-bench roadmap (Milestone A) and does not depend on or modify
roigbiv.eval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class DetectionMetrics:
    """Per-FOV spatial detection quality metrics against ground truth."""

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    mean_iou: Optional[float] = None
    median_iou: Optional[float] = None
    false_positive_count: Optional[int] = None
    false_negative_count: Optional[int] = None

    @classmethod
    def empty(cls) -> "DetectionMetrics":
        """Return an all-None instance for when ground truth is unavailable."""
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DetectionMetrics":
        return cls(**payload)


@dataclass
class TrackingMetrics:
    """Cross-session ROI identity tracking error counts."""

    split_count: Optional[int] = None
    merge_count: Optional[int] = None

    @classmethod
    def empty(cls) -> "TrackingMetrics":
        """Return an all-None instance for when ground truth is unavailable."""
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TrackingMetrics":
        return cls(**payload)


@dataclass
class RuntimeMetrics:
    """Pipeline run performance metrics."""

    runtime_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    @classmethod
    def empty(cls) -> "RuntimeMetrics":
        """Return an all-None instance for when runtime was not captured."""
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RuntimeMetrics":
        return cls(**payload)


@dataclass
class HitlMetrics:
    """Human-in-the-loop correction counts, one field per CorrectionOp type
    in roigbiv.pipeline.corrections (add/delete/merge/split/edit/relabel)."""

    add_count: Optional[int] = None
    delete_count: Optional[int] = None
    merge_count: Optional[int] = None
    split_count: Optional[int] = None
    edit_count: Optional[int] = None
    relabel_count: Optional[int] = None
    total_corrections: Optional[int] = None

    @classmethod
    def empty(cls) -> "HitlMetrics":
        """Return an all-None instance for when no HITL review occurred."""
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "HitlMetrics":
        return cls(**payload)


@dataclass
class TraceMetrics:
    """Calcium trace quality metrics against ground-truth traces."""

    mean_trace_correlation: Optional[float] = None
    median_trace_correlation: Optional[float] = None
    num_traces_compared: Optional[int] = None

    @classmethod
    def empty(cls) -> "TraceMetrics":
        """Return an all-None instance for when ground truth is unavailable."""
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TraceMetrics":
        return cls(**payload)
