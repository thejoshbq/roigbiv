"""Tests for roigbiv.benchmark.results — the BenchmarkRun/BenchmarkRunResult contract."""
from __future__ import annotations

import json

import pytest

from roigbiv.benchmark.metrics import DetectionMetrics, RuntimeMetrics, TrackingMetrics
from roigbiv.benchmark.results import (
    STAGE_DETECTOR_MAP,
    BenchmarkRun,
    BenchmarkRunResult,
    DetectorStageCount,
)


def _populated_result(fov_id: str = "fov1", with_gt: bool = True) -> BenchmarkRunResult:
    detection = (
        DetectionMetrics(precision=0.9, recall=0.8, f1=0.85, mean_iou=0.7,
                          false_positive_count=2, false_negative_count=3)
        if with_gt else DetectionMetrics.empty()
    )
    return BenchmarkRunResult(
        dataset_id="ds1",
        fov_id=fov_id,
        quality_tier="high",
        lens_type="generic",
        detection=detection,
        tracking=TrackingMetrics(split_count=1, merge_count=0),
        runtime=RuntimeMetrics(runtime_seconds=12.5, peak_memory_mb=256.0),
        detector_stage_counts=[
            DetectorStageCount(source_stage=1, source_detector=STAGE_DETECTOR_MAP[1], count=5),
        ],
        notes="test note",
    )


def test_detector_stage_count_roundtrip():
    obj = DetectorStageCount(source_stage=2, source_detector="suite2p", count=7)
    rebuilt = DetectorStageCount.from_dict(json.loads(json.dumps(obj.to_dict())))
    assert rebuilt == obj


def test_benchmark_run_result_roundtrip_with_gt():
    obj = _populated_result(with_gt=True)
    payload = json.loads(json.dumps(obj.to_dict()))
    rebuilt = BenchmarkRunResult.from_dict(payload)
    assert rebuilt == obj
    assert rebuilt.has_ground_truth is True


def test_benchmark_run_result_roundtrip_without_gt():
    obj = _populated_result(with_gt=False)
    payload = json.loads(json.dumps(obj.to_dict()))
    rebuilt = BenchmarkRunResult.from_dict(payload)
    assert rebuilt == obj
    assert rebuilt.has_ground_truth is False


def test_benchmark_run_result_detector_stage_counts_none_roundtrips():
    obj = BenchmarkRunResult(dataset_id="ds1", fov_id="fov1", quality_tier="low", lens_type="grin")
    assert obj.detector_stage_counts is None
    payload = json.loads(json.dumps(obj.to_dict()))
    rebuilt = BenchmarkRunResult.from_dict(payload)
    assert rebuilt.detector_stage_counts is None
    assert rebuilt == obj


def test_benchmark_run_defaults_are_empty_metrics():
    obj = BenchmarkRunResult(dataset_id="ds1", fov_id="fov1", quality_tier="high", lens_type="generic")
    assert obj.detection == DetectionMetrics.empty()
    assert obj.has_ground_truth is False


def test_benchmark_run_roundtrip():
    run = BenchmarkRun(
        pipeline_mode="cascade_legacy",
        git_commit="a" * 40,
        git_dirty=False,
        config_hash="sha256:" + "b" * 64,
        created_at="2026-07-02T00:00:00Z",
        manifest_path="manifest.yaml",
        results=[_populated_result("fov1"), _populated_result("fov2", with_gt=False)],
    )
    payload = json.loads(json.dumps(run.to_dict()))
    rebuilt = BenchmarkRun.from_dict(payload)
    assert rebuilt == run
    assert rebuilt.schema_version == 1


def test_benchmark_run_empty_results_roundtrips():
    run = BenchmarkRun(pipeline_mode="benchmark_only")
    payload = json.loads(json.dumps(run.to_dict()))
    rebuilt = BenchmarkRun.from_dict(payload)
    assert rebuilt == run
    assert rebuilt.results == []


def test_benchmark_run_from_dict_defaults_schema_version():
    payload = {"pipeline_mode": "cascade_legacy", "results": []}
    run = BenchmarkRun.from_dict(payload)
    assert run.schema_version == 1
    assert run.git_commit is None
    assert run.created_at == ""
