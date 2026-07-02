"""Tests for roigbiv.benchmark.report — Markdown + JSON comparison report builders."""
from __future__ import annotations

from roigbiv.benchmark.metrics import DetectionMetrics, RuntimeMetrics, TrackingMetrics
from roigbiv.benchmark.report import build_json_report, build_markdown_report
from roigbiv.benchmark.results import (
    STAGE_DETECTOR_MAP,
    BenchmarkRun,
    BenchmarkRunResult,
    DetectorStageCount,
)


def _result(fov_id, quality_tier="high", lens_type="generic", with_gt=True,
            with_detector_stage=False) -> BenchmarkRunResult:
    detection = (
        DetectionMetrics(precision=0.9, recall=0.8, f1=0.85, mean_iou=0.7,
                          false_positive_count=2, false_negative_count=1)
        if with_gt else DetectionMetrics.empty()
    )
    return BenchmarkRunResult(
        dataset_id="dsA",
        fov_id=fov_id,
        quality_tier=quality_tier,
        lens_type=lens_type,
        detection=detection,
        tracking=TrackingMetrics(split_count=1, merge_count=2),
        runtime=RuntimeMetrics(runtime_seconds=10.0, peak_memory_mb=100.0),
        detector_stage_counts=(
            [DetectorStageCount(source_stage=1, source_detector=STAGE_DETECTOR_MAP[1], count=3)]
            if with_detector_stage else None
        ),
    )


def test_build_markdown_report_empty_runs():
    md = build_markdown_report([])
    assert "No benchmark runs provided" in md


def test_build_markdown_report_single_mode_no_warnings():
    run = BenchmarkRun(pipeline_mode="cascade_legacy", results=[_result("fov1")])
    md = build_markdown_report([run])
    assert "cascade_legacy" in md
    assert "## Warnings" not in md  # no missing-GT FOVs, section omitted


def test_build_markdown_report_two_modes_compare():
    run_a = BenchmarkRun(pipeline_mode="cascade_legacy", results=[_result("fov1")])
    run_b = BenchmarkRun(pipeline_mode="candidate_union", results=[_result("fov1")])
    md = build_markdown_report([run_a, run_b])
    assert "cascade_legacy" in md
    assert "candidate_union" in md


def test_build_markdown_report_missing_gt_produces_warning():
    run = BenchmarkRun(pipeline_mode="cascade_legacy", results=[_result("fov1", with_gt=False)])
    md = build_markdown_report([run])
    assert "## Warnings" in md
    assert "dsA/fov1" in md
    assert "ground truth unavailable" in md


def test_build_markdown_report_empty_results_notes_zero_fovs():
    run = BenchmarkRun(pipeline_mode="cascade_legacy", results=[])
    md = build_markdown_report([run])
    assert "0 FOVs in this run" in md


def test_build_markdown_report_detector_stage_present():
    run = BenchmarkRun(pipeline_mode="cascade_legacy",
                        results=[_result("fov1", with_detector_stage=True)])
    md = build_markdown_report([run])
    assert "cellpose" in md


def test_build_markdown_report_detector_stage_absent_notes_not_captured():
    run = BenchmarkRun(pipeline_mode="cascade_legacy",
                        results=[_result("fov1", with_detector_stage=False)])
    md = build_markdown_report([run])
    assert "Not captured by any FOV in this run" in md


def test_missing_gt_fov_excluded_from_detection_averages():
    run = BenchmarkRun(
        pipeline_mode="cascade_legacy",
        results=[_result("fov1", with_gt=True), _result("fov2", with_gt=False)],
    )
    report = build_json_report([run])
    overall_rows = [
        row for row in report["detection_by_quality_tier"]["cascade_legacy"]
        if row["stratum"] == "Overall"
    ]
    assert len(overall_rows) == 1
    # Only fov1 (with GT) contributes -> n_fovs == 1, not 2
    assert overall_rows[0]["n_fovs"] == 1
    assert overall_rows[0]["precision"] == 0.9


def test_build_json_report_structure_keys():
    run = BenchmarkRun(pipeline_mode="cascade_legacy", results=[_result("fov1")])
    report = build_json_report([run])
    assert set(report.keys()) == {
        "provenance", "detection_by_quality_tier", "detection_by_lens_type",
        "detector_stage_breakdown", "runtime", "tracking", "warnings", "raw_runs",
    }
    assert report["provenance"][0]["pipeline_mode"] == "cascade_legacy"
    assert report["raw_runs"][0]["pipeline_mode"] == "cascade_legacy"


def test_stratified_by_quality_tier_groups_separately():
    run = BenchmarkRun(
        pipeline_mode="cascade_legacy",
        results=[
            _result("fov1", quality_tier="high"),
            _result("fov2", quality_tier="low"),
        ],
    )
    report = build_json_report([run])
    strata = {row["stratum"] for row in report["detection_by_quality_tier"]["cascade_legacy"]}
    assert strata == {"high", "low", "Overall"}


def test_stratum_with_no_ground_truth_still_appears():
    """A stratum where every FOV lacks GT still gets a row (n_fovs=0, all-None
    metrics) instead of silently disappearing from the table."""
    run = BenchmarkRun(
        pipeline_mode="cascade_legacy",
        results=[
            _result("fov1", quality_tier="high", with_gt=True),
            _result("fov2", quality_tier="low", with_gt=False),
        ],
    )
    report = build_json_report([run])
    rows = report["detection_by_quality_tier"]["cascade_legacy"]
    strata = {row["stratum"]: row for row in rows}
    assert "low" in strata
    assert strata["low"]["n_fovs"] == 0
    assert strata["low"]["precision"] is None
    # "high" still aggregates normally
    assert strata["high"]["n_fovs"] == 1
    assert strata["high"]["precision"] == 0.9
    # Overall only reflects the one GT-bearing FOV
    assert strata["Overall"]["n_fovs"] == 1


def test_runtime_and_tracking_aggregation():
    run = BenchmarkRun(
        pipeline_mode="cascade_legacy",
        results=[_result("fov1"), _result("fov2")],
    )
    report = build_json_report([run])
    assert report["runtime"]["cascade_legacy"]["mean_runtime_seconds"] == 10.0
    assert report["tracking"]["cascade_legacy"]["total_split_count"] == 2
    assert report["tracking"]["cascade_legacy"]["total_merge_count"] == 4
