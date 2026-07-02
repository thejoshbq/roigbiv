"""Markdown + JSON comparison report builders for benchmark runs (issue #32).

Pure functions — no I/O. Callers (the `roigbiv-benchmark report` CLI
subcommand) are responsible for reading `BenchmarkRun` JSON files and writing
the returned Markdown string / JSON dict to disk.

Style matches experiments/reports/comparison.md (hand-written prior art):
3-decimal floats, bold "Overall" rows, plain GFM tables (no Jinja2 — none
exists in this repo).
"""
from __future__ import annotations

from typing import Callable, Optional

from roigbiv.benchmark.results import BenchmarkRun, BenchmarkRunResult

_MISSING = "—"  # em dash, used for None cells in tables


def _mean(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _sum_ints(values: list[Optional[int]]) -> Optional[int]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def _fmt_float(x: Optional[float], decimals: int = 3) -> str:
    return _MISSING if x is None else f"{x:.{decimals}f}"


def _fmt_int(x: Optional[int]) -> str:
    return _MISSING if x is None else str(x)


def _gt_results(results: list[BenchmarkRunResult]) -> list[BenchmarkRunResult]:
    """Results with ground truth available (excludes DetectionMetrics.empty())."""
    return [r for r in results if r.has_ground_truth]


def _detection_row(label: str, results: list[BenchmarkRunResult]) -> dict:
    return {
        "stratum": label,
        "n_fovs": len(results),
        "precision": _mean([r.detection.precision for r in results]),
        "recall": _mean([r.detection.recall for r in results]),
        "f1": _mean([r.detection.f1 for r in results]),
        "mean_iou": _mean([r.detection.mean_iou for r in results]),
        "false_positive_count": _sum_ints([r.detection.false_positive_count for r in results]),
        "false_negative_count": _sum_ints([r.detection.false_negative_count for r in results]),
    }


def _stratified_detection_table(
    results: list[BenchmarkRunResult], key_fn: Callable[[BenchmarkRunResult], str]
) -> list[dict]:
    """Detection-metric rows grouped by `key_fn`, plus a trailing Overall row.

    Every stratum present in `results` gets a row, even if every FOV in it is
    missing ground truth (that row's metrics are then all-None/n_fovs=0
    rather than the stratum silently disappearing). Only ground-truth-bearing
    FOVs contribute to the averages — missing-GT FOVs are excluded (not
    counted as 0, which would skew them) and surfaced separately via
    `_warnings`.
    """
    gt_results = _gt_results(results)
    all_strata = sorted({key_fn(r) for r in results})
    strata: dict[str, list[BenchmarkRunResult]] = {s: [] for s in all_strata}
    for r in gt_results:
        strata[key_fn(r)].append(r)

    rows = [_detection_row(stratum, strata[stratum]) for stratum in all_strata]
    rows.append(_detection_row("Overall", gt_results))
    return rows


def _detector_stage_table(results: list[BenchmarkRunResult]) -> Optional[list[dict]]:
    """Aggregate (stage, detector) -> count across all FOVs in a run.

    Returns None if no result in this run captured detector_stage_counts.
    """
    captured = [r for r in results if r.detector_stage_counts is not None]
    if not captured:
        return None

    counts: dict[tuple[int, str], int] = {}
    for r in captured:
        for dsc in r.detector_stage_counts:
            key = (dsc.source_stage, dsc.source_detector)
            counts[key] = counts.get(key, 0) + dsc.count

    rows = [
        {"source_stage": stage, "source_detector": detector, "count": count}
        for (stage, detector), count in sorted(counts.items())
    ]
    return rows


def _runtime_row(results: list[BenchmarkRunResult]) -> dict:
    return {
        "mean_runtime_seconds": _mean([r.runtime.runtime_seconds for r in results]),
        "mean_peak_memory_mb": _mean([r.runtime.peak_memory_mb for r in results]),
    }


def _tracking_row(results: list[BenchmarkRunResult]) -> dict:
    return {
        "total_split_count": _sum_ints([r.tracking.split_count for r in results]),
        "total_merge_count": _sum_ints([r.tracking.merge_count for r in results]),
    }


def _warnings(runs: list[BenchmarkRun]) -> list[str]:
    warnings: list[str] = []
    for run in runs:
        for r in run.results:
            if not r.has_ground_truth:
                warnings.append(
                    f"{run.pipeline_mode}: {r.dataset_id}/{r.fov_id} — "
                    "ground truth unavailable; detection metrics omitted."
                )
    return warnings


def build_json_report(runs: list[BenchmarkRun]) -> dict:
    """Structured report payload — this is report.json's exact content."""
    sections: dict = {
        "provenance": [
            {
                "pipeline_mode": run.pipeline_mode,
                "git_commit": run.git_commit,
                "git_dirty": run.git_dirty,
                "config_hash": run.config_hash,
                "manifest_path": run.manifest_path,
                "created_at": run.created_at,
                "n_fovs": len(run.results),
            }
            for run in runs
        ],
        "detection_by_quality_tier": {
            run.pipeline_mode: _stratified_detection_table(run.results, lambda r: r.quality_tier)
            for run in runs
        },
        "detection_by_lens_type": {
            run.pipeline_mode: _stratified_detection_table(run.results, lambda r: r.lens_type)
            for run in runs
        },
        "detector_stage_breakdown": {
            run.pipeline_mode: _detector_stage_table(run.results) for run in runs
        },
        "runtime": {run.pipeline_mode: _runtime_row(run.results) for run in runs},
        "tracking": {run.pipeline_mode: _tracking_row(run.results) for run in runs},
        "warnings": _warnings(runs),
        "raw_runs": [run.to_dict() for run in runs],
    }
    return sections


def _render_detection_table_md(rows: list[dict]) -> str:
    header = "| Stratum | N FOVs | Precision | Recall | F1 | Mean IoU | FP | FN |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        cells = [
            row["stratum"],
            str(row["n_fovs"]),
            _fmt_float(row["precision"]),
            _fmt_float(row["recall"]),
            _fmt_float(row["f1"]),
            _fmt_float(row["mean_iou"]),
            _fmt_int(row["false_positive_count"]),
            _fmt_int(row["false_negative_count"]),
        ]
        if row["stratum"] == "Overall":
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_provenance_table_md(runs: list[BenchmarkRun]) -> str:
    header = "| Pipeline Mode | Git Commit | Dirty | Config Hash | Manifest | Created | N FOVs |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for run in runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    run.pipeline_mode,
                    run.git_commit or _MISSING,
                    _MISSING if run.git_dirty is None else ("yes" if run.git_dirty else "no"),
                    run.config_hash or _MISSING,
                    run.manifest_path or _MISSING,
                    run.created_at or _MISSING,
                    str(len(run.results)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_detector_stage_md(runs: list[BenchmarkRun]) -> str:
    parts = []
    for run in runs:
        rows = _detector_stage_table(run.results)
        parts.append(f"### {run.pipeline_mode}\n")
        if rows is None:
            parts.append("*Not captured by any FOV in this run.*\n")
            continue
        lines = ["| Stage | Detector | Count |", "|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['source_stage']} | {row['source_detector']} | {row['count']} |")
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


def _render_runtime_md(runs: list[BenchmarkRun]) -> str:
    lines = ["| Pipeline Mode | Mean Runtime (s) | Mean Peak Memory (MB) |", "|---|---|---|"]
    for run in runs:
        row = _runtime_row(run.results)
        lines.append(
            f"| {run.pipeline_mode} | {_fmt_float(row['mean_runtime_seconds'])} "
            f"| {_fmt_float(row['mean_peak_memory_mb'], decimals=1)} |"
        )
    return "\n".join(lines)


def _render_tracking_md(runs: list[BenchmarkRun]) -> str:
    lines = ["| Pipeline Mode | Total Splits | Total Merges |", "|---|---|---|"]
    for run in runs:
        row = _tracking_row(run.results)
        lines.append(
            f"| {run.pipeline_mode} | {_fmt_int(row['total_split_count'])} "
            f"| {_fmt_int(row['total_merge_count'])} |"
        )
    return "\n".join(lines)


def build_markdown_report(runs: list[BenchmarkRun]) -> str:
    """GFM Markdown report string — this is report.md's exact content."""
    parts = ["# Benchmark Comparison Report\n"]

    if not runs:
        parts.append("*No benchmark runs provided.*\n")
        return "\n".join(parts)

    parts.append("## Run Provenance\n")
    parts.append(_render_provenance_table_md(runs) + "\n")

    parts.append("## Detection Metrics — by Quality Tier\n")
    for run in runs:
        parts.append(f"### {run.pipeline_mode}\n")
        if not run.results:
            parts.append("*0 FOVs in this run.*\n")
            continue
        rows = _stratified_detection_table(run.results, lambda r: r.quality_tier)
        parts.append(_render_detection_table_md(rows) + "\n")

    parts.append("## Detection Metrics — by Lens Type\n")
    for run in runs:
        parts.append(f"### {run.pipeline_mode}\n")
        if not run.results:
            parts.append("*0 FOVs in this run.*\n")
            continue
        rows = _stratified_detection_table(run.results, lambda r: r.lens_type)
        parts.append(_render_detection_table_md(rows) + "\n")

    parts.append("## Detector / Stage Breakdown\n")
    parts.append(_render_detector_stage_md(runs))

    parts.append("## Runtime\n")
    parts.append(_render_runtime_md(runs) + "\n")

    parts.append("## Tracking (Split/Merge Counts)\n")
    parts.append(_render_tracking_md(runs) + "\n")

    warnings = _warnings(runs)
    if warnings:
        parts.append("## Warnings\n")
        parts.append("\n".join(f"- {w}" for w in warnings) + "\n")

    return "\n".join(parts)
