"""Tests for roigbiv.cli_benchmark — validate + report CLI subcommands."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from roigbiv.cli_benchmark import main


def _write_yaml(data: dict, path: Path) -> None:
    """Write a dict as YAML to a file."""
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_json(data: dict, path: Path) -> None:
    """Write a dict as JSON to a file."""
    with open(path, "w") as f:
        json.dump(data, f)


def _valid_entry_dict(base_dir: Path, dataset_id: str = "ds1", fov_id: str = "fov1") -> dict:
    """Return a valid manifest entry dict with real path pointing to a created directory."""
    data_dir = base_dir / "data" / dataset_id
    data_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_id": dataset_id,
        "fov_id": fov_id,
        "path": str(data_dir),
        "fs": 30.0,
        "has_manual_masks": True,
        "has_longitudinal_ids": False,
        "has_synthetic_injections": True,
        "quality_tier": "high",
        "frame_averaging": 1,
        "lens_type": "generic",
        "notes": None,
    }


def _valid_manifest_dict(base_dir: Path) -> dict:
    """Return a valid manifest dict with two entries."""
    return {
        "entries": [
            _valid_entry_dict(base_dir, dataset_id="ds1", fov_id="fov1"),
            _valid_entry_dict(base_dir, dataset_id="ds2", fov_id="fov2"),
        ]
    }


def test_main_valid_manifest_returns_zero(tmp_path: Path, capsys) -> None:
    """Valid manifest file returns exit code 0 with 'OK' in stdout."""
    manifest_dict = _valid_manifest_dict(tmp_path)
    manifest_file = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, manifest_file)

    result = main(["validate", str(manifest_file)])

    assert result == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert len(manifest_dict["entries"]) == 2
    assert "2 entries" in captured.out


def test_main_validation_errors_returns_one(tmp_path: Path, capsys) -> None:
    """Manifest with validation errors returns exit code 1 with errors in stderr."""
    entry = _valid_entry_dict(tmp_path)
    del entry["fs"]  # missing required field
    manifest_dict = {"entries": [entry]}
    manifest_file = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, manifest_file)

    result = main(["validate", str(manifest_file)])

    assert result == 1
    captured = capsys.readouterr()
    assert "fs" in captured.err  # field name in error
    assert "validation failed" in captured.err.lower()


def test_main_nonexistent_file_returns_two(tmp_path: Path, capsys) -> None:
    """Nonexistent manifest file returns exit code 2 with 'not found' in stderr."""
    result = main(["validate", str(tmp_path / "nonexistent.yaml")])

    assert result == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_main_malformed_yaml_returns_two(tmp_path: Path, capsys) -> None:
    """Malformed YAML content returns exit code 2."""
    manifest_file = tmp_path / "malformed.yaml"
    manifest_file.write_text("{ invalid: yaml: content::")

    result = main(["validate", str(manifest_file)])

    assert result == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_main_malformed_json_returns_two(tmp_path: Path, capsys) -> None:
    """Malformed JSON content returns exit code 2."""
    manifest_file = tmp_path / "malformed.json"
    manifest_file.write_text("{invalid json")

    result = main(["validate", str(manifest_file)])

    assert result == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_main_allow_missing_paths_flag_passes(tmp_path: Path, capsys) -> None:
    """With --allow-missing-paths flag, manifest with nonexistent paths validates (returns 0)."""
    entry = _valid_entry_dict(tmp_path)
    entry["path"] = str(tmp_path / "nonexistent" / "path")
    manifest_dict = {"entries": [entry]}
    manifest_file = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, manifest_file)

    result = main(["validate", str(manifest_file), "--allow-missing-paths"])

    assert result == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_main_without_allow_missing_paths_fails(tmp_path: Path, capsys) -> None:
    """Without --allow-missing-paths flag, manifest with nonexistent paths fails (returns 1)."""
    entry = _valid_entry_dict(tmp_path)
    entry["path"] = str(tmp_path / "nonexistent" / "path")
    manifest_dict = {"entries": [entry]}
    manifest_file = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, manifest_file)

    result = main(["validate", str(manifest_file)])

    assert result == 1
    captured = capsys.readouterr()
    assert "path" in captured.err.lower()


def test_main_multiple_validation_errors_reported(tmp_path: Path, capsys) -> None:
    """Multiple validation errors are all reported in stderr."""
    entry1 = _valid_entry_dict(tmp_path, fov_id="same_id")
    entry2 = _valid_entry_dict(tmp_path, dataset_id="ds2", fov_id="same_id")
    del entry2["fs"]  # also missing fs
    manifest_dict = {"entries": [entry1, entry2]}
    manifest_file = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, manifest_file)

    result = main(["validate", str(manifest_file)])

    assert result == 1
    captured = capsys.readouterr()
    # Should report duplicate fov_id errors (2) and missing fs (1)
    assert "fov_id" in captured.err
    assert "fs" in captured.err


def test_main_json_manifest_file(tmp_path: Path, capsys) -> None:
    """JSON manifest files are parsed and validated correctly."""
    manifest_dict = _valid_manifest_dict(tmp_path)
    manifest_file = tmp_path / "manifest.json"
    _write_json(manifest_dict, manifest_file)

    result = main(["validate", str(manifest_file)])

    assert result == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_main_stdout_contains_filename(tmp_path: Path, capsys) -> None:
    """Success message includes the manifest filename."""
    manifest_dict = _valid_manifest_dict(tmp_path)
    manifest_file = tmp_path / "my_benchmark.yaml"
    _write_yaml(manifest_dict, manifest_file)

    result = main(["validate", str(manifest_file)])

    assert result == 0
    captured = capsys.readouterr()
    assert "my_benchmark.yaml" in captured.out


def _write_run(path: Path, pipeline_mode: str, fovs: list[dict]) -> None:
    """Write a minimal benchmark_run.json fixture to `path`."""
    payload = {
        "schema_version": 1,
        "pipeline_mode": pipeline_mode,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config_hash": "sha256:" + "b" * 64,
        "created_at": "2026-07-02T00:00:00Z",
        "manifest_path": "manifest.yaml",
        "results": fovs,
    }
    path.write_text(json.dumps(payload))


def _fov(dataset_id: str, fov_id: str, quality_tier: str = "high", lens_type: str = "generic",
         with_gt: bool = True) -> dict:
    detection = (
        {"precision": 0.9, "recall": 0.8, "f1": 0.85, "mean_iou": 0.7,
         "false_positive_count": 2, "false_negative_count": 3}
        if with_gt else {}
    )
    return {
        "dataset_id": dataset_id,
        "fov_id": fov_id,
        "quality_tier": quality_tier,
        "lens_type": lens_type,
        "detection": detection,
        "tracking": {"split_count": 1, "merge_count": 0},
        "runtime": {"runtime_seconds": 12.5, "peak_memory_mb": 256.0},
        "hitl": {},
        "trace": {},
        "detector_stage_counts": None,
        "notes": None,
    }


def test_report_two_modes_writes_report_md_and_json(tmp_path: Path, capsys) -> None:
    """--run given twice compares two pipeline modes and writes report.md + report.json."""
    run_a = tmp_path / "run_cascade.json"
    run_b = tmp_path / "run_candidate.json"
    _write_run(run_a, "cascade_legacy", [_fov("ds1", "fov1")])
    _write_run(run_b, "candidate_union", [_fov("ds1", "fov1")])
    output_dir = tmp_path / "report_out"

    result = main([
        "report", "--run", str(run_a), "--run", str(run_b), "--output-dir", str(output_dir),
    ])

    assert result == 0
    captured = capsys.readouterr()
    assert "report.md" in captured.out
    assert "report.json" in captured.out

    md = (output_dir / "report.md").read_text()
    assert "cascade_legacy" in md
    assert "candidate_union" in md
    assert "a" * 40 in md  # git commit surfaced

    report_json = json.loads((output_dir / "report.json").read_text())
    assert {p["pipeline_mode"] for p in report_json["provenance"]} == {"cascade_legacy", "candidate_union"}


def test_report_single_mode_still_succeeds(tmp_path: Path, capsys) -> None:
    """A single --run still produces a valid report."""
    run_a = tmp_path / "run.json"
    _write_run(run_a, "cascade_legacy", [_fov("ds1", "fov1")])
    output_dir = tmp_path / "report_out"

    result = main(["report", "--run", str(run_a), "--output-dir", str(output_dir)])

    assert result == 0
    assert (output_dir / "report.md").exists()
    assert (output_dir / "report.json").exists()


def test_report_empty_results_run_succeeds(tmp_path: Path, capsys) -> None:
    """A run with zero FOVs still produces a report, not a crash."""
    run_a = tmp_path / "run.json"
    _write_run(run_a, "cascade_legacy", [])
    output_dir = tmp_path / "report_out"

    result = main(["report", "--run", str(run_a), "--output-dir", str(output_dir)])

    assert result == 0
    md = (output_dir / "report.md").read_text()
    assert "0 FOVs" in md


def test_report_missing_gt_fov_surfaces_warning(tmp_path: Path, capsys) -> None:
    """A FOV with no ground truth (empty detection metrics) is listed under Warnings."""
    run_a = tmp_path / "run.json"
    _write_run(run_a, "cascade_legacy", [_fov("ds1", "fov1", with_gt=False)])
    output_dir = tmp_path / "report_out"

    result = main(["report", "--run", str(run_a), "--output-dir", str(output_dir)])

    assert result == 0
    md = (output_dir / "report.md").read_text()
    assert "## Warnings" in md
    assert "ds1/fov1" in md
    assert "ground truth unavailable" in md


def test_report_missing_run_file_returns_two(tmp_path: Path, capsys) -> None:
    """A --run path that doesn't exist returns exit code 2."""
    output_dir = tmp_path / "report_out"

    result = main([
        "report", "--run", str(tmp_path / "nonexistent.json"), "--output-dir", str(output_dir),
    ])

    assert result == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_report_malformed_json_run_returns_two(tmp_path: Path, capsys) -> None:
    """A --run file with malformed JSON returns exit code 2."""
    run_a = tmp_path / "run.json"
    run_a.write_text("{not valid json")
    output_dir = tmp_path / "report_out"

    result = main(["report", "--run", str(run_a), "--output-dir", str(output_dir)])

    assert result == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_report_non_utf8_run_file_returns_two(tmp_path: Path, capsys) -> None:
    """A --run file with invalid UTF-8 bytes returns exit code 2, not a crash."""
    run_a = tmp_path / "run.json"
    run_a.write_bytes(b"\xff\xfe not valid utf-8")
    output_dir = tmp_path / "report_out"

    result = main(["report", "--run", str(run_a), "--output-dir", str(output_dir)])

    assert result == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_report_output_dir_collides_with_file_returns_two(tmp_path: Path, capsys) -> None:
    """If --output-dir is an existing regular file (not a directory), fail with exit code 2, not a crash."""
    run_a = tmp_path / "run.json"
    _write_run(run_a, "cascade_legacy", [_fov("ds1", "fov1")])
    output_dir = tmp_path / "not_a_dir"
    output_dir.write_text("i am a file, not a directory")

    result = main(["report", "--run", str(run_a), "--output-dir", str(output_dir)])

    assert result == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
