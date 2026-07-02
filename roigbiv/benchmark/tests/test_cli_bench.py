"""Tests for roigbiv.cli_bench — the `roigbiv-bench run` CLI entry point (issue #28)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from roigbiv.cli_bench import main


def _write_yaml(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


def _valid_entry_dict(base_dir: Path, fov_id: str = "fov1") -> dict:
    data_dir = base_dir / fov_id
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "data.tif").write_bytes(b"")
    return {
        "dataset_id": "ds1",
        "fov_id": fov_id,
        "path": fov_id,
        "fs": 7.5,
        "has_manual_masks": False,
        "has_longitudinal_ids": False,
        "has_synthetic_injections": False,
        "quality_tier": "high",
    }


def _fake_fov():
    return SimpleNamespace(rois=[SimpleNamespace(gate_outcome="accept")])


def test_run_all_success_returns_zero(tmp_path: Path, capsys) -> None:
    manifest_dict = {"entries": [_valid_entry_dict(tmp_path, "fov1")]}
    manifest_path = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, manifest_path)
    output_dir = tmp_path / "bench_out"

    with patch("roigbiv.pipeline.run.run_pipeline", return_value=_fake_fov()):
        result = main(["run", "--manifest", str(manifest_path), "--output-dir", str(output_dir)])

    assert result == 0
    report_path = output_dir / "benchmark_run.json"
    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
    for key in ("manifest_path", "output_dir", "git_commit", "hardware",
                "total_runtime_s", "fov_results", "roigbiv_version"):
        assert key in report
    assert len(report["fov_results"]) == 1
    assert report["fov_results"][0]["status"] == "success"


def test_run_partial_failure_returns_one(tmp_path: Path) -> None:
    manifest_dict = {"entries": [
        _valid_entry_dict(tmp_path, "fov_ok"),
        _valid_entry_dict(tmp_path, "fov_bad"),
    ]}
    manifest_path = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, manifest_path)
    output_dir = tmp_path / "bench_out"

    def _side_effect(tif_path, cfg):
        if "fov_bad" in str(tif_path):
            raise RuntimeError("boom")
        return _fake_fov()

    with patch("roigbiv.pipeline.run.run_pipeline", side_effect=_side_effect):
        result = main(["run", "--manifest", str(manifest_path), "--output-dir", str(output_dir)])

    assert result == 1
    report_path = output_dir / "benchmark_run.json"
    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
    statuses = {r["fov_id"]: r["status"] for r in report["fov_results"]}
    assert statuses["fov_ok"] == "success"
    assert statuses["fov_bad"] == "error"


def test_run_manifest_not_found_returns_two(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "bench_out"
    result = main(["run", "--manifest", str(tmp_path / "nope.yaml"),
                   "--output-dir", str(output_dir)])

    assert result == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err
    assert not (output_dir / "benchmark_run.json").exists()


def test_run_manifest_invalid_returns_two(tmp_path: Path) -> None:
    # Missing required fields -> validate_manifest errors -> ManifestError.
    manifest_path = tmp_path / "manifest.yaml"
    _write_yaml({"entries": [{"dataset_id": "ds1", "fov_id": "fov1", "path": "fov1.tif"}]},
                manifest_path)
    output_dir = tmp_path / "bench_out"

    result = main(["run", "--manifest", str(manifest_path), "--output-dir", str(output_dir)])

    assert result == 2
    assert not (output_dir / "benchmark_run.json").exists()


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
