"""Tests for roigbiv.cli_benchmark — manifest validation CLI entry point."""
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

    result = main([str(manifest_file)])

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

    result = main([str(manifest_file)])

    assert result == 1
    captured = capsys.readouterr()
    assert "fs" in captured.err  # field name in error
    assert "validation failed" in captured.err.lower()


def test_main_nonexistent_file_returns_two(tmp_path: Path, capsys) -> None:
    """Nonexistent manifest file returns exit code 2 with 'not found' in stderr."""
    result = main([str(tmp_path / "nonexistent.yaml")])

    assert result == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_main_malformed_yaml_returns_two(tmp_path: Path, capsys) -> None:
    """Malformed YAML content returns exit code 2."""
    manifest_file = tmp_path / "malformed.yaml"
    manifest_file.write_text("{ invalid: yaml: content::")

    result = main([str(manifest_file)])

    assert result == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_main_malformed_json_returns_two(tmp_path: Path, capsys) -> None:
    """Malformed JSON content returns exit code 2."""
    manifest_file = tmp_path / "malformed.json"
    manifest_file.write_text("{invalid json")

    result = main([str(manifest_file)])

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

    result = main([str(manifest_file), "--allow-missing-paths"])

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

    result = main([str(manifest_file)])

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

    result = main([str(manifest_file)])

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

    result = main([str(manifest_file)])

    assert result == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_main_stdout_contains_filename(tmp_path: Path, capsys) -> None:
    """Success message includes the manifest filename."""
    manifest_dict = _valid_manifest_dict(tmp_path)
    manifest_file = tmp_path / "my_benchmark.yaml"
    _write_yaml(manifest_dict, manifest_file)

    result = main([str(manifest_file)])

    assert result == 0
    captured = capsys.readouterr()
    assert "my_benchmark.yaml" in captured.out
