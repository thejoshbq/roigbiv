"""Tests for :mod:`roigbiv.pipeline.manifest`.

Covers:
- build_manifest() returns complete dict with required fields and correct types
- input.tif_hashes[<filename>] is sha256 hash when file exists, None when missing
- Fail-open behavior: git binary missing, input file missing, exceptions in helpers
- write_manifest() creates output_dir, writes JSON, returns path on success / None on failure
- JSON round-trip and seeds field correctness
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from roigbiv.pipeline import manifest
from roigbiv.pipeline.foundation import RNG_SEED
from roigbiv.pipeline.manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    run_manifest_path,
    write_manifest,
)
from roigbiv.pipeline.types import PipelineConfig


# ──────────────────────────── fixtures ────────────────────────────────────


def _write_tif(path: Path, content: bytes = b"raw_tif_payload") -> Path:
    """Write a minimal file at path (doesn't need to be valid TIF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    tif_path = _write_tif(tmp_path / "input.tif")
    return {"output_dir": output_dir, "tif_path": tif_path}


# ─────────────────────── build_manifest tests ──────────────────────────────


def test_build_manifest_returns_dict_with_all_required_keys(workspace):
    """build_manifest() returns a dict with every required top-level key."""
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])

    required_keys = {
        "schema_version",
        "generated_at",
        "roigbiv_version",
        "git",
        "python",
        "platform",
        "cuda",
        "packages",
        "seeds",
        "config",
        "input",
        "output_dir",
    }
    assert set(manifest_dict.keys()) == required_keys


def test_build_manifest_schema_version_is_string(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    assert isinstance(manifest_dict["schema_version"], str)
    assert manifest_dict["schema_version"] == "1.0"


def test_build_manifest_generated_at_is_valid_isoformat(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    generated_at = manifest_dict["generated_at"]
    assert isinstance(generated_at, str)
    # Should be parseable as an ISO format timestamp
    import datetime
    datetime.datetime.fromisoformat(generated_at)


def test_build_manifest_roigbiv_version_is_string(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    assert isinstance(manifest_dict["roigbiv_version"], str)
    assert len(manifest_dict["roigbiv_version"]) > 0


def test_build_manifest_git_field_has_required_keys(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    git = manifest_dict["git"]
    assert isinstance(git, dict)
    assert set(git.keys()) == {"commit", "dirty", "branch"}
    # Each should be str|None or bool|None
    assert git["commit"] is None or isinstance(git["commit"], str)
    assert git["dirty"] is None or isinstance(git["dirty"], bool)
    assert git["branch"] is None or isinstance(git["branch"], str)


def test_build_manifest_python_field_has_required_keys(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    python_info = manifest_dict["python"]
    assert isinstance(python_info, dict)
    assert set(python_info.keys()) == {"version", "implementation", "executable"}
    assert isinstance(python_info["version"], str)
    assert isinstance(python_info["implementation"], str)
    assert isinstance(python_info["executable"], str)


def test_build_manifest_platform_field_has_required_keys(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    platform_info = manifest_dict["platform"]
    assert isinstance(platform_info, dict)
    assert set(platform_info.keys()) == {"system", "release", "machine"}
    assert isinstance(platform_info["system"], str)
    assert isinstance(platform_info["release"], str)
    assert isinstance(platform_info["machine"], str)


def test_build_manifest_cuda_field_has_required_keys(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    cuda_info = manifest_dict["cuda"]
    assert isinstance(cuda_info, dict)
    assert set(cuda_info.keys()) == {
        "available",
        "torch_version",
        "cuda_version",
        "device_count",
        "device_names",
    }
    assert isinstance(cuda_info["available"], bool)
    assert cuda_info["torch_version"] is None or isinstance(cuda_info["torch_version"], str)
    assert cuda_info["cuda_version"] is None or isinstance(cuda_info["cuda_version"], str)
    assert isinstance(cuda_info["device_count"], int)
    assert isinstance(cuda_info["device_names"], list)


def test_build_manifest_packages_field_is_dict(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    packages = manifest_dict["packages"]
    assert isinstance(packages, dict)
    # Should include at least these known package names
    expected_packages = ["torch", "cellpose", "suite2p", "numpy", "roicat"]
    assert set(packages.keys()) == set(expected_packages)
    # Each value is str|None
    for val in packages.values():
        assert val is None or isinstance(val, str)


def test_build_manifest_seeds_is_correct(workspace):
    """seeds field tracks foundation.RNG_SEED (the actual value torch is seeded with)."""
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    seeds = manifest_dict["seeds"]
    assert seeds == {
        "torch_manual_seed": RNG_SEED,
        "torch_cuda_manual_seed_all": RNG_SEED,
    }


def test_build_manifest_config_is_dict(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    config = manifest_dict["config"]
    assert isinstance(config, dict)
    # Should contain at least the fs parameter
    assert "fs" in config or len(config) > 0


def test_build_manifest_input_field_structure(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    input_info = manifest_dict["input"]
    assert isinstance(input_info, dict)
    assert set(input_info.keys()) == {"path", "tif_hashes"}
    assert isinstance(input_info["path"], str)
    assert isinstance(input_info["tif_hashes"], dict)


def test_build_manifest_tif_hashes_contains_filename_key(workspace):
    """input.tif_hashes dict should have the input filename as key."""
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    tif_hashes = manifest_dict["input"]["tif_hashes"]
    assert workspace["tif_path"].name in tif_hashes


def test_build_manifest_tif_hash_is_sha256_hex_when_file_exists(workspace):
    """When input file exists and is readable, hash is "sha256:<hex>" of its actual content."""
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    hash_value = manifest_dict["input"]["tif_hashes"][workspace["tif_path"].name]
    expected = "sha256:" + hashlib.sha256(workspace["tif_path"].read_bytes()).hexdigest()
    assert hash_value == expected


def test_build_manifest_tif_hash_is_none_when_file_missing(workspace):
    """When input file does not exist, hash is None (fail-open)."""
    cfg = PipelineConfig(fs=7.5)
    missing_tif = workspace["output_dir"] / "nonexistent.tif"
    manifest_dict = build_manifest(cfg, missing_tif, workspace["output_dir"])
    # Should not raise, and tif_hashes[filename] should be None
    assert manifest_dict["input"]["tif_hashes"][missing_tif.name] is None


def test_build_manifest_output_dir_is_string(workspace):
    cfg = PipelineConfig(fs=7.5)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    assert isinstance(manifest_dict["output_dir"], str)
    assert manifest_dict["output_dir"] == str(workspace["output_dir"])


def test_build_manifest_never_raises_on_missing_file(workspace):
    """build_manifest() is pure and fail-open: does not raise even if file missing."""
    cfg = PipelineConfig(fs=7.5)
    missing_tif = workspace["output_dir"] / "nonexistent.tif"
    # Should not raise
    result = build_manifest(cfg, missing_tif, workspace["output_dir"])
    assert result is not None
    assert isinstance(result, dict)


def test_build_manifest_git_is_none_dict_when_git_binary_missing(workspace, monkeypatch):
    """When git binary is missing (FileNotFoundError), git field is all-None."""
    cfg = PipelineConfig(fs=7.5)

    def raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError("git binary not found")

    monkeypatch.setattr("subprocess.run", raise_file_not_found)
    manifest_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    # Should not raise, and git should be all None
    git = manifest_dict["git"]
    assert git == {"commit": None, "dirty": None, "branch": None}


# ────────────────────── write_manifest tests ───────────────────────────────


def test_write_manifest_creates_output_dir(tmp_path):
    """write_manifest() creates output_dir if it doesn't exist."""
    output_dir = tmp_path / "deep" / "nested" / "output"
    tif_path = _write_tif(tmp_path / "input.tif")
    cfg = PipelineConfig(fs=7.5)

    result = write_manifest(cfg, tif_path, output_dir)

    assert result is not None
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_write_manifest_writes_file_at_correct_path(workspace):
    """write_manifest() writes the manifest file at run_manifest_path(output_dir)."""
    cfg = PipelineConfig(fs=7.5)
    result = write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    expected_path = run_manifest_path(workspace["output_dir"])
    assert result == expected_path
    assert expected_path.exists()
    assert expected_path.name == MANIFEST_FILENAME


def test_write_manifest_file_is_valid_json(workspace):
    """The written manifest file is valid JSON."""
    cfg = PipelineConfig(fs=7.5)
    write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    manifest_path = run_manifest_path(workspace["output_dir"])
    manifest_dict = json.loads(manifest_path.read_text())
    assert isinstance(manifest_dict, dict)


def test_write_manifest_json_round_trip(workspace):
    """Manifest JSON round-trips to an equivalent dict with same top-level keys."""
    cfg = PipelineConfig(fs=7.5)
    write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    manifest_path = run_manifest_path(workspace["output_dir"])

    # Load from file
    loaded_dict = json.loads(manifest_path.read_text())
    # Verify all required keys are present
    required_keys = {
        "schema_version",
        "generated_at",
        "roigbiv_version",
        "git",
        "python",
        "platform",
        "cuda",
        "packages",
        "seeds",
        "config",
        "input",
        "output_dir",
    }
    assert set(loaded_dict.keys()) == required_keys


def test_write_manifest_returns_none_on_failure(workspace, monkeypatch):
    """write_manifest() returns None when an exception occurs (fail-open)."""
    cfg = PipelineConfig(fs=7.5)

    # Monkeypatch _python_info to raise an exception
    def raise_error():
        raise RuntimeError("Simulated failure in _python_info")

    monkeypatch.setattr(manifest, "_python_info", raise_error)
    result = write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    assert result is None


def test_write_manifest_never_raises(workspace, monkeypatch):
    """write_manifest() never raises, even on exception (fail-open)."""
    cfg = PipelineConfig(fs=7.5)

    def raise_error(*args, **kwargs):
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr(manifest, "_python_info", raise_error)
    # Should not raise
    try:
        write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    except Exception:
        pytest.fail("write_manifest() raised an exception when it should fail-open")


def test_write_manifest_logs_warning_on_failure(workspace, monkeypatch, caplog):
    """write_manifest() logs a warning when an exception occurs."""
    import logging
    cfg = PipelineConfig(fs=7.5)

    def raise_error():
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr(manifest, "_python_info", raise_error)

    # Capture logging output using caplog
    with caplog.at_level(logging.WARNING):
        result = write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])

    assert result is None
    # Verify that a warning was logged
    assert "Failed to write manifest" in caplog.text or any(
        "Failed to write manifest" in record.message
        for record in caplog.records
    )


def test_write_manifest_with_write_text_failure_returns_none(workspace, monkeypatch):
    """write_manifest() returns None if the final Path.write_text() call fails."""
    cfg = PipelineConfig(fs=7.5)

    def fail_write(*args, **kwargs):
        raise IOError("Simulated write failure")

    monkeypatch.setattr(Path, "write_text", fail_write)
    result = write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    assert result is None


def test_write_manifest_with_indent_2(workspace):
    """The written JSON should be formatted with indent=2."""
    cfg = PipelineConfig(fs=7.5)
    write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    manifest_path = run_manifest_path(workspace["output_dir"])
    content = manifest_path.read_text()

    # Check for indentation by looking for newlines and leading spaces
    lines = content.split("\n")
    # Should have multiple lines (not minified)
    assert len(lines) > 1
    # Should have some lines with leading spaces (indentation)
    indented_lines = [l for l in lines if l.startswith("  ")]
    assert len(indented_lines) > 0


# ────────────────── run_manifest_path tests ────────────────────────────────


def test_run_manifest_path_returns_correct_filename(workspace):
    """run_manifest_path() returns path with MANIFEST_FILENAME."""
    result = run_manifest_path(workspace["output_dir"])
    assert result.name == MANIFEST_FILENAME
    assert result.parent == workspace["output_dir"]


def test_run_manifest_path_accepts_string_path(tmp_path):
    """run_manifest_path() accepts both str and Path arguments."""
    result_from_str = run_manifest_path(str(tmp_path))
    result_from_path = run_manifest_path(tmp_path)
    assert result_from_str == result_from_path


# ───────────────────── integration tests ───────────────────────────────────


def test_build_and_write_manifest_full_integration(workspace):
    """Full integration: build_manifest + write_manifest in sequence."""
    cfg = PipelineConfig(fs=7.5)

    # Build first
    built_dict = build_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    assert built_dict["schema_version"] == "1.0"

    # Then write
    written_path = write_manifest(cfg, workspace["tif_path"], workspace["output_dir"])
    assert written_path is not None

    # Verify the written file contains the same structure
    loaded_dict = json.loads(written_path.read_text())
    assert loaded_dict["schema_version"] == built_dict["schema_version"]
    assert loaded_dict["roigbiv_version"] == built_dict["roigbiv_version"]
    assert loaded_dict["seeds"] == built_dict["seeds"]


def test_manifest_idempotent_write_overwrites(workspace):
    """Calling write_manifest twice on the same output_dir overwrites."""
    cfg1 = PipelineConfig(fs=7.5)
    cfg2 = PipelineConfig(fs=30.0)

    path1 = write_manifest(cfg1, workspace["tif_path"], workspace["output_dir"])
    dict1 = json.loads(path1.read_text())

    path2 = write_manifest(cfg2, workspace["tif_path"], workspace["output_dir"])
    dict2 = json.loads(path2.read_text())

    # Paths are the same (same file)
    assert path1 == path2
    # fs parameter changed in the config
    assert dict1["config"]["fs"] != dict2["config"]["fs"]
