"""Tests for roigbiv.benchmark.runner — the roigbiv-bench run orchestrator (issue #28)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from roigbiv.benchmark.runner import (
    FovRunResult,
    _git_commit_hash,
    _hardware_info,
    _resolve_entry_tif,
    _roi_counts,
    _run_one_fov,
    run_benchmark,
)
from roigbiv.benchmark.schema import ManifestEntry, ManifestError


def _entry(fov_id="fov1", dataset_id="ds1", path="fov1.tif", **kw) -> ManifestEntry:
    defaults = dict(
        dataset_id=dataset_id, fov_id=fov_id, path=path, fs=7.5,
        has_manual_masks=False, has_longitudinal_ids=False,
        has_synthetic_injections=False, quality_tier="high",
    )
    defaults.update(kw)
    return ManifestEntry(**defaults)


def _fake_fov(outcomes=("accept", "accept", "flag", "reject")):
    rois = [SimpleNamespace(gate_outcome=o) for o in outcomes]
    return SimpleNamespace(rois=rois)


# ---------------------------------------------------------------------------
# _roi_counts
# ---------------------------------------------------------------------------

def test_roi_counts_tallies_gate_outcomes():
    fov = _fake_fov(("accept", "accept", "flag", "reject"))
    assert _roi_counts(fov) == {"accept": 2, "flag": 1, "reject": 1}


def test_roi_counts_none_fov_returns_empty_dict():
    assert _roi_counts(None) == {}


# ---------------------------------------------------------------------------
# _resolve_entry_tif
# ---------------------------------------------------------------------------

def test_resolve_entry_tif_single_file_mode(tmp_path: Path):
    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    resolved = _resolve_entry_tif(entry, tmp_path)
    assert resolved == tif.resolve()


def test_resolve_entry_tif_directory_zero_tifs(tmp_path: Path):
    (tmp_path / "empty_dir").mkdir()
    entry = _entry(path="empty_dir")
    with pytest.raises(ValueError, match="found 0|no TIF files"):
        _resolve_entry_tif(entry, tmp_path)


def test_resolve_entry_tif_directory_multiple_tifs(tmp_path: Path):
    d = tmp_path / "multi_dir"
    d.mkdir()
    (d / "a.tif").write_bytes(b"")
    (d / "b.tif").write_bytes(b"")
    entry = _entry(path="multi_dir")
    with pytest.raises(ValueError, match="found 2"):
        _resolve_entry_tif(entry, tmp_path)


def test_resolve_entry_tif_directory_single_tif(tmp_path: Path):
    d = tmp_path / "one_dir"
    d.mkdir()
    tif = d / "only.tif"
    tif.write_bytes(b"")
    entry = _entry(path="one_dir")
    resolved = _resolve_entry_tif(entry, tmp_path)
    assert resolved == tif.resolve()


def test_resolve_entry_tif_missing_path_raises(tmp_path: Path):
    entry = _entry(path="does_not_exist.tif")
    with pytest.raises(FileNotFoundError):
        _resolve_entry_tif(entry, tmp_path)


# ---------------------------------------------------------------------------
# _run_one_fov
# ---------------------------------------------------------------------------

def test_run_one_fov_success(tmp_path: Path):
    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    with patch("roigbiv.pipeline.run.run_pipeline", return_value=_fake_fov()):
        result = _run_one_fov(entry, tmp_path, out_dir, log_path)

    assert result.status == "success"
    assert result.error is None
    assert result.roi_counts == {"accept": 2, "flag": 1, "reject": 1}
    assert result.config_used["fs"] == 7.5
    assert result.config_used["frame_averaging"] == 1
    assert result.duration_s is not None
    assert log_path.exists()


def test_run_one_fov_optics_confirmation_required(tmp_path: Path):
    from roigbiv.pipeline.run import OpticsConfirmationRequired

    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    exc = OpticsConfirmationRequired(out_dir, {"candidate_profile": "prism"})
    with patch("roigbiv.pipeline.run.run_pipeline", side_effect=exc):
        result = _run_one_fov(entry, tmp_path, out_dir, log_path)

    assert result.status == "error"
    assert "optics_confirmation_required" in result.error
    assert "prism" in result.error
    assert result.duration_s is not None


def test_run_one_fov_pipeline_aborted(tmp_path: Path):
    from roigbiv.pipeline.run import PipelineAborted

    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    with patch("roigbiv.pipeline.run.run_pipeline", side_effect=PipelineAborted()):
        result = _run_one_fov(entry, tmp_path, out_dir, log_path)

    assert result.status == "error"
    assert result.error == "aborted"


def test_run_one_fov_generic_exception_logged_to_file(tmp_path: Path):
    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    with patch("roigbiv.pipeline.run.run_pipeline", side_effect=RuntimeError("boom")):
        result = _run_one_fov(entry, tmp_path, out_dir, log_path)

    assert result.status == "error"
    assert result.error == "RuntimeError: boom"
    log_text = log_path.read_text()
    assert "RuntimeError" in log_text
    assert "Traceback" in log_text


def test_run_one_fov_discovery_failure_has_no_log_file(tmp_path: Path):
    entry = _entry(path="missing.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    result = _run_one_fov(entry, tmp_path, out_dir, log_path)

    assert result.status == "error"
    assert result.error.startswith("discovery:")
    assert result.log_path is None
    assert not log_path.exists()


def test_run_one_fov_log_open_failure_leaves_log_path_none(tmp_path: Path):
    """If open(log_path, "w") itself raises, result.log_path must stay None
    — it must never reference a log file that was never created."""
    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    with patch("builtins.open", side_effect=PermissionError("denied")):
        result = _run_one_fov(entry, tmp_path, out_dir, log_path)

    assert result.status == "error"
    assert result.log_path is None
    assert result.error.startswith("setup:")


def test_run_one_fov_never_raises_on_setup_failure(tmp_path: Path):
    """A failure before run_pipeline is ever called (e.g. mkdir permissions,
    bad profile config) must be recorded on the result, not propagate —
    the benchmark run must continue to the next entry."""
    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    with patch("roigbiv.pipeline.profiles.merged_overrides",
               side_effect=RuntimeError("profile explosion")):
        result = _run_one_fov(entry, tmp_path, out_dir, log_path)

    assert result.status == "error"
    assert result.error.startswith("setup:")
    assert "profile explosion" in result.error


# ---------------------------------------------------------------------------
# run_benchmark (end-to-end, pipeline mocked)
# ---------------------------------------------------------------------------

def _write_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump({"entries": entries}, f)
    return manifest_path


def test_run_benchmark_end_to_end_mixed_results(tmp_path: Path):
    ok_dir = tmp_path / "ok_fov"
    ok_dir.mkdir()
    (ok_dir / "data.tif").write_bytes(b"")
    (tmp_path / "bad_fov").mkdir()  # exists but has no TIF inside -> discovery error

    entries = [
        {
            "dataset_id": "ds1", "fov_id": "ok_fov", "path": "ok_fov", "fs": 7.5,
            "has_manual_masks": False, "has_longitudinal_ids": False,
            "has_synthetic_injections": False, "quality_tier": "high",
        },
        {
            "dataset_id": "ds1", "fov_id": "bad_fov", "path": "bad_fov", "fs": 7.5,
            "has_manual_masks": False, "has_longitudinal_ids": False,
            "has_synthetic_injections": False, "quality_tier": "high",
        },
    ]
    manifest_path = _write_manifest(tmp_path, entries)
    output_dir = tmp_path / "bench_out"

    with patch("roigbiv.pipeline.run.run_pipeline", return_value=_fake_fov()):
        report = run_benchmark(manifest_path, output_dir)

    assert len(report.fov_results) == 2
    by_id = {r.fov_id: r for r in report.fov_results}
    assert by_id["ok_fov"].status == "success"
    assert by_id["bad_fov"].status == "error"

    assert (output_dir / "ok_fov").exists()
    assert (output_dir / "logs" / "ok_fov.log").exists()
    assert not (output_dir / "logs" / "bad_fov.log").exists()

    assert isinstance(report.git_commit, (str, type(None)))
    assert isinstance(report.hardware, dict)


def test_run_benchmark_manifest_validation_failure_raises(tmp_path: Path):
    # Missing several required fields (fs, quality_tier, ...).
    entries = [{"dataset_id": "ds1", "fov_id": "fov1", "path": "fov1.tif"}]
    manifest_path = _write_manifest(tmp_path, entries)
    output_dir = tmp_path / "bench_out"

    with pytest.raises(ManifestError):
        run_benchmark(manifest_path, output_dir)

    assert not (output_dir / "benchmark_run.json").exists()


def test_run_benchmark_manifest_not_found_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_benchmark(tmp_path / "nope.yaml", tmp_path / "bench_out")


# ---------------------------------------------------------------------------
# Best-effort env probes
# ---------------------------------------------------------------------------

def test_git_commit_hash_best_effort_no_raise_on_subprocess_error(tmp_path: Path):
    with patch("subprocess.run", side_effect=OSError("git not found")):
        assert _git_commit_hash(tmp_path) is None


def test_git_commit_hash_non_repo_dir_returns_none(tmp_path: Path):
    # tmp_path is not a git repo -> `git -C tmp_path rev-parse HEAD` fails cleanly.
    assert _git_commit_hash(tmp_path) is None


def test_hardware_info_best_effort_no_raise(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("platform.platform", _boom)
    info = _hardware_info()
    assert info["platform"] is None
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Ablation support (issue #33)
# ---------------------------------------------------------------------------

def test_run_one_fov_applies_ablation_override(tmp_path: Path):
    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    captured_cfg = {}

    def _capture(tif_path, cfg):
        captured_cfg["cfg"] = cfg
        return _fake_fov()

    with patch("roigbiv.pipeline.run.run_pipeline", side_effect=_capture):
        result = _run_one_fov(entry, tmp_path, out_dir, log_path, "stage3_off")

    assert result.status == "success"
    assert result.ablation == "stage3_off"
    assert captured_cfg["cfg"].enable_stage_3 is False


def test_run_one_fov_unknown_ablation_recorded_as_setup_error(tmp_path: Path):
    tif = tmp_path / "fov1.tif"
    tif.write_bytes(b"")
    entry = _entry(path="fov1.tif")
    out_dir = tmp_path / "out" / entry.fov_id
    log_path = tmp_path / "out" / "logs" / f"{entry.fov_id}.log"

    result = _run_one_fov(entry, tmp_path, out_dir, log_path, "not-a-real-ablation")

    assert result.status == "error"
    assert result.error.startswith("setup:")


def test_run_benchmark_groups_output_by_ablation_name(tmp_path: Path):
    ok_dir = tmp_path / "ok_fov"
    ok_dir.mkdir()
    (ok_dir / "data.tif").write_bytes(b"")
    entries = [{
        "dataset_id": "ds1", "fov_id": "ok_fov", "path": "ok_fov", "fs": 7.5,
        "has_manual_masks": False, "has_longitudinal_ids": False,
        "has_synthetic_injections": False, "quality_tier": "high",
    }]
    manifest_path = _write_manifest(tmp_path, entries)
    output_dir = tmp_path / "bench_out"

    with patch("roigbiv.pipeline.run.run_pipeline", return_value=_fake_fov()):
        report = run_benchmark(manifest_path, output_dir,
                                ablations=["raw_only", "stage3_off"])

    assert report.ablations == ["raw_only", "stage3_off"]
    assert len(report.fov_results) == 2
    assert {r.ablation for r in report.fov_results} == {"raw_only", "stage3_off"}
    assert (output_dir / "raw_only" / "ok_fov").exists()
    assert (output_dir / "raw_only" / "logs" / "ok_fov.log").exists()
    assert (output_dir / "stage3_off" / "ok_fov").exists()
    assert (output_dir / "stage3_off" / "logs" / "ok_fov.log").exists()
    # Legacy flat layout must not appear when ablations are requested.
    assert not (output_dir / "ok_fov").exists()


def test_run_benchmark_all_sentinel_expands_to_every_registered_ablation(tmp_path: Path):
    from roigbiv.benchmark.ablations import ABLATIONS

    ok_dir = tmp_path / "ok_fov"
    ok_dir.mkdir()
    (ok_dir / "data.tif").write_bytes(b"")
    entries = [{
        "dataset_id": "ds1", "fov_id": "ok_fov", "path": "ok_fov", "fs": 7.5,
        "has_manual_masks": False, "has_longitudinal_ids": False,
        "has_synthetic_injections": False, "quality_tier": "high",
    }]
    manifest_path = _write_manifest(tmp_path, entries)
    output_dir = tmp_path / "bench_out"

    with patch("roigbiv.pipeline.run.run_pipeline", return_value=_fake_fov()):
        report = run_benchmark(manifest_path, output_dir, ablations=["all"])

    assert set(report.ablations) == set(ABLATIONS)
    assert len(report.fov_results) == len(ABLATIONS)


def test_run_benchmark_unknown_ablation_raises_before_any_fov_runs(tmp_path: Path):
    ok_dir = tmp_path / "ok_fov"
    ok_dir.mkdir()
    (ok_dir / "data.tif").write_bytes(b"")
    entries = [{
        "dataset_id": "ds1", "fov_id": "ok_fov", "path": "ok_fov", "fs": 7.5,
        "has_manual_masks": False, "has_longitudinal_ids": False,
        "has_synthetic_injections": False, "quality_tier": "high",
    }]
    manifest_path = _write_manifest(tmp_path, entries)
    output_dir = tmp_path / "bench_out"

    with pytest.raises(ValueError):
        run_benchmark(manifest_path, output_dir, ablations=["nope"])

    assert not (output_dir / "nope").exists()
    assert not (output_dir / "benchmark_run.json").exists()


def test_run_benchmark_legacy_layout_unchanged_when_ablations_omitted(tmp_path: Path):
    ok_dir = tmp_path / "ok_fov"
    ok_dir.mkdir()
    (ok_dir / "data.tif").write_bytes(b"")
    entries = [{
        "dataset_id": "ds1", "fov_id": "ok_fov", "path": "ok_fov", "fs": 7.5,
        "has_manual_masks": False, "has_longitudinal_ids": False,
        "has_synthetic_injections": False, "quality_tier": "high",
    }]
    manifest_path = _write_manifest(tmp_path, entries)
    output_dir = tmp_path / "bench_out"

    with patch("roigbiv.pipeline.run.run_pipeline", return_value=_fake_fov()):
        report = run_benchmark(manifest_path, output_dir)

    assert report.ablations == []
    assert report.fov_results[0].ablation is None
    assert (output_dir / "ok_fov").exists()
    assert (output_dir / "logs" / "ok_fov.log").exists()
