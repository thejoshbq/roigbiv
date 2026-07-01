"""
Tests for pipeline_mode field and CLI flag (issue #26 / ADR-0001).

Tests cover:
  - Default value of pipeline_mode in PipelineConfig
  - Inclusion of pipeline_mode in summary_for_log
  - Construction of PipelineConfig with all valid pipeline modes (parametrized)
  - CLI parsing for --pipeline-mode flag, including default, valid choices, and invalid choices
  - Resume fingerprint exclusion of pipeline_mode (so pre-existing workspaces without
    the field in their manifest still match new configs with it set)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline import run as run_mod
from roigbiv.pipeline.resume import compute_cfg_fingerprint
from roigbiv.pipeline.types import (
    DEFAULT_PIPELINE_MODE,
    PIPELINE_MODES,
    PipelineConfig,
)


def test_pipeline_mode_default():
    cfg = PipelineConfig()
    assert cfg.pipeline_mode == DEFAULT_PIPELINE_MODE
    assert cfg.pipeline_mode == "cascade_legacy"


def test_pipeline_mode_in_summary_for_log():
    cfg = PipelineConfig(pipeline_mode="candidate_union")
    log_summary = cfg.summary_for_log()
    assert "pipeline_mode" in log_summary
    assert log_summary["pipeline_mode"] == "candidate_union"


def test_pipeline_mode_default_in_summary_for_log():
    cfg = PipelineConfig()  # Uses default
    log_summary = cfg.summary_for_log()
    assert log_summary["pipeline_mode"] == "cascade_legacy"


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_pipeline_config_constructs_all_modes(mode):
    cfg = PipelineConfig(pipeline_mode=mode)
    assert cfg.pipeline_mode == mode


def test_cli_pipeline_mode_default():
    # Omitting --pipeline-mode must not trip argparse. A nonexistent --input
    # short-circuits main() at input-path resolution (run.py) before any real
    # pipeline work, matching the convention in test_run_modes.py — this test
    # only exercises CLI parsing, not the pipeline itself.
    try:
        rc = run_mod.main(["--input", "/nonexistent", "--fs", "7.5"])
    except SystemExit as e:
        pytest.fail(f"argparse should accept omitted --pipeline-mode; got SystemExit({e.code})")
    assert rc == 2, "expected short-circuit at nonexistent --input, not an argparse rejection"


def test_cli_pipeline_mode_valid_choice():
    try:
        rc = run_mod.main([
            "--input", "/nonexistent", "--fs", "7.5",
            "--pipeline-mode", "benchmark_only",
        ])
    except SystemExit as e:
        pytest.fail(f"argparse should accept --pipeline-mode benchmark_only; got SystemExit({e.code})")
    assert rc == 2, "expected short-circuit at nonexistent --input, not an argparse rejection"


def test_cli_pipeline_mode_invalid_choice():
    with pytest.raises(SystemExit) as exc_info:
        run_mod.main([
            "--input", "/nonexistent", "--fs", "7.5",
            "--pipeline-mode", "invalid_mode",
        ])
    assert exc_info.value.code == 2, (
        f"argparse should reject invalid --pipeline-mode with exit code 2; "
        f"got {exc_info.value.code}")


def test_resume_fingerprint_ignores_pipeline_mode():
    with tempfile.TemporaryDirectory() as td:
        tif = Path(td) / "test.tif"
        tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))

        # Two configs differing only in pipeline_mode should have the same fingerprint
        cfg_cascade = PipelineConfig(fs=7.5, pipeline_mode="cascade_legacy")
        cfg_union = PipelineConfig(fs=7.5, pipeline_mode="candidate_union")

        fp_cascade = compute_cfg_fingerprint(cfg_cascade, tif)
        fp_union = compute_cfg_fingerprint(cfg_union, tif)

        assert fp_cascade == fp_union, (
            "pipeline_mode must be excluded from the resume fingerprint so a "
            "workspace can be resumed with a different mode without invalidation")


def test_old_manifest_matches_new_cfg_fingerprint():
    with tempfile.TemporaryDirectory() as td:
        tif = Path(td) / "test.tif"
        tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))

        # Simulate an old config that existed before pipeline_mode was added.
        # It will have pipeline_mode set to the default by the constructor.
        old_cfg = PipelineConfig(fs=7.5)

        # A new config with explicit pipeline_mode should have the same fingerprint.
        new_cfg = PipelineConfig(fs=7.5, pipeline_mode=DEFAULT_PIPELINE_MODE)

        fp_old = compute_cfg_fingerprint(old_cfg, tif)
        fp_new = compute_cfg_fingerprint(new_cfg, tif)

        assert fp_old == fp_new, (
            "A pre-existing workspace (old cfg without pipeline_mode explicitly "
            "set) must match a new cfg with pipeline_mode set to the default. "
            "This allows --resume to work across the feature introduction.")


if __name__ == "__main__":
    import traceback

    tests = [
        test_pipeline_mode_default,
        test_pipeline_mode_in_summary_for_log,
        test_pipeline_mode_default_in_summary_for_log,
        # test_pipeline_config_constructs_all_modes is pytest-parametrized
        # (takes a `mode` arg) — pytest handles it; skip in this manual runner.
        test_cli_pipeline_mode_default,
        test_cli_pipeline_mode_valid_choice,
        test_cli_pipeline_mode_invalid_choice,
        test_resume_fingerprint_ignores_pipeline_mode,
        test_old_manifest_matches_new_cfg_fingerprint,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            traceback.print_exc()
            failed.append(test.__name__)
    print()
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)
    print(f"Manual run completed ({len(tests)} tests attempted).")
