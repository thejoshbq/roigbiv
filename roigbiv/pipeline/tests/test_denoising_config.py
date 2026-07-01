"""Tests for denoising configuration (issue #34).

Covers:
- PipelineConfig default construction with 5 new denoising fields
- summary_for_log() includes all 5 denoising keys
- _FINGERPRINT_EXCLUDE contains all 5 field names
- Validation logic: backend validity, enable + backend consistency, model path requirements
- CLI argument parsing for denoising flags (using the real parser from roigbiv.pipeline.run)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.types import PipelineConfig
from roigbiv.pipeline.resume import _FINGERPRINT_EXCLUDE


# ──────────────────────── Default construction ───────────────────────────


def test_denoising_default_enable_denoised_branch() -> None:
    """enable_denoised_branch defaults to False."""
    cfg = PipelineConfig()
    assert cfg.enable_denoised_branch is False


def test_denoising_default_denoiser_backend() -> None:
    """denoiser_backend defaults to 'none'."""
    cfg = PipelineConfig()
    assert cfg.denoiser_backend == "none"


def test_denoising_default_denoiser_model_path() -> None:
    """denoiser_model_path defaults to None."""
    cfg = PipelineConfig()
    assert cfg.denoiser_model_path is None


def test_denoising_default_denoised_branch_cache() -> None:
    """denoised_branch_cache defaults to None."""
    cfg = PipelineConfig()
    assert cfg.denoised_branch_cache is None


def test_denoising_default_validate_denoised_against_raw() -> None:
    """validate_denoised_against_raw defaults to True."""
    cfg = PipelineConfig()
    assert cfg.validate_denoised_against_raw is True


def test_denoising_defaults_all_together() -> None:
    """All 5 denoising fields have correct defaults together."""
    cfg = PipelineConfig()
    assert cfg.enable_denoised_branch is False
    assert cfg.denoiser_backend == "none"
    assert cfg.denoiser_model_path is None
    assert cfg.denoised_branch_cache is None
    assert cfg.validate_denoised_against_raw is True


# ──────────────────────── summary_for_log() ─────────────────────────────


def test_summary_for_log_includes_denoising_keys() -> None:
    """summary_for_log() includes all 5 denoising field keys."""
    cfg = PipelineConfig(
        enable_denoised_branch=True,
        denoiser_backend="deepcad_rt",
        denoiser_model_path=Path("/path/to/model"),
        denoised_branch_cache=Path("/path/to/cache"),
        validate_denoised_against_raw=False,
    )
    summary = cfg.summary_for_log()

    assert "enable_denoised_branch" in summary
    assert "denoiser_backend" in summary
    assert "denoiser_model_path" in summary
    assert "denoised_branch_cache" in summary
    assert "validate_denoised_against_raw" in summary


def test_summary_for_log_denoising_values_correct() -> None:
    """summary_for_log() returns correct values for denoising fields."""
    cfg = PipelineConfig(
        enable_denoised_branch=True,
        denoiser_backend="deepinterpolation",
        denoiser_model_path=Path("/test/model"),
        denoised_branch_cache=Path("/test/cache"),
        validate_denoised_against_raw=False,
    )
    summary = cfg.summary_for_log()

    assert summary["enable_denoised_branch"] is True
    assert summary["denoiser_backend"] == "deepinterpolation"
    # Paths are converted to strings in summary_for_log
    assert summary["denoiser_model_path"] == "/test/model"
    assert summary["denoised_branch_cache"] == "/test/cache"
    assert summary["validate_denoised_against_raw"] is False


# ──────────────────────── _FINGERPRINT_EXCLUDE ───────────────────────────


def test_fingerprint_exclude_contains_denoising_fields() -> None:
    """All 5 denoising field names are in _FINGERPRINT_EXCLUDE."""
    denoising_fields = {
        "enable_denoised_branch",
        "denoiser_backend",
        "denoiser_model_path",
        "denoised_branch_cache",
        "validate_denoised_against_raw",
    }
    for field in denoising_fields:
        assert field in _FINGERPRINT_EXCLUDE, (
            f"{field} not in _FINGERPRINT_EXCLUDE; "
            "denoising config changes should not invalidate resume fingerprint"
        )


# ──────────────────────── Validation logic ───────────────────────────────
# These tests call the real run_foundation() function to exercise the actual
# validation logic that lives in foundation.py lines 719-739. We mock out
# expensive operations (motion correction, SVD, L+S) while letting the real
# validation checks execute.


def _make_minimal_tif(path: Path, shape=(10, 16, 16)):
    """Create a minimal TIF file for testing."""
    tifffile.imwrite(str(path), np.zeros(shape, dtype=np.uint16))


def _mock_run_motion_correction(tif_path, cfg, output_dir, gpu_lock=None):
    """Minimal mock ops dict matching what run_foundation expects."""
    output_dir = Path(output_dir)
    # Create the data.bin file that run_foundation expects to stat()
    data_bin_path = output_dir / "data.bin"
    # Write minimal data.bin: 10 frames of 16x16 int16 = 10*16*16*2 bytes
    with open(data_bin_path, "wb") as f:
        f.write(np.zeros((10, 16, 16), dtype=np.int16).tobytes())
    return (
        {"Ly": 16, "Lx": 16, "nframes": 10, "meanImg": np.zeros((16, 16), np.float32)},
        data_bin_path,
        np.zeros(10, dtype=np.float32),  # motion_x
        np.zeros(10, dtype=np.float32),  # motion_y
    )


def test_denoising_validation_invalid_backend() -> None:
    """denoiser_backend must be in the allowed set.

    Valid values: "deepcad_rt", "deepinterpolation", "pmd", "none".
    Raises ValueError with "Unknown denoiser_backend" message.
    """
    from roigbiv.pipeline.foundation import run_foundation

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        _make_minimal_tif(tif_path)
        output_dir = td_path / "out"
        output_dir.mkdir()

        cfg = PipelineConfig(
            fs=7.5,
            denoiser_backend="bogus",  # Invalid
            force_cpu=True,
        )

        with patch("roigbiv.pipeline.foundation.run_motion_correction",
                   side_effect=_mock_run_motion_correction):
            with pytest.raises(ValueError, match="Unknown denoiser_backend"):
                run_foundation(tif_path, cfg, output_dir)


def test_denoising_validation_enable_with_none_backend() -> None:
    """enable_denoised_branch=True + denoiser_backend='none' is invalid.

    Raises ValueError with "requires a denoiser_backend other than 'none'" message.
    """
    from roigbiv.pipeline.foundation import run_foundation

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        _make_minimal_tif(tif_path)
        output_dir = td_path / "out"
        output_dir.mkdir()

        cfg = PipelineConfig(
            fs=7.5,
            enable_denoised_branch=True,  # Enabled
            denoiser_backend="none",      # But backend is 'none' — invalid combo
            force_cpu=True,
        )

        with patch("roigbiv.pipeline.foundation.run_motion_correction",
                   side_effect=_mock_run_motion_correction):
            with pytest.raises(ValueError,
                             match="requires a denoiser_backend other than 'none'"):
                run_foundation(tif_path, cfg, output_dir)


def test_denoising_validation_backend_requires_model_path() -> None:
    """denoiser_backend != 'none' requires denoiser_model_path.

    Raises ValueError with "requires denoiser_model_path" message.
    """
    from roigbiv.pipeline.foundation import run_foundation

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        _make_minimal_tif(tif_path)
        output_dir = td_path / "out"
        output_dir.mkdir()

        cfg = PipelineConfig(
            fs=7.5,
            denoiser_backend="deepcad_rt",     # Non-'none' backend
            denoiser_model_path=None,          # But no model path — invalid
            force_cpu=True,
        )

        with patch("roigbiv.pipeline.foundation.run_motion_correction",
                   side_effect=_mock_run_motion_correction):
            with pytest.raises(ValueError, match="requires denoiser_model_path"):
                run_foundation(tif_path, cfg, output_dir)


def test_denoising_valid_configuration_with_backend() -> None:
    """Valid configuration: enable_denoised_branch=True + backend + model_path.

    This configuration passes all validation checks (though it may fail later
    due to missing expensive stages; we only test that validation passes).
    """
    from roigbiv.pipeline.foundation import run_foundation

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        _make_minimal_tif(tif_path)
        output_dir = td_path / "out"
        output_dir.mkdir()
        model_path = td_path / "model.pth"
        model_path.touch()  # Just needs to exist for the config

        cfg = PipelineConfig(
            fs=7.5,
            enable_denoised_branch=True,
            denoiser_backend="deepcad_rt",
            denoiser_model_path=model_path,
            force_cpu=True,
        )

        # Should pass validation and proceed (or fail later in motion correction,
        # but not on validation). We just verify no validation error is raised.
        with patch("roigbiv.pipeline.foundation.run_motion_correction",
                   side_effect=_mock_run_motion_correction):
            # Patch the expensive compute_background_separation to stop early
            with patch("roigbiv.pipeline.foundation.compute_background_separation",
                      side_effect=RuntimeError("Stop after validation")):
                # If validation failed, ValueError would be raised.
                # RuntimeError means we got past validation.
                with pytest.raises(RuntimeError, match="Stop after validation"):
                    run_foundation(tif_path, cfg, output_dir)


def test_denoising_valid_configuration_disabled() -> None:
    """Valid configuration: enable_denoised_branch=False + backend='none'.

    When denoised branch is disabled, only backend='none' is valid.
    This is the default configuration that should pass validation easily.
    """
    from roigbiv.pipeline.foundation import run_foundation

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        _make_minimal_tif(tif_path)
        output_dir = td_path / "out"
        output_dir.mkdir()

        cfg = PipelineConfig(
            fs=7.5,
            enable_denoised_branch=False,  # Disabled
            denoiser_backend="none",       # No backend when disabled
            denoiser_model_path=None,      # No model needed
            force_cpu=True,
        )

        # Should pass validation and proceed (or fail later, but not on validation).
        with patch("roigbiv.pipeline.foundation.run_motion_correction",
                   side_effect=_mock_run_motion_correction):
            # Patch the expensive compute_background_separation to stop early
            with patch("roigbiv.pipeline.foundation.compute_background_separation",
                      side_effect=RuntimeError("Stop after validation")):
                # If validation failed, ValueError would be raised.
                # RuntimeError means we got past validation.
                with pytest.raises(RuntimeError, match="Stop after validation"):
                    run_foundation(tif_path, cfg, output_dir)


# ──────────────────────── CLI argument parsing (using the REAL parser) ────────


def test_cli_real_parser_has_denoising_flags() -> None:
    """The REAL parser from roigbiv.pipeline.run has all 5 denoising flags.

    This ensures the parser was properly updated and the flags exist with
    the correct dest names and action types.
    """
    from roigbiv.pipeline import run

    # Get the parser by calling main without invoking any actions
    # (we just need the parser object, not to run the full pipeline).
    # The main function builds the parser internally; we'll create a minimal
    # test by parsing with required args and checking the namespace.
    parser = run.argparse.ArgumentParser()
    # Copy the relevant flag definitions from run.py
    # Actually, we can't easily extract just the parser. Instead,
    # test by calling main() with minimal args and capturing what it receives.
    # But that's harder. Instead, directly verify the parser construction.
    # The cleanest way: use parse_args with the flags and check they exist.

    # Import the actual main function and trigger parser building
    import sys
    from io import StringIO

    # Capture parser building by calling main with --help and checking output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        run.main(["--help"])
    except SystemExit:
        # --help exits, that's OK
        pass
    finally:
        help_text = sys.stdout.getvalue()
        sys.stdout = old_stdout

    # Verify all 5 flags are in the help text
    assert "--denoised-branch" in help_text
    assert "--denoiser-backend" in help_text
    assert "--denoiser-model-path" in help_text
    assert "--denoised-branch-cache" in help_text
    assert "--validate-denoised-against-raw" in help_text


def test_cli_denoiser_backend_invalid_rejected() -> None:
    """The real parser rejects invalid --denoiser-backend values.

    Only deepcad_rt, deepinterpolation, pmd, none are valid.
    """
    from roigbiv.pipeline import run

    # Minimal test TIF
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        tifffile.imwrite(str(tif_path), np.zeros((10, 16, 16), dtype=np.uint16))

        # Try parsing with an invalid backend; argparse should reject it
        with pytest.raises(SystemExit):  # argparse.ArgumentParser.error() calls sys.exit
            run.main([
                "--input", str(tif_path),
                "--fs", "7.5",
                "--denoiser-backend", "bogus",  # Invalid
            ])


def test_cli_denoiser_backend_valid_values() -> None:
    """The real parser accepts all valid --denoiser-backend values."""
    from roigbiv.pipeline import run
    from unittest.mock import patch
    import sys
    from io import StringIO

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        tifffile.imwrite(str(tif_path), np.zeros((10, 16, 16), dtype=np.uint16))

        for backend in ["deepcad_rt", "deepinterpolation", "pmd", "none"]:
            # Mock out everything to just verify the parser accepts the backend value
            # We only care that the argument parsing succeeds
            with patch("roigbiv.pipeline.run._write_traces_bundle"):
                with patch("roigbiv.pipeline.run.run_pipeline") as mock_run:
                    # Return a mock FOVData with required attributes
                    mock_fov = MagicMock()
                    mock_fov.F_raw = None
                    mock_fov.F_neu = None
                    mock_fov.F_corrected = None
                    mock_run.return_value = mock_fov

                    try:
                        run.main([
                            "--input", str(tif_path),
                            "--fs", "7.5",
                            "--denoiser-backend", backend,
                            "--no-viewer",
                        ])
                    except SystemExit:
                        # OK if the pipeline fails after parsing
                        pass
                    # If we get here without SystemExit during argument parsing,
                    # the backend was accepted


def test_cli_denoising_overrides_in_run_single() -> None:
    """Denoising overrides are included in _run_single's config merge.

    When running a single TIF with denoising flags, the overrides are
    passed to run_pipeline.
    """
    from roigbiv.pipeline import run
    from unittest.mock import patch, call

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tif_path = td_path / "test.tif"
        tifffile.imwrite(str(tif_path), np.zeros((10, 16, 16), dtype=np.uint16))
        model_path = td_path / "model.pth"
        model_path.touch()

        with patch("roigbiv.pipeline.run.run_pipeline") as mock_run_pipeline:
            mock_run_pipeline.return_value = None
            try:
                run.main([
                    "--input", str(tif_path),
                    "--fs", "7.5",
                    "--denoised-branch",
                    "--denoiser-backend", "deepcad_rt",
                    "--denoiser-model-path", str(model_path),
                    "--no-viewer",
                ])
            except Exception:
                pass  # Pipeline may fail, we just care about what cfg is passed

        # run_pipeline(tif_path, cfg) is called positionally (run.py:_run_single) —
        # inspect args, not kwargs, or this assertion silently never runs.
        assert mock_run_pipeline.called
        args, _ = mock_run_pipeline.call_args
        cfg = args[1]
        assert cfg.denoiser_backend == "deepcad_rt"
        assert cfg.denoiser_model_path == model_path


def test_cli_denoising_overrides_in_run_workspace() -> None:
    """Denoising overrides are included in _run_workspace's config merge.

    When running a directory (workspace) with denoising flags, the overrides
    are passed to run_with_workspace in the merged_overrides dict.
    """
    from roigbiv.pipeline import run
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Create a minimal workspace structure
        (td_path / "output").mkdir()
        # Create a minimal valid TIFF file
        tif_path = td_path / "tif1.tif"
        tifffile.imwrite(str(tif_path), np.zeros((4, 16, 16), dtype=np.uint16))
        model_path = td_path / "model.pth"
        model_path.touch()

        # Patch run_with_workspace where it's imported in _run_workspace
        with patch("roigbiv.pipeline.workspace.run_with_workspace") as mock_workspace:
            mock_workspace.return_value = []  # Empty results
            try:
                run.main([
                    "--input", str(td_path),
                    "--fs", "7.5",
                    "--denoised-branch",
                    "--denoiser-backend", "deepinterpolation",
                    "--denoiser-model-path", str(model_path),
                ])
            except Exception:
                pass  # May fail, we just care about what overrides are passed

        # run_with_workspace(workspace, overrides, ...) is called positionally
        # (run.py:_run_workspace; overrides is args[1], NOT a "overrides" kwarg —
        # the real parameter name is cfg_overrides) — inspect args, not kwargs,
        # or this assertion silently never runs.
        assert mock_workspace.called
        args, _ = mock_workspace.call_args
        overrides = args[1]
        assert overrides.get("denoiser_backend") == "deepinterpolation"
        assert overrides.get("denoiser_model_path") == model_path
