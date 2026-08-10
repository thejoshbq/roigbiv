"""
Contract tests for centroid-discovery calibration
(:mod:`roigbiv.pipeline.calibration`).

Covers the write/load round-trip that
`roigbiv.pipeline.centroids.run_centroid_discovery` builds on, and that
pre-Cellpose calibration files still load.

The former diameter -> Suite2p `spatial_scale` bucket mapping is gone: it was
never a size control (Suite2p's `spatial_scale` only scales the accept
threshold, `Th2 = threshold_scaling * 5 * max(1, scale)`), so it encoded a
mechanism Suite2p does not implement. Every field here now maps onto a real
Cellpose control.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def test_write_and_load_calibration_round_trip():
    from roigbiv.pipeline.calibration import load_calibration, write_calibration

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        written = write_calibration(output_dir, 45.0, cellprob_threshold=-1.5,
                                    cellpose_model="cyto3")

        assert written.diameter_px == pytest.approx(45.0)
        assert written.cellprob_threshold == pytest.approx(-1.5)
        assert written.cellpose_model == "cyto3"

        loaded = load_calibration(output_dir)
        assert loaded.diameter_px == pytest.approx(45.0)
        assert loaded.cellprob_threshold == pytest.approx(-1.5)
        assert loaded.cellpose_model == "cyto3"
        assert loaded.generated_at == pytest.approx(written.generated_at)
    print("  [PASS] test_write_and_load_calibration_round_trip")


def test_write_calibration_defaults():
    """Omitted knobs fall back to the config defaults, not to a magic number."""
    from roigbiv.pipeline.calibration import (DEFAULT_CELLPROB_THRESHOLD,
                                              write_calibration)

    with tempfile.TemporaryDirectory() as td:
        calib = write_calibration(Path(td), 40.0)
        assert calib.cellprob_threshold == pytest.approx(DEFAULT_CELLPROB_THRESHOLD)
        assert calib.cellpose_model is None, (
            "no model override means 'use cfg.cellpose_model', not a hardcoded one")
    print("  [PASS] test_write_calibration_defaults")


def test_load_calibration_reads_legacy_suite2p_era_file():
    """A pre-Cellpose calibration.json still loads, keeping its diameter.

    The measured diameter is the one field that stayed meaningful across the
    detector swap — the spatial_scale/threshold_scaling pair did not, and is
    ignored rather than migrated into a Cellpose knob it does not correspond to.
    """
    from roigbiv.pipeline.calibration import (DEFAULT_CELLPROB_THRESHOLD,
                                              load_calibration)

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        (output_dir / "calibration.json").write_text(json.dumps({
            "diameter_px": 40.0,
            "spatial_scale": 4,
            "threshold_scaling": 0.5,
            "generated_at": 1234.0,
        }))

        calib = load_calibration(output_dir)
        assert calib.diameter_px == pytest.approx(40.0)
        assert calib.cellprob_threshold == pytest.approx(DEFAULT_CELLPROB_THRESHOLD)
        assert calib.cellpose_model is None
        assert not hasattr(calib, "spatial_scale")
    print("  [PASS] test_load_calibration_reads_legacy_suite2p_era_file")


def test_load_calibration_missing_file_returns_none():
    from roigbiv.pipeline.calibration import load_calibration

    with tempfile.TemporaryDirectory() as td:
        assert load_calibration(Path(td)) is None
    print("  [PASS] test_load_calibration_missing_file_returns_none")


def test_load_calibration_corrupt_json_returns_none():
    from roigbiv.pipeline.calibration import load_calibration

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        (output_dir / "calibration.json").write_text("{not json")
        assert load_calibration(output_dir) is None
    print("  [PASS] test_load_calibration_corrupt_json_returns_none")


if __name__ == "__main__":
    import traceback

    tests = [
        test_write_and_load_calibration_round_trip,
        test_write_calibration_defaults,
        test_load_calibration_reads_legacy_suite2p_era_file,
        test_load_calibration_missing_file_returns_none,
        test_load_calibration_corrupt_json_returns_none,
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
    print(f"All {len(tests)} tests passed.")
