"""Regression tests for the resume gate in roigbiv.suite2p.run_suite2p_fov.

A stale suite2p/plane0/stat.npy on disk must not silently short-circuit a
plain (non-``--resume``) run — see diagnostics for the incident where a
leftover stat.npy from an earlier run caused the live motion-correction
preview to show no frames.
"""
import sys
import types
from pathlib import Path

import pytest

import roigbiv.io as io_mod
import roigbiv.suite2p as s2p


@pytest.fixture
def fake_run_s2p(monkeypatch):
    """Stub out suite2p.run_s2p.run_s2p so no real registration runs."""
    calls = {"n": 0}

    def _run_s2p(ops):
        calls["n"] += 1
        plane0 = Path(ops["save_path0"]) / "suite2p" / "plane0"
        plane0.mkdir(parents=True, exist_ok=True)
        (plane0 / "stat.npy").write_bytes(b"")
        (plane0 / "ops.npy").write_bytes(b"")
        (plane0 / "data.bin").write_bytes(b"")

    fake_pkg = types.ModuleType("suite2p")
    fake_run_s2p_mod = types.ModuleType("suite2p.run_s2p")
    fake_run_s2p_mod.run_s2p = _run_s2p
    fake_default_ops_mod = types.ModuleType("suite2p.default_ops")
    fake_default_ops_mod.default_ops = lambda: {}
    monkeypatch.setitem(sys.modules, "suite2p", fake_pkg)
    monkeypatch.setitem(sys.modules, "suite2p.run_s2p", fake_run_s2p_mod)
    monkeypatch.setitem(sys.modules, "suite2p.default_ops", fake_default_ops_mod)
    monkeypatch.setattr(io_mod, "validate_tif", lambda *_a, **_k: None)
    return calls


def _seed_existing_stat_npy(output_dir: Path, stem: str):
    plane0 = output_dir / stem / "suite2p" / "plane0"
    plane0.mkdir(parents=True, exist_ok=True)
    (plane0 / "stat.npy").write_bytes(b"stale")


def test_resume_false_reregisters_despite_stale_stat_npy(tmp_path, fake_run_s2p):
    output_dir = tmp_path / "output"
    tif_path = tmp_path / "fov.tif"
    tif_path.write_bytes(b"")
    _seed_existing_stat_npy(output_dir, "fov")

    processed = s2p.run_suite2p_fov(tif_path, output_dir, fs=7.5, resume=False)

    assert processed is True
    assert fake_run_s2p["n"] == 1, "a plain run must re-register, not skip"


def test_resume_true_skips_existing_stat_npy(tmp_path, fake_run_s2p):
    output_dir = tmp_path / "output"
    tif_path = tmp_path / "fov.tif"
    tif_path.write_bytes(b"")
    _seed_existing_stat_npy(output_dir, "fov")

    processed = s2p.run_suite2p_fov(tif_path, output_dir, fs=7.5, resume=True)

    assert processed is False
    assert fake_run_s2p["n"] == 0, "--resume must still take the shortcut"


def test_resume_defaults_true_for_standalone_callers(tmp_path, fake_run_s2p):
    """Scripts (bench_motion_correction.py, process_external_data.py) call
    run_suite2p_fov directly and rely on its documented always-on
    resumability; the default must stay True so their behavior is unchanged.
    """
    output_dir = tmp_path / "output"
    tif_path = tmp_path / "fov.tif"
    tif_path.write_bytes(b"")
    _seed_existing_stat_npy(output_dir, "fov")

    processed = s2p.run_suite2p_fov(tif_path, output_dir, fs=7.5)

    assert processed is False
    assert fake_run_s2p["n"] == 0
