"""Tests for multi-user session isolation in run_with_workspace.

Verifies that passing an explicit RegistryConfig does not mutate os.environ,
and that the correct registry paths are used regardless of env-var state.
"""
from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path

import pytest

from roigbiv.pipeline.workspace import run_with_workspace, resolve_workspace
from roigbiv.registry.config import RegistryConfig


def _make_workspace(tmp_path: Path):
    """Create a minimal workspace with one tiny valid TIF."""
    import numpy as np
    import tifffile

    tif = tmp_path / "test_mc.tif"
    # Write a 10-frame 64×64 stack — small enough to pass validate_tif
    tifffile.imwrite(str(tif), np.zeros((10, 64, 64), dtype=np.uint16))
    return resolve_workspace(tmp_path)


def _make_cfg(tmp_path: Path) -> RegistryConfig:
    db = tmp_path / "registry.db"
    blobs = tmp_path / "blobs"
    cal = tmp_path / "calibration.json"
    return RegistryConfig(
        dsn=f"sqlite:///{db}",
        blob_backend="local",
        blob_root=blobs,
        endpoint=None,
        api_key=None,
        calibration_path=cal,
    )


def test_run_with_workspace_no_env_mutation(tmp_path):
    """With explicit registry_config, os.environ must not be written."""
    workspace = _make_workspace(tmp_path)
    cfg = _make_cfg(tmp_path)

    before = {
        k: os.environ.get(k)
        for k in ("ROIGBIV_REGISTRY_DSN", "ROIGBIV_BLOB_ROOT", "ROIGBIV_CALIBRATION_PATH")
    }

    run_with_workspace(
        workspace, {},
        registry_config=cfg,
        skip_registry=True,
        skip_backfill=True,
    )

    after = {
        k: os.environ.get(k)
        for k in ("ROIGBIV_REGISTRY_DSN", "ROIGBIV_BLOB_ROOT", "ROIGBIV_CALIBRATION_PATH")
    }
    assert before == after, (
        "run_with_workspace mutated os.environ even though registry_config was supplied"
    )


def test_explicit_cfg_takes_precedence_over_env(tmp_path):
    """Registry writes go to explicit cfg.dsn, not whatever ROIGBIV_REGISTRY_DSN says."""
    workspace = _make_workspace(tmp_path)

    correct_dir = tmp_path / "correct_registry"
    correct_dir.mkdir()
    cfg = _make_cfg(correct_dir)

    wrong_dir = tmp_path / "wrong_registry"
    wrong_dir.mkdir()
    wrong_dsn = f"sqlite:///{wrong_dir / 'registry.db'}"

    old_dsn = os.environ.get("ROIGBIV_REGISTRY_DSN")
    try:
        os.environ["ROIGBIV_REGISTRY_DSN"] = wrong_dsn
        run_with_workspace(
            workspace, {},
            registry_config=cfg,
            skip_backfill=True,
        )
    finally:
        if old_dsn is None:
            os.environ.pop("ROIGBIV_REGISTRY_DSN", None)
        else:
            os.environ["ROIGBIV_REGISTRY_DSN"] = old_dsn

    correct_db = correct_dir / "registry.db"
    wrong_db = wrong_dir / "registry.db"
    assert correct_db.exists(), "Expected registry.db at the explicit cfg path"
    assert not wrong_db.exists(), "Registry.db must NOT be created at the env-var path"


def _make_two_tif_workspace(tmp_path: Path):
    """Workspace with two distinct pre-corrected stacks (a_mc, b_mc)."""
    import numpy as np
    import tifffile

    for name in ("a_mc.tif", "b_mc.tif"):
        tifffile.imwrite(str(tmp_path / name),
                         np.zeros((6, 32, 32), dtype=np.uint16))
    return resolve_workspace(tmp_path)


def test_run_with_workspace_selected_subset(tmp_path, monkeypatch):
    """selected_tifs restricts the run to that subset and sizes the log/results
    from it — without mutating the frozen workspace."""
    import roigbiv.pipeline.workspace as wsmod

    workspace = _make_two_tif_workspace(tmp_path)
    cfg = _make_cfg(tmp_path)
    assert len(workspace.tifs) == 2

    seen: list[Path] = []

    def _stub(tif, ws, cfg_overrides, log, *, skip_registry, registry_cfg,
              **kwargs):
        seen.append(tif)
        return wsmod.FOVRunResult(tif=tif, output_dir=ws.output_root / tif.stem)

    monkeypatch.setattr(wsmod, "_process_one", _stub)
    logs: list[str] = []
    target = workspace.tifs[0]
    results = wsmod.run_with_workspace(
        workspace, {}, log_cb=logs.append,
        registry_config=cfg, skip_registry=True, skip_backfill=True,
        selected_tifs=[target],
    )
    assert seen == [target]
    assert len(results) == 1 and results[0].tif == target
    assert any("Found 1 TIF stack(s)" in line for line in logs)
    # Frozen workspace is untouched.
    assert len(workspace.tifs) == 2


def test_run_with_workspace_selected_none_runs_all(tmp_path, monkeypatch):
    """Omitting selected_tifs (the CLI path) still runs every TIF."""
    import roigbiv.pipeline.workspace as wsmod

    workspace = _make_two_tif_workspace(tmp_path)
    cfg = _make_cfg(tmp_path)

    seen: list[Path] = []

    def _stub(tif, ws, cfg_overrides, log, *, skip_registry, registry_cfg,
              **kwargs):
        seen.append(tif)
        return wsmod.FOVRunResult(tif=tif, output_dir=ws.output_root / tif.stem)

    monkeypatch.setattr(wsmod, "_process_one", _stub)
    results = wsmod.run_with_workspace(
        workspace, {}, registry_config=cfg,
        skip_registry=True, skip_backfill=True,
    )
    assert set(seen) == set(workspace.tifs)
    assert len(results) == 2


def test_runner_start_sizes_n_fovs_from_selection(tmp_path, monkeypatch):
    """PipelineRunner.start sizes the progress denominator from the subset."""
    import threading

    import roigbiv.ui.services.pipeline_runner as runner_mod

    workspace = _make_two_tif_workspace(tmp_path)
    # Stub the heavy run so the daemon thread returns immediately.
    monkeypatch.setattr(runner_mod, "run_with_workspace",
                        lambda *a, **k: [])
    runner = runner_mod.PipelineRunner(threading.Lock())
    ok = runner.start(workspace, {}, selected_tifs=[workspace.tifs[0]])
    assert ok is True
    assert runner.snapshot().n_fovs == 1
    if runner._thread is not None:
        runner._thread.join(timeout=5)


def test_secret_key_warning(monkeypatch):
    """build_app() must emit a UserWarning when ROIGBIV_SECRET_KEY is absent."""
    monkeypatch.delenv("ROIGBIV_SECRET_KEY", raising=False)

    from roigbiv.ui.app import build_app

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_app()

    secret_key_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "ROIGBIV_SECRET_KEY" in str(w.message)
    ]
    assert secret_key_warnings, "Expected a UserWarning about ROIGBIV_SECRET_KEY"
