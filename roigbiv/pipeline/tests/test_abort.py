"""Cooperative-abort plumbing for the workspace runner (sequential path).

The UI's Stop button sets a ``threading.Event`` that the pipeline checks at
FOV boundaries (``run_with_workspace``) and stage boundaries (``run_pipeline``
via ``_check_abort`` → ``PipelineAborted``). An aborted FOV must NOT leave a
half-written registry row, so ``_process_one`` catches ``PipelineAborted``
before ``_register_session`` is reached. The UI only ever runs sequentially
(``n_workers=1``); the threading event cannot reach the separate batch worker
processes, so ``_run_parallel`` only honors a stop requested *before* the pool
launches (a full mid-batch terminate is a CLI-only follow-up).
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline import workspace as ws_mod
from roigbiv.pipeline.run import PipelineAborted, _check_abort


def _write_fake_tif(path: Path, T: int = 4, H: int = 8, W: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((T, H, W), dtype=np.int16))


# ── _check_abort / PipelineAborted ───────────────────────────────────────────

def test_check_abort_noop_when_none_or_clear():
    _check_abort(None)                 # no event → no-op
    ev = threading.Event()
    _check_abort(ev)                   # clear event → no-op
    ev.set()
    with pytest.raises(PipelineAborted):
        _check_abort(ev)


def test_pipeline_aborted_is_baseexception_not_exception():
    # Must escape per-stage ``except Exception`` handlers inside run_pipeline.
    assert issubclass(PipelineAborted, BaseException)
    assert not issubclass(PipelineAborted, Exception)


# ── run_with_workspace FOV-boundary abort ────────────────────────────────────

def _workspace(tmp_path: Path, n: int) -> "ws_mod.WorkspacePaths":
    for i in range(n):
        _write_fake_tif(tmp_path / f"fov{i}_mc.tif")
    return ws_mod.resolve_workspace(tmp_path)


def _patch_light(monkeypatch):
    """Stub the registry schema check + backfill so only the loop runs."""
    monkeypatch.setattr(ws_mod, "_ensure_registry_schema", lambda *a, **k: None)
    monkeypatch.setattr(ws_mod, "_safety_backfill", lambda *a, **k: None)


def test_preset_abort_runs_zero_fovs(tmp_path: Path, monkeypatch):
    _patch_light(monkeypatch)
    calls: list[Path] = []

    def fake_process_one(tif, workspace, cfg_overrides, log, **kwargs):
        calls.append(tif)
        return ws_mod.FOVRunResult(tif=tif, output_dir=workspace.output_root)

    monkeypatch.setattr(ws_mod, "_process_one", fake_process_one)
    abort = threading.Event()
    abort.set()                        # stop requested before the loop starts

    results = ws_mod.run_with_workspace(
        _workspace(tmp_path, 2), {}, registry_config=SimpleNamespace(),
        skip_backfill=True, abort_event=abort,
    )
    assert calls == []
    assert results == []


def test_abort_after_first_fov_skips_the_rest(tmp_path: Path, monkeypatch):
    _patch_light(monkeypatch)
    abort = threading.Event()
    calls: list[Path] = []

    def fake_process_one(tif, workspace, cfg_overrides, log, *, abort_event=None,
                         **kwargs):
        calls.append(tif)
        abort_event.set()              # request stop while processing FOV 1
        return ws_mod.FOVRunResult(tif=tif, output_dir=workspace.output_root)

    monkeypatch.setattr(ws_mod, "_process_one", fake_process_one)

    results = ws_mod.run_with_workspace(
        _workspace(tmp_path, 3), {}, registry_config=SimpleNamespace(),
        skip_backfill=True, abort_event=abort,
    )
    assert len(calls) == 1             # FOV 1 ran; loop broke before FOV 2
    assert len(results) == 1


# ── _process_one stage-boundary abort skips the registry write ───────────────

def test_process_one_pipeline_aborted_skips_registry(tmp_path: Path, monkeypatch):
    tif = tmp_path / "fov0_mc.tif"
    _write_fake_tif(tif)
    workspace = ws_mod.resolve_workspace(tmp_path)

    # run_pipeline raises PipelineAborted at a stage seam; _register_session
    # must never be reached (no half-written registry row).
    monkeypatch.setattr(ws_mod, "validate_tif", lambda *_a, **_k: None)
    register = mock.Mock()
    monkeypatch.setattr(ws_mod, "_register_session", register)

    import roigbiv.pipeline.run as run_mod
    monkeypatch.setattr(
        run_mod, "run_pipeline",
        lambda *a, **k: (_ for _ in ()).throw(PipelineAborted()),
    )

    result = ws_mod._process_one(
        tif, workspace, {}, lambda _m: None,
        skip_registry=False, registry_cfg=SimpleNamespace(),
        abort_event=threading.Event(),
    )
    assert result.error == "aborted"
    register.assert_not_called()


# ── _run_parallel pre-dispatch guard (no in-process abort across workers) ─────

def test_run_parallel_preset_abort_returns_aborted_without_launch(
        tmp_path: Path, monkeypatch):
    # A stop requested before the pool launches short-circuits: every FOV is
    # reported as aborted and run_batch is never called.
    from roigbiv.pipeline import batch as batch_mod
    called = {"run_batch": False}

    def _boom(*a, **k):                # must not be invoked
        called["run_batch"] = True
        raise AssertionError("run_batch should not launch under preset abort")

    monkeypatch.setattr(batch_mod, "run_batch", _boom)

    workspace = _workspace(tmp_path, 2)
    abort = threading.Event()
    abort.set()
    results = ws_mod._run_parallel(
        workspace, workspace.tifs, {}, lambda _m: None,
        skip_registry=True, n_workers=2, registry_cfg=SimpleNamespace(),
        abort_event=abort,
    )
    assert called["run_batch"] is False
    assert len(results) == 2
    assert all(r.error == "aborted" for r in results)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
