"""Guards for :mod:`roigbiv.ui.services.extraction_runner`.

Exercises the runner's threading/state machine in isolation from the actual
extraction logic (already covered end-to-end by
``roigbiv/pipeline/tests/test_discovery_extract.py``) by monkeypatching
``extract_from_merged_masks``.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from roigbiv.ui.services import extraction_runner as mod
from roigbiv.ui.services.extraction_runner import ExtractionRunner


def _wait_until_idle(runner: ExtractionRunner, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while runner.snapshot().active:
        assert time.monotonic() < deadline, "extraction runner never went idle"
        time.sleep(0.01)


def test_start_runs_extraction_and_records_the_bundle(monkeypatch, tmp_path: Path):
    called = {}

    def _fake_extract(fov_output_dir, *, cfg=None, stats=(), skip_overlap_correction=False):
        called["fov_output_dir"] = fov_output_dir
        called["stats"] = stats
        return fov_output_dir / "traces"

    monkeypatch.setattr(mod, "extract_from_merged_masks", _fake_extract)

    runner = ExtractionRunner()
    started = runner.start(tmp_path, tmp_path.name, stats=("median", "mode"))
    assert started is True
    _wait_until_idle(runner)

    snap = runner.snapshot()
    assert snap.active is False
    assert snap.error is None
    assert snap.bundle_dir == str(tmp_path / "traces")
    assert snap.stem == tmp_path.name
    assert called["stats"] == ("median", "mode")
    assert any("Extracting" in line for line in snap.logs)
    assert any("Wrote" in line for line in snap.logs)


def test_a_second_start_is_refused_while_active(monkeypatch, tmp_path: Path):
    release = threading.Event()

    def _blocking_extract(fov_output_dir, *, cfg=None, stats=(),
                          skip_overlap_correction=False):
        release.wait(timeout=2.0)
        return fov_output_dir / "traces"

    monkeypatch.setattr(mod, "extract_from_merged_masks", _blocking_extract)

    runner = ExtractionRunner()
    assert runner.start(tmp_path, tmp_path.name) is True
    assert runner.start(tmp_path, tmp_path.name) is False  # already active

    release.set()
    _wait_until_idle(runner)


def test_a_failure_is_captured_not_raised(monkeypatch, tmp_path: Path):
    def _raising_extract(fov_output_dir, *, cfg=None, stats=(),
                         skip_overlap_correction=False):
        raise FileNotFoundError("no merged_masks.tif")

    monkeypatch.setattr(mod, "extract_from_merged_masks", _raising_extract)

    runner = ExtractionRunner()
    runner.start(tmp_path, tmp_path.name)
    _wait_until_idle(runner)

    snap = runner.snapshot()
    assert snap.error is not None
    assert "no merged_masks.tif" in snap.error
    assert snap.bundle_dir is None


def test_get_extraction_runner_is_stable_within_a_session(monkeypatch):
    monkeypatch.setattr(
        "roigbiv.ui.services.session.get_session_id", lambda: "sess-fixed")
    from roigbiv.ui.services.extraction_runner import get_extraction_runner

    a = get_extraction_runner()
    b = get_extraction_runner()
    assert a is b
