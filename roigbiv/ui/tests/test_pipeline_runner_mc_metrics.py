"""MC quality metrics are computed once per FOV and merged into the summary.

``PipelineRunner._compute_mc_metrics`` reads each completed FOV's
``summary/mean_M.tif`` and scores it with
:func:`roigbiv.pipeline.mc_metrics.compute_metrics`; ``_summarize`` merges the
cached result in under ``"mc_metrics"``. Computed once per run (cached on
``self._mc_metrics``), not recomputed on every snapshot/tick.
"""
import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.workspace import FOVRunResult
from roigbiv.ui.services.pipeline_runner import PipelineRunner


def _runner():
    return PipelineRunner(threading.Lock())


def _write_mean_m(output_dir: Path) -> None:
    (output_dir / "summary").mkdir(parents=True, exist_ok=True)
    img = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
    tifffile.imwrite(str(output_dir / "summary" / "mean_M.tif"), img)


def test_compute_mc_metrics_reads_mean_m_per_fov(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _write_mean_m(out_a)
    _write_mean_m(out_b)
    results = [
        FOVRunResult(tif=Path("/ws/a_mc.tif"), output_dir=out_a),
        FOVRunResult(tif=Path("/ws/b_mc.tif"), output_dir=out_b),
    ]
    metrics = PipelineRunner._compute_mc_metrics(results)
    assert set(metrics) == {"/ws/a_mc.tif", "/ws/b_mc.tif"}
    for m in metrics.values():
        assert "lap_var_smooth" in m and "banding_score" in m


def test_compute_mc_metrics_skips_errored_and_missing_fovs(tmp_path):
    ok_dir = tmp_path / "ok"
    _write_mean_m(ok_dir)
    missing_dir = tmp_path / "missing"  # no summary/mean_M.tif written
    results = [
        FOVRunResult(tif=Path("/ws/ok_mc.tif"), output_dir=ok_dir),
        FOVRunResult(tif=Path("/ws/failed_mc.tif"), output_dir=tmp_path / "failed",
                     error="RuntimeError: boom"),
        FOVRunResult(tif=Path("/ws/missing_mc.tif"), output_dir=missing_dir),
    ]
    metrics = PipelineRunner._compute_mc_metrics(results)
    assert set(metrics) == {"/ws/ok_mc.tif"}


def test_summarize_merges_cached_mc_metrics(tmp_path):
    out_dir = tmp_path / "fov1"
    out_dir.mkdir()
    runner = _runner()
    runner._mc_metrics = {"/ws/fov1_mc.tif": {"lap_var_smooth": 1.23}}
    r = FOVRunResult(tif=Path("/ws/fov1_mc.tif"), output_dir=out_dir)
    summary = runner._summarize(r)
    assert summary["mc_metrics"] == {"lap_var_smooth": 1.23}


def test_summarize_mc_metrics_none_when_uncached():
    runner = _runner()
    r = FOVRunResult(tif=Path("/ws/unseen_mc.tif"), output_dir=Path("/out/unseen"))
    summary = runner._summarize(r)
    assert summary["mc_metrics"] is None


def test_reset_locked_clears_mc_metrics():
    runner = _runner()
    runner._mc_metrics = {"/ws/x.tif": {"lap_var_smooth": 1.0}}
    runner._reset_locked()
    assert runner._mc_metrics == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
