"""When Suite2p cannot estimate the spatial scale, Stage 2 falls back to a set scale instead of Suite2p's 1.

Suite2p's scale only sets the detection threshold (``5 * scale`` at ``threshold_scaling`` 1). Its own fallback,
scale 1, halves the threshold that every movie with a working estimate gets (scale 2), and fills the 5,000-ROI
cap with fragments. The fallback must leave a working estimate and a forced scale alone.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

sparsedetect = pytest.importorskip("suite2p.detection.sparsedetect")

from roigbiv.suite2p import failed_scale_estimate_fallback, run_suite2p_fov  # noqa: E402

N_SCALES = 5
IMAGE_SIZE = 32
FALLBACK_SCALE = 2


def build_scale_stack(peak_scale: int) -> np.ndarray:
    """A per-scale correlation stack whose peaks all sit at peak_scale (0 makes Suite2p's estimate fail)."""
    rng = np.random.default_rng(peak_scale)
    stack = rng.uniform(0.0, 0.1, (N_SCALES, IMAGE_SIZE, IMAGE_SIZE))
    stack[peak_scale] += rng.uniform(1.0, 2.0, (IMAGE_SIZE, IMAGE_SIZE))
    return stack


def test_suite2p_falls_back_to_scale_1_without_the_wrapper():
    with pytest.warns(UserWarning):
        scale, _ = sparsedetect.find_best_scale(I=build_scale_stack(0), spatial_scale=0)
    assert scale == 1


def test_failed_estimate_uses_the_fallback_scale():
    with failed_scale_estimate_fallback(FALLBACK_SCALE), pytest.warns(UserWarning):
        scale, mode = sparsedetect.find_best_scale(I=build_scale_stack(0), spatial_scale=0)
    assert scale == FALLBACK_SCALE
    assert mode is sparsedetect.EstimateMode.Forced


def test_working_estimate_is_kept():
    with failed_scale_estimate_fallback(FALLBACK_SCALE):
        scale, mode = sparsedetect.find_best_scale(I=build_scale_stack(3), spatial_scale=0)
    assert scale == 3
    assert mode is sparsedetect.EstimateMode.Estimated


def test_forced_scale_is_kept():
    with failed_scale_estimate_fallback(FALLBACK_SCALE):
        scale, _ = sparsedetect.find_best_scale(I=build_scale_stack(0), spatial_scale=1)
    assert scale == 1


def test_wrapper_is_removed_after_an_error():
    original = sparsedetect.find_best_scale
    with pytest.raises(RuntimeError), failed_scale_estimate_fallback(FALLBACK_SCALE):
        raise RuntimeError("detection failed")
    assert sparsedetect.find_best_scale is original


def test_run_suite2p_fov_applies_the_fallback(tmp_path, monkeypatch):
    seen_scales: list[int] = []

    def fake_run_s2p(ops):
        """Stand in for Suite2p: ask for the scale the way sparsery does and record it."""
        with pytest.warns(UserWarning):
            scale, _ = sparsedetect.find_best_scale(I=build_scale_stack(0), spatial_scale=ops["spatial_scale"])
        seen_scales.append(scale)

    import importlib

    run_s2p_module = importlib.import_module("suite2p.run_s2p")  # the package attribute of that name is the function
    monkeypatch.setattr(run_s2p_module, "run_s2p", fake_run_s2p)
    movie_path = tmp_path / "movie_mc.tif"
    tifffile.imwrite(movie_path, np.zeros((10, IMAGE_SIZE, IMAGE_SIZE), np.uint16))
    run_suite2p_fov(movie_path, tmp_path / "out", fs=7.5, resume=False, failed_estimate_scale=FALLBACK_SCALE)
    assert seen_scales == [FALLBACK_SCALE]


def test_foundation_passes_the_configured_fallback(tmp_path, monkeypatch):
    import roigbiv.suite2p
    from roigbiv.pipeline.foundation import run_motion_correction
    from roigbiv.pipeline.types import PipelineConfig

    received: dict[str, object] = {}

    class StopAfterSuite2pCall(Exception):
        """Raised by the stub so foundation stops right after handing Suite2p its arguments."""

    def record_arguments(tif_path, output_dir, **keywords):
        """Stand in for run_suite2p_fov: keep the keywords foundation passed, then stop."""
        received.update(keywords)
        raise StopAfterSuite2pCall

    monkeypatch.setattr(roigbiv.suite2p, "run_suite2p_fov", record_arguments)
    movie_path = tmp_path / "fov.tif"
    movie_path.write_bytes(b"")
    with pytest.raises(StopAfterSuite2pCall):
        run_motion_correction(movie_path, PipelineConfig(fs=7.5, suite2p_failed_estimate_scale=3), tmp_path)
    assert received["failed_estimate_scale"] == 3
