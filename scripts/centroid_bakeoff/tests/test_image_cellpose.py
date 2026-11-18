"""Accept tests for U6 (``centroid_bakeoff.detectors.image_cellpose``).

Written by the contract; workers do not edit them. Cellpose is never run: a
fake ``run_fn`` records its arguments and returns fixed masks.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from centroid_bakeoff.detector import CentroidDetectorInputs  # noqa: E402
from centroid_bakeoff.detectors.image_cellpose import (  # noqa: E402
    SummaryImageCellposeDetector, activity_max_detector, deepcad_mean_detector, temporal_sd_detector,
)


@dataclasses.dataclass(frozen=True)
class FakeCfg:
    cellprob_threshold: float = -2.0
    cellpose_model: str = "fake"
    flow_threshold: float = 0.4


def two_masks():
    a = np.zeros((16, 16), dtype=bool)
    a[2:5, 3:6] = True        # centre (3, 4)
    b = np.zeros((16, 16), dtype=bool)
    b[10:12, 8:14] = True     # centre (10.5, 10.5)
    return [a, b]


class FakeRun:
    def __init__(self, masks=None, probs=(0.9, 0.4)):
        self.calls = []
        self.masks = two_masks() if masks is None else masks
        self.probs = list(probs)

    def __call__(self, image, ch2, cfg, max_S=None):
        self.calls.append({"image": image, "ch2": ch2, "cfg": cfg, "max_S": max_S})
        return self.masks, self.probs[: len(self.masks)], np.zeros((16, 16), np.int32), None


def inputs(summary, cfg=None):
    return CentroidDetectorInputs(summary=summary, fov_stem="f", shape=(16, 16), fs=7.5, cfg=cfg)


def summary():
    return {
        "activity_max": np.full((16, 16), 1.0, np.float32),
        "vcorr_S": np.full((16, 16), 2.0, np.float32),
        "max_S": np.full((16, 16), 3.0, np.float32),
    }


def test_detect_uses_named_image():
    run = FakeRun()
    s = summary()
    det = SummaryImageCellposeDetector("activity_max", cfg=FakeCfg(), run_fn=run)
    r = det.detect(inputs(s))
    call = run.calls[0]
    assert call["image"] is s["activity_max"] and call["ch2"] is s["vcorr_S"] and call["max_S"] is s["max_S"]
    assert call["cfg"] == FakeCfg()
    assert r.centroids.dtype == np.float32 and r.centroids.shape == (2, 2)
    np.testing.assert_allclose(r.centroids, [[3.0, 4.0], [10.5, 10.5]])
    assert r.scores.dtype == np.float32
    np.testing.assert_allclose(r.scores, [0.9, 0.4])
    assert r.meta["method"] == "cellpose_activity_max" and r.meta["image_key"] == "activity_max"
    assert r.meta["n"] == 2 and "runtime_s" in r.meta and r.meta["cellprob_threshold"] == -2.0


def test_second_channel_falls_back_to_image():
    run = FakeRun()
    s = {"temporal_sd": np.ones((16, 16), np.float32)}
    SummaryImageCellposeDetector("temporal_sd", cfg=FakeCfg(), run_fn=run).detect(inputs(s))
    assert run.calls[0]["ch2"] is s["temporal_sd"] and run.calls[0]["max_S"] is None


def test_missing_image_key():
    det = SummaryImageCellposeDetector("deepcad_mean", cfg=FakeCfg(), run_fn=FakeRun())
    with pytest.raises(KeyError, match="deepcad_mean"):
        det.detect(inputs(summary()))


def test_cellprob_override_copies_cfg():
    run = FakeRun()
    cfg = FakeCfg()
    r = SummaryImageCellposeDetector("activity_max", cfg=cfg, cellprob_threshold=-6.0, run_fn=run).detect(
        inputs(summary()))
    assert run.calls[0]["cfg"].cellprob_threshold == -6.0 and cfg.cellprob_threshold == -2.0
    assert r.meta["cellprob_threshold"] == -6.0


def test_inputs_cfg_used_when_detector_has_none():
    run = FakeRun()
    SummaryImageCellposeDetector("activity_max", run_fn=run).detect(inputs(summary(), cfg=FakeCfg(-1.0)))
    assert run.calls[0]["cfg"] == FakeCfg(-1.0)


def test_no_masks():
    r = SummaryImageCellposeDetector("activity_max", cfg=FakeCfg(), run_fn=FakeRun(masks=[])).detect(
        inputs(summary()))
    assert r.centroids.shape == (0, 2) and r.scores.shape == (0,) and r.meta["n"] == 0


def test_no_pipeline_import_when_injected(monkeypatch):
    monkeypatch.setitem(sys.modules, "roigbiv.pipeline.stage1", None)
    monkeypatch.setitem(sys.modules, "roigbiv.pipeline.types", None)
    r = SummaryImageCellposeDetector("activity_max", cfg=FakeCfg(), run_fn=FakeRun()).detect(inputs(summary()))
    assert r.n == 2


@pytest.mark.parametrize("factory,key", [
    (activity_max_detector, "activity_max"),
    (temporal_sd_detector, "temporal_sd"),
    (deepcad_mean_detector, "deepcad_mean"),
])
def test_factories(factory, key):
    run = FakeRun()
    det = factory(cfg=FakeCfg(), run_fn=run, cellprob_threshold=-3.0)
    assert det.image_key == key and det.name == f"cellpose_{key}"
    assert det.cfg == FakeCfg() and det.run_fn is run and det.cellprob_threshold == -3.0
    s = {key: np.zeros((16, 16), np.float32)}
    assert det.detect(inputs(s)).meta["method"] == f"cellpose_{key}"
