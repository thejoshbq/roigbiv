"""Tests for ``centroid_bakeoff.detectors.cpsam_image``.

Cellpose-SAM is never run. Unit tests inject a fake ``run_fn``; the subprocess
test runs the real worker under this interpreter with a stub ``cellpose``
package on ``PYTHONPATH``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from centroid_bakeoff.detector import CentroidDetectorInputs  # noqa: E402
from centroid_bakeoff.detectors.cpsam_image import (  # noqa: E402
    CellposeSamImageDetector, run_cpsam_subprocess,
)


def two_labels():
    lab = np.zeros((16, 16), np.int32)
    lab[2:5, 3:6] = 1        # centre (3, 4)
    lab[10:12, 8:14] = 7     # centre (10.5, 10.5)
    return lab


class FakeRun:
    def __init__(self, labels=None):
        self.calls = []
        self.labels = two_labels() if labels is None else labels

    def __call__(self, image, **kw):
        self.calls.append({"image": image, **kw})
        return self.labels, {"cellpose_version": "4.2.1.1", "device": "cuda"}


def inputs(summary):
    return CentroidDetectorInputs(summary=summary, fov_stem="f", shape=(16, 16), fs=7.5)


def test_detect_returns_label_centroids():
    run = FakeRun()
    det = CellposeSamImageDetector("max7", python="/x/python", run_fn=run)
    res = det.detect(inputs({"max7": np.ones((16, 16), np.float32)}))
    assert res.centroids.dtype == np.float32
    np.testing.assert_allclose(res.centroids, [[3.0, 4.0], [10.5, 10.5]])
    assert res.n == 2


def test_detect_passes_the_named_image_and_settings():
    run = FakeRun()
    img = np.full((16, 16), 2.0, np.float32)
    det = CellposeSamImageDetector("max7", python="/x/python", cellprob_threshold=-1.0,
                                   flow_threshold=0.6, gpu=False, run_fn=run)
    det.detect(inputs({"max7": img, "std": np.zeros((16, 16), np.float32)}))
    (call,) = run.calls
    assert call["image"] is img
    assert (call["python"], call["model"], call["cellprob_threshold"], call["flow_threshold"], call["gpu"]) == (
        "/x/python", "cpsam_v2", -1.0, 0.6, False)


def test_defaults_and_meta():
    det = CellposeSamImageDetector("std", python="/x/python", run_fn=FakeRun())
    assert det.name == "cpsam_std"
    res = det.detect(inputs({"std": np.ones((16, 16), np.float32)}))
    m = res.meta
    assert (m["method"], m["image_key"], m["model"], m["n"]) == ("cpsam_std", "std", "cpsam_v2", 2)
    assert (m["cellprob_threshold"], m["flow_threshold"]) == (0.0, 0.4)
    assert (m["cellpose_version"], m["device"]) == ("4.2.1.1", "cuda")
    assert m["runtime_s"] >= 0


def test_no_cells():
    det = CellposeSamImageDetector("std", python="/x/python", run_fn=FakeRun(np.zeros((16, 16), np.int32)))
    res = det.detect(inputs({"std": np.ones((16, 16), np.float32)}))
    assert res.centroids.shape == (0, 2) and res.centroids.dtype == np.float32


def test_missing_image_names_key_and_available():
    det = CellposeSamImageDetector("max7", python="/x/python", run_fn=FakeRun())
    with pytest.raises(KeyError, match="max7.*std"):
        det.detect(inputs({"std": np.ones((16, 16), np.float32)}))


STUB = '''
import numpy as np
__version__ = "9.9.stub"

class CellposeModel:
    def __init__(self, gpu=False, pretrained_model=None):
        self.gpu, self.pretrained_model = gpu, pretrained_model

    def eval(self, img, cellprob_threshold=0.0, flow_threshold=0.4):
        lab = np.zeros(img.shape, np.int32)
        lab[img > img.mean()] = 1
        log = {"model": self.pretrained_model, "gpu": self.gpu, "shape": list(img.shape),
               "dtype": str(img.dtype), "cellprob": cellprob_threshold, "flow": flow_threshold}
        open(__import__("os").environ["STUB_LOG"], "w").write(__import__("json").dumps(log))
        return lab, None, None
'''


@pytest.fixture
def stub_cellpose(tmp_path, monkeypatch):
    pkg = tmp_path / "stub" / "cellpose"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from importlib.metadata import version\n")
    (pkg / "models.py").write_text(STUB)
    log = tmp_path / "stub_log.json"
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))
    monkeypatch.setenv("STUB_LOG", str(log))
    return log


def test_subprocess_round_trip_on_cpu(stub_cellpose):
    img = np.zeros((8, 9), np.float64)
    img[2:4, 3:5] = 5.0
    lab, info = run_cpsam_subprocess(img, python=sys.executable, model="cpsam_v2",
                                     cellprob_threshold=-1.0, flow_threshold=0.6, gpu=False)
    assert lab.shape == (8, 9) and lab.dtype == np.int32
    assert np.argwhere(lab == 1).tolist() == [[2, 3], [2, 4], [3, 3], [3, 4]]
    log = json.loads(stub_cellpose.read_text())
    assert log == {"model": "cpsam_v2", "gpu": False, "shape": [8, 9], "dtype": "float32",
                   "cellprob": -1.0, "flow": 0.6}
    assert info["device"] == "cpu"


def test_subprocess_gpu_requested_without_cuda_fails(stub_cellpose, tmp_path, monkeypatch):
    torch = tmp_path / "stub" / "torch"
    torch.mkdir()
    (torch / "__init__.py").write_text("class cuda:\n    @staticmethod\n    def is_available():\n        return False\n")
    with pytest.raises(RuntimeError, match="CUDA"):
        run_cpsam_subprocess(np.ones((4, 4)), python=sys.executable, model="cpsam_v2",
                             cellprob_threshold=0.0, flow_threshold=0.4, gpu=True)
    assert not stub_cellpose.exists()
