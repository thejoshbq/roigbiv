"""Phase-M Stage-1 backend dispatch tests (cpsam sidecar wiring).

GPU-free / sidecar-free: the actual cpsam call is monkeypatched. These cover
(1) the default backend is unchanged, (2) the dispatch routes to the sidecar
without touching the in-process cellpose path, (3) the shared label-split and
diameter helpers behave, and (4) interpreter resolution + error surfaces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roigbiv.pipeline import stage1
from roigbiv.pipeline.types import PipelineConfig


def test_default_backend_is_cellpose3():
    assert PipelineConfig().stage1_backend == "cellpose3"


def test_split_labels_self_consistent():
    lab = np.zeros((6, 6), dtype=np.uint16)
    lab[0:2, 0:2] = 1
    lab[4:6, 4:6] = 2
    cp = np.full((6, 6), 0.5, dtype=np.float32)
    cp[lab == 2] = 0.9
    masks, probs, lab_out, cp_out = stage1._split_labels(lab, cp)
    assert len(masks) == 2 and len(probs) == 2
    assert masks[0].sum() == 4 and masks[1].sum() == 4
    assert probs[0] == pytest.approx(0.5) and probs[1] == pytest.approx(0.9)
    assert lab_out is lab and cp_out is cp


def test_effective_diameter_default_when_not_auto():
    cfg = PipelineConfig()
    cfg.diameter = 13
    cfg.diameter_auto = False
    assert stage1._effective_diameter(np.zeros((32, 32), np.float32), cfg) == 13


def test_unknown_backend_raises():
    cfg = PipelineConfig()
    cfg.stage1_backend = "bogus"
    cfg.force_cpu = True
    with pytest.raises(ValueError, match="unknown stage1_backend"):
        stage1.run_cellpose_detection(
            np.zeros((16, 16), np.float32), np.zeros((16, 16), np.float32), cfg
        )


def test_cpsam_backend_dispatches_to_sidecar(monkeypatch):
    """cpsam_sidecar must call _run_cpsam_sidecar with a (H,W,2) stack and an
    int diameter, and never touch the in-process CellposeModel path."""
    cfg = PipelineConfig()
    cfg.stage1_backend = "cpsam_sidecar"
    cfg.force_cpu = True
    cfg.diameter = 11
    cfg.diameter_auto = False

    H, W = 12, 12
    fake_lab = np.zeros((H, W), dtype=np.uint16)
    fake_lab[2:5, 2:5] = 1
    fake_cp = (fake_lab > 0).astype(np.float32)

    seen = {}

    def fake_sidecar(x, diameter, cfg_, gpu):
        seen["x_shape"] = x.shape
        seen["diameter"] = diameter
        seen["gpu"] = gpu
        return fake_lab, fake_cp

    monkeypatch.setattr(stage1, "_run_cpsam_sidecar", fake_sidecar)
    # Hard-fail if the in-process cellpose path is touched.
    monkeypatch.setattr(stage1, "_resolve_model_path",
                        lambda *_a, **_k: pytest.fail("cellpose3 path entered"))

    morph = np.random.RandomState(0).rand(H, W).astype(np.float32)
    vcorr = np.random.RandomState(1).rand(H, W).astype(np.float32)
    masks, probs, lab_out, cp_out = stage1.run_cellpose_detection(morph, vcorr, cfg)

    assert seen["x_shape"] == (H, W, 2)
    assert seen["diameter"] == 11
    assert seen["gpu"] is False                 # force_cpu honored
    assert len(masks) == 1 and lab_out.dtype == np.uint16


def test_resolve_cpsam_python_explicit_and_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("ROIGBIV_CPSAM_PYTHON", raising=False)
    cfg = PipelineConfig()

    # explicit, existing path wins
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh\n")
    cfg.cpsam_sidecar_python = str(fake_py)
    assert stage1._resolve_cpsam_python(cfg) == str(fake_py)

    # nothing resolvable → clear error (point the sibling probe at a missing path
    # by giving a bogus explicit + no env; sibling almost certainly absent in CI)
    cfg.cpsam_sidecar_python = str(tmp_path / "nope")
    monkeypatch.setattr(stage1.sys, "prefix", str(tmp_path / "envs" / "roigbiv"))
    with pytest.raises(FileNotFoundError, match="cpsam sidecar interpreter"):
        stage1._resolve_cpsam_python(cfg)
