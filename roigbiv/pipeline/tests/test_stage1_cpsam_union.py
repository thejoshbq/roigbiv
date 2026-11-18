"""Stage 1 union of the deployed Cellpose 3 model and Cellpose-SAM.

GPU-free and sidecar-free: Cellpose 3 and the sidecar call are monkeypatched.
Covers the merge rule, the max7 image, the union dispatch and its provenance
through Gate 1, the sidecar manifest, the resume fingerprint and the CLI flag.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from roigbiv.pipeline import cpsam_union, stage1
from roigbiv.pipeline.types import PipelineConfig

UNION = "cellpose3_cpsam_union"
REPO = Path(__file__).resolve().parents[3]


def _box(shape, r0, c0, h, w):
    m = np.zeros(shape, bool)
    m[r0:r0 + h, c0:c0 + w] = True
    return m


# ── merge rule ────────────────────────────────────────────────────────────


def test_secondary_within_radius_of_a_primary_centroid_is_dropped():
    shape = (40, 40)
    primary = [_box(shape, 10, 10, 4, 4)]                 # centroid (11.5, 11.5)
    at_radius = _box(shape, 10, 18, 4, 4)                 # centroid (11.5, 19.5): 8.0 px
    keep = cpsam_union.keep_secondary(primary, [at_radius], radius=8, iou_min=0.3)
    assert keep == []


def test_secondary_beyond_radius_with_no_overlap_is_kept():
    shape = (40, 40)
    primary = [_box(shape, 10, 10, 4, 4)]
    beyond = _box(shape, 10, 19, 4, 4)                    # 9.0 px, disjoint
    assert cpsam_union.keep_secondary(primary, [beyond], radius=8, iou_min=0.3) == [0]


def test_secondary_overlapping_a_primary_at_iou_min_or_more_is_dropped():
    shape = (20, 40)
    primary = [_box(shape, 0, 0, 10, 10)]
    iou_333 = _box(shape, 0, 5, 10, 10)                   # IoU 50/150
    iou_250 = _box(shape, 0, 6, 10, 10)                   # IoU 40/160
    # radius 1 so only IoU decides
    assert cpsam_union.keep_secondary(primary, [iou_333, iou_250], radius=1, iou_min=0.3) == [1]
    assert cpsam_union.keep_secondary(primary, [iou_250], radius=1, iou_min=0.25) == []


def test_every_secondary_is_kept_when_there_is_no_primary():
    shape = (20, 20)
    sec = [_box(shape, 0, 0, 3, 3), _box(shape, 10, 10, 3, 3)]
    assert cpsam_union.keep_secondary([], sec, radius=8, iou_min=0.3) == [0, 1]


# ── max7 image ────────────────────────────────────────────────────────────


def _movie(tmp_path, T=23, H=5, W=6, seed=0):
    rng = np.random.RandomState(seed)
    data = rng.randint(-200, 2000, size=(T, H, W)).astype(np.int16)
    p = tmp_path / "data.bin"
    data.tofile(p)
    return p, data


def test_binned_max_is_the_max_over_seven_frame_means_and_drops_the_tail(tmp_path):
    p, data = _movie(tmp_path, T=23)
    want = data[:21].astype(np.float64).reshape(3, 7, 5, 6).mean(1).max(0)
    got = cpsam_union.binned_max_movie(p, data.shape, frames_per_bin=7)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want.astype(np.float32), rtol=0, atol=1e-3)


def test_binned_max_does_not_depend_on_the_chunk_size(tmp_path):
    p, data = _movie(tmp_path, T=7 * 5 + 3, seed=1)
    a = cpsam_union.binned_max_movie(p, data.shape, frames_per_bin=7, chunk_bins=2)
    b = cpsam_union.binned_max_movie(p, data.shape, frames_per_bin=7, chunk_bins=64)
    np.testing.assert_array_equal(a, b)


def test_binned_max_refuses_a_movie_with_no_complete_bin(tmp_path):
    p, data = _movie(tmp_path, T=6)
    with pytest.raises(ValueError, match="no complete bin"):
        cpsam_union.binned_max_movie(p, data.shape, frames_per_bin=7)


# ── union dispatch ────────────────────────────────────────────────────────


H, W = 40, 40


def _flows(label_image):
    return stage1.CellposeFlows(
        label_image=label_image.astype(np.uint16),
        dP=np.zeros((2, H, W), np.float32),
        cellprob=(label_image > 0).astype(np.float32) * 0.5,
        niter=200, dp_scale=5.0, diameter=12.0,
    )


@pytest.fixture
def union_fakes(monkeypatch):
    """Cellpose 3 finds one cell at (10..14, 10..14); Cellpose-SAM finds a
    duplicate of it and one new cell at (25..30, 25..30)."""
    c3 = np.zeros((H, W), np.uint16)
    c3[10:14, 10:14] = 1
    cp = np.zeros((H, W), np.int32)
    cp[10:15, 10:15] = 1
    cp[25:30, 25:30] = 2
    cp_prob = np.where(cp > 0, 2.0, -3.0).astype(np.float32)
    seen = {}

    def fake_flows(mean_S, vcorr_S, cfg, *, max_S=None):
        seen["flows_backend"] = cfg.stage1_backend
        return _flows(c3)

    def fake_sidecar(x, fields, cfg):
        seen["image"] = x
        seen["fields"] = fields
        return cp.astype(np.uint16), cp_prob

    monkeypatch.setattr(stage1, "run_cellpose_flows", fake_flows)
    monkeypatch.setattr(stage1, "_sidecar", fake_sidecar)
    return seen


def _union_cfg():
    cfg = PipelineConfig()
    cfg.stage1_backend = UNION
    cfg.force_cpu = True
    cfg.roi_stamp_radius = 8
    return cfg


def test_union_appends_new_cpsam_cells_after_cellpose3_and_drops_duplicates(union_fakes):
    img = np.random.RandomState(2).rand(H, W).astype(np.float32)
    z = np.zeros((H, W), np.float32)
    out = stage1.run_stage1_candidates(z, z, _union_cfg(), cpsam_image=img)
    assert out.detectors == ["cellpose3", "cpsam_v2"]
    assert out.masks[0].sum() == 16 and out.masks[1][27, 27] and out.masks[1].sum() == 25
    assert out.probs[1] == pytest.approx(2.0)                # Cellpose-SAM's own cellprob
    assert out.counts == {"cellpose3": 1, "cpsam_v2": 2, "cpsam_v2_kept": 1}
    np.testing.assert_array_equal(union_fakes["image"], img)
    assert union_fakes["flows_backend"] == "cellpose3"


def test_union_sends_cpsam_its_own_settings_and_no_diameter(union_fakes):
    cfg = _union_cfg()
    cfg.cellprob_threshold = -2.0
    cfg.cpsam_cellprob_threshold = 0.5
    cfg.cpsam_flow_threshold = 0.3
    cfg.cpsam_model = "cpsam_v2"
    z = np.zeros((H, W), np.float32)
    stage1.run_stage1_candidates(z, z, cfg, cpsam_image=z)
    f = union_fakes["fields"]
    assert f["diameter"] is None and f["model"] == "cpsam_v2"
    assert f["cellprob_threshold"] == 0.5 and f["flow_threshold"] == 0.3
    assert f["gpu"] is False and f["channel_axis"] is None


def test_union_uses_the_configured_iou_and_radius(union_fakes):
    """The duplicate pair overlaps at IoU 16/25 = 0.64 with centroids 0.5 px apart."""
    z = np.zeros((H, W), np.float32)
    cfg = _union_cfg()
    cfg.roi_stamp_radius = 0
    cfg.cpsam_union_iou = 0.7
    assert stage1.run_stage1_candidates(z, z, cfg, cpsam_image=z).counts["cpsam_v2_kept"] == 2
    cfg.cpsam_union_iou = 0.6
    assert stage1.run_stage1_candidates(z, z, cfg, cpsam_image=z).counts["cpsam_v2_kept"] == 1


def test_union_without_an_image_raises(union_fakes):
    z = np.zeros((H, W), np.float32)
    with pytest.raises(ValueError, match="cpsam_image"):
        stage1.run_stage1_candidates(z, z, _union_cfg())


def test_cellpose3_backend_has_no_detector_provenance(monkeypatch):
    c3 = np.zeros((H, W), np.uint16)
    c3[10:14, 10:14] = 1
    monkeypatch.setattr(stage1, "run_cellpose_flows", lambda *a, **k: _flows(c3))
    monkeypatch.setattr(stage1, "_sidecar", lambda *a, **k: pytest.fail("sidecar called"))
    cfg = PipelineConfig()
    cfg.stage1_backend = "cellpose3"
    cfg.force_cpu = True
    z = np.zeros((H, W), np.float32)
    out = stage1.run_stage1_candidates(z, z, cfg)
    assert out.detectors is None and len(out.masks) == 1


def test_cellpose3_flows_accept_the_union_backend(monkeypatch):
    """Centroid discovery calls run_cellpose_flows directly; under the union
    backend it gets Cellpose 3's flows instead of an unknown-backend error."""
    cfg = _union_cfg()
    monkeypatch.setattr(stage1, "_resolve_model_path",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("model reached")))
    z = np.zeros((H, W), np.float32)
    with pytest.raises(RuntimeError, match="model reached"):
        stage1.run_cellpose_flows(z, z, cfg)


# ── provenance through Gate 1 ─────────────────────────────────────────────


def test_gate1_records_the_detector_in_features_and_the_serialized_roi():
    from roigbiv.pipeline.gate1 import evaluate_gate1
    cfg = PipelineConfig()
    masks = [_box((H, W), 5, 5, 10, 10), _box((H, W), 25, 25, 10, 10)]
    mean = np.random.RandomState(3).rand(H, W).astype(np.float32)
    z = np.zeros((H, W), np.float32)
    rois = evaluate_gate1(masks, [0.1, 0.2], mean, z, z, cfg, starting_label_id=1,
                          detectors=["cellpose3", "cpsam_v2"])
    assert [r.features["stage1_detector"] for r in rois] == ["cellpose3", "cpsam_v2"]
    assert rois[1].to_serializable()["features"]["stage1_detector"] == "cpsam_v2"
    plain = evaluate_gate1(masks, [0.1, 0.2], mean, z, z, cfg, starting_label_id=1)
    assert all("stage1_detector" not in r.features for r in plain)


# ── sidecar manifest ──────────────────────────────────────────────────────


def _load_sidecar():
    spec = importlib.util.spec_from_file_location("cpsam_sidecar_t", REPO / "scripts" / "cpsam_sidecar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_cellpose(monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self, gpu=False, **kw):
            calls["init"] = dict(gpu=gpu, **kw)

        def eval(self, x, **kw):
            calls["eval"] = kw
            lab = np.zeros(x.shape[:2], np.int32)
            lab[1:3, 1:3] = 1
            return lab, [None, None, np.zeros(x.shape[:2], np.float32)], None

    models = types.ModuleType("cellpose.models")
    models.CellposeModel = FakeModel
    pkg = types.ModuleType("cellpose")
    pkg.models = models
    monkeypatch.setitem(sys.modules, "cellpose", pkg)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return calls


def _manifest(tmp_path, **extra):
    np.save(tmp_path / "in.npy", np.zeros((6, 6), np.float32))
    m = {"input": str(tmp_path / "in.npy"), "labels_out": str(tmp_path / "lab.npy"),
         "cellprob_out": str(tmp_path / "cp.npy"), "diameter": 12.0, "cellprob_threshold": 0.0,
         "flow_threshold": 0.4, "gpu": False, "channel_axis": None}
    m.update(extra)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    return str(p)


def test_sidecar_passes_the_model_and_omits_a_null_diameter(tmp_path, fake_cellpose):
    rc = _load_sidecar().main(_manifest(tmp_path, model="cpsam_v2", diameter=None))
    assert rc == 0
    assert fake_cellpose["init"]["pretrained_model"] == "cpsam_v2"
    assert "diameter" not in fake_cellpose["eval"]


def test_sidecar_without_model_keeps_the_cellpose_default(tmp_path, fake_cellpose):
    assert _load_sidecar().main(_manifest(tmp_path)) == 0
    assert "pretrained_model" not in fake_cellpose["init"]
    assert fake_cellpose["eval"]["diameter"] == 12.0


def test_sidecar_refuses_gpu_without_cuda(tmp_path, fake_cellpose, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _load_sidecar().main(_manifest(tmp_path, gpu=True)) == 3
    assert "init" not in fake_cellpose


# ── config, fingerprint, CLI ──────────────────────────────────────────────


def test_cpsam_settings_default_to_the_benchmarked_values():
    cfg = PipelineConfig()
    assert cfg.stage1_backend == UNION
    assert (cfg.cpsam_model, cfg.cpsam_cellprob_threshold, cfg.cpsam_flow_threshold,
            cfg.cpsam_union_iou, cfg.cpsam_frames_per_bin) == ("cpsam_v2", 0.5, 0.4, 0.3, 7)


@pytest.mark.parametrize("field,value", [
    ("stage1_backend", "cellpose3"), ("cpsam_model", "cpsam"), ("cpsam_cellprob_threshold", 1.0),
    ("cpsam_flow_threshold", 0.6), ("cpsam_union_iou", 0.5), ("cpsam_frames_per_bin", 5),
])
def test_each_cpsam_setting_changes_the_resume_fingerprint(tmp_path, field, value):
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    tif = tmp_path / "fov.tif"
    tif.write_bytes(b"x")
    a = PipelineConfig()
    b = PipelineConfig()
    setattr(b, field, value)
    assert compute_cfg_fingerprint(a, tif) != compute_cfg_fingerprint(b, tif)


def test_cli_stage1_backend_flag_reaches_the_config(tmp_path):
    from roigbiv.pipeline import run
    import tifffile
    tif = tmp_path / "fov.tif"
    tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))

    def cfg_for(extra, out):
        # run_pipeline is mocked, so main fails after it on the missing FOV;
        # only the config it was handed matters (as in test_denoising_config).
        with patch("roigbiv.pipeline.run.run_pipeline") as rp:
            rp.return_value = None
            try:
                run.main(["--input", str(tif), "--fs", "7.5", "--no-viewer", "--profile", "grin",
                          "--output-dir", str(tmp_path / out), *extra])
            except Exception:
                pass
        assert rp.called
        return rp.call_args[0][1]

    assert cfg_for(["--stage1-backend", UNION], "o").stage1_backend == UNION
    assert cfg_for([], "o2").stage1_backend == UNION
    assert cfg_for(["--stage1-backend", "cellpose3"], "o3").stage1_backend == "cellpose3"


# ── run.py helpers ────────────────────────────────────────────────────────


def test_stage1_cpsam_image_is_max7_of_the_registered_movie_under_union_only(tmp_path):
    import tifffile
    from roigbiv.pipeline import run
    from roigbiv.pipeline.types import FOVData
    p, data = _movie(tmp_path, T=21)
    fov = FOVData(raw_path=tmp_path / "f.tif", output_dir=tmp_path, data_bin_path=p, shape=data.shape)
    c3 = PipelineConfig()
    c3.stage1_backend = "cellpose3"
    assert run._stage1_cpsam_image(fov, c3, tmp_path) is None
    assert not (tmp_path / "summary").exists()
    img = run._stage1_cpsam_image(fov, _union_cfg(), tmp_path)
    want = data.astype(np.float64).reshape(3, 7, 5, 6).mean(1).max(0).astype(np.float32)
    np.testing.assert_allclose(img, want, atol=1e-3)
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "summary" / "max7_M.tif"), img)


def test_detector_summary_counts_proposals_and_gate1_outcomes_per_detector():
    from roigbiv.pipeline import run
    from roigbiv.pipeline.types import ROI

    def roi(det, outcome):
        return ROI(mask=np.zeros((2, 2), bool), label_id=1, source_stage=1, confidence="high",
                   gate_outcome=outcome, features={"stage1_detector": det})

    s1 = stage1.Stage1Candidates([], [], np.zeros(1), np.zeros(1), detectors=[],
                                 counts={"cellpose3": 3, "cpsam_v2": 5, "cpsam_v2_kept": 2})
    rois = [roi("cellpose3", "accept"), roi("cellpose3", "reject"), roi("cellpose3", "accept"),
            roi("cpsam_v2", "flag"), roi("cpsam_v2", "accept")]
    assert run._stage1_detector_summary(s1, rois) == {
        "cellpose3": {"proposed": 3, "kept": 3, "accept": 2, "flag": 0, "reject": 1},
        "cpsam_v2": {"proposed": 5, "kept": 2, "accept": 1, "flag": 1, "reject": 0},
    }


def test_cpsam_python_falls_back_to_a_sibling_roisuite_env(tmp_path, monkeypatch):
    envs = tmp_path / "envs"
    (envs / "roigbiv").mkdir(parents=True)
    py = envs / "roisuite" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setattr(sys, "prefix", str(envs / "roigbiv"))
    monkeypatch.delenv("ROIGBIV_CPSAM_PYTHON", raising=False)
    assert stage1._resolve_cpsam_python(PipelineConfig()) == str(py)
    py.unlink()
    with pytest.raises(FileNotFoundError, match="--stage1-backend cellpose3"):
        stage1._resolve_cpsam_python(PipelineConfig())
