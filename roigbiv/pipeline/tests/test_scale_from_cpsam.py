"""Soma scale from Cellpose-SAM masks, with the DoG estimate and the profile
defaults as fallbacks, and Stage 1 reusing the same Cellpose-SAM masks.

GPU-free and sidecar-free: the sidecar call and Cellpose 3 are monkeypatched.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from roigbiv.pipeline import stage1
from roigbiv.pipeline.optics import SomaScale, measure_scale, scale_from_labels
from roigbiv.pipeline.types import PipelineConfig

UNION = "cellpose3_cpsam_union"
SIDES = (10, 12, 14, 16, 18)          # square masks; median area 14 * 14 = 196


def _squares(sides=SIDES, size=200):
    """Non-overlapping square masks, four per row, spaced by the largest side."""
    lab = np.zeros((size, size), np.uint16)
    step = max(sides) + 10
    for i, s in enumerate(sides):
        r, c = 10 + (i // 4) * step, 10 + (i % 4) * step
        assert r + s <= size and c + s <= size, "squares do not fit"
        lab[r:r + s, c:c + s] = i + 1
    assert [int((lab == i + 1).sum()) for i in range(len(sides))] == [s * s for s in sides]
    return lab


def _d(area):
    return 2.0 * np.sqrt(area / np.pi)


def _planted(n, radius=8, size=256, sep=50):
    """mean_M-like image the DoG estimate can size (as in test_optics)."""
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cols = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        cy, cx = sep + (i // cols) * sep, sep + (i % cols) * sep
        if cy < size - sep and cx < size - sep:
            img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius ** 2))
    return img


# ── scale_from_labels / measure_scale ─────────────────────────────────────


def test_scale_from_labels_is_the_mask_size_divided_by_k():
    s = scale_from_labels(_squares(), k=1.25)
    areas = np.array(SIDES, float) ** 2
    assert s.ok and s.source == "cpsam" and s.n_somata == 5
    assert s.diameter_med == pytest.approx(_d(196) / 1.25)
    assert s.area_med == pytest.approx(196 / 1.25 ** 2)
    assert s.diameter_p5 == pytest.approx(np.percentile(_d(areas), 5) / 1.25)
    assert s.diameter_p95 == pytest.approx(np.percentile(_d(areas), 95) / 1.25)
    assert s.area_p95 == pytest.approx(np.percentile(areas, 95) / 1.25 ** 2)


def test_scale_from_labels_needs_five_masks():
    s = scale_from_labels(_squares(SIDES[:4]))
    assert not s.ok and s.n_somata == 4 and s.source == "cpsam"


def test_measure_scale_uses_cpsam_masks_first():
    s = measure_scale(_planted(16), cpsam_labels=_squares(), cpsam_k=1.0, profile_name="grin")
    assert s.source == "cpsam" and s.ok
    assert s.diameter_med == pytest.approx(_d(196))


def test_measure_scale_falls_back_to_dog_with_four_masks():
    img = _planted(16)
    s = measure_scale(img, cpsam_labels=_squares(SIDES[:4]), profile_name="grin")
    assert s.source == "dog" and s.ok
    from roigbiv.pipeline.optics import measure_soma_scale
    assert s.diameter_med == pytest.approx(measure_soma_scale(img).diameter_med)


def test_measure_scale_falls_back_to_dog_when_the_masks_are_bimodal():
    s = measure_scale(_planted(16), cpsam_labels=_squares((3, 3, 4, 40, 40)), profile_name="grin")
    assert s.source == "dog"


def test_measure_scale_falls_back_to_dog_when_the_masks_contradict_the_profile():
    # 50 px somata are implausible for GRIN (scale_plausible caps it at 40).
    s = measure_scale(_planted(16), cpsam_labels=_squares((50, 52, 54, 56, 58), size=400),
                      profile_name="grin")
    assert s.source == "dog"


def test_measure_scale_without_masks_or_a_dog_estimate_keeps_the_profile():
    s = measure_scale(_planted(2), cpsam_labels=None, profile_name="grin")
    assert not s.ok and s.source == "profile"


def test_measure_scale_keeps_the_profile_when_the_dog_estimate_contradicts_it():
    from roigbiv.pipeline.optics import measure_soma_scale
    img = _planted(16, radius=4)
    dog = measure_soma_scale(img)
    assert dog.ok and dog.diameter_med < 18           # sized fine (about 12 px), but too small for PRISM
    s = measure_scale(img, cpsam_labels=None, profile_name="prism")
    assert not s.ok and s.source == "profile"


def test_dog_estimate_records_its_source():
    from roigbiv.pipeline.optics import measure_soma_scale
    assert measure_soma_scale(_planted(16)).source == "dog"
    assert SomaScale().source == "dog"


# ── _apply_auto_scale provenance ──────────────────────────────────────────


def _auto_cfg(**kw):
    cfg = PipelineConfig(profile="grin", auto_scale=True, **kw)
    cfg.stage1_backend = UNION
    cfg.force_cpu = True
    return cfg


def test_apply_auto_scale_derives_from_cpsam_masks_and_records_the_source():
    from roigbiv.pipeline.run import _apply_auto_scale
    cfg = _auto_cfg(cpsam_scale_k=1.25)
    _apply_auto_scale(cfg, SimpleNamespace(mean_M=_planted(16)), cpsam_labels=_squares())
    assert cfg.auto_adapt["scale_ok"] is True
    assert cfg.auto_adapt["scale_source"] == "cpsam"
    assert cfg.auto_adapt["cpsam_scale_k"] == 1.25
    assert cfg.auto_adapt["soma_diameter_med"] == round(_d(196) / 1.25, 1)
    assert cfg.diameter == round(_d(196) / 1.25)


def test_apply_auto_scale_ignores_cpsam_masks_when_the_source_is_dog():
    from roigbiv.pipeline.run import _apply_auto_scale
    cfg = _auto_cfg(scale_source="dog")
    _apply_auto_scale(cfg, SimpleNamespace(mean_M=_planted(16)), cpsam_labels=_squares())
    assert cfg.auto_adapt["scale_source"] == "dog"
    assert "cpsam_scale_k" not in cfg.auto_adapt


def test_apply_auto_scale_records_profile_when_nothing_sizes_the_fov():
    from roigbiv.pipeline.run import _apply_auto_scale
    cfg = _auto_cfg()
    d0 = cfg.diameter
    _apply_auto_scale(cfg, SimpleNamespace(mean_M=_planted(2)), cpsam_labels=_squares(SIDES[:2]))
    assert cfg.auto_adapt["scale_ok"] is False
    assert cfg.auto_adapt["scale_source"] == "profile"
    assert cfg.diameter == d0


# ── Stage 1 order: Cellpose-SAM, scale, derive, Cellpose 3, union ─────────


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """One fake sidecar call log, Cellpose 3 recording the diameter it gets,
    and a registered movie for the max7 image."""
    seen = {"sidecar": 0}
    lab = _squares(size=256)             # same frame as _planted(16), which the DoG estimate can size

    def fake_sidecar(x, fields, cfg):
        seen["sidecar"] += 1
        seen["fields"] = fields
        return lab.copy(), np.where(lab > 0, 2.0, -3.0).astype(np.float32)

    def fake_flows(mean_S, vcorr_S, cfg, *, max_S=None):
        seen["c3_diameter"] = cfg.diameter
        return stage1.CellposeFlows(
            label_image=np.zeros(lab.shape, np.uint16), dP=np.zeros((2, *lab.shape), np.float32),
            cellprob=np.zeros(lab.shape, np.float32), niter=200, dp_scale=5.0, diameter=12.0)

    monkeypatch.setattr(stage1, "_sidecar", fake_sidecar)
    monkeypatch.setattr(stage1, "run_cellpose_flows", fake_flows)
    T, (H, W) = 14, lab.shape
    p = tmp_path / "data.bin"
    np.random.default_rng(0).integers(0, 100, (T, H, W)).astype(np.int16).tofile(p)
    fov = SimpleNamespace(mean_M=_planted(16), data_bin_path=p, shape=(T, H, W),
                          vcorr_S=np.zeros((H, W), np.float32), max_S=None)
    seen["fov"] = fov
    return seen


def test_cpsam_runs_once_before_scale_and_cellpose3_gets_the_derived_diameter(fakes, tmp_path):
    from roigbiv.pipeline.run import _auto_scale_with_cpsam
    cfg = _auto_cfg(cpsam_scale_k=1.25)
    fov = fakes["fov"]
    pre = _auto_scale_with_cpsam(cfg, fov, tmp_path)
    assert fakes["sidecar"] == 1
    assert fakes["fields"]["diameter"] is None
    assert cfg.auto_adapt["scale_source"] == "cpsam"
    s1 = stage1.run_stage1_candidates(fov.mean_M, fov.vcorr_S, cfg, cpsam_labels=pre)
    assert fakes["sidecar"] == 1                       # Stage 1 reused the masks
    assert fakes["c3_diameter"] == round(_d(196) / 1.25)
    assert s1.counts[cfg.cpsam_model] == 5             # the same masks went into the union
    assert s1.detectors == [cfg.cpsam_model] * 5


def test_no_early_cpsam_call_when_the_scale_source_is_dog(fakes, tmp_path):
    from roigbiv.pipeline.run import _auto_scale_with_cpsam
    cfg = _auto_cfg(scale_source="dog")
    assert _auto_scale_with_cpsam(cfg, fakes["fov"], tmp_path) is None
    assert fakes["sidecar"] == 0
    assert cfg.auto_adapt["scale_source"] == "dog"


def test_no_early_cpsam_call_without_the_union_backend(fakes, tmp_path):
    from roigbiv.pipeline.run import _auto_scale_with_cpsam
    cfg = _auto_cfg()
    cfg.stage1_backend = "cellpose3"
    assert _auto_scale_with_cpsam(cfg, fakes["fov"], tmp_path) is None
    assert fakes["sidecar"] == 0
    assert cfg.auto_adapt["scale_source"] == "dog"


def test_no_scale_and_no_cpsam_call_when_auto_scale_is_off(fakes, tmp_path):
    from roigbiv.pipeline.run import _auto_scale_with_cpsam
    cfg = _auto_cfg()
    cfg.auto_scale = False
    d0 = cfg.diameter
    assert _auto_scale_with_cpsam(cfg, fakes["fov"], tmp_path) is None
    assert fakes["sidecar"] == 0 and cfg.diameter == d0 and "scale_source" not in cfg.auto_adapt


def test_a_foundation_only_dry_run_sizes_from_dog_with_no_cpsam_call(fakes, tmp_path):
    """A dry run stays cheap (no GPU, no movie pass) but still records a scale;
    the --resume run re-derives it from Cellpose-SAM."""
    from roigbiv.pipeline.run import _auto_scale_with_cpsam
    cfg = _auto_cfg(foundation_only=True)
    assert _auto_scale_with_cpsam(cfg, fakes["fov"], tmp_path) is None
    assert fakes["sidecar"] == 0
    assert not (tmp_path / "summary" / "max7_M.tif").exists()
    assert cfg.auto_adapt["scale_source"] == "dog" and cfg.auto_adapt["scale_ok"] is True


def test_stage1_runs_cpsam_itself_when_no_masks_are_passed(fakes):
    cfg = _auto_cfg()
    fov = fakes["fov"]
    s1 = stage1.run_stage1_candidates(fov.mean_M, fov.vcorr_S, cfg, cpsam_image=fov.mean_M)
    assert fakes["sidecar"] == 1 and s1.counts[cfg.cpsam_model] == 5


# ── config, fingerprint, CLI ──────────────────────────────────────────────


def test_scale_settings_default_to_cpsam_first():
    cfg = PipelineConfig()
    assert cfg.scale_source == "cpsam"
    assert cfg.cpsam_scale_k == 1.05             # tuned on the dev split (size calibration, 2026-11-18)


@pytest.mark.parametrize("field,value", [("scale_source", "dog"), ("cpsam_scale_k", 1.2)])
def test_each_scale_setting_changes_the_resume_fingerprint(tmp_path, field, value):
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    tif = tmp_path / "fov.tif"
    tif.write_bytes(b"x")
    a, b = _auto_cfg(), _auto_cfg()
    setattr(b, field, value)
    assert compute_cfg_fingerprint(a, tif) != compute_cfg_fingerprint(b, tif)


def test_cli_scale_source_flag_reaches_the_config(tmp_path):
    import tifffile

    from roigbiv.pipeline import run
    tif = tmp_path / "fov.tif"
    tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))

    def cfg_for(extra, out):
        with patch("roigbiv.pipeline.run.run_pipeline") as rp:
            rp.return_value = None
            try:
                run.main(["--input", str(tif), "--fs", "7.5", "--no-viewer", "--profile", "grin",
                          "--output-dir", str(tmp_path / out), *extra])
            except Exception:
                pass
        assert rp.called
        return rp.call_args[0][1]

    assert cfg_for([], "o1").scale_source == "cpsam"
    assert cfg_for(["--scale-source", "dog"], "o2").scale_source == "dog"
    with pytest.raises(SystemExit):
        run.main(["--input", str(tif), "--fs", "7.5", "--scale-source", "nope"])


def test_workspace_cli_scale_source_flag_reaches_the_config(tmp_path):
    import tifffile

    from roigbiv.pipeline import run
    ws = tmp_path / "ws"
    ws.mkdir()
    tifffile.imwrite(str(ws / "fov.tif"), np.zeros((4, 16, 16), np.uint16))
    with patch("roigbiv.pipeline.workspace.run_with_workspace") as rw:
        rw.return_value = []
        try:
            run.main(["--input", str(ws), "--fs", "7.5", "--profile", "grin", "--scale-source", "dog"])
        except Exception:
            pass
    assert rw.called
    assert rw.call_args[0][1]["scale_source"] == "dog"


# ── review fixes ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("k", [0.0, -1.0])
def test_scale_from_labels_refuses_a_non_positive_k(k):
    with pytest.raises(ValueError, match="cpsam_scale_k"):
        scale_from_labels(_squares(), k=k)


def test_scale_settings_can_change_on_an_optics_confirmation_resume():
    from roigbiv.pipeline.resume import _CONFIRM_RESUME_FIELDS
    assert {"scale_source", "cpsam_scale_k"} <= _CONFIRM_RESUME_FIELDS


def test_a_rejected_cpsam_estimate_is_recorded_next_to_the_winner():
    from roigbiv.pipeline.run import _apply_auto_scale
    cfg = _auto_cfg()
    _apply_auto_scale(cfg, SimpleNamespace(mean_M=_planted(16)), cpsam_labels=_squares(SIDES[:4]))
    assert cfg.auto_adapt["scale_source"] == "dog"
    assert cfg.auto_adapt["cpsam_rejected"] == {"n_masks": 4, "diameter_med": None}
    cfg = _auto_cfg(cpsam_scale_k=1.0)
    _apply_auto_scale(cfg, SimpleNamespace(mean_M=_planted(16)), cpsam_labels=_squares((50, 52, 54, 56, 58), size=400))
    assert cfg.auto_adapt["cpsam_rejected"] == {"n_masks": 5, "diameter_med": round(_d(54 ** 2), 1)}
    cfg = _auto_cfg()
    _apply_auto_scale(cfg, SimpleNamespace(mean_M=_planted(16)), cpsam_labels=_squares())
    assert "cpsam_rejected" not in cfg.auto_adapt


def test_no_early_cpsam_call_when_a_low_confidence_prior_will_pause_the_run(fakes, tmp_path):
    from roigbiv.pipeline.run import _auto_scale_with_cpsam
    cfg = _auto_cfg()
    cfg.auto_adapt = {"prior": {"confidence": "low", "reasons": ["ambiguous frame size"]}}
    assert _auto_scale_with_cpsam(cfg, fakes["fov"], tmp_path) is None
    assert fakes["sidecar"] == 0 and cfg.auto_adapt["scale_source"] == "dog"
    cfg = _auto_cfg(assume_optics=True)              # no pause, so Cellpose-SAM runs
    cfg.auto_adapt = {"prior": {"confidence": "low", "reasons": []}}
    assert _auto_scale_with_cpsam(cfg, fakes["fov"], tmp_path) is not None
    assert fakes["sidecar"] == 1 and cfg.auto_adapt["scale_source"] == "cpsam"


def test_run_pipeline_hands_the_auto_scale_masks_to_stage1(tmp_path):
    """The wiring inside run_pipeline: the masks auto-scale returns reach
    run_stage1_candidates, and no second max7 image is built."""
    from roigbiv.pipeline import run
    from roigbiv.pipeline.types import FOVData

    class StopAfterStage1(Exception):
        pass

    shape = (4, 16, 16)
    fov = FOVData(raw_path=tmp_path / "fov.tif", output_dir=tmp_path, data_bin_path=tmp_path / "data.bin",
                  shape=shape, mean_M=np.zeros(shape[1:], np.float32), vcorr_S=np.zeros(shape[1:], np.float32),
                  k_background=30, rois=[])
    pre = (np.zeros(shape[1:], np.uint16), np.zeros(shape[1:], np.float32))
    seen = {}

    def fake_stage1(mean_S, vcorr_S, cfg, **kw):
        seen.update(kw)
        raise StopAfterStage1

    tif = tmp_path / "fov.tif"
    tif.write_bytes(b"x")
    cfg = _auto_cfg(fs=7.5, no_viewer=True, output_dir=tmp_path / "out")
    with patch("roigbiv.pipeline.foundation.run_foundation", return_value=fov), \
            patch.object(run, "_auto_scale_with_cpsam", return_value=pre) as early, \
            patch.object(run, "_stage1_cpsam_image", side_effect=AssertionError("second max7")), \
            patch.object(stage1, "run_stage1_candidates", side_effect=fake_stage1):
        with pytest.raises(StopAfterStage1):
            run.run_pipeline(tif, cfg)
    assert early.call_count == 1
    assert seen["cpsam_labels"] is pre and seen["cpsam_image"] is None
