"""
Contract tests for standalone centroid discovery
(:mod:`roigbiv.pipeline.centroids`).

Cellpose is mocked out (heavy, GPU-bound, covered by Stage 1's own tests) —
these cover this module's contract: it detects on the anatomical mean image,
honors a per-FOV calibration (diameter / threshold / model), annotates each
centroid with Suite2p activity corroboration without re-running Suite2p, and
keys recompute on the resolved parameters plus a schema version.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile


def _write_summary(output_dir: Path, shape=(64, 64)) -> None:
    """Foundation's anatomical summary images."""
    summary = output_dir / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    mean_m = np.zeros(shape, dtype=np.float32)
    mean_m[5:35, 5:35] = 200.0   # the only "tissue" in this FOV
    tifffile.imwrite(summary / "mean_M.tif", mean_m)
    tifffile.imwrite(summary / "vcorr_S.tif", np.full(shape, 0.3, dtype=np.float32))
    tifffile.imwrite(summary / "max_S.tif", np.full(shape, 900.0, dtype=np.float32))


def _two_masks(shape=(64, 64)):
    """Two disjoint square masks matching _write_summary's bright blocks."""
    a = np.zeros(shape, dtype=bool)
    a[10:20, 10:20] = True
    b = np.zeros(shape, dtype=bool)
    b[40:50, 40:50] = True
    return [a, b], [0.91, 0.62]


def _fake_cellpose(masks, probs):
    def _run(morph, ch2, cfg, *, max_S=None):
        _fake_cellpose.seen_cfg = cfg
        _fake_cellpose.seen_morph = morph
        label = np.zeros(morph.shape, dtype=np.uint16)
        for i, m in enumerate(masks, start=1):
            label[m] = i
        return masks, probs, label, np.zeros(morph.shape, dtype=np.float32)
    return _run


def _write_suite2p_stat(output_dir: Path, stem: str, points) -> None:
    plane = output_dir / stem / "suite2p" / "plane0"
    plane.mkdir(parents=True, exist_ok=True)
    stat = np.array([
        {"ypix": np.array([int(y), int(y)]), "xpix": np.array([int(x), int(x) + 1])}
        for y, x in points
    ], dtype=object)
    np.save(str(plane / "stat.npy"), stat, allow_pickle=True)


class _FakeCfg:
    fs = 7.5
    tau = 1.0
    diameter = 12
    diameter_auto = True
    cellprob_threshold = -2.0
    cellpose_model = "models/deployed/current_model"
    centroid_tissue_mask = False
    centroid_tissue_mask_sigma = 8.0


def test_detects_on_mean_m_and_writes_expected_schema():
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            result = run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        payload = json.loads(result.output_path.read_text())
        assert payload["source"] == "cellpose"
        assert payload["schema"] == 3
        assert result.count == 2

        c0 = payload["centroids"][0]
        assert c0["y"] == pytest.approx(14.5)
        assert c0["x"] == pytest.approx(14.5)
        assert c0["npix"] == 100
        assert c0["cellpose_prob"] == pytest.approx(0.91)
        # 100 px square -> equivalent-circle diameter 2*sqrt(100/pi)
        assert c0["equiv_diameter_px"] == pytest.approx(11.28, abs=0.01)

        # mean_M is the morphological channel, not the ~0 L+S residual mean_S.
        assert float(_fake_cellpose.seen_morph.max()) == pytest.approx(200.0)
    print("  [PASS] test_detects_on_mean_m_and_writes_expected_schema")


def test_calibration_overrides_diameter_threshold_and_model():
    """Each calibrated field reaches Cellpose as a real inference control."""
    from roigbiv.pipeline.calibration import write_calibration
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        write_calibration(output_dir, 45.0, cellprob_threshold=-1.0,
                          cellpose_model="cyto3")
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            result = run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        cfg_used = _fake_cellpose.seen_cfg
        assert cfg_used.diameter == 45
        assert cfg_used.diameter_auto is False, (
            "an explicit measurement must beat the per-image estimator")
        assert cfg_used.cellprob_threshold == pytest.approx(-1.0)
        assert cfg_used.cellpose_model == "cyto3"

        params = json.loads(result.output_path.read_text())["params"]
        assert params["diameter_px"] == pytest.approx(45.0)
        assert params["cellpose_model"] == "cyto3"
    print("  [PASS] test_calibration_overrides_diameter_threshold_and_model")


def test_detection_substrate_is_pinned_single_channel_undenoised():
    """Centroid discovery pins its own Cellpose input handling.

    All three were measured on the reference prism FOV and each one alone is
    load-bearing: Stage 1's 2-channel convention (vcorr_S/max_S as the "nuclear"
    channel) took detection from 8 somata to 0-1, per-tile normalization
    stretches dark cell-free tiles to parity with tissue, and denoise_cyto3
    erases real structure on shot-noise-dominated data (8 without, 5 with).
    Stage 1's own settings are deliberately unchanged.
    """
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        cfg_used = _fake_cellpose.seen_cfg
        assert tuple(cfg_used.channels) == (0, 0)
        assert cfg_used.tile_norm_blocksize == 0
        assert cfg_used.use_denoise is False
    print("  [PASS] test_detection_substrate_is_pinned_single_channel_undenoised")


def test_uncalibrated_leaves_diameter_estimation_alone():
    """No calibration must not pin inference to cfg.diameter's generic default.

    cfg.diameter defaults to 12 px while these somata are 40-80 px, so silently
    disabling diameter_auto here would hand Cellpose the wrong scale.
    """
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            result = run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        cfg_used = _fake_cellpose.seen_cfg
        assert cfg_used.diameter == 12
        assert cfg_used.diameter_auto is True, "cfg's own estimator must survive"
        assert json.loads(result.output_path.read_text())["params"]["diameter_px"] is None
    print("  [PASS] test_uncalibrated_leaves_diameter_estimation_alone")


def test_caller_config_is_not_mutated():
    from roigbiv.pipeline.centroids import run_centroid_discovery
    from roigbiv.pipeline.calibration import write_calibration

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        write_calibration(output_dir, 45.0, cellpose_model="cyto3")
        cfg = _FakeCfg()
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            run_centroid_discovery(mc_tif, output_dir, cfg)

        assert cfg.diameter == 12
        assert cfg.cellpose_model == "models/deployed/current_model"
    print("  [PASS] test_caller_config_is_not_mutated")


def test_activity_cross_check_annotates_without_rerunning_suite2p():
    """Suite2p is read, never re-run, and only annotates — never filters."""
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        # One Suite2p candidate on the first mask, none near the second.
        _write_suite2p_stat(output_dir, "fovA", [(14, 14)])
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)), \
             patch("roigbiv.suite2p.run_suite2p_fov") as mock_runner:
            result = run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        assert mock_runner.call_count == 0
        payload = json.loads(result.output_path.read_text())
        assert result.count == 2, "cross-check annotates, it must not drop ROIs"
        assert payload["centroids"][0]["activity_support"] is True
        assert payload["centroids"][1]["activity_support"] is False
        assert payload["activity_cross_check"]["available"] is True
        assert payload["activity_cross_check"]["n_supported"] == 1
    print("  [PASS] test_activity_cross_check_annotates_without_rerunning_suite2p")


def test_missing_suite2p_output_means_no_cross_check_not_failure():
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            result = run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        payload = json.loads(result.output_path.read_text())
        assert result.count == 2
        assert payload["activity_cross_check"]["available"] is False
        assert "activity_support" not in payload["centroids"][0]
    print("  [PASS] test_missing_suite2p_output_means_no_cross_check_not_failure")


def test_no_summary_and_no_stack_fails_fast_with_guidance():
    """With neither summary images nor a usable stack there is nothing to detect
    on — say so, rather than surfacing a decoder error from deeper down."""
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()  # present but empty

        with pytest.raises(FileNotFoundError, match="run motion correction"):
            run_centroid_discovery(mc_tif, output_dir, _FakeCfg())
    print("  [PASS] test_no_summary_and_no_stack_fails_fast_with_guidance")


def test_falls_back_to_a_mean_projection_without_foundation_summaries():
    """Centroids-only on a pre-corrected stack Foundation never saw still works.

    That mode resolves its own ``_mc.tif`` and skips run_pipeline entirely
    (workspace._run_centroids_only), so there is no ``summary/`` to read — the
    anatomical image has to come from the stack itself.
    """
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        stack = np.zeros((12, 64, 64), dtype=np.uint16)
        stack[:, 5:35, 5:35] = 200
        tifffile.imwrite(mc_tif, stack)
        assert not (output_dir / "summary").exists()
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            result = run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        assert result.count == 2
        # The mean projection, not a zero array or the raw first frame.
        assert float(_fake_cellpose.seen_morph.max()) == pytest.approx(200.0)
        assert _fake_cellpose.seen_morph.shape == (64, 64)
    print("  [PASS] test_falls_back_to_a_mean_projection_without_foundation_summaries")


def test_resumes_on_unchanged_params_and_recomputes_on_change():
    from roigbiv.pipeline.calibration import write_calibration
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)) as mock_cp:
            run_centroid_discovery(mc_tif, output_dir, _FakeCfg())
            run_centroid_discovery(mc_tif, output_dir, _FakeCfg())
            assert mock_cp.call_count == 1, "identical params must reuse"

            write_calibration(output_dir, 45.0)
            run_centroid_discovery(mc_tif, output_dir, _FakeCfg())
            assert mock_cp.call_count == 2, "a new calibration must recompute"
    print("  [PASS] test_resumes_on_unchanged_params_and_recomputes_on_change")


def test_stale_schema_forces_recompute():
    """An artifact from an older schema must not be reused on matching params.

    Caught end-to-end during the Suite2p era: the pre-fix centroids.json
    recorded the same params the fixed code resolved to, so a params-only key
    silently handed back the broken result instead of re-detecting.
    """
    from roigbiv.pipeline.centroids import run_centroid_discovery

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)) as mock_cp:
            run_centroid_discovery(mc_tif, output_dir, _FakeCfg())
            payload = json.loads((output_dir / "centroids.json").read_text())
            payload["schema"] = 1
            (output_dir / "centroids.json").write_text(json.dumps(payload))
            run_centroid_discovery(mc_tif, output_dir, _FakeCfg())

        assert mock_cp.call_count == 2
    print("  [PASS] test_stale_schema_forces_recompute")


def test_tissue_mask_is_opt_in():
    """Default off; when enabled it drops candidates outside the tissue."""
    from roigbiv.pipeline.centroids import run_centroid_discovery

    class _MaskCfg(_FakeCfg):
        centroid_tissue_mask = True
        centroid_tissue_mask_sigma = 2.0  # 64 px fixture; 8.0 is the real-FOV default

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        mc_tif = output_dir / "fovA_mc.tif"
        mc_tif.touch()
        _write_summary(output_dir)
        # _two_masks puts the second mask in the dark quadrant, outside tissue.
        masks, probs = _two_masks()

        with patch("roigbiv.pipeline.stage1.run_cellpose_detection",
                   side_effect=_fake_cellpose(masks, probs)):
            off = run_centroid_discovery(mc_tif, output_dir, _FakeCfg())
            (output_dir / "centroids.json").unlink()
            on = run_centroid_discovery(mc_tif, output_dir, _MaskCfg())

        assert off.count == 2, "mask must be off by default"
        assert on.count == 1
        payload = json.loads(on.output_path.read_text())
        assert payload["n_detected"] == 2
        assert payload["n_outside_tissue"] == 1
        assert payload["tissue_mask"]["applied"] is True
    print("  [PASS] test_tissue_mask_is_opt_in")


def test_clear_centroid_output_leaves_foundation_suite2p_intact():
    """Suite2p output is Foundation's; centroid discovery only reads it."""
    from roigbiv.pipeline.centroids import clear_centroid_output

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_suite2p_stat(output_dir, "fovA", [(1, 1)])
        (output_dir / "centroids.json").write_text("{}")

        clear_centroid_output(output_dir, "fovA")

        assert not (output_dir / "centroids.json").exists()
        assert (output_dir / "fovA" / "suite2p" / "plane0" / "stat.npy").exists()
    print("  [PASS] test_clear_centroid_output_leaves_foundation_suite2p_intact")


def test_clear_centroid_output_no_op_when_nothing_to_clear():
    from roigbiv.pipeline.centroids import clear_centroid_output

    with tempfile.TemporaryDirectory() as td:
        clear_centroid_output(Path(td), "fovA")  # must not raise
    print("  [PASS] test_clear_centroid_output_no_op_when_nothing_to_clear")


if __name__ == "__main__":
    import traceback

    tests = [
        test_detects_on_mean_m_and_writes_expected_schema,
        test_calibration_overrides_diameter_threshold_and_model,
        test_detection_substrate_is_pinned_single_channel_undenoised,
        test_uncalibrated_leaves_diameter_estimation_alone,
        test_caller_config_is_not_mutated,
        test_activity_cross_check_annotates_without_rerunning_suite2p,
        test_missing_suite2p_output_means_no_cross_check_not_failure,
        test_no_summary_and_no_stack_fails_fast_with_guidance,
        test_falls_back_to_a_mean_projection_without_foundation_summaries,
        test_resumes_on_unchanged_params_and_recomputes_on_change,
        test_stale_schema_forces_recompute,
        test_tissue_mask_is_opt_in,
        test_clear_centroid_output_leaves_foundation_suite2p_intact,
        test_clear_centroid_output_no_op_when_nothing_to_clear,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            traceback.print_exc()
            failed.append(test.__name__)
    print()
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")
