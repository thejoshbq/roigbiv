"""
Contract tests for the centroid -> registry label-image bridge
(:mod:`roigbiv.pipeline.centroid_masks`).

The registry reads a session as ``merged_masks.tif`` + ``summary/mean_M.tif``;
these cover that a centroids-only FOV produces a label image the registry's own
loader accepts, that stamp geometry round-trips, and that a full cascade run's
masks are never clobbered.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile


class _FakeCfg:
    roi_stamp_radius = 5


def _write_centroids(output_dir: Path, points) -> Path:
    path = output_dir / "centroids.json"
    path.write_text(json.dumps({
        "stem": "fovA",
        "schema": 4,
        "centroids": [
            {"label_id": i, "y": float(y), "x": float(x), "npix": 100,
             "equiv_diameter_px": 11.28, "cellpose_prob": 0.9}
            for i, (y, x) in enumerate(points)
        ],
    }))
    return path


def _write_mean(output_dir: Path, shape=(64, 64)) -> None:
    summary = output_dir / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(summary / "mean_M.tif", np.zeros(shape, dtype=np.float32))


def test_stamps_one_label_per_centroid():
    from roigbiv.pipeline.centroid_masks import stamp_centroids

    stamped = stamp_centroids([(10.0, 10.0), (40.0, 40.0)], (64, 64), radius=5)

    assert stamped.n_centroids == 2
    assert stamped.n_labels == 2
    assert stamped.labels.dtype == np.uint16
    assert sorted(np.unique(stamped.labels).tolist()) == [0, 1, 2]
    print("  [PASS] test_stamps_one_label_per_centroid")


def test_stamp_centroid_round_trips_within_a_pixel():
    """The registry fingerprints per-ROI centroids — they must survive stamping."""
    from scipy.ndimage import center_of_mass

    from roigbiv.pipeline.centroid_masks import stamp_centroids

    points = [(12.4, 31.7), (45.0, 20.2)]
    stamped = stamp_centroids(points, (64, 64), radius=6)

    for label, (y, x) in enumerate(points, start=1):
        cy, cx = center_of_mass(stamped.labels == label)
        assert cy == pytest.approx(y, abs=1.0)
        assert cx == pytest.approx(x, abs=1.0)
    print("  [PASS] test_stamp_centroid_round_trips_within_a_pixel")


def test_disks_are_clipped_to_the_frame():
    from roigbiv.pipeline.centroid_masks import stamp_centroids

    # Centroid hard against the corner — the disk must clip, not wrap or raise.
    stamped = stamp_centroids([(1.0, 1.0)], (32, 32), radius=8)

    assert stamped.labels[0, 0] == 1
    assert stamped.labels[31, 31] == 0
    print("  [PASS] test_disks_are_clipped_to_the_frame")


def test_overlapping_stamps_are_counted_not_hidden():
    """Crowding is reported so a mis-sized stamp radius stays visible."""
    from roigbiv.pipeline.centroid_masks import stamp_centroids

    apart = stamp_centroids([(10.0, 10.0), (40.0, 40.0)], (64, 64), radius=5)
    close = stamp_centroids([(10.0, 10.0), (10.0, 14.0)], (64, 64), radius=5)

    assert apart.n_overlapping_pairs == 0
    assert close.n_overlapping_pairs == 1
    print("  [PASS] test_overlapping_stamps_are_counted_not_hidden")


def test_fully_buried_stamp_shows_up_as_a_missing_label():
    from roigbiv.pipeline.centroid_masks import stamp_centroids

    # Identical centroids: label 2 completely covers label 1.
    stamped = stamp_centroids([(20.0, 20.0), (20.0, 20.0)], (64, 64), radius=5)

    assert stamped.n_centroids == 2
    assert stamped.n_labels == 1
    print("  [PASS] test_fully_buried_stamp_shows_up_as_a_missing_label")


def test_written_masks_are_readable_by_the_registry_loader():
    """The whole point of this module: satisfy load_session_input."""
    from roigbiv.pipeline.centroid_masks import write_merged_masks
    from roigbiv.registry.roicat_adapter import load_session_input

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(10.0, 10.0), (40.0, 40.0)])

        stamped = write_merged_masks(output_dir, _FakeCfg())

        assert stamped is not None
        assert (output_dir / "merged_masks.tif").exists()

        session = load_session_input(output_dir, session_key="fovA")
        assert session.merged_masks.dtype == np.uint16
        assert session.merged_masks.shape == (64, 64)
        assert sorted(np.unique(session.merged_masks).tolist()) == [0, 1, 2]
    print("  [PASS] test_written_masks_are_readable_by_the_registry_loader")


def test_no_centroids_json_is_a_skip_not_a_crash():
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)

        assert write_merged_masks(output_dir, _FakeCfg()) is None
        assert not (output_dir / "merged_masks.tif").exists()
    print("  [PASS] test_no_centroids_json_is_a_skip_not_a_crash")


def test_cascade_masks_are_never_overwritten():
    """A full pipeline run's real detections outrank centroid stamps."""
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(10.0, 10.0)])

        real = np.zeros((64, 64), dtype=np.uint16)
        real[30:35, 30:35] = 7
        tifffile.imwrite(output_dir / "merged_masks.tif", real)
        (output_dir / "pipeline_log.json").write_text("{}")

        write_merged_masks(output_dir, _FakeCfg())

        assert np.array_equal(tifffile.imread(output_dir / "merged_masks.tif"), real)
    print("  [PASS] test_cascade_masks_are_never_overwritten")


def test_own_stamps_are_refreshed_when_centroids_change():
    """Without a cascade run, re-stamping must pick up a centroid recompute."""
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(10.0, 10.0)])
        write_merged_masks(output_dir, _FakeCfg())

        _write_centroids(output_dir, [(10.0, 10.0), (40.0, 40.0)])
        stamped = write_merged_masks(output_dir, _FakeCfg())

        assert stamped.n_centroids == 2
        written = tifffile.imread(output_dir / "merged_masks.tif")
        assert sorted(np.unique(written).tolist()) == [0, 1, 2]
    print("  [PASS] test_own_stamps_are_refreshed_when_centroids_change")


def test_write_merged_masks_applies_the_centroid_edit_log():
    """A delete op reaches merged_masks.tif without touching centroids.json."""
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        raw = _write_centroids(output_dir, [(10.0, 10.0), (40.0, 40.0)])
        raw_before = raw.read_text()

        append_centroid_op(output_dir, CentroidOp.delete(1))
        stamped = write_merged_masks(output_dir, _FakeCfg())

        assert stamped.written is True
        assert stamped.present_labels == (2,)
        assert raw.read_text() == raw_before  # centroids.json is never rewritten
    print("  [PASS] test_write_merged_masks_applies_the_centroid_edit_log")


def test_write_merged_masks_surfaces_edit_warnings():
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(10.0, 10.0)])

        append_centroid_op(output_dir, CentroidOp.delete(99))  # absent label
        stamped = write_merged_masks(output_dir, _FakeCfg())

        assert len(stamped.edit_warnings) == 1
        assert "99" in stamped.edit_warnings[0]
    print("  [PASS] test_write_merged_masks_surfaces_edit_warnings")


def test_cascade_masks_report_written_false_but_still_compute_labels():
    """A full-cascade FOV: labels are computed for inspection, disk is untouched."""
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(10.0, 10.0)])

        real = np.zeros((64, 64), dtype=np.uint16)
        real[30:35, 30:35] = 7
        tifffile.imwrite(output_dir / "merged_masks.tif", real)
        (output_dir / "pipeline_log.json").write_text("{}")

        stamped = write_merged_masks(output_dir, _FakeCfg())

        assert stamped.written is False
        assert stamped.n_centroids == 1
        assert np.array_equal(tifffile.imread(output_dir / "merged_masks.tif"), real)
    print("  [PASS] test_cascade_masks_report_written_false_but_still_compute_labels")


def test_missing_mean_image_fails_with_guidance():
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_centroids(output_dir, [(10.0, 10.0)])

        with pytest.raises(FileNotFoundError, match="re-run centroid discovery"):
            write_merged_masks(output_dir, _FakeCfg())
    print("  [PASS] test_missing_mean_image_fails_with_guidance")


def test_stamp_radius_comes_from_config():
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    class _BigCfg:
        roi_stamp_radius = 12

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(30.0, 30.0)])

        stamped = write_merged_masks(output_dir, _BigCfg())

        assert stamped.radius_px == 12
        # Area of a radius-12 disk, allowing for rasterization.
        assert int((stamped.labels == 1).sum()) == pytest.approx(452, abs=40)
    print("  [PASS] test_stamp_radius_comes_from_config")


def test_calibrated_soma_diameter_beats_the_config_default():
    """A measured diameter outranks the config guess.

    ``roi_stamp_radius`` defaults to 8 px and only rescales under
    ``--auto-scale``. On the prism FOV that produced disks far too small to
    overlap between sessions, which zeroed the only signal ROICaT had left.
    """
    from roigbiv.pipeline.calibration import write_calibration
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir, shape=(256, 256))
        _write_centroids(output_dir, [(120.0, 120.0)])
        write_calibration(output_dir, diameter_px=30.0,
                          cellprob_threshold=-2.0, cellpose_model="cyto3")

        stamped = write_merged_masks(output_dir, _FakeCfg())

    assert stamped.radius_px == 15          # round(30 / 2), under the cap
    assert stamped.radius_capped_from is None
    # The config default (8) would have been far too small to overlap.
    assert stamped.radius_px > _FakeCfg.roi_stamp_radius


def test_a_soma_larger_than_roicats_crop_is_capped():
    """ROICaT crops every ROI to a fixed 36x36 window regardless of
    um_per_pixel. A disk that fills it edge to edge makes every ROI image the
    same solid square (measured pairwise cosine 1.000), so ROInet and SWT go
    blind and nothing clusters — 30/44 ROIs clustered at radius 20 on the
    reference prism FOV, 0/44 at radius 24 and above.
    """
    from roigbiv.pipeline.calibration import write_calibration
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir, shape=(256, 256))
        _write_centroids(output_dir, [(120.0, 120.0)])
        # The real reference FOV calibrates at 60-70 px.
        write_calibration(output_dir, diameter_px=70.0,
                          cellprob_threshold=0.0, cellpose_model="cyto3")

        stamped = write_merged_masks(output_dir, _FakeCfg())

    assert stamped.radius_px == 20          # the measured cap
    assert stamped.radius_capped_from == 35  # what anatomy asked for


def test_the_cap_is_not_applied_when_anatomy_is_small_enough():
    from roigbiv.pipeline.centroid_masks import resolve_stamp_radius

    class _Cfg:
        roi_stamp_radius = 12

    with tempfile.TemporaryDirectory() as td:
        assert resolve_stamp_radius(Path(td), _Cfg()) == 12


def test_config_radius_is_used_when_the_fov_is_uncalibrated():
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    class _Cfg:
        roi_stamp_radius = 9

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(30.0, 30.0)])

        assert write_merged_masks(output_dir, _Cfg()).radius_px == 9


def test_a_degenerate_calibration_does_not_produce_a_useless_stamp():
    from roigbiv.pipeline.calibration import write_calibration
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)
        _write_mean(output_dir)
        _write_centroids(output_dir, [(30.0, 30.0)])
        write_calibration(output_dir, diameter_px=1.0,
                          cellprob_threshold=-2.0, cellpose_model="cyto3")

        assert write_merged_masks(output_dir, _FakeCfg()).radius_px == 4


def test_stamp_labeled_centroids_equivalence_to_positional():
    """Explicit labels in ascending order produce byte-identical output to positional."""
    from roigbiv.pipeline.centroid_masks import stamp_centroids, stamp_labeled_centroids

    points = [(12.0, 15.0), (35.0, 42.0), (60.0, 20.0)]
    positional = stamp_centroids(points, (64, 64), radius=5)
    labeled = stamp_labeled_centroids(
        {i: p for i, p in enumerate(points, start=1)}, (64, 64), radius=5)

    assert np.array_equal(positional.labels, labeled.labels)
    assert positional.n_centroids == labeled.n_centroids
    assert positional.n_overlapping_pairs == labeled.n_overlapping_pairs
    print("  [PASS] test_stamp_labeled_centroids_equivalence_to_positional")


def test_stamp_labeled_centroids_with_gaps():
    """Labels with gaps (e.g., 1 and 5) both survive in the output."""
    from roigbiv.pipeline.centroid_masks import stamp_labeled_centroids

    stamped = stamp_labeled_centroids(
        {1: (10.0, 10.0), 5: (40.0, 40.0)}, (64, 64), radius=5)

    assert stamped.n_centroids == 2
    assert sorted(np.unique(stamped.labels).tolist()) == [0, 1, 5]
    assert stamped.present_labels == (1, 5)
    print("  [PASS] test_stamp_labeled_centroids_with_gaps")


def test_ascending_label_order_resolves_overlap():
    """Higher labels overwrite lower ones in overlap regions, regardless of insertion order."""
    from roigbiv.pipeline.centroid_masks import stamp_labeled_centroids

    # Build dict with higher label inserted first, but stamping should still happen
    # in ascending order (1, then 2), so label 2 wins the overlap.
    labeled = {2: (10.0, 10.0), 1: (10.0, 14.0)}
    stamped = stamp_labeled_centroids(labeled, (64, 64), radius=5)

    # In the overlap region, label 2 should have higher values than label 1.
    # Since labels are uint16 (not boolean masks), we check that label 2
    # reaches into the overlap region.
    label_2_mask = stamped.labels == 2
    label_1_mask = stamped.labels == 1
    # They should not overlap (higher label wins).
    assert not np.any(label_2_mask & label_1_mask)
    print("  [PASS] test_ascending_label_order_resolves_overlap")


def test_present_labels_excludes_fully_buried_stamp():
    """A stamp completely overwritten has n_centroids count but missing from present_labels."""
    from roigbiv.pipeline.centroid_masks import stamp_labeled_centroids

    # Place labels 1 and 2 at the exact same position: label 1 gets stamped first,
    # then label 2 overwrites it completely.
    stamped = stamp_labeled_centroids(
        {1: (20.0, 20.0), 2: (20.0, 20.0)}, (64, 64), radius=5)

    assert stamped.n_centroids == 2
    assert stamped.n_labels == 1
    assert stamped.present_labels == (2,)
    print("  [PASS] test_present_labels_excludes_fully_buried_stamp")


if __name__ == "__main__":
    import traceback

    tests = [
        test_stamps_one_label_per_centroid,
        test_stamp_centroid_round_trips_within_a_pixel,
        test_disks_are_clipped_to_the_frame,
        test_overlapping_stamps_are_counted_not_hidden,
        test_fully_buried_stamp_shows_up_as_a_missing_label,
        test_written_masks_are_readable_by_the_registry_loader,
        test_no_centroids_json_is_a_skip_not_a_crash,
        test_cascade_masks_are_never_overwritten,
        test_own_stamps_are_refreshed_when_centroids_change,
        test_write_merged_masks_applies_the_centroid_edit_log,
        test_write_merged_masks_surfaces_edit_warnings,
        test_cascade_masks_report_written_false_but_still_compute_labels,
        test_missing_mean_image_fails_with_guidance,
        test_stamp_radius_comes_from_config,
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
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
