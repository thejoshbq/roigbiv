"""Candidate table built from a saved run's FOV directory."""

import dataclasses
import json
import math

import numpy as np
import pytest
import tifffile

from roigbiv.workbench import table as tb

H, W = 40, 50


def _roi(label, stage, outcome, **feats):
    reasons = feats.pop("reasons", [] if outcome != "reject" else ["area:3"])
    return {
        "label_id": label,
        "source_stage": stage,
        "confidence": "moderate",
        "gate_outcome": outcome,
        "area": feats.pop("area", 50),
        "solidity": 0.9,
        "eccentricity": 0.3,
        "nuclear_shadow_score": 0.0,
        "soma_surround_contrast": 0.0,
        "cellpose_prob": None,
        "iscell_prob": None,
        "event_count": None,
        "corr_contrast": None,
        "activity_type": None,
        "gate_reasons": reasons,
        "features": feats,
    }


def _report(rois):
    return {
        "detected": len(rois),
        "accepted": sum(r["gate_outcome"] == "accept" for r in rois),
        "flagged": sum(r["gate_outcome"] == "flag" for r in rois),
        "rejected": sum(r["gate_outcome"] == "reject" for r in rois),
        "rois": rois,
    }


def _disk(cy, cx, r):
    ys, xs = np.ogrid[:H, :W]
    return ((ys - cy) ** 2 + (xs - cx) ** 2) <= r**2


def make_fov(tmp_path, *, pool=4, stamp=3, drop_stat=False):
    """A saved-run FOV dir: Stage 1 two kept + one reject, Stage 2 from stat, Stage 3 disks, Stage 4 one kept."""
    fov = tmp_path / "run" / "fovA"
    for n in (1, 2, 3, 4):
        (fov / f"stage{n}").mkdir(parents=True)
    s1 = [
        _roi(1, 1, "accept", stage1_detector="cellpose3"),
        _roi(2, 1, "flag", stage1_detector="cpsam"),
        _roi(3, 1, "reject", stage1_detector="cellpose3"),
    ]
    m1 = np.zeros((H, W), np.uint16)
    m1[_disk(10, 10, stamp)] = 1
    m1[_disk(10, 30, stamp)] = 2
    s2 = [
        _roi(4, 2, "reject", suite2p_stat_index=1, gate2_min_corr=-0.1, area=2),
        _roi(5, 2, "accept", suite2p_stat_index=0, gate2_min_corr=0.2, area=3),
    ]
    m2 = np.zeros((H, W), np.uint16)
    m2[_disk(30, 10, stamp)] = 5
    stat = np.array(
        [
            {"ypix": np.array([29, 30, 31]), "xpix": np.array([10, 10, 10]), "npix": 3},
            {"ypix": np.array([5, 5, -1]), "xpix": np.array([40, 41, 41]), "npix": 3},
        ],
        dtype=object,
    )
    pool_area = int(_disk(20, 20, pool).sum())
    s3 = [
        _roi(
            6,
            3,
            "reject",
            area=pool_area,
            centroid_y=20.0,
            centroid_x=20.0,
            gate3_best_r2=0.3,
            gate3_rise_decay_ratio=float("nan"),
            events=[{"frame": 1}],
        ),
        _roi(7, 3, "flag", area=pool_area, centroid_y=30.0, centroid_x=30.0, gate3_best_r2=0.8),
    ]
    m3 = np.zeros((H, W), np.uint16)
    m3[_disk(30, 30, stamp)] = 7
    s4 = [
        _roi(8, 4, "flag", centroid_y=20.0, centroid_x=40.0, mean_intra_corr=0.4),
        _roi(9, 4, "reject", centroid_y=2.0, centroid_x=2.0, mean_intra_corr=0.1),
    ]
    m4 = np.zeros((H, W), np.uint16)
    m4[_disk(20, 40, stamp)] = 8
    for n, rois, masks in ((1, s1, m1), (2, s2, m2), (3, s3, m3), (4, s4, m4)):
        (fov / f"stage{n}" / f"stage{n}_report.json").write_text(json.dumps(_report(rois)))
        tifffile.imwrite(fov / f"stage{n}" / f"stage{n}_masks.tif", masks)
    if not drop_stat:
        (fov / "movie/suite2p/plane0").mkdir(parents=True)
        np.save(fov / "movie/suite2p/plane0/stat.npy", stat, allow_pickle=True)
    log = {"config": {"spatial_pool_radius": pool, "roi_stamp_radius": stamp}}
    (fov / "pipeline_log.json").write_text(json.dumps(log))
    return fov


def _by_id(t):
    return {r.candidate_id: r for r in t.rows}


def test_every_candidate_is_a_row_and_counts_match_the_reports(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path), run_id="run", fov_id="fovA")
    assert len(t.rows) == 9
    assert t.problems == []
    assert t.shape == (H, W)


def test_stage1_kept_rows_use_the_persisted_stamp_and_rejects_read_not_recorded(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path), run_id="run", fov_id="fovA")
    rows = _by_id(t)
    kept = rows["fovA/s1/1"]
    assert kept.footprint == "stamp"
    assert (kept.centroid_y, kept.centroid_x) == pytest.approx((10.0, 10.0))
    assert t.footprint("fovA/s1/1").sum() == _disk(10, 10, 3).sum()
    rej = rows["fovA/s1/3"]
    assert rej.footprint == "not_recorded" and rej.centroid_y is None
    assert t.footprint("fovA/s1/3") is None
    assert rows["fovA/s1/2"].detector == "cpsam"


def test_stage2_rows_take_the_suite2p_stat_pixels_rejects_included(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path), run_id="run", fov_id="fovA")
    rows = _by_id(t)
    rej = rows["fovA/s2/4"]
    assert rej.footprint == "suite2p_stat"
    fp = t.footprint("fovA/s2/4")
    assert sorted(zip(*np.nonzero(fp))) == [(5, 40), (5, 41)]  # the out-of-frame pixel is dropped
    assert (rej.centroid_y, rej.centroid_x) == pytest.approx((5.0, 40.5))
    assert dict(rej.gates)["gate2_min_corr"] == -0.1
    assert "suite2p_stat_index" in dict(rej.gates)


def test_stage3_rows_take_the_pool_radius_disk(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path, pool=4, stamp=3), run_id="run", fov_id="fovA")
    rej = _by_id(t)["fovA/s3/6"]
    assert rej.footprint == "pool_disk"
    assert np.array_equal(t.footprint("fovA/s3/6"), _disk(20, 20, 4))


def test_stage4_kept_rows_use_the_stamp_and_rejects_keep_only_the_centroid(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path), run_id="run", fov_id="fovA")
    rows = _by_id(t)
    assert rows["fovA/s4/8"].footprint == "stamp"
    rej = rows["fovA/s4/9"]
    assert rej.footprint == "not_recorded"
    assert (rej.centroid_y, rej.centroid_x) == (2.0, 2.0)
    assert t.footprint("fovA/s4/9") is None


def test_missing_stat_file_is_a_problem_and_stage2_reads_not_recorded(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path, drop_stat=True), run_id="run", fov_id="fovA")
    assert any("stat.npy" in p for p in t.problems)
    assert _by_id(t)["fovA/s2/4"].footprint == "not_recorded"


def test_two_stat_files_are_ambiguous_not_guessed(tmp_path):
    fov = make_fov(tmp_path)
    (fov / "other/suite2p/plane0").mkdir(parents=True)
    np.save(fov / "other/suite2p/plane0/stat.npy", np.array([], dtype=object), allow_pickle=True)
    t = tb.build_fov_table(fov, run_id="run", fov_id="fovA")
    assert "stage 2: expected one */suite2p/plane0/stat.npy, found 2" in t.problems
    assert _by_id(t)["fovA/s2/4"].footprint != "suite2p_stat"


def test_a_stat_index_outside_stat_npy_is_a_problem(tmp_path):
    fov = make_fov(tmp_path)
    p = fov / "stage2/stage2_report.json"
    rep = json.loads(p.read_text())
    rep["rois"][0]["features"]["suite2p_stat_index"] = 7
    p.write_text(json.dumps(rep))
    t = tb.build_fov_table(fov, run_id="run", fov_id="fovA")
    assert "stage 2 label 4: suite2p_stat_index 7 not in stat.npy" in t.problems
    assert _by_id(t)["fovA/s2/4"].footprint == "not_recorded"


def test_a_stamped_row_keeps_the_report_centroid_over_the_stamp_centre(tmp_path):
    fov = make_fov(tmp_path)
    p = fov / "stage4/stage4_report.json"
    rep = json.loads(p.read_text())
    rep["rois"][0]["features"].update(centroid_y=20.4, centroid_x=40.3)
    p.write_text(json.dumps(rep))
    row = _by_id(tb.build_fov_table(fov, run_id="run", fov_id="fovA"))["fovA/s4/8"]
    assert (row.centroid_y, row.centroid_x) == (20.4, 40.3)


def test_a_kept_label_absent_from_the_stage_masks_is_a_problem(tmp_path):
    fov = make_fov(tmp_path)
    m = tifffile.imread(fov / "stage1/stage1_masks.tif")
    m[m == 2] = 0
    tifffile.imwrite(fov / "stage1/stage1_masks.tif", m)
    t = tb.build_fov_table(fov, run_id="run", fov_id="fovA")
    assert "stage 1 label 2 is flag but absent from stage1_masks.tif" in t.problems
    assert _by_id(t)["fovA/s1/2"].footprint == "not_recorded"


def test_a_report_whose_counts_disagree_with_its_rois_is_a_problem(tmp_path):
    fov = make_fov(tmp_path)
    p = fov / "stage3/stage3_report.json"
    rep = json.loads(p.read_text())
    rep["detected"] = 5
    p.write_text(json.dumps(rep))
    t = tb.build_fov_table(fov, run_id="run", fov_id="fovA")
    assert "stage 3 detected: table 2, report 5" in t.problems


def test_a_stage_without_a_report_is_named_not_silently_empty(tmp_path):
    fov = make_fov(tmp_path)
    (fov / "stage4/stage4_report.json").unlink()
    t = tb.build_fov_table(fov, run_id="run", fov_id="fovA")
    assert t.missing_stages == (4,)
    assert not any(r.stage == 4 for r in t.rows)


def test_write_then_read_round_trips_rows_and_footprints(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path), run_id="run", fov_id="fovA")
    out = tmp_path / "wb"
    tb.write_fov_table(t, out)
    assert (out / "candidates.parquet").exists() and (out / "footprints.npz").exists()
    back = tb.read_fov_table(out)
    assert [r.candidate_id for r in back.rows] == [r.candidate_id for r in t.rows]
    for a, b in zip(t.rows, back.rows):  # NaN != NaN, so compare gates with NaN as a value
        assert dataclasses.replace(a, gates=()) == dataclasses.replace(b, gates=())
        assert [k for k, _ in a.gates] == [k for k, _ in b.gates]
        np.testing.assert_array_equal([v for _, v in a.gates], [v for _, v in b.gates])
    assert math.isnan(dict(next(r for r in back.rows if r.label_id == 6).gates)["gate3_rise_decay_ratio"])
    assert back.shape == t.shape
    for r in t.rows:
        a, b = t.footprint(r.candidate_id), back.footprint(r.candidate_id)
        assert (a is None and b is None) or np.array_equal(a, b)


def test_a_detected_footprint_whose_size_differs_from_the_gate_area_is_a_problem(tmp_path):
    fov = make_fov(tmp_path)
    for n, label, area in ((2, 4, 50), (3, 6, 50)):
        p = fov / f"stage{n}" / f"stage{n}_report.json"
        rep = json.loads(p.read_text())
        next(r for r in rep["rois"] if r["label_id"] == label)["area"] = area
        p.write_text(json.dumps(rep))
    t = tb.build_fov_table(fov, run_id="run", fov_id="fovA")
    assert "stage 3 label 6: pool_disk has 49 px, gate area 50" in t.problems
    assert "stage 2 label 4: suite2p_stat has 2 px, gate area 50" in t.problems
    assert len(t.problems) == 2  # stamps (Stages 1 and 4, area 50 in the fixture) are not gate geometry


def test_write_refuses_a_table_with_problems_unless_told(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path, drop_stat=True), run_id="run", fov_id="fovA")
    with pytest.raises(ValueError, match="problems"):
        tb.write_fov_table(t, tmp_path / "wb")
    tb.write_fov_table(t, tmp_path / "wb", allow_problems=True)
    assert tb.read_fov_table(tmp_path / "wb").problems == t.problems


def test_rows_are_ordered_by_stage_then_label(tmp_path):
    t = tb.build_fov_table(make_fov(tmp_path), run_id="run", fov_id="fovA")
    assert [(r.stage, r.label_id) for r in t.rows] == sorted((r.stage, r.label_id) for r in t.rows)


def test_stage3_rows_without_a_pool_radius_are_a_problem(tmp_path):
    fov = make_fov(tmp_path)
    log = json.loads((fov / "pipeline_log.json").read_text())
    del log["config"]["spatial_pool_radius"]
    (fov / "pipeline_log.json").write_text(json.dumps(log))
    t = tb.build_fov_table(fov, run_id="run", fov_id="fovA")
    assert "stage 3: spatial_pool_radius not in pipeline_log config; Stage 3 footprints not rebuilt" in t.problems
    assert _by_id(t)["fovA/s3/6"].footprint == "not_recorded"
