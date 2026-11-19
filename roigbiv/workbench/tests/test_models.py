"""Candidate table and run manifest models (experimental, schema v1)."""

import json
import math

import pytest

from roigbiv.workbench import models as m


def _row(**kw):
    base = {
        "run_id": "outH",
        "fov_id": "carl-1m",
        "stage": 3,
        "label_id": 12,
        "outcome": "reject",
        "detector": "template_sweep",
        "centroid_y": 8.1,
        "centroid_x": 8.7,
        "area": 465,
        "footprint": "pool_disk",
        "reasons": ("waveform_r2:0.315<0.6", "rise_decay:0.975"),
        "gates": (("gate3_best_r2", 0.315), ("gate3_rise_decay_ratio", math.inf)),
    }
    base.update(kw)
    return m.CandidateRow(**base)


# ---- CandidateRow ---------------------------------------------------------


def test_schema_is_versioned_and_marked_experimental():
    assert m.SCHEMA_VERSION == 1
    assert m.EXPERIMENTAL is True
    assert _row().schema_version == 1


def test_candidate_id_names_fov_stage_and_label():
    assert _row().candidate_id == "carl-1m/s3/12"


def test_reason_code_is_the_token_before_the_colon():
    assert m.reason_code("waveform_r2:0.513<0.6") == "waveform_r2"
    assert m.reason_code("area:7") == "area"
    assert m.reason_code("dog_contrast_conjunction:score=-1.2") == "dog_contrast_conjunction"
    assert m.reason_code("no_colon") == "no_colon"


def test_reason_codes_follow_reasons_in_order():
    assert _row().reason_codes == ("waveform_r2", "rise_decay")


def test_gates_are_sorted_by_name():
    row = _row(gates=(("b", 2.0), ("a", 1.0)))
    assert row.gates == (("a", 1.0), ("b", 2.0))


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"outcome": "maybe"}, "outcome"),
        ({"stage": 5}, "stage"),
        ({"stage": 0}, "stage"),
        ({"footprint": "guessed"}, "footprint"),
        ({"centroid_y": None}, "centroid"),
        ({"centroid_y": None, "footprint": "not_recorded"}, "both be set"),
        ({"gates": (("a", 1.0), ("a", 2.0))}, "duplicate gate"),
    ],
)
def test_invalid_rows_are_refused_with_the_field_named(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _row(**kw)


def test_missing_centroid_is_allowed_only_when_footprint_not_recorded():
    row = _row(centroid_y=None, centroid_x=None, footprint="not_recorded")
    assert row.centroid_y is None and row.centroid_x is None
    with pytest.raises(ValueError, match="centroid"):
        _row(centroid_y=None, centroid_x=None, footprint="stamp")


def test_record_round_trip_keeps_none_inf_and_gate_columns():
    row = _row(area=None)
    rec = row.to_record()
    assert rec["candidate_id"] == "carl-1m/s3/12"
    assert rec["gate.gate3_best_r2"] == 0.315
    assert math.isinf(rec["gate.gate3_rise_decay_ratio"])
    assert rec["reason_codes"] == ["waveform_r2", "rise_decay"]
    assert m.CandidateRow.from_record(rec) == row


def test_record_keeps_a_nan_gate_apart_from_an_absent_one():
    rec = _row(gates=(("unresolved", float("nan")),)).to_record()
    rec["gate.other_rows_gate"] = float("nan")  # a column another row filled, as in a table
    back = m.CandidateRow.from_record(rec)
    assert [k for k, _ in back.gates] == ["unresolved"]
    assert math.isnan(dict(back.gates)["unresolved"])


def test_from_record_refuses_an_unknown_schema_version():
    rec = _row().to_record()
    rec["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version 2"):
        m.CandidateRow.from_record(rec)


# ---- rows from a saved stage report ----------------------------------------


STAGE3_ROI = {
    "label_id": 12,
    "source_stage": 3,
    "confidence": "requires_review",
    "gate_outcome": "reject",
    "area": 465,
    "solidity": 0.97,
    "eccentricity": 0.37,
    "nuclear_shadow_score": 0.0,
    "soma_surround_contrast": 0.0,
    "cellpose_prob": None,
    "iscell_prob": None,
    "event_count": 24,
    "corr_contrast": None,
    "activity_type": None,
    "gate_reasons": ["waveform_r2:0.315<0.6", "activity_contrast:0.19"],
    "features": {
        "centroid_y": 8.1,
        "centroid_x": 8.7,
        "mean_event_score": 19.1,
        "n_raw_events": 2464,
        "events": [{"y": 0, "x": 7, "frame": 93, "score": 29.7}],
        "picked_events": [],
        "gate3_waveform_source": "event_patch",
        "gate3_best_r2": 0.315,
        "gate3_rise_decay_ratio": float("inf"),
        "gate3_event_r2_values": [0.1, 0.2],
    },
}

STAGE1_ROI = {
    "label_id": 1,
    "source_stage": 1,
    "confidence": "high",
    "gate_outcome": "accept",
    "area": 1426,
    "solidity": 0.96,
    "eccentricity": 0.68,
    "nuclear_shadow_score": -11.8,
    "soma_surround_contrast": 2.9,
    "cellpose_prob": 0.70,
    "iscell_prob": None,
    "event_count": None,
    "corr_contrast": None,
    "activity_type": None,
    "gate_reasons": [],
    "features": {"vcorr_mean": 0.28, "mask_peak_count": 0, "stage1_detector": "cellpose3"},
}


def test_report_roi_takes_numeric_scalars_as_gates_and_skips_lists_strings_and_none():
    row = m.row_from_report_roi(STAGE3_ROI, run_id="outH", fov_id="carl-1m", stage=3)
    gates = dict(row.gates)
    assert gates["gate3_best_r2"] == 0.315
    assert math.isinf(gates["gate3_rise_decay_ratio"])
    assert gates["event_count"] == 24.0
    assert gates["solidity"] == 0.97
    for absent in (
        "events",
        "picked_events",
        "gate3_event_r2_values",
        "gate3_waveform_source",
        "cellpose_prob",
        "label_id",
        "source_stage",
        "centroid_y",
        "area",
    ):
        assert absent not in gates
    assert row.area == 465
    assert (row.centroid_y, row.centroid_x) == (8.1, 8.7)
    assert row.footprint == "not_recorded"
    assert row.reason_codes == ("waveform_r2", "activity_contrast")
    assert row.detector == "template_sweep"


def test_report_roi_without_a_centroid_reads_not_recorded():
    row = m.row_from_report_roi(STAGE1_ROI, run_id="outH", fov_id="carl-1m", stage=1)
    assert row.footprint == "not_recorded"
    assert row.centroid_y is None
    assert row.detector == "cellpose3"
    assert "mask_peak_count" in dict(row.gates)


def test_report_roi_stage_must_match_source_stage():
    with pytest.raises(ValueError, match="source_stage 3"):
        m.row_from_report_roi(STAGE3_ROI, run_id="outH", fov_id="carl-1m", stage=2)


def test_report_roi_bools_are_not_gate_values():
    roi = json.loads(json.dumps(STAGE1_ROI))
    roi["features"]["flag_bool"] = True
    assert "flag_bool" not in dict(m.row_from_report_roi(roi, run_id="r", fov_id="f", stage=1).gates)


# ---- outcome counts against the stage report ------------------------------


def test_outcome_counts_match_report():
    rows = [_row(label_id=i, outcome=o) for i, o in enumerate(["reject", "reject", "flag"])]
    report = {3: {"detected": 3, "accepted": 0, "flagged": 1, "rejected": 2}}
    assert m.check_outcome_counts(rows, report) == []


def test_outcome_count_mismatch_is_reported_per_stage_and_field():
    rows = [_row(label_id=1, outcome="reject")]
    report = {3: {"detected": 2, "accepted": 0, "flagged": 0, "rejected": 2}, 4: {"detected": 1}}
    problems = m.check_outcome_counts(rows, report)
    assert "stage 3 detected: table 1, report 2" in problems
    assert "stage 3 rejected: table 1, report 2" in problems
    assert "stage 4 detected: table 0, report 1" in problems


# ---- RunManifest ----------------------------------------------------------

SHA = "a" * 64


def _manifest(**kw):
    base = {
        "run_id": "outH",
        "fov_id": "carl-1m",
        "source": "pipeline",
        "created_utc": "2026-11-19T14:11:22+00:00",
        "commit": "9c3a3f1" + "0" * 33,
        "dirty": False,
        "environment": (("python", "3.10.14"), ("numpy", "1.26.4")),
        "inputs": (m.FileHash(role="movie", path="fovs/carl/movie.tif", sha256=SHA, size=10),),
        "model_weights": (),
        "annotations": (),
        "effective": (("diameter", 28), ("scale_source", "cpsam")),
        "requested_config_sha256": SHA,
        "not_recorded": (),
    }
    base.update(kw)
    return m.RunManifest(**base)


def test_manifest_json_round_trip():
    man = _manifest()
    text = man.to_json()
    assert json.loads(text)["schema_version"] == 1
    assert json.loads(text)["experimental"] is True
    assert m.RunManifest.from_json(text) == man


def test_manifest_effective_values_sorted_by_name():
    man = _manifest(effective=(("z", 1), ("a", 2)))
    assert man.effective == (("a", 2), ("z", 1))


def test_pipeline_manifest_needs_commit_and_dirty_flag():
    with pytest.raises(ValueError, match="commit"):
        _manifest(commit=None)
    with pytest.raises(ValueError, match="dirty"):
        _manifest(dirty=None)


def test_backfill_manifest_lists_what_it_could_not_record():
    man = _manifest(source="backfill", commit=None, dirty=None, not_recorded=("commit", "dirty"))
    assert man.not_recorded == ("commit", "dirty")
    with pytest.raises(ValueError, match="not_recorded"):
        _manifest(source="backfill", commit=None, dirty=None, not_recorded=())


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"source": "guess"}, "source"),
        ({"requested_config_sha256": "abc"}, "sha256"),
    ],
)
def test_manifest_refuses_bad_fields(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _manifest(**kw)


def test_file_hash_refuses_a_malformed_sha256():
    with pytest.raises(ValueError, match="movie x: sha256"):
        m.FileHash(role="movie", path="x", sha256="zz", size=1)
    assert m.FileHash(role="movie", path="x", sha256=None).sha256 is None


def test_manifest_from_json_refuses_unknown_schema_version():
    d = json.loads(_manifest().to_json())
    d["schema_version"] = 9
    with pytest.raises(ValueError, match="schema_version 9"):
        m.RunManifest.from_json(json.dumps(d))
