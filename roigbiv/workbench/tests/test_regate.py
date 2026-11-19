"""regate: re-apply gate thresholds to a saved candidate table (frozen pool, non-causal downstream)."""

import math

import pytest

from roigbiv.workbench import regate as rg
from roigbiv.workbench.models import CandidateRow
from roigbiv.workbench.table import FovTable

CFG = {
    "min_area": 200,
    "max_area": 1800,
    "min_solidity": 0.55,
    "max_eccentricity": 0.9,
    "min_contrast": 0.1,
    "flag_area_margin": 100,
    "flag_solidity_margin": 0.05,
    "flag_eccentricity_margin": 0.03,
    "flag_contrast_margin": 0.03,
    "gate2_min_area": 300,
    "gate2_max_area": 2000,
    "gate2_min_solidity": 0.4,
    "gate2_max_eccentricity": 0.85,
    "gate2_max_correlation": 0.7,
    "gate2_anticorr_threshold": -0.5,
    "gate2_near_corr_threshold": 0.5,
    "gate2_flag_corr_threshold": 0.5,
    "gate3_min_solidity": 0.5,
    "gate3_min_waveform_r2": 0.6,
    "gate3_min_waveform_r2_single_event": 0.5,
    "gate3_max_rise_decay_ratio": 0.5,
    "gate3_anticorr_threshold": -0.5,
    "gate3_max_negative_excursion_share": 0.2,
    "gate3_min_activity_contrast": 0.5,
    "gate4_min_corr_contrast": 0.1,
    "stage4_max_eccentricity": 0.85,
    "stage4_min_solidity": 0.6,
    "gate4_max_motion_corr": 0.3,
    "gate4_anticorr_threshold": -0.5,
    "annulus_outer_radius": 35,
    "roi_stamp_radius": 14,
    "spatial_pool_radius": 14,
}


def row(stage, label, outcome, area=500, reasons=(), **gates):
    return CandidateRow(
        run_id="r",
        fov_id="f",
        stage=stage,
        label_id=label,
        outcome=outcome,
        detector=None,
        centroid_y=10.0,
        centroid_x=10.0,
        area=area,
        footprint="stamp",
        reasons=tuple(reasons),
        gates=tuple(gates.items()),
    )


def s1(label=1, outcome="accept", area=500, **kw):
    g = {
        "solidity": 0.9,
        "eccentricity": 0.5,
        "soma_surround_contrast": 0.5,
        "nuclear_shadow_score": 0.0,
        "dog_thresh_strong_neg": -5.0,
        "mask_peak_count": 0,
    }
    g.update(kw)
    return row(1, label, outcome, area, **g)


def s3(label=1, **kw):
    g = {
        "solidity": 0.9,
        "event_count": 4,
        "gate3_best_r2": 0.8,
        "gate3_rise_decay_ratio": 0.2,
        "gate3_min_corr": 0.0,
        "gate3_negative_excursion_share": 0.0,
        "gate3_activity_contrast": 1.0,
    }
    g.update(kw)
    return row(3, label, "accept", **g)


def one(r, **over):
    return rg.decide(r, {**CFG, **over})


# ---- Gate 1 -----------------------------------------------------------------


def test_gate1_clean_candidate_is_accepted():
    assert one(s1()) == ("accept", ())


def test_gate1_one_failure_within_margin_flags_and_beyond_rejects():
    assert one(s1(area=150)) == ("flag", ("area_low",))
    assert one(s1(area=99)) == ("reject", ("area_low",))
    assert one(s1(solidity=0.52)) == ("flag", ("solidity",))
    assert one(s1(soma_surround_contrast=0.08)) == ("flag", ("contrast",))
    assert one(s1(soma_surround_contrast=0.0)) == ("reject", ("contrast",))


def test_gate1_two_shape_failures_reject_even_within_margin():
    assert one(s1(area=150, solidity=0.52)) == ("reject", ("area_low", "solidity"))


def test_gate1_contrast_plus_one_shape_failure_rejects():
    assert one(s1(area=150, soma_surround_contrast=0.08))[0] == "reject"


def test_gate1_dog_and_contrast_conjunction_rejects():
    out = one(s1(nuclear_shadow_score=-6.0, soma_surround_contrast=0.08))
    assert out == ("reject", ("contrast", "dog_contrast_conjunction"))
    assert one(s1(nuclear_shadow_score=-6.0)) == ("accept", ())  # a strong DoG alone, with good contrast, passes


def test_gate1_a_failure_exactly_at_its_margin_flags():
    assert one(s1(area=100)) == ("flag", ("area_low",))  # 100 below min_area, margin 100


def test_gate1_merge_peaks_demote_an_accept_to_flag():
    assert one(s1(mask_peak_count=2)) == ("flag", ("merge_peaks",))
    assert one(s1(mask_peak_count=2, area=99))[0] == "reject"


def test_gate1_boundaries_match_the_pipeline_comparisons():
    assert one(s1(area=200))[0] == "accept"  # area < min_area fails, so min_area itself passes
    assert one(s1(soma_surround_contrast=0.1))[0] == "flag"  # contrast <= min_contrast fails


# ---- Gate 2 -----------------------------------------------------------------


def s2(**kw):
    g = {
        "solidity": 0.9,
        "eccentricity": 0.5,
        "gate2_max_abs_corr": 0.1,
        "gate2_min_corr": 0.0,
        "gate2_near_duplicate_corr": 0.0,
    }
    g.update(kw)
    return row(2, 1, "accept", kw.pop("area", 500), **g)


def test_gate2_rules():
    assert one(s2()) == ("accept", ())
    assert one(s2(gate2_max_abs_corr=0.6)) == ("flag", ())
    assert one(s2(gate2_max_abs_corr=0.7)) == ("reject", ("corr",))
    assert one(s2(gate2_min_corr=-0.5)) == ("reject", ("anticorr",))
    assert one(s2(gate2_near_duplicate_corr=0.51)) == ("reject", ("near_duplicate",))
    assert one(s2(eccentricity=0.86)) == ("reject", ("eccentricity",))
    assert one(row(2, 1, "accept", 250, solidity=0.9, eccentricity=0.5))[0] == "reject"


# ---- Gate 3 -----------------------------------------------------------------


def test_gate3_rules():
    assert one(s3()) == ("accept", ())
    assert one(s3(gate3_best_r2=0.65)) == ("flag", ())  # within 0.1 above the threshold
    assert one(s3(gate3_best_r2=0.55)) == ("reject", ("waveform_r2",))
    assert one(s3(gate3_best_r2=0.55, event_count=1)) == ("flag", ())  # single-event threshold 0.5
    assert one(s3(gate3_rise_decay_ratio=0.5)) == ("reject", ("rise_decay",))
    assert one(s3(gate3_min_corr=-0.5)) == ("reject", ("anticorr",))
    assert one(s3(gate3_negative_excursion_share=0.3)) == ("reject", ("negative_excursions",))
    assert one(s3(gate3_activity_contrast=0.4)) == ("reject", ("activity_contrast",))
    assert one(s3(solidity=0.4)) == ("reject", ("solidity",))


def test_gate3_unresolved_rise_decay_fails_and_nan_noise_measures_fail():
    r = s3()
    no_rd = CandidateRow(**{**r.__dict__, "gates": tuple((k, v) for k, v in r.gates if k != "gate3_rise_decay_ratio")})
    assert one(no_rd) == ("reject", ("rise_decay",))  # saved as None when the ratio was infinite
    assert one(s3(gate3_negative_excursion_share=math.nan))[1] == ("negative_excursions",)
    assert one(s3(gate3_activity_contrast=math.nan))[1] == ("activity_contrast",)


def test_gate3_checks_a_run_did_not_record_are_not_applied():
    r = s3()
    old = CandidateRow(
        **{
            **r.__dict__,
            "gates": tuple(
                (k, v) for k, v in r.gates if k not in ("gate3_negative_excursion_share", "gate3_activity_contrast")
            ),
        }
    )
    assert one(old, gate3_min_activity_contrast=99.0) == ("accept", ())


# ---- Gate 4 -----------------------------------------------------------------


def s4(**kw):
    g = {
        "corr_contrast": 0.3,
        "eccentricity": 0.5,
        "solidity": 0.9,
        "motion_corr_x": 0.0,
        "motion_corr_y": 0.0,
        "gate4_mean_intensity": 100.0,
        "gate4_mean_intensity_floor": 50.0,
    }
    g.update(kw)
    return row(4, 1, "flag", **g)


def test_gate4_rules_have_no_accept_tier():
    assert one(s4()) == ("flag", ())
    assert one(s4(corr_contrast=0.1)) == ("reject", ("corr_contrast",))
    assert one(s4(motion_corr_y=-0.3)) == ("reject", ("motion",))
    assert one(s4(gate4_anticorr_min=-0.5)) == ("reject", ("anticorr",))
    assert one(s4(gate4_mean_intensity=40.0)) == ("reject", ("intensity",))


# ---- crowding, profile checks, funnel ---------------------------------------


def at(r, cy, cx):
    return CandidateRow(**{**r.__dict__, "centroid_y": float(cy), "centroid_x": float(cx), "footprint": "pool_disk"})


CROWD = {**CFG, "roi_stamp_radius": 6}


def stamped(rows, shape=(60, 60)):
    import numpy as np

    ys, xs = np.ogrid[: shape[0], : shape[1]]
    fps = {r.candidate_id: np.flatnonzero(((ys - r.centroid_y) ** 2 + (xs - r.centroid_x) ** 2) <= 36) for r in rows}
    return FovTable(run_id="r", fov_id="f", shape=shape, rows=list(rows), footprints=fps)


def test_crowding_is_recomputed_for_the_new_kept_set():
    a = at(s1(1, "accept"), 20, 20)
    b = at(s1(2, "reject", area=150), 20, 24)  # 4 px away, inside the stamp radius
    res = rg.regate(stamped([a, b]), CROWD, {"flag_area_margin": 200})
    out = {r.label_id: (r.outcome, r.reason_codes) for r in res.table.rows}
    # b becomes a flag (moderate), a stays high, so the pair is crowded but a outranks b: no demotion.
    assert out == {1: ("accept", ()), 2: ("flag", ("area_low",))}
    res = rg.regate(stamped([a, at(s1(2, "reject", area=99), 20, 24)]), CROWD, {"min_area": 50})
    out = {r.label_id: (r.outcome, r.reason_codes) for r in res.table.rows}
    # Both accept (both high): the tie goes to the earlier label, so label 2 is demoted.
    assert out == {1: ("accept", ()), 2: ("flag", ("crowded_neighbor",))}


def test_a_recorded_demotion_lapses_when_the_neighbour_is_rejected():
    a = at(s1(1, "accept"), 20, 20)
    b = at(s1(2, "flag", reasons=("crowded_neighbor:dist=4.0px<radius=6",)), 20, 24)
    assert {r.label_id: r.outcome for r in rg.regate(stamped([a, b]), CROWD).table.rows} == {1: "accept", 2: "flag"}
    res = rg.regate(stamped([at(s1(1, "accept", area=150), 20, 20), b]), CROWD, {"flag_area_margin": 0})
    assert {r.label_id: r.outcome for r in res.table.rows} == {1: "reject", 2: "accept"}


def test_a_later_stage_crowds_an_earlier_one_but_the_earlier_report_is_not_rewritten():
    a = at(s1(1, "accept"), 20, 20)
    c = at(s3(5), 20, 23)  # Stage 3 accept (high confidence, 4 events -> moderate): weaker, so it is demoted
    res = rg.regate(stamped([a, c]), CROWD)
    out = {r.label_id: (r.outcome, r.reason_codes) for r in res.table.rows}
    assert out == {1: ("accept", ()), 5: ("flag", ("crowded_neighbor",))}


def test_a_newly_kept_candidate_with_no_centroid_is_counted_as_unplaced_for_crowding():
    r = CandidateRow(
        run_id="r",
        fov_id="f",
        stage=1,
        label_id=9,
        outcome="reject",
        detector=None,
        centroid_y=None,
        centroid_x=None,
        area=99,
        footprint="not_recorded",
        gates=s1().gates,
    )
    res = rg.regate(
        stamped([at(s1(1, "accept"), 20, 20)]) if False else FovTable(run_id="r", fov_id="f", shape=(60, 60), rows=[r]),
        CROWD,
        {"min_area": 50},
    )
    assert res.crowding_unplaced == ("f/s1/9",)


def test_overrides_that_change_how_a_quantity_was_measured_are_refused():
    with pytest.raises(ValueError, match="annulus_outer_radius.*rerun"):
        rg.check_overrides({"annulus_outer_radius": 20})
    for k in ("gate4_min_mean_intensity_pct", "baseline_window_s", "baseline_percentile"):
        with pytest.raises(ValueError, match="rerun"):
            rg.check_overrides({k: 1})
    with pytest.raises(ValueError, match="not a gate threshold"):
        rg.check_overrides({"no_such_field": 1})


@pytest.mark.parametrize(
    "over",
    [
        {"gate2_anticorr_threshold": 0.0},
        {"gate3_anticorr_threshold": 0.1},
        {"gate2_near_corr_threshold": -0.1},
        {"gate2_flag_corr_threshold": -0.1},
        {"gate3_min_waveform_r2": 0.0},
        {"gate4_max_motion_corr": 0.0},
        {"gate2_max_correlation": 0.0},
    ],
)
def test_overrides_outside_the_range_the_saved_values_can_answer_are_refused(over):
    with pytest.raises(ValueError, match="saved"):
        rg.check_overrides(over)


def table(rows):
    return FovTable(run_id="r", fov_id="f", shape=(40, 40), rows=rows)


def test_the_run_profile_reproduces_every_saved_outcome():
    rows = [
        s1(1, "accept"),
        s1(2, "flag", area=150, reasons=("area_low:-50.000",)),
        s1(3, "reject", area=99, reasons=("area_low:-101.000",)),
    ]
    res = rg.regate(table(rows), CFG)
    assert res.mismatches == ()
    assert res.funnel[1] == {"detected": 3, "accepted": 1, "flagged": 1, "rejected": 1}
    assert res.label == "frozen pool, non-causal downstream"


def test_a_changed_profile_reports_the_changed_rows_and_funnel():
    rows = [s1(1, "accept"), s1(2, "flag", area=150), s1(3, "reject", area=99)]
    res = rg.regate(table(rows), CFG, {"min_area": 100})
    assert res.funnel[1] == {"detected": 3, "accepted": 2, "flagged": 1, "rejected": 0}
    assert {c.candidate_id: (c.before, c.after) for c in res.changed} == {
        "f/s1/2": ("flag", "accept"),
        "f/s1/3": ("reject", "flag"),
    }
    assert res.overrides == {"min_area": 100}


def test_regate_rows_keep_geometry_and_carry_the_new_outcome():
    res = rg.regate(table([s1(3, "reject", area=99)]), CFG, {"min_area": 100})
    (r,) = res.table.rows
    assert r.outcome == "flag" and r.reason_codes == ("area_low",) and r.area == 99


def test_an_override_for_a_check_no_row_recorded_is_listed_as_not_applied():
    r = s3()
    old = CandidateRow(
        **{
            **r.__dict__,
            "gates": tuple(
                (k, v) for k, v in r.gates if k not in ("gate3_negative_excursion_share", "gate3_activity_contrast")
            ),
        }
    )
    res = rg.regate(table([old]), CFG, {"gate3_min_activity_contrast": 2.0})
    assert res.not_applied == ("gate3_min_activity_contrast",)


def test_a_pair_exactly_one_stamp_radius_apart_is_not_crowded():
    a, b = at(s1(1, "accept"), 20, 20), at(s1(2, "accept"), 20, 26)  # 6 px apart, radius 6
    assert {r.label_id: r.outcome for r in rg.regate(stamped([a, b]), CROWD).table.rows} == {1: "accept", 2: "accept"}


def test_stage3_rank_follows_its_event_count():
    f = at(s1(1, "flag", area=150), 20, 20)  # Stage 1 flag: moderate
    few, many = at(s3(5, event_count=3), 20, 23), at(s3(5, event_count=6), 20, 23)
    # 3 events is moderate too: a tie, so the later stage loses its accept.
    assert {r.label_id: r.outcome for r in rg.regate(stamped([f, few]), CROWD).table.rows}[5] == "flag"
    # 6 events is high: it outranks the Stage 1 flag and keeps its accept.
    assert {r.label_id: r.outcome for r in rg.regate(stamped([f, many]), CROWD).table.rows}[5] == "accept"
