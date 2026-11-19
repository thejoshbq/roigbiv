"""Miss ledger: per stage, how many annotated cells were proposed and kept, and where each miss went."""

import numpy as np
import pytest

from roigbiv.workbench import ledger as lg
from roigbiv.workbench.match import Annotation
from roigbiv.workbench.models import CandidateRow
from roigbiv.workbench.table import FovTable

SHAPE = (60, 80)


def disk(cy, cx, r=4):
    ys, xs = np.ogrid[: SHAPE[0], : SHAPE[1]]
    return np.flatnonzero(((ys - cy) ** 2 + (xs - cx) ** 2) <= r**2)


def row(stage, label, outcome, cy, cx, reasons=(), footprint="pool_disk"):
    return CandidateRow(
        run_id="r",
        fov_id="f",
        stage=stage,
        label_id=label,
        outcome=outcome,
        detector=None,
        centroid_y=cy,
        centroid_x=cx,
        area=49,
        footprint=footprint,
        reasons=reasons,
    )


def table(rows):
    fps = {r.candidate_id: disk(r.centroid_y, r.centroid_x) for r in rows if r.footprint != "not_recorded"}
    return FovTable(run_id="r", fov_id="f", shape=SHAPE, rows=rows, footprints=fps)


ANNS = [Annotation(i, c, disk(*c)) for i, c in enumerate([(10, 10), (10, 40), (30, 20), (30, 70), (50, 40)])]
ROWS = [
    row(1, 1, "accept", 10, 10),  # cell 0 found at Stage 1
    row(1, 2, "reject", 10, 40, ("area:300",)),  # cell 1 proposed by Stage 1, rejected for area
    row(3, 3, "reject", 10, 41, ("waveform_r2:0.3<0.6", "activity_contrast:0.1")),  # and again by Stage 3
    row(3, 4, "reject", 30, 20, ("activity_contrast:0.2",)),  # cell 2 proposed only by Stage 3, rejected
    row(2, 5, "flag", 30, 71),  # cell 3 found at Stage 2, one pixel off
    # cell 4 never proposed
]


def test_per_stage_counts_for_proposed_and_kept_candidates():
    led = lg.fov_ledger(table(ROWS), ANNS, radius=5.0, shift=(17, 23))
    s1, s2, s3 = (led.stages[n] for n in (1, 2, 3))
    assert (s1.proposed.localized, s1.kept.localized) == (2, 1)
    assert (s2.proposed.localized, s2.kept.localized) == (1, 1)
    assert (s3.proposed.localized, s3.kept.localized) == (2, 0)
    assert led.stages[4].proposed.n_candidates == 0


def test_the_cumulative_pool_counts_each_cell_once():
    led = lg.fov_ledger(table(ROWS), ANNS, radius=5.0, shift=(17, 23))
    assert led.all_proposed.localized == 4
    assert led.all_kept.localized == 2
    assert led.all_kept.localization_coverage == pytest.approx(2 / 5)


def test_each_cell_is_found_rejected_with_its_stages_and_reasons_or_never_proposed():
    led = lg.fov_ledger(table(ROWS), ANNS, radius=5.0, shift=(17, 23))
    fate = {f.ann_id: f for f in led.fates}
    assert fate[0].status == "found" and fate[0].kept_by == "f/s1/1"
    assert fate[3].status == "found" and fate[3].kept_by == "f/s2/5"
    assert fate[1].status == "rejected"
    assert fate[1].rejections == (("f/s1/2", ("area",)), ("f/s3/3", ("waveform_r2", "activity_contrast")))
    assert fate[2].status == "rejected" and fate[2].rejections == (("f/s3/4", ("activity_contrast",)),)
    assert fate[4].status == "never_proposed" and fate[4].rejections == ()


def test_reason_histogram_counts_cells_not_candidates():
    led = lg.fov_ledger(table(ROWS), ANNS, radius=5.0, shift=(17, 23))
    # Cell 1 is rejected twice (Stages 1 and 3); each reason counts the cell once per stage.
    assert led.reason_counts() == {(1, "area"): 1, (3, "waveform_r2"): 1, (3, "activity_contrast"): 2}


def test_status_counts_sum_to_the_annotations():
    led = lg.fov_ledger(table(ROWS), ANNS, radius=5.0, shift=(17, 23))
    assert led.status_counts() == {"found": 2, "rejected": 2, "never_proposed": 1}


def test_a_candidate_without_a_centroid_is_left_out_and_counted():
    rows = [*ROWS, row(1, 9, "reject", None, None, ("area:3",), footprint="not_recorded")]
    led = lg.fov_ledger(table(rows), ANNS, radius=5.0, shift=(17, 23))
    assert led.unplaced == {1: 1}
    assert led.all_proposed.localized == 4
    assert led.all_proposed.n_candidates == len(ROWS)


def test_a_cell_whose_kept_candidate_went_to_a_neighbour_is_shared_not_missed():
    anns = [Annotation(0, (40, 10), disk(40, 10)), Annotation(1, (40, 16), disk(40, 16))]
    rows = [row(1, 1, "accept", 40, 13), row(1, 2, "reject", 40, 17, ("area:9",))]
    fates = {f.ann_id: f.status for f in lg.fov_ledger(table(rows), anns, radius=5.0, shift=(17, 23)).fates}
    assert sorted(fates.values()) == ["found", "shared"]


def dense_stage3(stage=3):
    return [
        row(stage, 100 + i, "reject", 2.0 + 5 * y, 2.0 + 5 * x, ("activity_contrast:0.1",))
        for i, (y, x) in enumerate(np.ndindex(12, 16))
    ]


def test_each_pool_is_readable_only_when_its_own_baseline_is_at_most_point_three():
    led = lg.fov_ledger(table(ROWS + dense_stage3()), ANNS, radius=5.0, shift=(17, 23))
    assert led.stages[1].proposed.readable and led.stages[1].kept.readable
    assert not led.stages[3].proposed.readable
    assert not led.all_proposed.readable
    assert lg.unreadable_pools(led) == ["all_proposed", "s3_proposed"]


def test_the_localization_baseline_alone_makes_a_pool_unreadable():
    t = table(dense_stage3())
    t.footprints.clear()  # no footprints: recoverable is unknown everywhere, so only localization can match
    led = lg.fov_ledger(t, ANNS, radius=5.0, shift=(17, 23))
    assert led.all_proposed.baseline == 0.0
    assert not led.stages[3].proposed.readable


def test_rejections_from_a_chance_level_pool_are_set_apart_not_counted():
    rows = [row(1, 1, "accept", 10, 10), row(1, 2, "reject", 10, 40, ("area:300",))] + dense_stage3()
    led = lg.fov_ledger(table(rows), ANNS, radius=5.0, shift=(17, 23))
    fate = {f.ann_id: f for f in led.fates}
    assert fate[0].status == "found"
    assert fate[1].status == "rejected" and fate[1].rejections == (("f/s1/2", ("area",)),)
    assert 3 in fate[1].chance_level
    others = [fate[i] for i in (2, 3, 4)]
    assert all(f.status == "chance_only" and f.rejections == () and f.chance_level == (3,) for f in others)
    assert led.reason_counts() == {(1, "area"): 1}
    assert led.status_counts() == {"found": 1, "rejected": 1, "chance_only": 3}


def test_a_rejected_candidate_in_reach_of_two_cells_counts_for_both():
    anns = [Annotation(0, (40, 10), disk(40, 10)), Annotation(1, (40, 16), disk(40, 16))]
    rows = [row(3, 1, "reject", 40, 13, ("activity_contrast:0.1",))]
    led = lg.fov_ledger(table(rows), anns, radius=5.0, shift=(17, 23))
    assert led.stages[3].proposed.readable
    fates = {f.ann_id: f for f in led.fates}
    assert [fates[i].status for i in (0, 1)] == ["rejected", "rejected"]
    assert all(fates[i].rejections == (("f/s3/1", ("activity_contrast",)),) for i in (0, 1))
    assert led.reason_counts() == {(3, "activity_contrast"): 2}


def test_the_nearest_in_reach_reject_of_a_stage_is_the_one_listed():
    anns = [Annotation(0, (40, 10), disk(40, 10))]
    rows = [row(1, 1, "reject", 40, 14, ("area:9",)), row(1, 2, "reject", 40, 11, ("eccentricity:0.9",))]
    (f,) = lg.fov_ledger(table(rows), anns, radius=5.0, shift=(17, 23)).fates
    assert f.rejections == (("f/s1/2", ("eccentricity",)),)


def test_a_reject_with_no_reason_is_counted_as_unspecified():
    anns = [Annotation(0, (40, 10), disk(40, 10))]
    led = lg.fov_ledger(table([row(1, 1, "reject", 40, 10)]), anns, radius=5.0, shift=(17, 23))
    assert led.reason_counts() == {(1, "unspecified"): 1}


def test_a_pool_can_be_copied():
    import copy

    led = lg.fov_ledger(table(ROWS), ANNS, radius=5.0, shift=(17, 23))
    p = copy.copy(led.all_kept)
    assert p.localized == led.all_kept.localized
