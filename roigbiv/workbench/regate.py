"""regate: re-apply gate thresholds to a saved candidate table.

Results are labelled "frozen pool, non-causal downstream": the candidate pool is the saved one, and a changed
outcome here does not change what later stages saw (subtraction, the residual, prior-ROI correlations). Only a
real rerun answers that.

Each stage's decision is the pipeline's own rule (``roigbiv/pipeline/gate{1,2,3,4}.py``) applied to the gate
quantities the run recorded. ``REGATEABLE`` lists the settings that are pure thresholds on those quantities.
Settings that change how a quantity is measured (annulus radii, correlation search radii, waveform windows,
percentiles) are refused, since the saved value no longer answers them. Some saved values stand in for "check not
run" (a correlation of 0.0 when no prior ROI was near), so thresholds are also kept on the side where that
stand-in can never fail (``_RANGE``). Checks a run did not record (Gate 3 noise checks before they existed) are not
applied, and an override aimed at one is listed in ``RegateResult.not_applied``.

Crowding (``roi_stamp.resolve_crowding``) is replayed as the pipeline runs it: after each stage's gate, over the
kept stamps of that stage and every earlier one, with confidence ranks from each gate's own rules. A stage report
is written right after its own crowding pass, so a later pass can demote an earlier stage's accept without the
earlier report showing it. Rows keep the report's view; such later demotions are listed in
``RegateResult.later_demoted``. A newly kept candidate with no saved centroid cannot be placed and is listed in
``crowding_unplaced``.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from roigbiv.workbench.table import STAGES, FovTable

LABEL = "frozen pool, non-causal downstream"
REGATEABLE = {
    1: (
        "min_area",
        "max_area",
        "min_solidity",
        "max_eccentricity",
        "min_contrast",
        "flag_area_margin",
        "flag_solidity_margin",
        "flag_eccentricity_margin",
        "flag_contrast_margin",
    ),
    2: (
        "gate2_min_area",
        "gate2_max_area",
        "gate2_min_solidity",
        "gate2_max_eccentricity",
        "gate2_max_correlation",
        "gate2_anticorr_threshold",
        "gate2_near_corr_threshold",
        "gate2_flag_corr_threshold",
    ),
    3: (
        "gate3_min_solidity",
        "gate3_min_waveform_r2",
        "gate3_min_waveform_r2_single_event",
        "gate3_max_rise_decay_ratio",
        "gate3_anticorr_threshold",
        "gate3_max_negative_excursion_share",
        "gate3_min_activity_contrast",
    ),
    4: (
        "gate4_min_corr_contrast",
        "stage4_max_eccentricity",
        "stage4_min_solidity",
        "gate4_max_motion_corr",
        "gate4_anticorr_threshold",
    ),
}
_ALL = {k for v in REGATEABLE.values() for k in v}
# Settings that change a measured quantity: answering them needs a rerun.
_MEASUREMENT = {
    "annulus_inner_buffer",
    "annulus_outer_radius",
    "gate2_spatial_radius",
    "gate2_near_distance",
    "gate4_spatial_radius",
    "spatial_pool_radius",
    "roi_stamp_radius",
    "dog_strong_negative_percentile",
    "gate1_merge_peak_min_area",
    "gate1_merge_peak_min_separation",
    "gate3_waveform_window_tau_multiple",
    "gate3_waveform_source",
    "gate3_excursion_z",
    "gate4_min_mean_intensity_pct",
    "baseline_window_s",
    "baseline_percentile",
    "diameter",
    "tau",
    "fs",
}
# (field, test the value must pass, why): the saved stand-in value must never fail the check.
_RANGE = {
    "gate2_anticorr_threshold": (lambda v: v < 0, "< 0: a saved min correlation of 0.0 means no Stage 1 ROI near"),
    "gate3_anticorr_threshold": (lambda v: v < 0, "< 0: a saved min correlation of 0.0 means no prior ROI near"),
    "gate2_max_correlation": (lambda v: v > 0, "> 0: a saved max correlation of 0.0 means no Stage 1 ROI near"),
    "gate2_near_corr_threshold": (lambda v: v >= 0, ">= 0: a saved 0.0 means no Stage 1 ROI within near distance"),
    "gate2_flag_corr_threshold": (lambda v: v >= 0, ">= 0: a saved 0.0 means no Stage 1 ROI near"),
    "gate3_min_waveform_r2": (lambda v: v > 0, "> 0: a saved best R2 of 0 also stands for no waveform at all"),
    "gate3_min_waveform_r2_single_event": (lambda v: v > 0, "> 0, as for gate3_min_waveform_r2"),
    "gate4_max_motion_corr": (lambda v: v > 0, "> 0: a saved motion correlation of 0.0 means it was not computed"),
}


def check_overrides(overrides: dict) -> None:
    for k, v in overrides.items():
        if k in _MEASUREMENT:
            raise ValueError(f"{k} changes how a gate quantity is measured; answering it needs a rerun")
        if k not in _ALL:
            raise ValueError(f"{k} is not a gate threshold regate can apply (see REGATEABLE)")
        if k in _RANGE and not _RANGE[k][0](v):
            raise ValueError(f"{k}={v}: must be {_RANGE[k][1]} (the saved values cannot answer otherwise)")


def _gate1(g, area, cfg):
    fails = []
    if area < cfg["min_area"]:
        fails.append(("area_low", area - cfg["min_area"]))
    elif area > cfg["max_area"]:
        fails.append(("area_high", area - cfg["max_area"]))
    if g["solidity"] < cfg["min_solidity"]:
        fails.append(("solidity", cfg["min_solidity"] - g["solidity"]))
    if g["eccentricity"] > cfg["max_eccentricity"]:
        fails.append(("eccentricity", g["eccentricity"] - cfg["max_eccentricity"]))
    contrast_fails = g["soma_surround_contrast"] <= cfg["min_contrast"]
    if contrast_fails:
        fails.append(("contrast", cfg["min_contrast"] - g["soma_surround_contrast"]))
    dog_reject = g["nuclear_shadow_score"] < g["dog_thresh_strong_neg"] and contrast_fails
    margins = {
        "area_low": "flag_area_margin",
        "area_high": "flag_area_margin",
        "solidity": "flag_solidity_margin",
        "eccentricity": "flag_eccentricity_margin",
        "contrast": "flag_contrast_margin",
    }
    if dog_reject or sum(n != "contrast" for n, _ in fails) >= 2:
        outcome = "reject"
    elif not fails:
        outcome = "accept"
    elif len(fails) == 1:
        name, delta = fails[0]
        outcome = "flag" if abs(delta) <= cfg[margins[name]] else "reject"
    else:
        outcome = "reject"
    codes = [n for n, _ in fails] + (["dog_contrast_conjunction"] if dog_reject else [])
    if g.get("mask_peak_count", 0) >= 2:
        codes.append("merge_peaks")
        if outcome == "accept":
            outcome = "flag"
    return outcome, codes


def _gate2(g, area, cfg):
    fails = []
    if area < cfg["gate2_min_area"] or area > cfg["gate2_max_area"]:
        fails.append("area")
    if g["solidity"] < cfg["gate2_min_solidity"]:
        fails.append("solidity")
    if g["eccentricity"] > cfg["gate2_max_eccentricity"]:
        fails.append("eccentricity")
    max_abs = g.get("gate2_max_abs_corr", 0.0)
    if max_abs >= cfg["gate2_max_correlation"]:
        fails.append("corr")
    if g.get("gate2_min_corr", 0.0) <= cfg["gate2_anticorr_threshold"]:
        fails.append("anticorr")
    if g.get("gate2_near_duplicate_corr", 0.0) > cfg["gate2_near_corr_threshold"]:
        fails.append("near_duplicate")
    if fails:
        return "reject", fails
    return ("flag" if max_abs > cfg["gate2_flag_corr_threshold"] else "accept"), fails


def _gate3(g, area, cfg):
    fails = []
    if g["solidity"] < cfg["gate3_min_solidity"]:
        fails.append("solidity")
    n_events = int(g.get("event_count", 0))
    min_r2 = cfg["gate3_min_waveform_r2_single_event"] if n_events == 1 else cfg["gate3_min_waveform_r2"]
    best = g.get("gate3_best_r2", 0.0)
    if best < min_r2:
        fails.append("waveform_r2")
    if g.get("gate3_rise_decay_ratio", math.inf) >= cfg["gate3_max_rise_decay_ratio"]:  # None was saved for inf
        fails.append("rise_decay")
    if g.get("gate3_min_corr", 0.0) <= cfg["gate3_anticorr_threshold"]:
        fails.append("anticorr")
    if "gate3_negative_excursion_share" in g and not (
        g["gate3_negative_excursion_share"] <= cfg["gate3_max_negative_excursion_share"]
    ):
        fails.append("negative_excursions")
    if "gate3_activity_contrast" in g and not g["gate3_activity_contrast"] >= cfg["gate3_min_activity_contrast"]:
        fails.append("activity_contrast")
    if fails:
        return "reject", fails
    return ("flag" if best < min_r2 + 0.1 else "accept"), fails


def _gate4(g, area, cfg):
    fails = []
    if g.get("corr_contrast", 0.0) <= cfg["gate4_min_corr_contrast"]:
        fails.append("corr_contrast")
    if g["eccentricity"] > cfg["stage4_max_eccentricity"]:
        fails.append("eccentricity")
    if g["solidity"] < cfg["stage4_min_solidity"]:
        fails.append("solidity")
    if max(abs(g.get("motion_corr_x", 0.0)), abs(g.get("motion_corr_y", 0.0))) >= cfg["gate4_max_motion_corr"]:
        fails.append("motion")
    if "gate4_anticorr_min" in g and g["gate4_anticorr_min"] <= cfg["gate4_anticorr_threshold"]:
        fails.append("anticorr")
    if g.get("gate4_mean_intensity", 0.0) < g.get("gate4_mean_intensity_floor", 0.0):
        fails.append("intensity")
    return ("reject" if fails else "flag"), fails


_GATES = {1: _gate1, 2: _gate2, 3: _gate3, 4: _gate4}


def decide(row, cfg: dict) -> tuple[str, tuple]:
    """(outcome, reason codes) for one table row under ``cfg``, before crowding."""
    outcome, codes = _GATES[row.stage](dict(row.gates), row.area or 0, cfg)
    return outcome, tuple(codes)


_RANK = {"high": 2, "moderate": 1}  # roi_stamp._CONFIDENCE_RANK; requires_review and Gate 3's "low" rank 0
# Override -> the saved quantity it acts on, for checks some runs never recorded.
_NEEDS = {
    "gate3_max_negative_excursion_share": "gate3_negative_excursion_share",
    "gate3_min_activity_contrast": "gate3_activity_contrast",
    "gate4_anticorr_threshold": "gate4_anticorr_min",
}


def _rank(row, outcome: str) -> int:
    if outcome == "reject" or row.stage == 4:
        return 0
    if row.stage == 3:
        n = int(dict(row.gates).get("event_count", 0))
        return 2 if n >= 6 else (1 if n >= 2 else 0)
    return 2 if outcome == "accept" else 1


def _com(flat, shape):
    ys, xs = np.unravel_index(flat, shape)
    return float(ys.mean()), float(xs.mean())


def _stamp_centre(table, row, radius):
    """Centre of mass of the candidate's persisted stamp, as ``resolve_crowding`` sees it."""
    flat = table.footprints.get(row.candidate_id)
    if flat is None or len(flat) == 0:
        return None
    if row.footprint == "stamp":
        return _com(flat, table.shape)
    cy, cx = _com(flat, table.shape)  # canonicalize: a disk at the gate mask's centre of mass, clipped
    ys, xs = np.ogrid[: table.shape[0], : table.shape[1]]
    return _com(np.flatnonzero(((ys - cy) ** 2 + (xs - cx) ** 2) <= radius**2), table.shape)


def _replay_crowding(table, rows, outcomes, radius):
    """Mutates ``outcomes`` (cid -> [outcome, rank, codes]) stage by stage; returns report snapshots and more."""
    snap, unplaced = {}, []
    centre = {r.candidate_id: _stamp_centre(table, r, radius) for r in rows}
    for s in STAGES:
        active = []
        for r in rows:
            if r.stage > s or outcomes[r.candidate_id][0] == "reject":
                continue
            if centre[r.candidate_id] is None:
                if r.stage == s and r.outcome == "reject":
                    unplaced.append(r.candidate_id)
                continue
            active.append(r)
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                (ya, xa), (yb, xb) = centre[a.candidate_id], centre[b.candidate_id]
                if float(np.hypot(ya - yb, xa - xb)) >= radius:
                    continue
                oa, ob = outcomes[a.candidate_id], outcomes[b.candidate_id]
                if oa[1] == ob[1]:
                    weaker = a if (a.stage, a.label_id) > (b.stage, b.label_id) else b
                else:
                    weaker = a if oa[1] < ob[1] else b
                w = outcomes[weaker.candidate_id]
                if w[0] == "accept":
                    w[0], w[1], w[2] = "flag", 1, (*w[2], "crowded_neighbor")
        for r in rows:
            if r.stage == s:
                snap[r.candidate_id] = tuple(outcomes[r.candidate_id])
    return snap, tuple(unplaced)


@dataclass(frozen=True)
class Change:
    candidate_id: str
    before: str
    after: str


@dataclass(frozen=True)
class RegateResult:
    table: FovTable
    funnel: dict
    changed: tuple
    mismatches: tuple  # with no overrides: rows whose saved outcome the rules do not reproduce
    overrides: dict
    later_demoted: tuple = ()  # candidates a later stage's crowding pass demoted after their report was written
    crowding_unplaced: tuple = ()
    not_applied: tuple = ()
    label: str = LABEL


def _funnel(rows) -> dict:
    out = {}
    for s in STAGES:
        c = Counter(r.outcome for r in rows if r.stage == s)
        out[s] = {"detected": sum(c.values()), "accepted": c["accept"], "flagged": c["flag"], "rejected": c["reject"]}
    return out


def regate(table: FovTable, run_config: dict, overrides: dict | None = None) -> RegateResult:
    overrides = dict(overrides or {})
    check_overrides(overrides)
    cfg = {**run_config, **overrides}
    outcomes = {}
    for r in table.rows:
        outcome, codes = decide(r, cfg)
        outcomes[r.candidate_id] = [outcome, _rank(r, outcome), codes]
    snap, unplaced = _replay_crowding(table, table.rows, outcomes, cfg["roi_stamp_radius"])
    rows, changed = [], []
    for r in table.rows:
        outcome, _, codes = snap[r.candidate_id]
        if outcome != r.outcome:
            changed.append(Change(r.candidate_id, r.outcome, outcome))
        rows.append(dataclasses.replace(r, outcome=outcome, reasons=codes))
    later = tuple(cid for cid, o in outcomes.items() if o[0] != snap[cid][0])
    recorded = {k for r in table.rows for k, _ in r.gates}
    not_applied = tuple(sorted(k for k in overrides if k in _NEEDS and _NEEDS[k] not in recorded))
    return RegateResult(
        table=dataclasses.replace(table, rows=rows),
        funnel=_funnel(rows),
        changed=tuple(changed),
        mismatches=tuple(changed) if not overrides else (),
        overrides=overrides,
        later_demoted=later,
        crowding_unplaced=unplaced,
        not_applied=not_applied,
    )
