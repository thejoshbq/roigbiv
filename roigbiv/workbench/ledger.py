"""Miss ledger: per stage, how many annotated cells were proposed and kept, and where each miss went.

For each stage, two pools are matched against the annotations: every candidate the stage proposed (rejects
included) and the ones it kept (accept or flag). The cumulative pools pool all stages. Each annotated cell gets
a fate from centroid localization: ``found`` (a kept candidate was assigned to it), ``shared`` (a kept candidate
is within reach but the one-to-one assignment gave it to a neighbour), ``rejected`` (only rejected candidates were
within reach; for each stage, the nearest in-reach reject and its reason codes are listed, reach not assignment), ``chance_only`` (the only candidates in reach come from
chance-level pools) or ``never_proposed``.

A pool is readable only when its own chance baselines (both measures, both transforms) are at most
``STOP_BASELINE`` (phase T stop rule, applied per pool: Josh, 19 Nov). A rejection found through a chance-level
stage pool is listed in ``Fate.chance_level`` and never counted as a reason; such misses are judged from traces.
Candidates with no centroid in the saved run cannot be placed; they are counted in ``unplaced``, not matched.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from roigbiv.workbench.match import Annotation, Candidate, Summary, _distances, match, summarize
from roigbiv.workbench.table import STAGES, FovTable

STOP_BASELINE = 0.3


@dataclass(frozen=True)
class Pool:
    n_candidates: int
    summary: Summary

    def __getattr__(self, name):
        if name.startswith("_"):  # copy and pickle probe dunders before ``summary`` exists
            raise AttributeError(name)
        return getattr(self.summary, name)

    @property
    def max_baseline(self) -> float:
        s = self.summary
        return max(s.baseline_rotate, s.baseline_shift, s.loc_baseline_rotate, s.loc_baseline_shift)

    @property
    def readable(self) -> bool:
        return self.max_baseline <= STOP_BASELINE


@dataclass(frozen=True)
class StageLedger:
    stage: int
    proposed: Pool
    kept: Pool


@dataclass(frozen=True)
class Fate:
    ann_id: int
    status: str  # found | shared | rejected | chance_only | never_proposed
    kept_by: str | None
    rejections: tuple  # ((candidate_id, reason codes), ...) from readable stage pools, in stage order
    chance_level: tuple = ()  # stages whose pool had a candidate in reach but is chance-level


@dataclass(frozen=True)
class FovLedger:
    fov_id: str
    stages: dict
    all_proposed: Pool
    all_kept: Pool
    fates: tuple
    unplaced: dict  # stage -> candidates with no centroid

    def status_counts(self) -> dict:
        return dict(Counter(f.status for f in self.fates))

    def reason_counts(self) -> dict:
        """(stage, reason code) -> annotated cells rejected for it at that stage."""
        c: Counter = Counter()
        for f in self.fates:
            for cid, codes in f.rejections:
                stage = int(cid.rsplit("/", 2)[1][1:])
                c.update((stage, code) for code in (set(codes) or {"unspecified"}))
        return dict(c)


def _candidates(table: FovTable, rows) -> list[Candidate]:
    return [
        Candidate(r.candidate_id, (r.centroid_y, r.centroid_x), table.footprints.get(r.candidate_id))
        for r in rows
        if r.centroid_y is not None
    ]


def _pool(table, rows, anns, radius, shift) -> Pool:
    cands = _candidates(table, rows)
    return Pool(len(cands), summarize(anns, cands, radius=radius, shape=table.shape, shift=shift))


def _nearest_in_reach(anns, cands, radius) -> list:
    """For each annotation, the nearest candidate id within ``radius`` (not one-to-one), or None."""
    if not anns or not cands:
        return [None] * len(anns)
    d = _distances(anns, cands)
    j = d.argmin(axis=1)
    return [cands[k].candidate_id if d[i, k] <= radius else None for i, k in enumerate(j)]


def fov_ledger(table: FovTable, anns: Sequence[Annotation], *, radius: float, shift: tuple) -> FovLedger:
    """``shift`` is the fixed offset of the shifted chance baseline, not a registration shift."""
    rows_by = {s: [r for r in table.rows if r.stage == s] for s in STAGES}
    kept = [r for r in table.rows if r.outcome != "reject"]
    stages = {
        s: StageLedger(
            s,
            _pool(table, rows_by[s], anns, radius, shift),
            _pool(table, [r for r in rows_by[s] if r.outcome != "reject"], anns, radius, shift),
        )
        for s in STAGES
    }
    by_id = {r.candidate_id: r for r in table.rows}
    kept_links = match(anns, _candidates(table, kept), radius=radius, shape=table.shape)
    nearest = {s: _nearest_in_reach(anns, _candidates(table, rows_by[s]), radius) for s in STAGES}
    fates = []
    for i, a in enumerate(anns):
        if kept_links[i].candidate_id is not None:
            fates.append(Fate(a.ann_id, "found", kept_links[i].candidate_id, ()))
            continue
        if kept_links[i].localized:
            fates.append(Fate(a.ann_id, "shared", None, ()))
            continue
        # No kept candidate is in reach here, so every candidate a stage links to this cell is a reject.
        # Reach, not the one-to-one assignment: a reject between two cells is in reach of both.
        hits = [(s, nearest[s][i]) for s in STAGES if nearest[s][i] is not None]
        rej = tuple((cid, by_id[cid].reason_codes) for s, cid in hits if stages[s].proposed.readable)
        chance = tuple(s for s, _ in hits if not stages[s].proposed.readable)
        status = "rejected" if rej else ("chance_only" if chance else "never_proposed")
        fates.append(Fate(a.ann_id, status, None, rej, chance))
    unplaced = dict(Counter(r.stage for r in table.rows if r.centroid_y is None))
    return FovLedger(
        fov_id=table.fov_id,
        stages=stages,
        all_proposed=_pool(table, table.rows, anns, radius, shift),
        all_kept=_pool(table, kept, anns, radius, shift),
        fates=tuple(fates),
        unplaced=unplaced,
    )


def unreadable_pools(led: FovLedger) -> list[str]:
    """Names of the pools whose chance baseline is above ``STOP_BASELINE``; their numbers are not read."""
    pools = {"all_proposed": led.all_proposed, "all_kept": led.all_kept}
    for s, st in led.stages.items():
        pools[f"s{s}_proposed"], pools[f"s{s}_kept"] = st.proposed, st.kept
    return sorted(name for name, p in pools.items() if not p.readable)
