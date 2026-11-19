"""Candidate table rows and run manifests (experimental, schema v1).

A ``CandidateRow`` is one Stage 1 to 4 candidate on one FOV, rejects included, with every gate quantity the
stage report holds as a number. A ``RunManifest`` records what a run was made from: input, weight and
annotation hashes, commit, environment, and the effective values the run used (scale-derived thresholds,
not only the requested config). Readers accept ``READABLE_VERSIONS``; a version bump adds a migration.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Union

SCHEMA_VERSION = 1
READABLE_VERSIONS = (1,)
EXPERIMENTAL = True

OUTCOMES = ("accept", "flag", "reject")
STAGES = (1, 2, 3, 4)
# Where a row's footprint comes from:
#   detected      the gate-time mask, from the run's stage{N}_gate_masks.npz (roigbiv.pipeline.gate_masks)
#   suite2p_stat  Stage 2: ypix/xpix of the Suite2p stat entry the candidate came from (its gate-time mask)
#   pool_disk     Stage 3: the spatial_pool_radius disk at the centroid, which is the gate-time mask
#   stamp         the persisted canonical disk (roi_stamp_radius) of an accept or flag; not the detected shape
#   not_recorded  the saved run holds no geometry for this candidate (never read as "no candidate")
FOOTPRINTS = ("detected", "suite2p_stat", "pool_disk", "stamp", "not_recorded")
STAGE_DETECTORS = {2: "suite2p", 3: "template_sweep", 4: "corr_contrast"}
SOURCES = ("pipeline", "backfill")

# Report fields that are identity, geometry or text, not gate quantities.
_NOT_GATES = frozenset({"label_id", "source_stage", "area", "centroid_y", "centroid_x"})
_REPORT_COUNT_FIELDS = (("detected", None), ("accepted", "accept"), ("flagged", "flag"), ("rejected", "reject"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

Scalar = Union[int, float, str, bool]


def reason_code(reason: str) -> str:
    """``"waveform_r2:0.513<0.6"`` -> ``"waveform_r2"``: the gate check that failed, without its value."""
    return reason.split(":", 1)[0]


def _check_version(v: object, what: str) -> None:
    if v not in READABLE_VERSIONS:
        raise ValueError(f"{what}: schema_version {v} not readable (reads {READABLE_VERSIONS})")


@dataclass(frozen=True)
class CandidateRow:
    run_id: str
    fov_id: str
    stage: int
    label_id: int
    outcome: str
    detector: str | None
    centroid_y: float | None
    centroid_x: float | None
    area: int | None
    footprint: str
    reasons: tuple[str, ...] = ()
    gates: tuple[tuple[str, float], ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        if self.footprint not in FOOTPRINTS:
            raise ValueError(f"footprint must be one of {FOOTPRINTS}, got {self.footprint!r}")
        has_y, has_x = self.centroid_y is not None, self.centroid_x is not None
        if has_y != has_x:
            raise ValueError("centroid_y and centroid_x must both be set or both be None")
        if not has_y and self.footprint != "not_recorded":
            raise ValueError(f"centroid missing but footprint is {self.footprint!r}; use 'not_recorded'")
        gates = tuple(sorted((str(k), float(v)) for k, v in self.gates))
        names = [k for k, _ in gates]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate gate names in {names}")
        object.__setattr__(self, "gates", gates)
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def candidate_id(self) -> str:
        return f"{self.fov_id}/s{self.stage}/{self.label_id}"

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(reason_code(r) for r in self.reasons)

    def to_record(self) -> dict:
        """Flat dict for one table row: gate quantities become ``gate.<name>`` columns."""
        rec = {k: v for k, v in asdict(self).items() if k not in ("gates", "reasons")}
        rec["candidate_id"] = self.candidate_id
        rec["reasons"] = list(self.reasons)
        rec["reason_codes"] = list(self.reason_codes)
        # Explicit names keep a NaN gate value (unresolved) apart from a gate this row never had.
        rec["gate_names"] = [k for k, _ in self.gates]
        rec.update({f"gate.{k}": v for k, v in self.gates})
        return rec

    @classmethod
    def from_record(cls, rec: Mapping) -> CandidateRow:
        _check_version(rec.get("schema_version"), "CandidateRow")
        names = rec.get("gate_names")
        if names is None:
            names = [k[len("gate.") :] for k, v in rec.items() if k.startswith("gate.") and v is not None]
        gates = tuple((k, rec[f"gate.{k}"]) for k in names)
        return cls(
            run_id=rec["run_id"],
            fov_id=rec["fov_id"],
            stage=int(rec["stage"]),
            label_id=int(rec["label_id"]),
            outcome=rec["outcome"],
            detector=rec.get("detector"),
            centroid_y=rec.get("centroid_y"),
            centroid_x=rec.get("centroid_x"),
            area=rec.get("area"),
            footprint=rec["footprint"],
            reasons=tuple(rec.get("reasons") or ()),
            gates=gates,
            schema_version=int(rec["schema_version"]),
        )


def _gate_values(*sources: Mapping) -> Iterable[tuple[str, float]]:
    for src in sources:
        for k, v in src.items():
            # bool is an int subclass; a flag is not a gate quantity.
            if k in _NOT_GATES or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            yield k, float(v)


def row_from_report_roi(roi: Mapping, *, run_id: str, fov_id: str, stage: int) -> CandidateRow:
    """One ``stage{N}_report.json`` ROI entry as a row, footprint ``not_recorded``; the table builder sets it."""
    if roi.get("source_stage") != stage:
        raise ValueError(f"ROI {roi.get('label_id')} has source_stage {roi.get('source_stage')}, expected {stage}")
    feats = roi.get("features") or {}
    cy, cx = feats.get("centroid_y"), feats.get("centroid_x")
    if cy is None or cx is None:
        cy = cx = None
    top = {k: v for k, v in roi.items() if k != "features"}
    return CandidateRow(
        run_id=run_id,
        fov_id=fov_id,
        stage=stage,
        label_id=int(roi["label_id"]),
        outcome=roi["gate_outcome"],
        detector=feats.get("stage1_detector") if stage == 1 else STAGE_DETECTORS[stage],
        centroid_y=None if cy is None else float(cy),
        centroid_x=None if cx is None else float(cx),
        area=roi.get("area"),
        footprint="not_recorded",
        reasons=tuple(roi.get("gate_reasons") or ()),
        gates=tuple(_gate_values(top, feats)),
    )


def check_outcome_counts(rows: Iterable[CandidateRow], reports: Mapping[int, Mapping]) -> list:
    """Compare per-stage outcome counts with each stage report; an empty list means they agree."""
    rows = list(rows)
    problems = []
    for stage in sorted(reports):
        mine = [r for r in rows if r.stage == stage]
        for name, outcome in _REPORT_COUNT_FIELDS:
            if name not in reports[stage]:
                continue
            n = len(mine) if outcome is None else sum(r.outcome == outcome for r in mine)
            if n != reports[stage][name]:
                problems.append(f"stage {stage} {name}: table {n}, report {reports[stage][name]}")
    return problems


@dataclass(frozen=True)
class FileHash:
    role: str
    path: str
    sha256: str | None
    size: int | None = None

    def __post_init__(self) -> None:
        if self.sha256 is not None and not _SHA256.match(self.sha256):
            raise ValueError(f"{self.role} {self.path}: sha256 must be 64 lowercase hex, got {self.sha256!r}")


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    fov_id: str
    source: str
    created_utc: str
    commit: str | None
    dirty: bool | None
    environment: tuple[tuple[str, str], ...]
    inputs: tuple[FileHash, ...]
    model_weights: tuple[FileHash, ...]
    annotations: tuple[FileHash, ...]
    effective: tuple[tuple[str, Scalar], ...]
    requested_config_sha256: str | None
    not_recorded: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    experimental: bool = field(default=EXPERIMENTAL)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.source == "pipeline":
            if self.commit is None:
                raise ValueError("a pipeline manifest needs commit")
            if self.dirty is None:
                raise ValueError("a pipeline manifest needs the dirty flag")
        missing = [k for k in ("commit", "dirty", "requested_config_sha256") if getattr(self, k) is None]
        if self.source == "backfill" and not set(missing) <= set(self.not_recorded):
            raise ValueError(f"backfill manifest: {missing} are None and must be listed in not_recorded")
        sha = self.requested_config_sha256
        if sha is not None and not _SHA256.match(sha):
            raise ValueError(f"requested_config_sha256 must be 64 lowercase hex, got {sha!r}")
        for name in ("environment", "effective"):
            object.__setattr__(self, name, tuple(sorted(tuple(p) for p in getattr(self, name))))
        for name in ("inputs", "model_weights", "annotations", "not_recorded"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        d = json.loads(text)
        _check_version(d.get("schema_version"), "RunManifest")
        for name in ("inputs", "model_weights", "annotations"):
            d[name] = tuple(FileHash(**h) for h in d[name])
        for name in ("environment", "effective", "not_recorded"):
            d[name] = tuple(tuple(p) if isinstance(p, list) else p for p in d[name])
        return cls(**d)
