"""Candidate table for one FOV of a saved run: every Stage 1 to 4 candidate, rejects included.

Rows come from the ``stage{N}_report.json`` files; footprints come from the best source the saved run holds
(see ``models.FOOTPRINTS``). The builder reads a run directory and never writes to it. ``write_fov_table``
puts ``candidates.parquet`` (one row per candidate, gate quantities as ``gate.<name>`` columns) and
``footprints.npz`` (flat pixel indices per candidate, CSR layout) in a separate directory.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roigbiv.workbench.models import SCHEMA_VERSION, CandidateRow, check_outcome_counts, row_from_report_roi

STAGES = (1, 2, 3, 4)
TABLE_FILE = "candidates.parquet"
FOOTPRINT_FILE = "footprints.npz"
META_FILE = "table_meta.json"


@dataclass
class FovTable:
    run_id: str
    fov_id: str
    shape: tuple
    rows: list
    footprints: dict = field(default_factory=dict)  # candidate_id -> sorted flat pixel indices (int64)
    problems: list = field(default_factory=list)
    missing_stages: tuple = ()

    def footprint(self, candidate_id: str) -> np.ndarray | None:
        """The candidate's footprint as an (H, W) bool mask, or None when the saved run holds none."""
        idx = self.footprints.get(candidate_id)
        if idx is None:
            return None
        mask = np.zeros(int(np.prod(self.shape)), dtype=bool)
        mask[idx] = True
        return mask.reshape(self.shape)


def _disk_flat(cy: float, cx: float, radius: int, shape: tuple) -> np.ndarray:
    # Same disk as roigbiv.pipeline.roi_stamp.disk_mask (the pipeline's own Stage 3 pooling mask).
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    return np.flatnonzero(((ys - cy) ** 2 + (xs - cx) ** 2) <= radius**2)


def _centroid(flat: np.ndarray, shape: tuple) -> tuple[float, float]:
    ys, xs = np.unravel_index(flat, shape)
    return float(ys.mean()), float(xs.mean())


def _find_stat(fov_dir: Path) -> tuple[Path | None, str | None]:
    hits = sorted(fov_dir.glob("*/suite2p/plane0/stat.npy"))
    if len(hits) == 1:
        return hits[0], None
    return None, f"stage 2: expected one */suite2p/plane0/stat.npy, found {len(hits)}"


def _stat_flat(entry: dict, shape: tuple) -> np.ndarray:
    ys = np.asarray(entry["ypix"], dtype=np.int64)
    xs = np.asarray(entry["xpix"], dtype=np.int64)
    ok = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
    return np.unique(np.ravel_multi_index((ys[ok], xs[ok]), shape))


class _Builder:
    def __init__(self, fov_dir: Path, run_id: str, fov_id: str):
        import tifffile

        self.dir, self.run_id, self.fov_id = fov_dir, run_id, fov_id
        self._imread = tifffile.imread
        cfg = json.loads((fov_dir / "pipeline_log.json").read_text()).get("config", {})
        self.pool_radius = cfg.get("spatial_pool_radius")
        self._pool_problem_logged = False
        self.shape = tuple(self._imread(fov_dir / "stage1" / "stage1_masks.tif").shape)
        self.rows: list = []
        self.footprints: dict = {}
        self.problems: list = []
        self.reports: dict = {}
        self.missing: list = []
        self._stat = None

    def stat(self):
        if self._stat is None:
            path, problem = _find_stat(self.dir)
            if problem:
                self.problems.append(problem)
                self._stat = []
            else:
                self._stat = list(np.load(path, allow_pickle=True))
        return self._stat

    def stage(self, n: int) -> None:
        rp = self.dir / f"stage{n}" / f"stage{n}_report.json"
        if not rp.exists():
            self.missing.append(n)
            return
        report = json.loads(rp.read_text())
        self.reports[n] = report
        labels = None
        for roi in report["rois"]:
            row = row_from_report_roi(roi, run_id=self.run_id, fov_id=self.fov_id, stage=n)
            if n == 3 or (n == 2 and self.stat()):
                row = self._detected(row, roi)
            elif row.outcome != "reject":
                if labels is None:
                    labels = self._imread(self.dir / f"stage{n}" / f"stage{n}_masks.tif").ravel()
                row = self._stamp(row, labels)
            self.rows.append(row)

    def _stamp(self, row: CandidateRow, labels: np.ndarray) -> CandidateRow:
        flat = np.flatnonzero(labels == row.label_id)
        if flat.size == 0:
            self.problems.append(
                f"stage {row.stage} label {row.label_id} is {row.outcome} but absent from stage{row.stage}_masks.tif"
            )
            return row
        self.footprints[row.candidate_id] = flat
        cy, cx = (row.centroid_y, row.centroid_x) if row.centroid_y is not None else _centroid(flat, self.shape)
        return dataclasses.replace(row, footprint="stamp", centroid_y=cy, centroid_x=cx)

    def _detected(self, row: CandidateRow, roi: dict) -> CandidateRow:
        if row.stage == 2:
            idx = (roi.get("features") or {}).get("suite2p_stat_index")
            stat = self.stat()
            if idx is None or not 0 <= idx < len(stat):
                if stat:
                    self.problems.append(f"stage 2 label {row.label_id}: suite2p_stat_index {idx} not in stat.npy")
                return row
            flat = _stat_flat(stat[idx], self.shape)
            self.footprints[row.candidate_id] = flat
            self._check_area(row, "suite2p_stat", flat)
            cy, cx = _centroid(flat, self.shape)
            return dataclasses.replace(row, footprint="suite2p_stat", centroid_y=cy, centroid_x=cx)
        if row.stage == 3 and not self.pool_radius and not self._pool_problem_logged:
            self.problems.append(
                "stage 3: spatial_pool_radius not in pipeline_log config; Stage 3 footprints not rebuilt"
            )
            self._pool_problem_logged = True
        if row.stage == 3 and row.centroid_y is not None and row.centroid_x is not None and self.pool_radius:
            flat = _disk_flat(row.centroid_y, row.centroid_x, int(self.pool_radius), self.shape)
            self.footprints[row.candidate_id] = flat
            self._check_area(row, "pool_disk", flat)
            return dataclasses.replace(row, footprint="pool_disk")
        return row

    def _check_area(self, row: CandidateRow, source: str, flat: np.ndarray) -> None:
        # A rebuilt gate-time mask must have the gate's own area; a mismatch means a stale stat.npy or radius.
        if row.area is not None and flat.size != row.area:
            self.problems.append(
                f"stage {row.stage} label {row.label_id}: {source} has {flat.size} px, gate area {row.area}"
            )


def build_fov_table(fov_dir: Path, *, run_id: str, fov_id: str) -> FovTable:
    """Read one FOV directory of a saved run (read-only) into a candidate table."""
    b = _Builder(Path(fov_dir), run_id, fov_id)
    for n in STAGES:
        b.stage(n)
    b.problems.extend(check_outcome_counts(b.rows, b.reports))
    rows = sorted(b.rows, key=lambda r: (r.stage, r.label_id))
    return FovTable(
        run_id=run_id,
        fov_id=fov_id,
        shape=b.shape,
        rows=rows,
        footprints=b.footprints,
        problems=b.problems,
        missing_stages=tuple(b.missing),
    )


def write_fov_table(table: FovTable, dest: Path, *, allow_problems: bool = False) -> None:
    """Write ``candidates.parquet``, ``footprints.npz`` and ``table_meta.json`` into ``dest``."""
    import pandas as pd

    if table.problems and not allow_problems:
        raise ValueError(f"{table.fov_id}: table has problems, not written: {table.problems}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_record() for r in table.rows]).to_parquet(dest / TABLE_FILE, engine="pyarrow", index=False)
    ids = [r.candidate_id for r in table.rows if r.candidate_id in table.footprints]
    parts = [np.asarray(table.footprints[i], dtype=np.int64) for i in ids]
    offsets = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    np.savez_compressed(
        dest / FOOTPRINT_FILE,
        candidate_id=np.array(ids, dtype=str),
        offsets=offsets,
        flat_idx=np.concatenate(parts) if parts else np.zeros(0, np.int64),
        shape=np.array(table.shape, dtype=np.int64),
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": table.run_id,
        "fov_id": table.fov_id,
        "shape": list(table.shape),
        "n_rows": len(table.rows),
        "problems": table.problems,
        "missing_stages": list(table.missing_stages),
    }
    (dest / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def read_fov_table(src: Path) -> FovTable:
    import pandas as pd

    src = Path(src)
    meta = json.loads((src / META_FILE).read_text())
    df = pd.read_parquet(src / TABLE_FILE, engine="pyarrow")
    rows = []
    for rec in df.to_dict("records"):
        for k in ("centroid_y", "centroid_x", "area", "detector"):
            if _is_nan(rec.get(k)):
                rec[k] = None
        for k in ("centroid_y", "centroid_x"):
            rec[k] = None if rec[k] is None else float(rec[k])
        rec["area"] = None if rec["area"] is None else int(rec["area"])
        rec["reasons"] = tuple(rec["reasons"])
        rec["gate_names"] = list(rec["gate_names"])
        rows.append(CandidateRow.from_record(rec))
    z = np.load(src / FOOTPRINT_FILE, allow_pickle=False)
    off = z["offsets"]
    fps = {str(c): z["flat_idx"][off[i] : off[i + 1]] for i, c in enumerate(z["candidate_id"])}
    return FovTable(
        run_id=meta["run_id"],
        fov_id=meta["fov_id"],
        shape=tuple(meta["shape"]),
        rows=rows,
        footprints=fps,
        problems=meta["problems"],
        missing_stages=tuple(meta["missing_stages"]),
    )


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)
