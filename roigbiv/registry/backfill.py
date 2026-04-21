"""Retroactively register/match every FOV already under inference/pipeline/.

Walks the directory, reads summary/mean_M.tif + merged_masks.tif +
roi_metadata.json per FOV, and feeds them through `register_or_match` in
chronological order (by session_date parsed from the filename stem, falling
back to pipeline_log.json's ISO timestamp).

The `dry_run=True` mode reports intended actions without mutating state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from roigbiv.registry import build_blob_store, build_store
from roigbiv.registry.filename import parse_filename_metadata
from roigbiv.registry.fingerprint import cells_from_masks
from roigbiv.registry.orchestrator import register_or_match


@dataclass
class BackfillCandidate:
    output_dir: Path
    stem: str
    session_date: date
    sort_key: tuple


def discover(root: Path) -> list[BackfillCandidate]:
    """Find every FOV output dir that has the files we need."""
    root = Path(root)
    out: list[BackfillCandidate] = []
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not _has_required_outputs(d):
            continue
        stem = d.name
        meta = parse_filename_metadata(stem)
        sess_date = meta.session_date or _fallback_date_from_log(d) or date.today()
        out.append(BackfillCandidate(
            output_dir=d,
            stem=stem,
            session_date=sess_date,
            sort_key=(sess_date, stem),
        ))
    out.sort(key=lambda c: c.sort_key)
    return out


def run_backfill(root: Path, *, dry_run: bool = False) -> list[dict]:
    candidates = discover(root)
    if not candidates:
        return []

    if dry_run:
        return [{
            "output_dir": str(c.output_dir),
            "stem": c.stem,
            "session_date": c.session_date.isoformat(),
            "action": "would_register",
        } for c in candidates]

    store = build_store()
    blob_store = build_blob_store()
    reports: list[dict] = []
    for cand in candidates:
        try:
            mean_m, cells = _load_fov_inputs(cand.output_dir)
        except Exception as exc:
            reports.append({
                "output_dir": str(cand.output_dir),
                "stem": cand.stem,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        try:
            report = register_or_match(
                fov_stem=cand.stem,
                mean_m=mean_m,
                cells=cells,
                output_dir=cand.output_dir,
                store=store,
                blob_store=blob_store,
                session_date_override=cand.session_date,
            )
            report["stem"] = cand.stem
            reports.append(report)
        except Exception as exc:
            reports.append({
                "output_dir": str(cand.output_dir),
                "stem": cand.stem,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return reports


def _has_required_outputs(d: Path) -> bool:
    return (
        (d / "summary" / "mean_M.tif").exists()
        and (d / "merged_masks.tif").exists()
        and (d / "roi_metadata.json").exists()
    )


def _load_fov_inputs(d: Path):
    mean_m = tifffile.imread(str(d / "summary" / "mean_M.tif")).astype(np.float32)
    merged_masks = tifffile.imread(str(d / "merged_masks.tif"))
    roi_metadata = json.loads((d / "roi_metadata.json").read_text())
    cells = cells_from_masks(merged_masks, roi_metadata)
    return mean_m, cells


def _fallback_date_from_log(d: Path) -> Optional[date]:
    log = d / "pipeline_log.json"
    if not log.exists():
        return None
    try:
        data = json.loads(log.read_text())
        ts = data.get("timestamp")
        if not ts:
            return None
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except Exception:
        return None
