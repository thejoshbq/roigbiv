"""Retroactively register / match pipeline output directories against the registry.

Walks a root directory, collects every subdirectory that has the minimum set
of pipeline outputs (``merged_masks.tif`` + ``summary/mean_M.tif``), parses
their filename metadata, sorts by ``(session_date, stem)``, and calls
:func:`register_or_match` on each in chronological order.

Because :func:`register_or_match` is idempotent for the hash-pre-filter
shortcut, re-running the backfill on an already-populated registry is safe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from roigbiv.registry.config import (
    RegistryConfig,
    build_adapter_config,
    build_blob_store,
    build_store,
    load_calibration,
)
from roigbiv.registry.filename import parse_filename_metadata
from roigbiv.registry.orchestrator import register_or_match
from roigbiv.registry.roicat_adapter import load_session_input


@dataclass
class BackfillCandidate:
    output_dir: Path
    stem: str
    session_date: date
    sort_key: tuple


def discover(root: Path) -> list[BackfillCandidate]:
    """Find every FOV output dir under ``root`` that has the required files.

    Required files (per the roigbiv pipeline output contract):
      * ``merged_masks.tif`` — the uint16 unified label image.
      * ``summary/mean_M.tif`` — the mean projection.
    """
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


def run_backfill(
    root: Path,
    *,
    dry_run: bool = False,
    cfg: Optional[RegistryConfig] = None,
) -> list[dict]:
    """Register/match every discovered session in chronological order."""
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

    cfg = cfg or RegistryConfig.from_env()
    store = build_store(cfg)
    blob_store = build_blob_store(cfg)
    adapter_cfg = build_adapter_config(cfg)
    calibration = load_calibration(cfg)

    reports: list[dict] = []
    for cand in candidates:
        try:
            query = load_session_input(cand.output_dir, session_key=cand.stem)
        except Exception as exc:  # noqa: BLE001
            reports.append({
                "output_dir": str(cand.output_dir),
                "stem": cand.stem,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        try:
            report = register_or_match(
                fov_stem=cand.stem,
                query=query,
                output_dir=cand.output_dir,
                store=store,
                blob_store=blob_store,
                session_date_override=cand.session_date,
                calibration=calibration,
                adapter_config=adapter_cfg,
                accept_threshold=cfg.fov_accept_threshold,
                review_threshold=cfg.fov_review_threshold,
            )
            report["stem"] = cand.stem
            reports.append(report)
        except Exception as exc:  # noqa: BLE001
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
    )


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
