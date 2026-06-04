"""Wrappers for section 5.2 post-subtraction diagnostics.

Reads subtraction_report_residual_S{N}.json output files and returns
structured aggregate artifact metrics per stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_subtraction_report(report_path: str | Path) -> dict:
    """Load and aggregate one subtraction report JSON.

    The report is keyed by ROI label_id and contains per-ROI:
        mean_ratio, std_ratio, anticorr_max, pass (bool).

    Returns
    -------
    dict with:
        n_rois         : int
        pass_count     : int
        fail_count     : int
        pass_rate      : float
        mean_anticorr  : float   (mean across ROIs)
        std_ratio_p90  : float   (90th percentile of std_ratio values)
        ring_candidates: int     (ROIs with std_ratio > 3.0 — over-subtraction)
        halo_candidates: int     (ROIs with std_ratio < 0.3 — under-subtraction)
    """
    data: dict = json.loads(Path(report_path).read_text())
    if not data:
        return {
            "n_rois": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": float("nan"),
            "mean_anticorr": float("nan"),
            "std_ratio_p90": float("nan"),
            "ring_candidates": 0,
            "halo_candidates": 0,
        }

    entries = list(data.values())
    passes = [e for e in entries if e.get("pass", False)]
    anticorrs = [e["anticorr_max"] for e in entries if "anticorr_max" in e]
    std_ratios = [e["std_ratio"] for e in entries if "std_ratio" in e]

    return {
        "n_rois": len(entries),
        "pass_count": len(passes),
        "fail_count": len(entries) - len(passes),
        "pass_rate": len(passes) / len(entries) if entries else float("nan"),
        "mean_anticorr": float(np.mean(anticorrs)) if anticorrs else float("nan"),
        "std_ratio_p90": float(np.percentile(std_ratios, 90)) if std_ratios else float("nan"),
        "ring_candidates": sum(1 for r in std_ratios if r > 3.0),
        "halo_candidates": sum(1 for r in std_ratios if r < 0.3),
    }


def load_all_stage_reports(pipeline_dir: str | Path) -> dict[str, dict]:
    """Load subtraction reports for all three stages that exist in pipeline_dir.

    Returns dict keyed by stage name ("S1", "S2", "S3") for present reports.
    """
    pipeline_dir = Path(pipeline_dir)
    results: dict[str, dict] = {}
    for stage_idx in (1, 2, 3):
        key = f"S{stage_idx}"
        report_path = pipeline_dir / f"subtraction_report_residual_S{stage_idx}.json"
        if report_path.exists():
            results[key] = load_subtraction_report(report_path)
    return results
