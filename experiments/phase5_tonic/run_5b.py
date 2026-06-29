#!/usr/bin/env python
"""Phase-5b tonic accept-tier — gate-aware A/B driver (recall-refinement).

WHY a single arm: the accept tier flips gate_outcome (flag→accept) for
anatomical tonic ROIs; it changes NO mask in merged_masks.tif (which carries
every non-rejected ROI). So the stock recall/precision/FP harness is identical
across arms — the tier is invisible to it. The thing that actually changes is
review-queue membership, and its risk is FP escaping review. We therefore run
ONCE on the settled default (tier OFF — but the 5a elevation feature is logged),
IoU-match every ROI to anatomical GT, and dump per-ROI records. summarize_5b.py
then simulates the tier at swept thresholds post-hoc (mask-identical → exact).

Per-ROI record: stem, label_id, source_stage, activity_type, gate_outcome,
confidence, neuropil_baseline_elevation, matched (IoU≥0.3 to a GT cell).

Env knobs:
  P5B_MAX_FOVS     limit FOVs (smoke); default all
  P5B_MIN_FREE_GB  abort if free disk below; default 25
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")   # avoid ollama VRAM contention

import numpy as np
import tifffile

ROOT = Path("/home/thejoshbq/Otis-Lab/Projects/Phoxel-Workbench/roigbiv")
sys.path.insert(0, str(ROOT))

from roigbiv.pipeline.types import PipelineConfig
from roigbiv.pipeline.run import run_pipeline
from roigbiv.eval.harness import _load_roi_metadata
from roigbiv.eval.match import iou_match

MANIFEST = ROOT / "experiments/harness/heldout_fovs.txt"
OUT_ROOT = ROOT / "experiments/runs/phase5_5b"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_ROOT / "roi_records.json"
PROG = OUT_ROOT / "run.log"
FS = 7.5
MAX_FOVS = int(os.environ.get("P5B_MAX_FOVS", "0")) or None
MIN_FREE_GB = float(os.environ.get("P5B_MIN_FREE_GB", "25"))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def free_gb() -> float:
    st = os.statvfs(str(OUT_ROOT))
    return st.f_bavail * st.f_frsize / 1e9


def resolve_manifest() -> list[tuple[str, str, str]]:
    out = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stem, movie, gt = (line.split("|") + ["", ""])[:3]
        mh = next(iter(sorted((ROOT / "data").rglob(Path(movie).name))), None)
        gh = next(iter(sorted((ROOT / "data").rglob(Path(gt).name))), None)
        if mh and gh:
            out.append((stem, str(mh), str(gh)))
        else:
            log(f"UNRESOLVED {stem} (movie={'ok' if mh else 'MISS'} gt={'ok' if gh else 'MISS'})")
    return out


def load_results() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save_results(res: dict) -> None:
    RESULTS.write_text(json.dumps(res, indent=2))


def reclaim(out_dir: Path) -> None:
    try:
        for d in out_dir.rglob("suite2p"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        keep = {"merged_masks.tif", "roi_metadata.json"}
        for f in out_dir.rglob("*"):
            if f.is_file() and f.name not in keep and f.stat().st_size > 20_000_000:
                f.unlink(missing_ok=True)
    except Exception as exc:
        log(f"  reclaim warning ({out_dir.name}): {exc}")


def run_fov(stem: str, movie: str, gt: str, res: dict) -> None:
    if stem in res and "error" not in res[stem]:
        log(f"skip {stem} (already recorded)")
        return
    out_dir = OUT_ROOT / stem
    if free_gb() < MIN_FREE_GB:
        log(f"ABORT {stem}: free disk {free_gb():.1f}GB < {MIN_FREE_GB}GB")
        raise SystemExit(2)

    cfg = PipelineConfig(fs=FS)
    cfg.output_dir = out_dir
    cfg.no_viewer = True
    # settled default: fused ch2 + denoise ON; tier OFF (default) — feature logged.

    t0 = time.time()
    try:
        run_pipeline(Path(movie), cfg)
        dt = time.time() - t0
        pred = tifffile.imread(str(out_dir / "merged_masks.tif")).astype(np.uint16)
        gt_img = tifffile.imread(str(gt)).astype(np.uint16)
        meta = _load_roi_metadata(out_dir) or {}
        m = iou_match(gt_img, pred, min_iou=0.3)
        matched_pred = {pred_label for _, pred_label, _ in m.tp}

        records = []
        for label_id, e in meta.items():
            feats = e.get("features", {}) or {}
            records.append({
                "label_id": int(label_id),
                "source_stage": int(e.get("source_stage", -1)),
                "activity_type": e.get("activity_type") or "unknown",
                "gate_outcome": e.get("gate_outcome"),
                "confidence": e.get("confidence"),
                "elevation": float(feats.get("neuropil_baseline_elevation", 0.0)),
                "matched": int(label_id) in matched_pred,
            })
        res[stem] = {
            "stem": stem, "runtime_s": round(dt, 1),
            "n_gt": int(m.n_tp + m.n_fn), "n_pred": int(m.n_tp + m.n_fp),
            "tp": int(m.n_tp), "fp": int(m.n_fp), "fn": int(m.n_fn),
            "records": records,
        }
        save_results(res)
        n_tonic12 = sum(1 for r in records
                        if r["activity_type"] == "tonic" and r["source_stage"] in (1, 2))
        log(f"OK {stem} {dt:.0f}s  rois={len(records)} tonic(s1/2)={n_tonic12} "
            f"tp={m.n_tp} fp={m.n_fp} fn={m.n_fn}")
    except Exception as exc:
        res[stem] = {"stem": stem, "error": str(exc)}
        save_results(res)
        log(f"FAIL {stem}: {exc}\n{traceback.format_exc()}")
    finally:
        reclaim(out_dir)
        gc.collect()


def main() -> int:
    fovs = resolve_manifest()
    if MAX_FOVS:
        fovs = fovs[:MAX_FOVS]
    log(f"=== 5b gate-aware run: {len(fovs)} FOVs, free={free_gb():.0f}GB ===")
    res = load_results()
    for i, (stem, movie, gt) in enumerate(fovs, 1):
        log(f"--- FOV {i}/{len(fovs)}: {stem} ---")
        run_fov(stem, movie, gt, res)
    log("=== ALL RUNS DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
