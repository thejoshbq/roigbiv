#!/usr/bin/env python
"""Phase-2 PMD A/B driver (recall-refinement engagement).

One variable: cfg.use_pmd_denoise (off vs on). Identical config otherwise.
For each held-out FOV runs both arms, scores merged_masks.tif vs anatomical GT
with the repo's stratified harness, then reclaims large intermediates (disk is
tight). Resumable: skips arms already present in ab_results.json.

Env knobs:
  PMD_AB_MAX_FOVS   limit number of FOVs (smoke testing); default all
  PMD_AB_MIN_FREE_GB abort a run if free disk below this; default 25
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
from roigbiv.eval.harness import score_one, _load_roi_metadata
from roigbiv.eval.diagnostics import load_all_stage_reports

MANIFEST = ROOT / "experiments/harness/heldout_fovs.txt"
OUT_ROOT = ROOT / "experiments/runs/phase2_pmd"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_ROOT / "ab_results.json"
PROG = OUT_ROOT / "ab_progress.log"
FS = 7.5
MAX_FOVS = int(os.environ.get("PMD_AB_MAX_FOVS", "0")) or None
MIN_FREE_GB = float(os.environ.get("PMD_AB_MIN_FREE_GB", "25"))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def free_gb() -> float:
    st = os.statvfs(str(OUT_ROOT))
    return st.f_bavail * st.f_frsize / 1e9


def resolve_manifest() -> list[tuple[str, str, str]]:
    """Return [(stem, movie_path, gt_path)] resolving stale JOSH/ paths by basename."""
    out = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stem, movie, gt = (line.split("|") + ["", ""])[:3]
        mb, gb = Path(movie).name, Path(gt).name
        mh = next(iter(sorted((ROOT / "data").rglob(mb))), None)
        gh = next(iter(sorted((ROOT / "data").rglob(gb))), None)
        if mh and gh:
            out.append((stem, str(mh), str(gh)))
        else:
            log(f"UNRESOLVED {stem} (movie={'ok' if mh else 'MISS'} gt={'ok' if gh else 'MISS'})")
    return out


def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {}


def save_results(res: dict) -> None:
    RESULTS.write_text(json.dumps(res, indent=2))


def reclaim(out_dir: Path) -> None:
    """Delete large intermediates, keep small artifacts the harness needs."""
    try:
        for d in out_dir.rglob("suite2p"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        pmd = out_dir / "pmd"
        if pmd.exists():
            shutil.rmtree(pmd, ignore_errors=True)
        for pat in ("**/data.bin", "**/*.dat", "**/*.bin"):
            for f in out_dir.glob(pat):
                if f.name not in ("merged_masks.tif",):
                    f.unlink(missing_ok=True)
        # input-movie copy + SVD factors: regenerable, not needed post-scoring.
        svd = out_dir / "svd_factors.npz"
        if svd.exists():
            svd.unlink()
        for f in out_dir.glob("*.tif"):
            if f.name != "merged_masks.tif" and f.stat().st_size > 50_000_000:
                f.unlink(missing_ok=True)
        # Final sweep: drop any remaining big file (e.g. stage3 raw-event blobs),
        # keeping only the small artifacts the harness needs.
        keep = {"merged_masks.tif", "roi_metadata.json"}
        for f in out_dir.rglob("*"):
            if f.is_file() and f.name not in keep and f.stat().st_size > 20_000_000:
                f.unlink(missing_ok=True)
    except Exception as exc:
        log(f"  reclaim warning ({out_dir.name}): {exc}")


def run_arm(stem: str, movie: str, gt: str, pmd_on: bool, res: dict) -> None:
    tag = "pmdon" if pmd_on else "pmdoff"
    key = f"{stem}|{tag}"
    if key in res:
        log(f"skip {key} (already scored)")
        return
    out_dir = OUT_ROOT / f"{stem}_{tag}"
    if free_gb() < MIN_FREE_GB:
        log(f"ABORT {key}: free disk {free_gb():.1f}GB < {MIN_FREE_GB}GB")
        raise SystemExit(2)

    cfg = PipelineConfig(fs=FS)
    cfg.output_dir = out_dir
    cfg.use_pmd_denoise = pmd_on
    cfg.no_viewer = True

    t0 = time.time()
    try:
        run_pipeline(Path(movie), cfg)
        dt = time.time() - t0
        pred = tifffile.imread(str(out_dir / "merged_masks.tif")).astype(np.uint16)
        gt_img = tifffile.imread(str(gt)).astype(np.uint16)
        meta = _load_roi_metadata(out_dir)
        reports = load_all_stage_reports(out_dir)
        scored = score_one(pred, gt_img, meta, reports)
        stage_counts = {}
        for sname, rep in (reports or {}).items():
            if isinstance(rep, dict):
                sub = {k: rep.get(k) for k in ("detected", "accepted", "flagged", "rejected")
                       if k in rep}
                if sub:
                    stage_counts[sname] = sub
        res[key] = {
            "stem": stem, "arm": tag, "runtime_s": round(dt, 1),
            "detection": scored["detection"], "counts": scored["counts"],
            "stage_counts": stage_counts,
        }
        save_results(res)
        ov = scored["detection"]["overall"]
        log(f"OK {key} {dt:.0f}s  recall={ov['recall']:.3f} "
            f"tp={ov['tp']} fp={ov['fp']} fn={ov['fn']}")
    except Exception as exc:
        res[key] = {"stem": stem, "arm": tag, "error": str(exc)}
        save_results(res)
        log(f"FAIL {key}: {exc}\n{traceback.format_exc()}")
    finally:
        reclaim(out_dir)
        gc.collect()


def main() -> int:
    fovs = resolve_manifest()
    if MAX_FOVS:
        fovs = fovs[:MAX_FOVS]
    log(f"=== A/B start: {len(fovs)} FOVs × 2 arms, free={free_gb():.0f}GB ===")
    res = load_results()
    for i, (stem, movie, gt) in enumerate(fovs, 1):
        log(f"--- FOV {i}/{len(fovs)}: {stem} ---")
        run_arm(stem, movie, gt, False, res)   # baseline first
        run_arm(stem, movie, gt, True, res)    # PMD on
    log("=== ALL RUNS DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
