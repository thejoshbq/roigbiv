#!/usr/bin/env python
"""Confirmatory denoise A/B — denoise-OFF arm on the NEW fused default.

Phase-3's secondary finding (denoise-OFF beats deployed by recall +0.013 / prec
+0.019 / FP -10%) was measured at the OLD ch2=vcorr_S baseline. The default has
since moved to ch2=vcorr_max_fused (Phase-4), and use_denoise acts on channel-1
(mean_M) while the flip changed channel-2 — the two interact at the detector, so
the Phase-3 delta does NOT transfer. Per one_variable_per_experiment, this re-runs
the comparison on the new default:

  denoise_on   ch1 = denoise_cyto3(mean_M)   (current default)   -- REUSED from
               ch2 = vcorr_max_fused                                Phase-4 `fused`
  denoise_off  ch1 = mean_M (raw)            (candidate flip)     -- THIS SCRIPT
               ch2 = vcorr_max_fused

The single variable is use_denoise. ch2=vcorr_max_fused, backend=cellpose3, model,
thresholds all fixed = the post-Phase-4 deployed config. denoise_on already exists
as experiments/runs/phase4_channel/{stem}|fused, so only the OFF arm runs here
(13 incremental full-pipeline runs).

Recall-first bar (same as Phase 4): no per-FOV recall regression AND pooled
post-review FP increase <= +15%. If any FOV regresses, KEEP denoise ON (the
checkpoint was fine-tuned WITH denoise in-loop, so OFF is off-distribution).

Env knobs mirror run_ab.py: PD_AB_MAX_FOVS, PD_AB_MIN_FREE_GB.
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
OUT_ROOT = ROOT / "experiments/runs/phase_denoise_ab"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_ROOT / "off_results.json"
PROG = OUT_ROOT / "off_progress.log"
FS = 7.5
MAX_FOVS = int(os.environ.get("PD_AB_MAX_FOVS", "0")) or None
MIN_FREE_GB = float(os.environ.get("PD_AB_MIN_FREE_GB", "25"))


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
    try:
        for d in out_dir.rglob("suite2p"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        for pat in ("**/data.bin", "**/*.dat", "**/*.bin"):
            for f in out_dir.glob(pat):
                if f.name not in ("merged_masks.tif",):
                    f.unlink(missing_ok=True)
        svd = out_dir / "svd_factors.npz"
        if svd.exists():
            svd.unlink()
        for f in out_dir.glob("*.tif"):
            if f.name != "merged_masks.tif" and f.stat().st_size > 50_000_000:
                f.unlink(missing_ok=True)
        keep = {"merged_masks.tif", "roi_metadata.json"}
        for f in out_dir.rglob("*"):
            if f.is_file() and f.name not in keep and f.stat().st_size > 20_000_000:
                f.unlink(missing_ok=True)
    except Exception as exc:
        log(f"  reclaim warning ({out_dir.name}): {exc}")


def run_off(stem: str, movie: str, gt: str, res: dict) -> None:
    key = f"{stem}|off"
    if key in res and "error" not in res[key]:
        log(f"skip {key} (already scored)")
        return
    out_dir = OUT_ROOT / f"{stem}_off"
    if free_gb() < MIN_FREE_GB:
        log(f"ABORT {key}: free disk {free_gb():.1f}GB < {MIN_FREE_GB}GB")
        raise SystemExit(2)

    cfg = PipelineConfig(fs=FS)
    cfg.output_dir = out_dir
    cfg.stage1_backend = "cellpose3"           # fixed
    cfg.stage1_ch2_source = "vcorr_max_fused"  # fixed = new default
    cfg.use_denoise = False                     # the ONE variable
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
        res[key] = {
            "stem": stem, "arm": "off", "use_denoise": False,
            "stage1_ch2_source": "vcorr_max_fused",
            "runtime_s": round(dt, 1),
            "detection": scored["detection"], "counts": scored["counts"],
        }
        save_results(res)
        ov = scored["detection"]["overall"]
        log(f"OK {key} {dt:.0f}s  recall={ov['recall']:.3f} "
            f"tp={ov['tp']} fp={ov['fp']} fn={ov['fn']}")
    except Exception as exc:
        res[key] = {"stem": stem, "arm": "off", "error": str(exc)}
        save_results(res)
        log(f"FAIL {key}: {exc}\n{traceback.format_exc()}")
    finally:
        reclaim(out_dir)
        gc.collect()


def main() -> int:
    fovs = resolve_manifest()
    if MAX_FOVS:
        fovs = fovs[:MAX_FOVS]
    log(f"=== denoise-OFF arm: {len(fovs)} FOVs (fused ch2, use_denoise=False), "
        f"free={free_gb():.0f}GB ===")
    res = load_results()
    for i, (stem, movie, gt) in enumerate(fovs, 1):
        log(f"--- FOV {i}/{len(fovs)}: {stem} ---")
        run_off(stem, movie, gt, res)
    log("=== ALL OFF RUNS DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
