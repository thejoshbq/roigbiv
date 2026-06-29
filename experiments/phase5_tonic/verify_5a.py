#!/usr/bin/env python
"""Phase-5a end-to-end check: run ONE FOV on the settled default config and
confirm the neuropil_baseline_elevation feature is computed and logged into
roi_metadata.json. No A/B here — 5a's gate is "feature present and logged".
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")
ROOT = Path("/home/thejoshbq/Otis-Lab/Projects/Phoxel-Workbench/roigbiv")
sys.path.insert(0, str(ROOT))

from roigbiv.pipeline.types import PipelineConfig
from roigbiv.pipeline.run import run_pipeline

MANIFEST = ROOT / "experiments/harness/heldout_fovs.txt"
OUT = ROOT / "experiments/runs/phase5_5a_verify"
OUT.mkdir(parents=True, exist_ok=True)


def resolve_first():
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stem, movie = (line.split("|") + ["", ""])[:2]
        mh = next(iter(sorted((ROOT / "data").rglob(Path(movie).name))), None)
        if mh:
            return stem, str(mh)
    raise SystemExit("no FOV resolved")


def main():
    stem, movie = resolve_first()
    out_dir = OUT / stem
    cfg = PipelineConfig(fs=7.5)            # defaults = fused ch2 + denoise ON
    cfg.output_dir = out_dir
    cfg.no_viewer = True
    print(f"running {stem}")
    run_pipeline(Path(movie), cfg)

    meta_path = next(iter(out_dir.rglob("roi_metadata.json")), None)
    if meta_path is None:
        raise SystemExit("FAIL: no roi_metadata.json produced")
    meta = json.loads(meta_path.read_text())
    rois = meta if isinstance(meta, list) else meta.get("rois", meta)
    n = len(rois)
    have = [r for r in rois if "neuropil_baseline_elevation" in r.get("features", {})]
    vals = [r["features"]["neuropil_baseline_elevation"] for r in have]
    no_array = all("F_neuropil" not in r.get("features", {}) for r in rois)
    print(f"\n=== 5a verify @ {meta_path.relative_to(out_dir)} ===")
    print(f"  ROIs: {n}  with neuropil_baseline_elevation: {len(have)}")
    print(f"  F_neuropil array dropped from JSON: {no_array}")
    if vals:
        import statistics as st
        print(f"  elevation min/median/max: {min(vals):.3f} / "
              f"{st.median(vals):.3f} / {max(vals):.3f}")
        tonic = [r for r in rois if r.get("activity_type") == "tonic"
                 and "neuropil_baseline_elevation" in r.get("features", {})]
        if tonic:
            tv = [r["features"]["neuropil_baseline_elevation"] for r in tonic]
            print(f"  tonic ROIs: {len(tonic)}  elevation median: "
                  f"{st.median(tv):.3f}")
    ok = (len(have) == n) and no_array and n > 0
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(all ROIs logged + array dropped)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
