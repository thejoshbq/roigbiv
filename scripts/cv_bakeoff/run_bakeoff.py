"""CV bake-off CLI — run several detectors on FOV summary images, emit a grid.

Examples
--------
List CP3 model options::

    python scripts/cv_bakeoff/run_bakeoff.py --list-models

Compare the in-process methods on one FOV's summaries::

    python scripts/cv_bakeoff/run_bakeoff.py \\
        --summary-dir inference/pipeline/<stem>/summary \\
        --methods cp3,classical --background mean_M

Sweep every processed FOV in a workspace::

    python scripts/cv_bakeoff/run_bakeoff.py \\
        --workspace /path/to/fov_dir --methods cp3,classical

Outputs (grid PNG + meta JSON) land in ``experiments/runs/cv_bakeoff/``.
Nothing is written to ``inference/``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

# scripts/ on path so ``cv_bakeoff`` imports as a package (repo convention).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_bakeoff.detector import DetectorInputs  # noqa: E402
from cv_bakeoff.grid import render_grid  # noqa: E402

_DEFAULT_OUT = Path("experiments/runs/cv_bakeoff")


def _load_summary(summary_dir: Path) -> dict[str, np.ndarray]:
    """Load every ``*.tif`` in a summary dir, keyed by stem (mean_M, vcorr_S, …)."""
    from roigbiv.pipeline.loaders import _maybe_read_tif

    out: dict[str, np.ndarray] = {}
    for tif in sorted(summary_dir.glob("*.tif")):
        arr = _maybe_read_tif(tif)
        if arr is not None:
            out[tif.stem] = arr
    return out


def _build_detector(method: str, args):
    if method == "cp3":
        from cv_bakeoff.detectors.cp3 import CP3Detector
        return CP3Detector(
            model=args.model, diameter=args.diameter,
            flow_threshold=args.flow_threshold, force_cpu=args.cpu,
        )
    if method == "classical":
        from cv_bakeoff.detectors.classical import ClassicalDetector
        return ClassicalDetector(
            channel=args.background, diameter=args.diameter or 12.0,
        )
    if method == "cp-sam":
        from cv_bakeoff.detectors.cp_sam import CPSAMDetector
        return CPSAMDetector(
            channel=args.background, diameter=args.diameter,
            flow_threshold=args.flow_threshold,
        )
    if method == "stardist":
        from cv_bakeoff.detectors.stardist import StarDistDetector
        return StarDistDetector(channel=args.background)
    raise SystemExit(f"unknown method {method!r}")


def _discover_summary_dirs(workspace: Path) -> list[Path]:
    """Find every ``*/summary`` (containing mean_M.tif) under a workspace."""
    found = []
    for cand in sorted(workspace.rglob("summary")):
        if cand.is_dir() and (cand / "mean_M.tif").exists():
            found.append(cand)
    return found


def _apply_enhance(summary: dict[str, np.ndarray], spec: str | None):
    """Apply a comma-list of enhancement transforms (Part 2). No-op if unset."""
    if not spec:
        return summary, None
    try:
        from cv_bakeoff.enhance import apply_enhancements
    except ImportError:
        raise SystemExit("--enhance requested but cv_bakeoff.enhance is unavailable.")
    return apply_enhancements(summary, spec), spec


def _process_one(summary_dir: Path, args, methods: list[str], out_dir: Path) -> Path:
    summary = _load_summary(summary_dir)
    if args.background not in summary:
        raise SystemExit(
            f"background channel {args.background!r} not in {summary_dir} "
            f"(have {sorted(summary)})"
        )
    summary, enhance_label = _apply_enhance(summary, args.enhance)

    fov_stem = summary_dir.parent.name
    inputs = DetectorInputs(summary=summary, fov_stem=fov_stem, fs=args.fs)

    results = []
    for method in methods:
        det = _build_detector(method, args)
        print(f"  [{fov_stem}] running {method}…", flush=True)
        results.append(det.detect(inputs))

    ts = _dt.datetime.now()
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    png = out_dir / f"{fov_stem}_bakeoff_{stamp}.png"
    render_grid(
        summary[args.background], results,
        fov_stem=fov_stem, background_name=args.background,
        out_path=png, enhance_label=enhance_label,
    )
    meta = {
        "fov_stem": fov_stem,
        "summary_dir": str(summary_dir),
        "background": args.background,
        "enhance": enhance_label,
        "timestamp": ts.isoformat(timespec="seconds"),
        "results": [r.meta for r in results],
    }
    (out_dir / f"{fov_stem}_bakeoff_{stamp}.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"  → {png}", flush=True)
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CV bake-off on FOV summary images.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--summary-dir", type=Path,
                     help="A single FOV's .../summary directory.")
    src.add_argument("--workspace", type=Path,
                     help="Discover every */summary under this directory.")
    p.add_argument("--methods", default="cp3,classical",
                   help="Comma list: cp3,classical,cp-sam,stardist. Default cp3,classical.")
    p.add_argument("--background", default="mean_M",
                   help="Summary channel for the overlay background / classical input.")
    p.add_argument("--model", default="models/deployed/current_model",
                   help="Cellpose CP3 model spec (path or builtin).")
    p.add_argument("--diameter", type=float, default=None,
                   help="Soma diameter (px). Default: CP3 config default / 12 for classical.")
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--fs", type=float, default=7.5)
    p.add_argument("--enhance", default=None,
                   help="Comma list of enhancement transforms (see cv_bakeoff.enhance).")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                   help=f"Output directory. Default {_DEFAULT_OUT}.")
    p.add_argument("--cpu", action="store_true", help="Force CPU for Cellpose.")
    p.add_argument("--list-models", action="store_true",
                   help="List available CP3 models and exit.")
    p.add_argument("--list-enhance", action="store_true",
                   help="List available --enhance transforms and exit.")
    args = p.parse_args(argv)

    if args.list_models:
        from roigbiv.pipeline.stage1 import list_available_models
        for opt in list_available_models():
            print(f"  {opt['value']}\t({opt['label']})")
        return 0

    if args.list_enhance:
        from cv_bakeoff.enhance import available
        print("Enhancement transforms (chain with commas, args with colons):")
        for name in available():
            print(f"  {name}")
        return 0

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        p.error("no methods selected")

    if args.summary_dir:
        summary_dirs = [args.summary_dir]
    elif args.workspace:
        summary_dirs = _discover_summary_dirs(args.workspace)
        if not summary_dirs:
            p.error(f"no */summary dirs with mean_M.tif under {args.workspace}")
        print(f"Discovered {len(summary_dirs)} FOV(s).", flush=True)
    else:
        p.error("pass --summary-dir or --workspace")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for sdir in summary_dirs:
        _process_one(sdir, args, methods, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
