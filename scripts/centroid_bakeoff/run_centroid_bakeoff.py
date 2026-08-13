"""Centroid bake-off CLI — OpenCV vs. Cellpose vs. Suite2p on real + synthetic FOVs.

First step toward cross-session cell registration (see
docs/adr/0003-centroid-canonical-roi-stamps.md): benchmarks bare centroid
localization against ground truth, since pyramidal neurons' apical dendrites
make boundary segmentation unreliable for this cell type. Point-first
counterpart to scripts/cv_bakeoff/ (which compares segmentation masks).

Examples
--------
Real FOVs only, all three methods::

    conda run -n roigbiv python scripts/centroid_bakeoff/run_centroid_bakeoff.py \\
        --real-fov-dir data/BEGINNER_ROIS/LM_RoiSets/LM_RoiSets/TDT4_ENSURESA \\
        --fs 30.0

Add the synthetic-injection arm::

    conda run -n roigbiv python scripts/centroid_bakeoff/run_centroid_bakeoff.py \\
        --real-fov-dir data/BEGINNER_ROIS/LM_RoiSets/LM_RoiSets/TDT4_ENSURESA \\
        --synthetic --synthetic-seed 0 --fs 30.0

One method only, forcing CPU::

    conda run -n roigbiv python scripts/centroid_bakeoff/run_centroid_bakeoff.py \\
        --real-fov-dir data/BEGINNER_ROIS/LM_RoiSets/LM_RoiSets/TDT4_ENSURESA \\
        --methods cellpose --cpu

Sweep each method's operating-point knob(s) instead of one fixed point —
adds a PR-curve PNG + best-F1 point per (FOV, method), on top of the default
single-point report (additive, not a replacement)::

    conda run -n roigbiv python scripts/centroid_bakeoff/run_centroid_bakeoff.py \\
        --real-fov-dir data/BEGINNER_ROIS/LM_RoiSets/LM_RoiSets/TDT4_ENSURESA \\
        --sweep --fs 30.0

Add ``--sweep-quick`` while iterating — skips Suite2p's structural
threshold_scaling grid (the expensive ~12s/run part) and only rescores its
iscell probabilities on a single default run.

Outputs (overlay PNGs + JSON report) land in experiments/runs/centroid_bakeoff/.
Nothing is written to inference/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

# scripts/ on path so ``centroid_bakeoff`` imports as a package (repo convention,
# matches scripts/cv_bakeoff/run_bakeoff.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from centroid_bakeoff.detector import CentroidDetectorInputs  # noqa: E402
from centroid_bakeoff.ground_truth import (  # noqa: E402
    build_synthetic_fov, discover_real_pairs, imagej_roiset_to_centroids,
)
from centroid_bakeoff.point_match import match_points  # noqa: E402
from centroid_bakeoff.report import (  # noqa: E402
    build_aggregate, build_fov_report, build_sweep_report,
    print_max_distance_sensitivity, print_summary_table, print_sweep_best_table,
    render_overlay_grid, render_pr_curve, write_json_report, write_sweep_json_report,
)
from centroid_bakeoff.sweep import (  # noqa: E402
    SweepPoint, SweepResult, max_distance_sensitivity, param_grid_sweep, rescore_sweep,
)

_DEFAULT_OUT = Path("experiments/runs/centroid_bakeoff")
_GRIN_FALLBACK_DIAMETER = 12.0  # roigbiv/pipeline/types.py's own GRIN-profile default

# Sweep grids — see docs/adr n/a; rationale lives in the Phase 2 plan
# (now-that-we-seem-parallel-wolf.md). Centered on production defaults:
# cellprob_threshold=-2.0 (types.py), threshold_scaling=1.0 (suite2p default_ops).
_CELLPROB_GRID = [-6.0, -4.0, -2.0, -1.0, 0.0, 2.0, 4.0, 6.0]
_THRESHOLD_SCALING_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]
_THRESHOLD_SCALING_GRID_QUICK = [1.0]
_MAX_DISTANCE_MULTIPLIERS = [0.5, 1.0, 1.5]
_OPENCV_SWEEP_GRID = {
    "min_circularity": [0.1, 0.3, 0.5],
    "min_convexity": [0.3, 0.5, 0.8],
    "min_threshold": [10.0, 50.0],
    "max_threshold": [220.0, 250.0],
}


def _score_thresholds(scores: np.ndarray, n: int = 12) -> list:
    """Adaptive rescore grid from a result's own score distribution — avoids
    hardcoding magic numbers for confidences whose natural range differs by
    method (iscell probability in [0,1] vs. Cellpose's mean cellprob)."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return [0.0]
    pct = np.linspace(0.0, 95.0, n)
    return sorted({round(float(v), 4) for v in np.percentile(scores, pct)})


def _build_detector(method: str, args, work_dir: Path):
    if method == "opencv":
        from centroid_bakeoff.detectors.opencv_blob import OpenCVBlobDetector
        return OpenCVBlobDetector(channel=args.opencv_channel)
    if method == "cellpose":
        from centroid_bakeoff.detectors.cellpose_centroid import CellposeCentroidDetector
        from roigbiv.pipeline.types import PipelineConfig
        cfg = PipelineConfig(fs=args.fs, force_cpu=args.cpu)
        return CellposeCentroidDetector(cfg=cfg)
    if method == "suite2p":
        from centroid_bakeoff.detectors.suite2p_centroid import Suite2pCentroidDetector
        return Suite2pCentroidDetector(
            work_dir=work_dir / "suite2p_bench",
            iscell_threshold=args.iscell_threshold,
            lean=not args.suite2p_full,
        )
    if method == "consensus":
        from centroid_bakeoff.consensus import ConsensusCentroidDetector, ConsensusModel
        from roigbiv.pipeline.types import PipelineConfig
        # Missing/nonexistent model path falls back to the hand-prior default
        # (ConsensusModel.load's own behavior, mirroring CalibrationModel.load)
        # -- never hard-fails just because fit_consensus.py hasn't been run yet.
        model = ConsensusModel.load(args.consensus_model_path)
        cp_cfg = PipelineConfig(fs=args.fs, force_cpu=args.cpu)
        return ConsensusCentroidDetector(
            model=model, cellpose_cfg=cp_cfg,
            suite2p_work_dir=work_dir / "suite2p_consensus",
        )
    raise SystemExit(f"unknown method {method!r}")


def _sweep_cellpose(inputs, gt, max_distance, cfg, fov_stem) -> SweepResult:
    """2-stage sweep: structural cellprob_threshold grid (rerun per value),
    plus a free rescore over each run's own mean-cellprob scores."""
    from centroid_bakeoff.detectors.cellpose_centroid import CellposeCentroidDetector

    points = []
    for cpt in _CELLPROB_GRID:
        det = CellposeCentroidDetector(cfg=cfg, cellprob_threshold=cpt)
        t0 = time.time()
        result = det.detect(inputs)
        elapsed = time.time() - t0
        m = match_points(gt, result.centroids, max_distance=max_distance)
        points.append(SweepPoint(
            params={"cellprob_threshold": cpt}, match=m, n_pred=result.n,
            runtime_s=round(elapsed, 2), centroids=result.centroids,
        ))
        if result.n > 0:
            rescored = rescore_sweep(
                result, gt, max_distance, thresholds=_score_thresholds(result.scores),
                method="cellpose", fov_stem=fov_stem, param_name="min_mean_cellprob",
            )
            for p in rescored.points:
                p.params = {"cellprob_threshold": cpt, **p.params}
            points.extend(rescored.points)
    return SweepResult(method="cellpose", fov_stem=fov_stem, points=points)


def _sweep_suite2p(inputs, gt, max_distance, work_dir, args, fov_stem) -> SweepResult:
    """2-stage sweep: structural threshold_scaling grid (rerun per value, each
    unfiltered via iscell_threshold=0.0), plus a free rescore over each run's
    iscell probabilities."""
    from centroid_bakeoff.detectors.suite2p_centroid import Suite2pCentroidDetector

    ts_grid = _THRESHOLD_SCALING_GRID_QUICK if args.sweep_quick else _THRESHOLD_SCALING_GRID
    points = []
    for ts in ts_grid:
        det = Suite2pCentroidDetector(
            work_dir=work_dir / "suite2p_sweep", iscell_threshold=0.0,
            lean=not args.suite2p_full, threshold_scaling=ts,
        )
        t0 = time.time()
        result = det.detect(inputs)
        elapsed = time.time() - t0
        m = match_points(gt, result.centroids, max_distance=max_distance)
        points.append(SweepPoint(
            params={"threshold_scaling": ts}, match=m, n_pred=result.n,
            runtime_s=round(elapsed, 2), centroids=result.centroids,
        ))
        if result.n > 0:
            rescored = rescore_sweep(
                result, gt, max_distance, thresholds=_score_thresholds(result.scores),
                method="suite2p", fov_stem=fov_stem, param_name="iscell_threshold",
            )
            for p in rescored.points:
                p.params = {"threshold_scaling": ts, **p.params}
            points.extend(rescored.points)
    return SweepResult(method="suite2p", fov_stem=fov_stem, points=points)


def _sweep_opencv(inputs, gt, max_distance, args, fov_stem) -> SweepResult:
    """Pure structural grid — no reusable per-candidate confidence to rescore."""
    from centroid_bakeoff.detectors.opencv_blob import OpenCVBlobDetector

    def factory(**combo):
        return OpenCVBlobDetector(channel=args.opencv_channel, **combo)

    return param_grid_sweep(
        factory, _OPENCV_SWEEP_GRID, inputs, gt, max_distance,
        method="opencv", fov_stem=fov_stem,
    )


def _run_foundation_for(tif_path: Path, args, work_dir: Path):
    """Run Foundation to get summary images. Returns (mean_M, vcorr_S, max_S, dog_map)."""
    from roigbiv.pipeline.foundation import run_foundation
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=args.fs, do_registration=False, force_cpu=args.cpu)
    fov_out = work_dir / "foundation"
    fov_data = run_foundation(tif_path, cfg, fov_out)
    return fov_data.mean_M, fov_data.vcorr_S, fov_data.max_S, fov_data.dog_map


def _load_summary_dir(summary_dir: Path) -> dict[str, np.ndarray]:
    """Load a precomputed */summary dir's *.tif files (cv_bakeoff's escape hatch)."""
    out: dict[str, np.ndarray] = {}
    for tif in sorted(summary_dir.glob("*.tif")):
        out[tif.stem] = tifffile.imread(str(tif)).astype(np.float32)
    return out


def _process_fov(
    fov_stem: str, gt_source: str, gt: np.ndarray, tif_path: Path,
    summary: dict, args, out_dir: Path,
) -> tuple:
    """Returns (fov_report, sweep_report_or_None, sensitivity_or_None)."""
    from roigbiv.pipeline.optics import measure_soma_scale

    work_dir = out_dir / "_work" / fov_stem
    work_dir.mkdir(parents=True, exist_ok=True)

    mean_M = summary.get("mean_M")
    dog_map = summary.get("dog_map")
    soma_scale = measure_soma_scale(mean_M, dog_map) if mean_M is not None else None

    if args.max_distance_px is not None:
        max_distance = args.max_distance_px
    elif soma_scale is not None and soma_scale.ok:
        max_distance = soma_scale.diameter_med / 2.0
    else:
        max_distance = _GRIN_FALLBACK_DIAMETER / 2.0

    shape = mean_M.shape if mean_M is not None else next(iter(summary.values())).shape
    inputs = CentroidDetectorInputs(
        summary=summary, fov_stem=fov_stem, shape=shape, fs=args.fs,
        raw_tif_path=tif_path, soma_scale=soma_scale,
    )

    method_results = {}
    for method in args.methods:
        print(f"  [{fov_stem}] running {method}…", flush=True)
        det = _build_detector(method, args, work_dir)
        t0 = time.time()
        try:
            result = det.detect(inputs)
        except Exception as exc:  # noqa: BLE001 — diagnostic: log + continue
            print(f"    ! {method} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        match = match_points(gt, result.centroids, max_distance=max_distance)
        method_results[method] = (result, match)
        print(f"    {method}: n_pred={result.n} tp={match.n_tp} fp={match.n_fp} "
              f"fn={match.n_fn} f1={match.f1} ({time.time()-t0:.1f}s)", flush=True)

    png_path = out_dir / f"{fov_stem}_overlay.png"
    bg = mean_M if mean_M is not None else next(iter(summary.values()))
    if method_results:
        render_overlay_grid(bg, gt, method_results, fov_stem=fov_stem,
                            gt_source=gt_source, out_path=png_path)
        print(f"  → {png_path}", flush=True)

    fov_report = build_fov_report(fov_stem, gt_source, gt, method_results)

    if not args.sweep:
        return fov_report, None, None

    print(f"  [{fov_stem}] sweeping…", flush=True)
    sweep_results: dict[str, SweepResult] = {}
    if "cellpose" in args.methods:
        from roigbiv.pipeline.types import PipelineConfig
        cp_cfg = PipelineConfig(fs=args.fs, force_cpu=args.cpu)
        sweep_results["cellpose"] = _sweep_cellpose(inputs, gt, max_distance, cp_cfg, fov_stem)
    if "suite2p" in args.methods:
        sweep_results["suite2p"] = _sweep_suite2p(inputs, gt, max_distance, work_dir, args, fov_stem)
    if "opencv" in args.methods:
        sweep_results["opencv"] = _sweep_opencv(inputs, gt, max_distance, args, fov_stem)

    sweep_report = None
    sensitivity = None
    if sweep_results:
        pr_png = out_dir / f"{fov_stem}_pr_curve.png"
        render_pr_curve(sweep_results, fov_stem=fov_stem, out_path=pr_png)
        print(f"  → {pr_png}", flush=True)
        sweep_report = build_sweep_report(fov_stem, gt_source, len(gt), sweep_results)

        if max_distance is not None:
            # Multiples of whatever max_distance the single-point path actually
            # used (soma-radius derived, or the GRIN fallback) — not a
            # separately-derived radius, so a FOV where measure_soma_scale
            # isn't `.ok` still gets a sensitivity table instead of silently
            # skipping it.
            distances = [round(max_distance * mult, 2) for mult in _MAX_DISTANCE_MULTIPLIERS]
            per_method = {}
            for method, sweep in sweep_results.items():
                best = sweep.best
                if best is None or best.centroids is None:
                    continue
                rematches = max_distance_sensitivity(best, gt, distances)
                per_method[method] = {d: r.to_dict() for d, r in zip(distances, rematches)}
            if per_method:
                sensitivity = {fov_stem: per_method}

    return fov_report, sweep_report, sensitivity


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Centroid-detection bake-off: OpenCV vs. Cellpose vs. Suite2p.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--real-fov-dir", type=Path, default=None,
                   help="Root to recursively discover *_mc.tif + *_RoiSet[_FINAL].zip pairs.")
    p.add_argument("--summary-dir", type=Path, default=None,
                   help="A single already-processed FOV's .../summary directory "
                        "(escape hatch; requires --raw-tif for the suite2p method).")
    p.add_argument("--raw-tif", type=Path, default=None,
                   help="Raw/mc TIF paired with --summary-dir (suite2p needs the movie).")
    p.add_argument("--synthetic", action="store_true",
                   help="Add a synthetic soma-injection FOV as a GT arm.")
    p.add_argument("--synthetic-seed", type=int, default=0)
    p.add_argument("--synthetic-shape", default="300,512,512",
                   help="T,H,W for the synthetic movie. Default 300,512,512.")
    p.add_argument("--methods", default="opencv,cellpose,suite2p",
                   help="Comma list: opencv,cellpose,suite2p,consensus.")
    p.add_argument("--consensus-model-path", type=Path,
                   default=Path("experiments/runs/centroid_bakeoff_consensus/consensus_model.json"),
                   help="Fitted ConsensusModel JSON from fit_consensus.py. Missing/nonexistent "
                        "path falls back to a hand-prior default (no crash). Only used by "
                        "--methods consensus.")
    p.add_argument("--fs", type=float, default=30.0)
    p.add_argument("--max-distance-px", type=float, default=None,
                   help="Point-match tolerance. Default: one soma radius via "
                        "measure_soma_scale, else 6px (GRIN-profile fallback).")
    p.add_argument("--iscell-threshold", type=float, default=0.3,
                   help="Suite2p iscell[:,1] cutoff. Matches PipelineConfig default.")
    p.add_argument("--suite2p-full", action="store_true",
                   help="Disable the lean spikedetect=False override (parity check).")
    p.add_argument("--opencv-channel", default="dog_map", choices=["dog_map", "mean_M"])
    p.add_argument("--cpu", action="store_true", help="Force CPU for Cellpose/Foundation.")
    p.add_argument("--sweep", action="store_true",
                   help="Also sweep each method's operating-point knob(s) and report "
                        "a precision/recall curve + best-F1 point, instead of only the "
                        "single fixed (production-default) point. Additive: the default "
                        "single-point report/JSON is unchanged and always produced.")
    p.add_argument("--sweep-quick", action="store_true",
                   help="Shrink Suite2p's structural threshold_scaling grid to a single "
                        "value (rescore-only) for fast iteration. No effect without --sweep.")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args(argv)

    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not args.methods:
        p.error("no methods selected")

    if not args.real_fov_dir and not args.summary_dir and not args.synthetic:
        p.error("pass --real-fov-dir, --summary-dir, and/or --synthetic")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fov_reports: list[dict] = []
    sweep_reports: list[dict] = []
    sensitivity_all: dict = {}

    def _collect(result: tuple) -> None:
        fov_report, sweep_report, sensitivity = result
        fov_reports.append(fov_report)
        if sweep_report is not None:
            sweep_reports.append(sweep_report)
        if sensitivity is not None:
            sensitivity_all.update(sensitivity)

    if args.real_fov_dir:
        pairs = discover_real_pairs([args.real_fov_dir])
        print(f"Discovered {len(pairs)} real FOV(s) under {args.real_fov_dir}", flush=True)
        for mc_tif, roi_zip, stem in pairs:
            print(f"\n== {stem} (real) ==", flush=True)
            with tifffile.TiffFile(str(mc_tif)) as tf:
                shape = tf.pages[0].shape
            gt, _names = imagej_roiset_to_centroids(roi_zip, shape)
            work_dir = out_dir / "_work" / stem
            mean_M, vcorr_S, max_S, dog_map = _run_foundation_for(mc_tif, args, work_dir)
            summary = {"mean_M": mean_M, "vcorr_S": vcorr_S, "max_S": max_S, "dog_map": dog_map}
            _collect(_process_fov(stem, "real", gt, mc_tif, summary, args, out_dir))

    if args.summary_dir:
        stem = args.summary_dir.parent.name
        print(f"\n== {stem} (precomputed summary) ==", flush=True)
        summary = _load_summary_dir(args.summary_dir)
        gt = np.zeros((0, 2), dtype=np.float32)  # no GT available via this path
        _collect(_process_fov(stem, "none", gt, args.raw_tif, summary, args, out_dir))

    if args.synthetic:
        stem = f"synthetic_seed{args.synthetic_seed}"
        print(f"\n== {stem} (synthetic) ==", flush=True)
        T, H, W = (int(x) for x in args.synthetic_shape.split(","))
        movie, gt, _specs = build_synthetic_fov(
            shape=(T, H, W), fs=args.fs, seed=args.synthetic_seed,
        )
        work_dir = out_dir / "_work" / stem
        work_dir.mkdir(parents=True, exist_ok=True)
        synth_tif = work_dir / f"{stem}.tif"
        tifffile.imwrite(str(synth_tif), movie.astype(np.float32))
        mean_M, vcorr_S, max_S, dog_map = _run_foundation_for(synth_tif, args, work_dir)
        summary = {"mean_M": mean_M, "vcorr_S": vcorr_S, "max_S": max_S, "dog_map": dog_map}
        _collect(_process_fov(stem, "synthetic", gt, synth_tif, summary, args, out_dir))

    if not fov_reports:
        p.error("no FOVs processed")

    aggregate = build_aggregate(fov_reports)
    json_path = write_json_report(
        fov_reports, aggregate, out_dir / "centroid_bakeoff_report.json",
    )
    print(f"\n=== SUMMARY (micro-averaged across {len(fov_reports)} FOV(s)) ===", flush=True)
    print_summary_table(aggregate)
    print(f"\nReport: {json_path}", flush=True)
    print(f"Overlays: {out_dir}/*_overlay.png", flush=True)

    if args.sweep and sweep_reports:
        sweep_json_path = write_sweep_json_report(
            sweep_reports, out_dir / "centroid_bakeoff_sweep_report.json",
        )
        print(f"\n=== SWEEP: BEST OPERATING POINT PER (FOV, METHOD) ===", flush=True)
        print_sweep_best_table(sweep_reports)
        print(f"\nSweep report: {sweep_json_path}", flush=True)
        print(f"PR curves: {out_dir}/*_pr_curve.png", flush=True)
        if sensitivity_all:
            print_max_distance_sensitivity(sensitivity_all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
