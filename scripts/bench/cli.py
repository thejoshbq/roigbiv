"""Command line for the benchmark measurement tools.

Usage (from the repo root)::

    python scripts/bench/cli.py score --root R --pred-dir P [--split dev|test]
                                      [--include-flagged] [--fraction 0.5] [--allow-missing]
                                      [--no-verify] [--acquisition FILE] [--out FILE]
    python scripts/bench/cli.py audit --root R [--min-fovs 8] [--min-animals 3]
                                      [--min-movies 3] [--out FILE]
    python scripts/bench/cli.py inter-rater --root R [--split dev|test] [--include-flagged]
                                      [--fraction 0.5] [--no-verify] [--out FILE]

``--root`` defaults to ``$ROIGBIV_BENCH_ROOT``. ``score`` reads one
``<pred-dir>/<fov_id>.npy`` per FOV (``(N, 2)`` ``(y, x)`` centroids) and scores
it against the FOV's first annotation, at ``fraction`` of the soma diameter
measured from that annotation. ``score`` and ``inter-rater`` cover core FOVs of
both splits unless told otherwise, check every file's sha256 first unless
``--no-verify``, and record those settings in the output. A FOV whose parsed
ROI count differs from the manifest's ``n_rois`` is listed under
``annotation_count_mismatches`` (POINT RoiSets can legitimately differ).
Output is JSON (stdout, or ``--out``). Exit codes: 0 ok; 1 a FOV has no
prediction file (output still written; ``--allow-missing`` makes this 0);
2 bad arguments, manifest or prediction file.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
# Run as a script, this folder is sys.path[0]; drop it so its module names
# cannot shadow top-level imports, and put scripts/ on the path instead.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
sys.path.insert(0, str(_HERE.parent))

from bench.manifest import ManifestError, load_annotation, load_benchmark, select  # noqa: E402
from bench.schema import FOV, AnnotationPoints, Benchmark, FovScore, GroupScore  # noqa: E402
from bench.score import (  # noqa: E402
    DEFAULT_DISTANCE_FRACTION, aggregate, inter_rater, inter_rater_ceiling, max_distance_px,
    score_fov, soma_diameter_px,
)
from bench.split_audit import audit_split, check_g0  # noqa: E402


def _fov_score_dict(s: FovScore) -> dict:
    return {**asdict(s), "precision": s.precision, "recall": s.recall, "f1": s.f1}


def _group_dict(g: GroupScore) -> dict:
    return {**asdict(g), "precision": g.precision, "recall": g.recall}


def _settings(args, bench: Benchmark) -> dict:
    return {
        "benchmark": bench.name, "version": bench.version, "fraction": args.fraction,
        "split": args.split, "include_flagged": args.include_flagged, "verified": not args.no_verify,
    }


def _fovs(args, bench: Benchmark) -> tuple[FOV, ...]:
    return select(bench, split=args.split, tier=None if args.include_flagged else "core")


def _count_mismatch(fov: FOV, index: int, pts: AnnotationPoints) -> Optional[dict]:
    expected = fov.annotations[index].n_rois
    if expected == len(pts.centroids):
        return None
    return {"fov_id": fov.id, "index": index, "n_rois_manifest": expected, "n_parsed": len(pts.centroids)}


def _load_pred(path: Path, fov_id: str) -> np.ndarray:
    pred = np.load(path)
    if pred.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"{fov_id}: prediction must be (N, 2) (y, x), got shape {pred.shape}")
    return pred


def cmd_score(args) -> tuple[dict, int]:
    bench = load_benchmark(args.root, acquisition=args.acquisition, verify=not args.no_verify)
    pred_dir = Path(args.pred_dir)
    scores: list[FovScore] = []
    rows: list[dict] = []
    missing: list[str] = []
    mismatches: list[dict] = []
    for fov in _fovs(args, bench):
        pred_path = pred_dir / f"{fov.id}.npy"
        if not pred_path.exists():
            missing.append(fov.id)
            continue
        gt = load_annotation(bench, fov, 0)
        mismatch = _count_mismatch(fov, 0, gt)
        if mismatch:
            mismatches.append(mismatch)
        d = max_distance_px(soma_diameter_px(gt.areas), args.fraction)
        s = score_fov(fov.id, fov.lens, gt.centroids, _load_pred(pred_path, fov.id), d)
        scores.append(s)
        rows.append({**_fov_score_dict(s), "n_rois_manifest": fov.annotations[0].n_rois})
    overall = aggregate(scores, by="all")
    result = {
        **_settings(args, bench),
        "per_fov": rows,
        "by_lens": {k: _group_dict(v) for k, v in aggregate(scores, by="lens").items()},
        "all": _group_dict(overall["all"]) if overall else None,
        "missing_predictions": missing,
        "annotation_count_mismatches": mismatches,
    }
    if missing and not args.allow_missing:
        print(f"bench: no prediction file for {', '.join(missing)} (aggregates exclude them)",
              file=sys.stderr)
        return result, 1
    return result, 0


def cmd_audit(args) -> tuple[dict, int]:
    bench = load_benchmark(args.root)
    kw = {"min_fovs": args.min_fovs, "min_animals": args.min_animals, "min_movies": args.min_movies}
    core = check_g0(bench.fovs, **kw)
    flagged = check_g0(bench.fovs, include_flagged=True, **kw)
    audit = audit_split(bench.fovs)
    return {
        "benchmark": bench.name, "version": bench.version,
        "split_audit": {**asdict(audit), "clean": audit.clean},
        "g0_core": {**asdict(core), "passed": core.passed},
        "g0_incl_flagged": {**asdict(flagged), "passed": flagged.passed},
    }, 0


def cmd_inter_rater(args) -> tuple[dict, int]:
    bench = load_benchmark(args.root, verify=not args.no_verify)
    per_fov: dict[str, dict] = {}
    all_pairs = []
    mismatches: list[dict] = []
    for fov in _fovs(args, bench):
        if len(fov.annotations) < 2:
            continue
        points = [load_annotation(bench, fov, i) for i in range(len(fov.annotations))]
        raters: dict = {}
        for i, p in enumerate(points):
            name = p.rater or f"a{i}"
            if name in raters:
                name = f"{name}#{i}"
            raters[name] = p.centroids
            mismatch = _count_mismatch(fov, i, p)
            if mismatch:
                mismatches.append(mismatch)
        d = max_distance_px(soma_diameter_px(points[0].areas), args.fraction)
        pairs = inter_rater(fov.id, raters, d)
        all_pairs.extend(pairs)
        per_fov[fov.id] = {
            "pairs": [{**asdict(p), "f1": p.f1} for p in pairs],
            "ceiling": inter_rater_ceiling(pairs),
        }
    return {**_settings(args, bench), "per_fov": per_fov, "ceiling": inter_rater_ceiling(all_pairs),
            "annotation_count_mismatches": mismatches}, 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--root", default=None, help="benchmark root (default $ROIGBIV_BENCH_ROOT)")
        sp.add_argument("--out", default=None, help="write JSON here instead of stdout")

    def population(sp):
        sp.add_argument("--split", choices=("dev", "test"), default=None)
        sp.add_argument("--include-flagged", action="store_true")
        sp.add_argument("--fraction", type=float, default=DEFAULT_DISTANCE_FRACTION)
        sp.add_argument("--no-verify", action="store_true", help="skip the sha256 check")

    s = sub.add_parser("score", help="score predicted centroids per FOV and per lens")
    common(s)
    population(s)
    s.add_argument("--pred-dir", required=True)
    s.add_argument("--allow-missing", action="store_true", help="exit 0 when a FOV has no prediction")
    s.add_argument("--acquisition", default=None, help="frame-rate sidecar JSON")
    s.set_defaults(func=cmd_score)

    a = sub.add_parser("audit", help="split leak check and the G0 gate")
    common(a)
    a.add_argument("--min-fovs", type=int, default=8)
    a.add_argument("--min-animals", type=int, default=3)
    a.add_argument("--min-movies", type=int, default=3)
    a.set_defaults(func=cmd_audit)

    r = sub.add_parser("inter-rater", help="pairwise rater agreement on FOVs with 2+ annotations")
    common(r)
    population(r)
    r.set_defaults(func=cmd_inter_rater)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, code = args.func(args)
    except (ManifestError, FileNotFoundError, ValueError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
