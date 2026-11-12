"""Command line for the benchmark measurement tools.

Usage (from the repo root)::

    python scripts/bench/cli.py score --root R --pred-dir P [--split dev|test]
                                      [--include-flagged] [--fraction 0.5] [--out FILE]
    python scripts/bench/cli.py audit --root R [--min-fovs 8] [--min-animals 3]
                                      [--min-movies 3] [--out FILE]
    python scripts/bench/cli.py inter-rater --root R [--fraction 0.5] [--out FILE]

``--root`` defaults to ``$ROIGBIV_BENCH_ROOT``. ``score`` reads one
``<pred-dir>/<fov_id>.npy`` per FOV (``(N, 2)`` ``(y, x)`` centroids) and scores
it against the FOV's first annotation, at ``fraction`` of the soma diameter
measured from that annotation. FOVs without a prediction file are listed under
``missing_predictions``. Output is JSON (stdout, or ``--out``). Exit codes: 0
ok, 2 bad arguments or manifest.
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
from bench.schema import FovScore, GroupScore  # noqa: E402
from bench.score import (  # noqa: E402
    DEFAULT_DISTANCE_FRACTION, aggregate, inter_rater, inter_rater_ceiling, max_distance_px,
    score_fov, soma_diameter_px,
)
from bench.split_audit import audit_split, check_g0  # noqa: E402


def _fov_score_dict(s: FovScore) -> dict:
    return {**asdict(s), "precision": s.precision, "recall": s.recall, "f1": s.f1}


def _group_dict(g: GroupScore) -> dict:
    return {**asdict(g), "precision": g.precision, "recall": g.recall}


def cmd_score(args) -> dict:
    bench = load_benchmark(args.root, acquisition=args.acquisition)
    fovs = select(bench, split=args.split, tier=None if args.include_flagged else "core")
    pred_dir = Path(args.pred_dir)
    scores: list[FovScore] = []
    missing: list[str] = []
    for fov in fovs:
        pred_path = pred_dir / f"{fov.id}.npy"
        if not pred_path.exists():
            missing.append(fov.id)
            continue
        gt = load_annotation(bench, fov, 0)
        d = max_distance_px(soma_diameter_px(gt.areas), args.fraction)
        pred = np.load(pred_path).reshape(-1, 2)
        scores.append(score_fov(fov.id, fov.lens, gt.centroids, pred, d))
    overall = aggregate(scores, by="all")
    return {
        "benchmark": bench.name, "version": bench.version, "fraction": args.fraction,
        "per_fov": [_fov_score_dict(s) for s in scores],
        "by_lens": {k: _group_dict(v) for k, v in aggregate(scores, by="lens").items()},
        "all": _group_dict(overall["all"]) if overall else None,
        "missing_predictions": missing,
    }


def cmd_audit(args) -> dict:
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
    }


def cmd_inter_rater(args) -> dict:
    bench = load_benchmark(args.root)
    per_fov: dict[str, dict] = {}
    all_pairs = []
    for fov in bench.fovs:
        if len(fov.annotations) < 2:
            continue
        points = [load_annotation(bench, fov, i) for i in range(len(fov.annotations))]
        raters = {(p.rater or f"a{i}"): p.centroids for i, p in enumerate(points)}
        d = max_distance_px(soma_diameter_px(points[0].areas), args.fraction)
        pairs = inter_rater(fov.id, raters, d)
        all_pairs.extend(pairs)
        per_fov[fov.id] = {
            "pairs": [{**asdict(p), "f1": p.f1} for p in pairs],
            "ceiling": inter_rater_ceiling(pairs),
        }
    return {"benchmark": bench.name, "version": bench.version, "per_fov": per_fov,
            "ceiling": inter_rater_ceiling(all_pairs)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--root", default=None, help="benchmark root (default $ROIGBIV_BENCH_ROOT)")
        sp.add_argument("--out", default=None, help="write JSON here instead of stdout")

    s = sub.add_parser("score", help="score predicted centroids per FOV and per lens")
    common(s)
    s.add_argument("--pred-dir", required=True)
    s.add_argument("--split", choices=("dev", "test"), default=None)
    s.add_argument("--include-flagged", action="store_true")
    s.add_argument("--fraction", type=float, default=DEFAULT_DISTANCE_FRACTION)
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
    r.add_argument("--fraction", type=float, default=DEFAULT_DISTANCE_FRACTION)
    r.set_defaults(func=cmd_inter_rater)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except (ManifestError, FileNotFoundError, ValueError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
