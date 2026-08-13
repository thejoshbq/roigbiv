"""Diagnose why a set of same-FOV sessions does not match.

    python -m scripts.registry_diagnose.run_diagnose \
        --workspace data/logan_cousa_trial \
        --out /tmp/diagnose.json

Reads each session's ``merged_masks.tif`` and ``summary/mean_M.tif``, builds a
ROICaT-independent ground-truth pairing from the centroids, then runs the real
matcher under several configurations and reports where the true pairs are lost.

Read-only: no registry database is opened and nothing is written except the
report.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from roigbiv.registry.roicat_adapter import (
    AdapterConfig,
    centroids_from_merged_masks,
    load_session_input,
)
from scripts.registry_diagnose.ground_truth import pair_sessions, transitive_cells
from scripts.registry_diagnose.probe import probe


def build_configs(base: AdapterConfig, only: list[str] | None) -> dict:
    """The configurations worth separating, each isolating one suspect.

    ``as_shipped`` is the control. ``footprint_only`` zeroes the two channels
    that canonical-disk ROIs should render uninformative; if it recovers pairs
    the control loses, the mixing is the fault. ``phase_correlation`` swaps the
    alignment method only; if that recovers them, the warp is the fault.
    """
    configs = {
        "as_shipped": base,
        "footprint_only": replace(base, power_NN=0.0, power_SWT=0.0),
        "phase_correlation": replace(base, alignment_method="PhaseCorrelation"),
        "no_alignment": replace(base, alignment_method="NullRegistration"),
        # ROICaT blends a density image of the ROI footprints into each mean
        # projection before aligning. Canonical disks make that a field of
        # identical blobs; if it is what the aligner is choking on, dropping it
        # is the fix and these two say so.
        "no_roi_mixing": replace(base, roi_mixing_factor=0.0),
        "pc_no_roi_mixing": replace(base, alignment_method="PhaseCorrelation",
                                    roi_mixing_factor=0.0),
        "footprint_only_no_mixing": replace(base, power_NN=0.0, power_SWT=0.0,
                                            roi_mixing_factor=0.0),
    }
    if only:
        missing = [k for k in only if k not in configs]
        if missing:
            raise SystemExit(f"unknown config(s): {missing}; "
                             f"choose from {sorted(configs)}")
        configs = {k: configs[k] for k in only}
    return configs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path,
                    help="workspace root; sessions read from its output/ dir "
                         "in session_order.json order")
    ap.add_argument("--session", type=Path, action="append", default=[],
                    help="explicit session output dir (repeatable, ordered); "
                         "overrides --workspace discovery")
    ap.add_argument("--out", type=Path, required=True, help="report JSON path")
    ap.add_argument("--config", action="append", default=[],
                    help="run only these configs (repeatable)")
    ap.add_argument("--mode", choices=("bundle", "sequential", "both"),
                    default="bundle",
                    help="bundle: cluster all sessions at once. sequential: "
                         "replay how registration actually runs — session k "
                         "clustered against sessions 0..k-1 (default bundle)")
    ap.add_argument("--max-distance", type=float, default=25.0,
                    help="ground-truth pairing gate in px (default 25)")
    ap.add_argument("--d-cutoff", type=float, default=None,
                    help="bypass ROICaT's crossover inference")
    ap.add_argument("--prealign", action="store_true",
                    help="shift every session into session 0's frame using the "
                         "ground-truth translation before handing it to the "
                         "matcher — isolates how much the aligner is costing")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each config N times to expose nondeterminism")
    ap.add_argument("--device", default=None, help="torch device override")
    args = ap.parse_args(argv)

    dirs = args.session or _discover(args.workspace)
    if len(dirs) < 2:
        raise SystemExit("need at least two sessions to diagnose a match")

    sessions = [load_session_input(d, session_key=d.name) for d in dirs]
    names = [s.session_key for s in sessions]
    # Centroids in ascending-label_id order — the order cluster_sessions
    # concatenates ROIs in, so a ground-truth index is a matcher index.
    centroids = [centroids_from_merged_masks(s.merged_masks).astype(np.float64)
                 for s in sessions]

    print(f"sessions ({len(sessions)}):")
    for name, c in zip(names, centroids):
        print(f"  {name}  {len(c)} ROIs")

    pairings = {}
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            p = pair_sessions(names[i], centroids[i], sessions[i].mean_m,
                              names[j], centroids[j], sessions[j].mean_m,
                              max_distance=args.max_distance)
            pairings[(i, j)] = p
            print(f"  ground truth {i}<->{j}: {p.n_pairs} pairs via "
                  f"{p.shift_source} shift {np.round(p.shift_yx, 1).tolist()}, "
                  f"median residual {p.median_residual}, "
                  f"proposals {p.proposal_scores}")

    truth = transitive_cells(names, pairings)
    spans = {}
    for cell in truth:
        spans[len(cell)] = spans.get(len(cell), 0) + 1
    print(f"  ground truth cells: {len(truth)} "
          f"(by session span: {dict(sorted(spans.items()))})")

    if args.prealign:
        shifts = [(0.0, 0.0)] + [pairings[(0, k)].shift_yx
                                 for k in range(1, len(sessions))]
        sessions = [_shifted(s, sh) for s, sh in zip(sessions, shifts)]
        centroids = [c + np.asarray(sh) for c, sh in zip(centroids, shifts)]
        print("  pre-aligned by " + str([tuple(np.round(s, 1)) for s in shifts]))

    base = AdapterConfig(d_cutoff=args.d_cutoff)
    if args.device:
        base = replace(base, device=args.device)

    # Each entry is (label, session prefix). "bundle" is the diagnostic view —
    # everything in one clustering. "sequential" is what register_or_match
    # actually does, one session joining the ones already registered, and its
    # posterior is the number that decides a real run.
    slices = []
    if args.mode in ("bundle", "both"):
        slices.append(("bundle", len(sessions)))
    if args.mode in ("sequential", "both"):
        slices += [(f"seq{k}", k + 1) for k in range(1, len(sessions))]

    results = []
    for name, cfg in build_configs(base, args.config).items():
      for rep in range(args.repeat):
        for slice_label, k in slices:
            label = name if slice_label == "bundle" and args.mode == "bundle" \
                else f"{name}/{slice_label}"
            if args.repeat > 1:
                label = f"{label}#{rep + 1}"
            print(f"\n--- {label} ---", flush=True)
            r = probe(label, sessions[:k], cfg,
                      truth=[{s: r_ for s, r_ in cell.items() if s < k}
                             for cell in truth
                             if sum(1 for s in cell if s < k) >= 2],
                      raw_centroids=centroids[:k])
            results.append(r)
            _print_summary(r)

    report = {
        "sessions": names,
        "n_rois": [len(c) for c in centroids],
        "ground_truth": {
            "max_distance_px": args.max_distance,
            "pairings": [p.to_dict() for p in pairings.values()],
            "n_cells": len(truth),
            "cells_by_session_span": {str(k): v for k, v in sorted(spans.items())},
        },
        "probes": [r.to_dict() for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out}")
    return 0


def _shifted(session, shift_yx):
    """*session* translated by *shift_yx*, mean image and labels together.

    Integer translation only, and the label image moves by the same rounded
    amount as the mean projection — interpolating a label image would invent
    ROI ids that were never detected.
    """
    from roigbiv.registry.roicat_adapter import SessionInput

    dy, dx = int(round(shift_yx[0])), int(round(shift_yx[1]))
    if dy == 0 and dx == 0:
        return session

    def move(arr):
        out = np.zeros_like(arr)
        h, w = arr.shape
        ys, xs = slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx))
        sy, sx = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
        out[ys, xs] = arr[sy, sx]
        return out

    return SessionInput(session_key=session.session_key,
                        mean_m=move(session.mean_m),
                        merged_masks=move(session.merged_masks))


def _discover(workspace: Path | None) -> list[Path]:
    if workspace is None:
        raise SystemExit("pass --workspace or at least two --session paths")
    from roigbiv.pipeline.session_order import load_order

    output_root = workspace / "output"
    entries = load_order(workspace)
    if entries:
        return [output_root / e.stem for e in entries]
    return sorted(d for d in output_root.iterdir()
                  if (d / "merged_masks.tif").exists())


def _print_summary(r) -> None:
    if r.error:
        print(f"  FAILED after {r.elapsed_s:.0f}s: {r.error}")
        return
    print(f"  {r.elapsed_s:.0f}s  alignment={r.alignment_method} "
          f"inlier_rate={r.alignment_inlier_rate:.3f} "
          f"shifts={[None if s is None else tuple(round(v, 1) for v in s) for s in r.implied_shift_yx]}")
    print(f"  true-pair centroid distance  before={r.centroid_residual_before} ")
    print(f"                                after={r.centroid_residual_after}")
    for c in r.channels:
        d = c.to_dict()
        print(f"  {d['channel']:<8} true_med={d['true_median']} "
              f"rand_p90={d['random_p90']} AUC={d['auc']}")
    print(f"  pruning: {r.pruned_survival} (d_cutoff={r.d_cutoff})")
    print(f"  clusters: {r.clusters_recovered}")
    print(f"  -> posterior={r.posterior:.3f} decision={r.decision}")


if __name__ == "__main__":
    sys.exit(main())
