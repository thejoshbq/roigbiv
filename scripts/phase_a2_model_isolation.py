#!/usr/bin/env python
"""Phase A2 — isolating model test (deployed-finetuned vs cpsam), single variable.

Gated-workflow plan: confirm root cause #1 (the deployed CP3 model is overfit to
bright round GRIN somata and does not fire on diffuse PRISM cells) as ONE isolated
variable, instead of inferring it backward from a multi-param profile bundle.

Holds EVERYTHING constant — the identical raw ``mean_M`` (ch1) + ``vcorr_S`` (ch2),
diameter, denoise, thresholds, normalization — and varies only ``cellpose_model``.
Runs ``run_cellpose_detection`` **before Gate 1** and reports candidate counts +
prob/area distributions, plus an overlay PNG per model for visual A/B against the
cytoSAM-GUI baseline.

Usage
-----
    conda activate roigbiv
    python scripts/phase_a2_model_isolation.py \
        --summary data/logan_cousa_trial/052126_DS-Prism-3_VI15_D2_FOV2_post-007/output/052126_DS-Prism-3_VI15_D2_FOV2_post-007/summary \
        --diameter 56 --out experiments/runs/phase_a2

Compares the deployed model against cpsam by default; pass ``--models`` to override.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def _summarize(masks: list[np.ndarray], probs: list[float]) -> dict:
    areas = np.array([int(m.sum()) for m in masks], dtype=np.int64)
    p = np.array(probs, dtype=np.float64)
    n = len(masks)
    return {
        "n_candidates": n,
        "area_median": float(np.median(areas)) if n else float("nan"),
        "area_p5": float(np.percentile(areas, 5)) if n else float("nan"),
        "area_p95": float(np.percentile(areas, 95)) if n else float("nan"),
        "prob_median": float(np.median(p)) if n else float("nan"),
        "prob_min": float(p.min()) if n else float("nan"),
        "prob_max": float(p.max()) if n else float("nan"),
    }


def _save_overlay(mean_M: np.ndarray, label_image: np.ndarray, out_png: Path,
                  title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    lo, hi = np.percentile(mean_M, (1, 99))
    disp = np.clip((mean_M - lo) / max(hi - lo, 1e-6), 0, 1)
    bnd = find_boundaries(label_image, mode="outer")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(disp, cmap="gray", interpolation="nearest")
    overlay = np.zeros((*disp.shape, 4), dtype=np.float32)
    overlay[bnd] = (0.1, 0.9, 1.0, 1.0)  # cyan outlines
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True,
                    help="Foundation summary dir containing mean_M.tif + vcorr_S.tif")
    ap.add_argument("--models", nargs="+",
                    default=["models/deployed/current_model", "cpsam"],
                    help="Cellpose model specs to compare (identical otherwise)")
    ap.add_argument("--diameter", type=int, default=56)
    ap.add_argument("--diameter-auto", action="store_true",
                    help="Use per-image diameter estimate instead of --diameter")
    ap.add_argument("--fs", type=float, default=7.5)
    ap.add_argument("--cellprob-threshold", type=float, default=-1.0)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--tile-norm-blocksize", type=int, default=256)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/phase_a2"))
    args = ap.parse_args()

    from roigbiv.pipeline.types import PipelineConfig
    from roigbiv.pipeline.stage1 import run_cellpose_detection

    mean_M = tifffile.imread(str(args.summary / "mean_M.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(args.summary / "vcorr_S.tif")).astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\nPhase A2 — model isolation on {args.summary}")
    print(f"  mean_M {mean_M.shape} mean={mean_M.mean():.3g} max={mean_M.max():.3g}")
    print(f"  vcorr_S {vcorr_S.shape} mean={vcorr_S.mean():.3g}")
    print(f"  fixed: diameter={'auto' if args.diameter_auto else args.diameter} "
          f"cellprob={args.cellprob_threshold} flow={args.flow_threshold} "
          f"tile_norm={args.tile_norm_blocksize} denoise={not args.no_denoise}\n")

    results: dict[str, dict] = {}
    for spec in args.models:
        tag = spec.replace("/", "_")
        print(f"── model: {spec} ──", flush=True)
        cfg = PipelineConfig(
            fs=args.fs,
            cellpose_model=spec,
            diameter=args.diameter,
            diameter_auto=args.diameter_auto,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            tile_norm_blocksize=args.tile_norm_blocksize,
            use_denoise=not args.no_denoise,
        )
        masks, probs, label_image, _ = run_cellpose_detection(mean_M, vcorr_S, cfg)
        stats = _summarize(masks, probs)
        results[spec] = stats
        print(f"  → candidates (pre-Gate-1): {stats['n_candidates']}")
        print(f"    area  median={stats['area_median']:.0f} "
              f"p5={stats['area_p5']:.0f} p95={stats['area_p95']:.0f} px²")
        print(f"    prob  median={stats['prob_median']:.3f} "
              f"[{stats['prob_min']:.3f}, {stats['prob_max']:.3f}]\n", flush=True)
        out_png = args.out / f"overlay_{tag}.png"
        _save_overlay(mean_M, label_image, out_png,
                      f"{spec}  ·  {stats['n_candidates']} candidates (pre-Gate-1)")
        tifffile.imwrite(str(args.out / f"labels_{tag}.tif"), label_image)

    print("=" * 64)
    print("PHASE A2 VERDICT")
    for spec, s in results.items():
        print(f"  {spec:40s} {s['n_candidates']:>5d} candidates")
    if len(results) == 2:
        a, b = list(results.values())
        specs = list(results.keys())
        print(f"\n  {specs[1]} / {specs[0]} candidate ratio: "
              f"{(b['n_candidates'] / max(a['n_candidates'], 1)):.2f}×")
    print(f"\n  Overlays + label TIFFs → {args.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
