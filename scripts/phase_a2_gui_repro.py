#!/usr/bin/env python
"""Phase A2b — reproduce the Cellpose *GUI* path with a CP3 generalist.

Reconciles a contradiction: the working cytoSAM-GUI run was reported as CP3, yet
CP3 generalists find ~0 ROIs through the pipeline's Stage-1 path. The pipeline and
the GUI differ in PREPROCESSING, not just model: the pipeline feeds a dual-channel
(mean_M, vcorr_S) stack + denoise_cyto3 + tile_norm_blocksize. The Cellpose GUI by
default runs **single-channel grayscale, global 1–99 percentile normalize, no
denoise, no tiling**.

This script calls ``CellposeModel.eval`` directly in GUI-default style so we can
tell whether the gap is preprocessing (cheap fix, no env change) or genuinely the
model version (justifies a CP4 upgrade).

Usage
-----
    conda activate roigbiv
    python scripts/phase_a2_gui_repro.py \
        --summary data/logan_cousa_trial/.../summary \
        --models cyto3 cyto2 models/deployed/current_model \
        --out experiments/runs/phase_a2_guirepro
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def _run(model_spec, img, diameter, cellprob, flow, gpu):
    from cellpose.models import CellposeModel
    from roigbiv.pipeline.stage1 import _CELLPOSE_BUILTINS, _resolve_model_path

    mp = _resolve_model_path(model_spec)
    if mp in _CELLPOSE_BUILTINS:
        model = CellposeModel(gpu=gpu, model_type=mp)
    else:
        model = CellposeModel(gpu=gpu, pretrained_model=mp)
    # GUI-default eval: single grayscale channel, per-image global normalize, no tiling.
    masks, flows, styles = model.eval(
        img.astype(np.float32),
        diameter=diameter,
        channels=[0, 0],
        cellprob_threshold=cellprob,
        flow_threshold=flow,
        normalize=True,
    )
    lab = np.asarray(masks, dtype=np.uint16)
    ids = np.unique(lab)
    ids = ids[ids != 0]
    areas = np.array([int((lab == i).sum()) for i in ids], dtype=np.int64)
    return lab, len(ids), areas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--image", default="mean_M",
                    help="which summary image to segment (default mean_M)")
    ap.add_argument("--models", nargs="+",
                    default=["cyto3", "cyto2", "models/deployed/current_model"])
    ap.add_argument("--diameters", type=int, nargs="+", default=[56, 0],
                    help="diameters to try; 0 = let Cellpose auto-estimate")
    ap.add_argument("--cellprob", type=float, default=0.0, help="GUI default 0.0")
    ap.add_argument("--flow", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/phase_a2_guirepro"))
    args = ap.parse_args()

    from roigbiv.pipeline.device import cuda_compute_capable
    gpu = cuda_compute_capable()
    args.out.mkdir(parents=True, exist_ok=True)

    img = tifffile.imread(str(args.summary / f"{args.image}.tif")).astype(np.float32)
    print(f"\nPhase A2b — GUI-style CP3 repro on {args.image} {img.shape} "
          f"(mean={img.mean():.3g} max={img.max():.3g})")
    print(f"  GUI-default eval: channels=[0,0], normalize=True (global 1-99), "
          f"no denoise, no tiling, cellprob={args.cellprob} flow={args.flow}\n")

    rows = []
    for spec in args.models:
        for D in args.diameters:
            diam = None if D == 0 else D
            try:
                lab, n, areas = _run(spec, img, diam, args.cellprob, args.flow, gpu)
            except Exception as exc:
                print(f"  {spec:36s} diam={str(D):>4s}  ERROR: {exc}")
                continue
            amed = float(np.median(areas)) if n else float("nan")
            print(f"  {spec:36s} diam={str(D):>4s}  → {n:>4d} ROIs "
                  f"(area median={amed:.0f}px²)")
            rows.append((spec, D, n, amed))
            tag = spec.replace("/", "_")
            tifffile.imwrite(str(args.out / f"labels_{tag}_d{D}.tif"), lab)

    print("\n" + "=" * 64)
    print("PHASE A2b VERDICT — does GUI-style preprocessing rescue CP3?")
    best = max(rows, key=lambda r: r[2]) if rows else None
    if best:
        print(f"  best: {best[0]} diam={best[1]} → {best[2]} ROIs")
        if best[2] >= 20:
            print("  → CP3 reproduces 'plenty' with GUI preprocessing: fix is "
                  "PREPROCESSING, no CP4 upgrade needed.")
        else:
            print("  → CP3 still finds few even GUI-style: recollection likely "
                  "CP4-SAM; upgrade justified.")
    print(f"  Labels → {args.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
