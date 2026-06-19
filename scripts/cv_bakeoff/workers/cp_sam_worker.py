"""Cellpose-SAM (CP4) sidecar worker — runs in the ``cp-sam`` env.

Must NOT import roigbiv (it lives in a different conda env). Reads a scratch
``inputs.npz``, runs cellpose>=4, writes ``{stem}_cp-sam_masks.tif`` (uint16) +
``{stem}_cp-sam_meta.json``. See scripts/cv_bakeoff/detectors/cp_sam.py for the
driver side of the contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def _stretch01(img, lo_pct=1.0, hi_pct=99.5):
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--stem", required=True)
    p.add_argument("--channel", default="mean_M")
    p.add_argument("--diameter", type=float, default=None)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    args = p.parse_args()

    with np.load(args.in_path) as d:
        img = _stretch01(d[args.channel])

    # Cellpose-SAM (cellpose>=4): the generalist CPSAM model, no channel spec.
    from cellpose import models

    try:
        model = models.CellposeModel(gpu=True)
    except Exception:  # noqa: BLE001 — fall back to CPU if no usable GPU
        model = models.CellposeModel(gpu=False)

    eval_kwargs = {"flow_threshold": args.flow_threshold}
    if args.diameter is not None:
        eval_kwargs["diameter"] = args.diameter
    out = model.eval(img, **eval_kwargs)
    masks = out[0]  # cellpose>=4 returns (masks, flows, styles)

    label_mask = np.asarray(masks, dtype=np.uint16)
    out_dir = Path(args.out_dir)
    tifffile.imwrite(str(out_dir / f"{args.stem}_cp-sam_masks.tif"), label_mask)
    (out_dir / f"{args.stem}_cp-sam_meta.json").write_text(json.dumps({
        "method": "cp-sam",
        "channel": args.channel,
        "diameter": args.diameter,
        "flow_threshold": args.flow_threshold,
        "n_rois": int(label_mask.max()),
    }, indent=2))
    print(f"  [cp_sam_worker] {args.stem}: {int(label_mask.max())} ROIs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
