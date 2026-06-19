"""StarDist 2D sidecar worker — runs in the ``stardist`` env.

Must NOT import roigbiv. Reads a scratch ``inputs.npz``, runs a pretrained
StarDist model, writes ``{stem}_stardist_masks.tif`` (uint16) +
``{stem}_stardist_meta.json``. See scripts/cv_bakeoff/detectors/stardist.py for
the driver side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def _normalize(img, lo_pct=1.0, hi_pct=99.8):
    # StarDist's own recommended percentile normalization.
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
    p.add_argument("--model", default="2D_versatile_fluo")
    p.add_argument("--prob-thresh", type=float, default=None)
    p.add_argument("--nms-thresh", type=float, default=None)
    args = p.parse_args()

    with np.load(args.in_path) as d:
        img = _normalize(d[args.channel])

    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(args.model)
    predict_kwargs = {}
    if args.prob_thresh is not None:
        predict_kwargs["prob_thresh"] = args.prob_thresh
    if args.nms_thresh is not None:
        predict_kwargs["nms_thresh"] = args.nms_thresh
    labels, _details = model.predict_instances(img, **predict_kwargs)

    label_mask = np.asarray(labels, dtype=np.uint16)
    out_dir = Path(args.out_dir)
    tifffile.imwrite(str(out_dir / f"{args.stem}_stardist_masks.tif"), label_mask)
    (out_dir / f"{args.stem}_stardist_meta.json").write_text(json.dumps({
        "method": "stardist",
        "channel": args.channel,
        "model": args.model,
        "prob_thresh": args.prob_thresh,
        "nms_thresh": args.nms_thresh,
        "n_rois": int(label_mask.max()),
    }, indent=2))
    print(f"  [stardist_worker] {args.stem}: {int(label_mask.max())} ROIs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
