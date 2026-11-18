#!/usr/bin/env python
"""Cellpose-SAM (cellpose 4.x) sidecar runner — Phase M.

Runs OUT-OF-PROCESS, invoked by the `cp-sam` conda env's interpreter, because
cellpose 4.x requires numpy 2.x and cannot share the roigbiv (numpy 1.26 /
suite2p / cellpose 3.x) interpreter. This script therefore imports ONLY numpy
and cellpose — never roigbiv.

Contract (single argv: a JSON manifest path):
  {
    "input":            "<path>.npy",   # (H,W) or (H,W,C) float32 stack
    "labels_out":       "<path>.npy",   # written: (H,W) uint16 label image
    "cellprob_out":     "<path>.npy",   # written: (H,W) float32 cellprob map
    "diameter":         <float|null>,   # null: no diameter (cpsam sizes cells itself)
    "model":            "<name|path>",  # optional; omitted = the 4.x default model
    "cellprob_threshold": <float>,
    "flow_threshold":   <float>,
    "gpu":              <bool>,
    "channel_axis":     <int|null>      # -1 for (H,W,C), null for (H,W)
  }

cpsam is channel-invariant and noise-robust: no denoise step, and the
channels=(1,2) cyto/nucleus role convention does not apply. Normalization is
left to cpsam's robust default (normalize=True).
"""
import json
import sys

import numpy as np


def main(manifest_path: str) -> int:
    with open(manifest_path) as f:
        m = json.load(f)

    x = np.load(m["input"]).astype(np.float32)

    gpu = bool(m.get("gpu", True))
    if gpu:
        # Cellpose falls back to the CPU without saying so when CUDA is
        # missing; a GPU run that silently becomes a CPU run is refused.
        import torch

        if not torch.cuda.is_available():
            print("gpu requested but CUDA is not available", file=sys.stderr)
            return 3

    # Import here so a missing/broken cellpose surfaces as a clean stderr message
    # the parent can relay, not an import-time crash before argv is parsed.
    from cellpose.models import CellposeModel

    model_kwargs = {"pretrained_model": m["model"]} if m.get("model") else {}
    model = CellposeModel(gpu=gpu, **model_kwargs)   # 4.x default == cpsam
    print(f"cpsam loaded (gpu={gpu}, model={m.get('model') or 'default'})", flush=True)

    eval_kwargs = dict(
        cellprob_threshold=float(m["cellprob_threshold"]),
        flow_threshold=float(m["flow_threshold"]),
        normalize=True,
    )
    if m.get("diameter") is not None:
        eval_kwargs["diameter"] = float(m["diameter"])
    ch_axis = m.get("channel_axis", None)
    if ch_axis is not None:
        eval_kwargs["channel_axis"] = int(ch_axis)

    masks, flows, styles = model.eval(x, **eval_kwargs)

    label_image = np.asarray(masks, dtype=np.uint16)

    # cellpose returns flows as a length-3+ tuple; flows[2] is the dense cellprob
    # map. Fall back to a binary map if the shape doesn't line up.
    cellprob_map = None
    if isinstance(flows, (list, tuple)) and len(flows) >= 3:
        cp = np.asarray(flows[2], dtype=np.float32)
        if cp.shape == label_image.shape:
            cellprob_map = cp
    if cellprob_map is None:
        cellprob_map = (label_image > 0).astype(np.float32)

    np.save(m["labels_out"], label_image)
    np.save(m["cellprob_out"], cellprob_map)
    print(f"cpsam done: {int(label_image.max())} objects, "
          f"shape {label_image.shape}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: cpsam_sidecar.py <manifest.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
