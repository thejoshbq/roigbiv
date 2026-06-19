#!/usr/bin/env python
"""Locate high-motion windows in a raw stack (to pick a discriminative bench segment).

Reads frames at a stride, estimates each sampled frame's rigid shift vs a
reference via phase cross-correlation, and reports the contiguous windows with
the largest displacement spread. Use the printed --start-frame with
bench_motion_correction.py.

    conda run -n roigbiv python scripts/scan_motion.py \
        --stack data/logan_cousa_trial/_stacks/052126_DS-Prism-3_VI15_D2_FOV2_beh-006.tif \
        --stride 25 --window 1500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--window", type=int, default=1500, help="bench window size (frames)")
    ap.add_argument("--max-samples", type=int, default=1500)
    args = ap.parse_args()

    from skimage.registration import phase_cross_correlation

    with tifffile.TiffFile(str(args.stack)) as tf:
        T = len(tf.pages)
        idx = list(range(0, T, args.stride))[:args.max_samples]
        print(f"{args.stack.name}: {T} frames, sampling {len(idx)} @ stride {args.stride}",
              flush=True)
        ref = tf.asarray(key=idx[0]).astype(np.float32)
        shifts = np.zeros((len(idx), 2), dtype=np.float32)
        for i, fr in enumerate(idx):
            img = tf.asarray(key=fr).astype(np.float32)
            sh, _err, _phase = phase_cross_correlation(ref, img, upsample_factor=1)
            shifts[i] = sh  # (dy, dx)

    mag = np.linalg.norm(shifts, axis=1)
    print(f"global drift: |shift| min={mag.min():.1f} max={mag.max():.1f} "
          f"mean={mag.mean():.1f} std={mag.std():.1f} px", flush=True)
    print(f"y range [{shifts[:,0].min():.0f},{shifts[:,0].max():.0f}]  "
          f"x range [{shifts[:,1].min():.0f},{shifts[:,1].max():.0f}] px", flush=True)

    # Frame-to-frame jitter (local motion, not slow drift)
    jit = np.linalg.norm(np.diff(shifts, axis=0), axis=1)
    win_samples = max(2, args.window // args.stride)
    best = []
    for s in range(0, len(idx) - win_samples + 1, max(1, win_samples // 2)):
        seg = jit[s:s + win_samples]
        spread = np.linalg.norm(
            shifts[s:s + win_samples].max(0) - shifts[s:s + win_samples].min(0))
        best.append((float(seg.mean()), float(spread), idx[s]))
    best.sort(reverse=True)
    print("\nTop windows by mean frame-to-frame jitter (start_frame, jitter, drift_spread):",
          flush=True)
    for jmean, spread, start in best[:6]:
        print(f"  --start-frame {start:>6}  jitter={jmean:5.2f}px  spread={spread:5.1f}px",
              flush=True)


if __name__ == "__main__":
    main()
