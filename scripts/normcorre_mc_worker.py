#!/usr/bin/env python
"""NoRMCorre (CaImAn) motion-correction worker — runs in the ``caiman`` env.

One-off benchmark sidecar (loose-rigor ``scripts/``). Runs piecewise-rigid
NoRMCorre on a stack and writes the temporal-mean image of the corrected movie
so the roigbiv-side bench can score it with the shared metrics. Never imported
by roigbiv; invoked via ``conda run -n caiman``.

Two knobs matter for matching legacy SIMA quality:
  * ``--stride-y`` — patch spacing along rows. Small values push toward
    row-granularity (SIMA's edge). Default 24 ≈ fine vertical bands.
  * ``--gsig-filt`` — high-pass Gaussian σ applied before registration. This is
    NoRMCorre's robustness mechanism on dim, low-SNR frames (the regime where
    rowwise-pcc regressed). 0 disables it.

    conda run -n caiman python scripts/normcorre_mc_worker.py \
        --input subset.tif --out-mean mean_normcorre.tif \
        --max-shift 24 --stride-y 24 --stride-x 256 --gsig-filt 2
"""
import argparse
import sys
import warnings

import numpy as np
import tifffile

warnings.filterwarnings("ignore")


def _log(m):
    print(f"[normcorre_worker] {m}", file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-mean", required=True)
    ap.add_argument("--out-mc", default=None, help="optional corrected movie tif")
    ap.add_argument("--max-shift", type=int, default=24)
    ap.add_argument("--stride-y", type=int, default=24)
    ap.add_argument("--stride-x", type=int, default=256)
    ap.add_argument("--overlap-y", type=int, default=12)
    ap.add_argument("--overlap-x", type=int, default=128)
    ap.add_argument("--gsig-filt", type=float, default=2.0,
                    help="high-pass Gaussian sigma; 0 disables")
    ap.add_argument("--max-dev", type=int, default=8,
                    help="max patch deviation from rigid (px)")
    ap.add_argument("--rigid", action="store_true",
                    help="rigid-only (pw_rigid=False) — fair test when motion is bulk")
    args = ap.parse_args(argv)

    import caiman as cm
    from caiman.motion_correction import MotionCorrect

    _log(f"loading {args.input}")
    mov = cm.load(args.input)
    _log(f"movie shape {mov.shape} dtype {mov.dtype}")

    # write to a caiman memmap so MotionCorrect can operate out-of-core. Unique
    # per output so concurrent variant runs don't clobber a shared input mmap.
    import os as _os
    _tag = _os.path.splitext(_os.path.basename(args.out_mean))[0]
    fname = mov.save(f"/tmp/normcorre_input_{_tag}.mmap", order="C")
    _log(f"memmapped -> {fname}")

    # gSig_filt feeds a cv2 Gaussian kernel size — must be integer-valued.
    _g = int(round(args.gsig_filt))
    gsig = None if args.gsig_filt <= 0 else (_g, _g)
    mc = MotionCorrect(
        [fname], dview=None,
        max_shifts=(args.max_shift, args.max_shift),
        strides=(args.stride_y, args.stride_x),
        overlaps=(args.overlap_y, args.overlap_x),
        max_deviation_rigid=args.max_dev,
        pw_rigid=not args.rigid,
        gSig_filt=gsig,
        shifts_opencv=True,
        border_nan="copy",
    )
    _log(f"running {'rigid' if args.rigid else 'pw_rigid'} NoRMCorre: "
         f"max_shift={args.max_shift} "
         f"strides=({args.stride_y},{args.stride_x}) gSig_filt={gsig}")
    mc.motion_correct(save_movie=True)

    fcorr = mc.fname_tot_rig if args.rigid else mc.fname_tot_els
    if isinstance(fcorr, (list, tuple)):
        fcorr = fcorr[0]
    _log(f"corrected mmap: {fcorr}")
    corrected = cm.load(fcorr)
    _log(f"corrected loaded shape={corrected.shape} dtype={corrected.dtype}")
    # caiman's lazy movie object reduces incorrectly along axis 0 (per-frame
    # indexing is fine, but .mean(0) stripes); materialize to a plain ndarray
    # first so the temporal mean is a vanilla numpy reduction.
    # caiman's lazy movie object corrupts BULK reads (np.asarray / .mean(0) both
    # produce vertical-stripe garbage) while single-frame indexing is correct.
    # Accumulate the temporal mean one indexed frame at a time.
    T = corrected.shape[0]
    acc = np.zeros(corrected.shape[1:], dtype=np.float64)
    for t in range(T):
        acc += np.asarray(corrected[t], dtype=np.float64)
    mean_img = (acc / T).astype(np.float32)
    tifffile.imwrite(args.out_mean, mean_img)
    _log(f"wrote mean -> {args.out_mean}  shape={mean_img.shape}")

    if args.out_mc:
        tifffile.imwrite(args.out_mc, corr_arr, bigtiff=True)
        _log(f"wrote corrected movie -> {args.out_mc}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        _log(f"FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
