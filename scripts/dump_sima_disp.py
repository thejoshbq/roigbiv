#!/usr/bin/env python
"""Dump SIMA HMM2D's FULL per-row displacement field — runs in ``sima-legacy``.

The production worker (``sima_mc_worker.py``) only persists a row-*median*
per-frame trace, which hides the decisive quantity for the
"can a patch method replace SIMA" question: how much do the per-row
displacements vary *within* a single frame.

This is a one-off measurement script (loose-rigor ``scripts/``). It runs the
genuine legacy estimation and saves ``displacements`` verbatim as
``{stem}_dispfull.npz`` (shape ~(T, planes, rows, 2), order (y, x)). Movie
export is skipped — only the displacement field is wanted, so we pay for
estimation but not the per-frame write-back.

    conda run -n sima-legacy python scripts/dump_sima_disp.py \
        --input experiments/runs/mc_legacy_val/pre005_sub400.tif \
        --outdir experiments/runs/mc_legacy_val --stem pre005_sub400
"""
import argparse
import gc
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import h5py
import tifffile

warnings.filterwarnings("ignore")

import sima            # noqa: E402
import sima.motion     # noqa: E402
from sima.motion import HiddenMarkov2D  # noqa: E402


def _log(msg):
    print(f"[dump_sima_disp] {msg}", file=sys.stderr, flush=True)


def _stage_tif_to_hdf5(input_tif, h5_path, key="imaging"):
    with tifffile.TiffFile(input_tif) as tf:
        series = tf.series[0]
        shape = series.shape
        if len(shape) != 3:
            raise ValueError(f"expected 3D (T,Y,X), got {shape}")
        T, Ly, Lx = (int(s) for s in shape)
        multipage = len(tf.pages) >= T
        vol = None if multipage else series.asarray()
        with h5py.File(h5_path, "w") as hf:
            dset = hf.create_dataset(key, shape=(T, Ly, Lx), dtype="uint16",
                                     chunks=(1, Ly, Lx))
            for t in range(T):
                frame = tf.pages[t].asarray() if multipage else vol[t]
                dset[t] = np.asarray(frame, dtype=np.uint16)
    return T, Ly, Lx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--max-displacement", type=int, default=50)
    ap.add_argument("--granularity", default="row")
    args = ap.parse_args(argv)

    input_tif = os.path.abspath(args.input)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    out_npz = os.path.join(outdir, f"{args.stem}_dispfull.npz")

    scratch = tempfile.mkdtemp(prefix=f"{args.stem}_dispdump_", dir=outdir)
    h5_path = os.path.join(scratch, f"{args.stem}.h5")
    sima_dir = os.path.join(scratch, f"{args.stem}.sima")
    mc_sima_dir = os.path.join(scratch, f"{args.stem}_mc.sima")

    seq = corrected = ds = None
    try:
        _log(f"staging {input_tif} -> {h5_path}")
        T, Ly, Lx = _stage_tif_to_hdf5(input_tif, h5_path)
        _log(f"staged T={T} Ly={Ly} Lx={Lx}; running HMM2D "
             f"granularity={args.granularity!r} max_disp={args.max_displacement}")
        seq = sima.Sequence.create("HDF5", h5_path, "tyx", key="imaging")
        ds = sima.ImagingDataset([seq], sima_dir)
        mc = HiddenMarkov2D(granularity=args.granularity,
                            max_displacement=[args.max_displacement] * 2,
                            verbose=False)
        corrected = mc.correct([seq], mc_sima_dir)
        disp = np.asarray(corrected.sequences[0].displacements)
        _log(f"displacement field shape={disp.shape} dtype={disp.dtype}")
        # store as int16 — SIMA displacements are integer-valued canvas offsets
        np.savez_compressed(out_npz, displacements=disp.astype(np.int16),
                            shape=np.asarray(disp.shape))
        _log(f"wrote {out_npz}")
        return 0
    finally:
        seq = corrected = ds = None
        gc.collect()
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        _log(f"FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
