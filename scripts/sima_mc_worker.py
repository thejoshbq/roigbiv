#!/usr/bin/env python
"""Legacy SIMA motion-correction worker — runs in the ``sima-legacy`` py3.8 env.

This module is executed *by the sidecar interpreter*, never imported by roigbiv
(it may import only sima-legacy-env packages). The roigbiv-side driver is
``roigbiv/pipeline/legacy_mc.py``, which invokes this via
``conda run -n sima-legacy python scripts/sima_mc_worker.py ...``.

It reproduces the legacy notebook's motion correction exactly:

    seq = sima.Sequence.create('HDF5', tmp.h5, 'tyx')
    ds  = sima.ImagingDataset([seq], <stem>.sima)
    mc  = sima.motion.HiddenMarkov2D(granularity='row', max_displacement=[d,d])
    cor = mc.correct([seq], <stem>_mc.sima)
    # export corrected frames as <stem>_mc.tif

with two deliberate deviations forced by running SIMA on modern Python:

  * **Input is staged to a temporary HDF5 'tyx' file**, not fed as a TIFF
    Sequence. SIMA's ``_Sequence_TIFF_Interleaved.__iter__`` uses
    ``next(base_iter)`` inside a generator, which Python 3.7+ (PEP 479) turns
    into a ``RuntimeError``. The HDF5 reader uses the safe base iterator, so the
    notebook's HDF5 path is the only one that works here.
  * **Frames are exported by us with modern tifffile**, reading each corrected
    frame via the dataset API — SIMA's own ``export_frames`` raised
    ``"integer out of range for 'I'"`` in the legacy notebook itself.

Outputs written to ``--outdir``:
  * ``{stem}_mc.tif``       uint16 (T, Ly', Lx') corrected movie (BigTIFF).
                            Note SIMA pads/crops to a common canvas, so the
                            corrected dims differ from the input dims.
  * ``{stem}_mc_disp.npz``  ``motion_x`` (T,), ``motion_y`` (T,) float32 traces.
  * ``{stem}_mc_meta.json`` shape + parameter provenance.

Exit code is non-zero with a clear stderr message on any failure.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import h5py
import tifffile

warnings.filterwarnings("ignore")  # silence SIMA's np.int/np.float deprecations

import sima            # noqa: E402
import sima.motion     # noqa: E402
from sima.motion import HiddenMarkov2D, PlaneTranslation2D  # noqa: E402


def _log(msg):
    print(f"[sima_mc_worker] {msg}", file=sys.stderr, flush=True)


def _stage_tif_to_hdf5(input_tif, h5_path, key="imaging"):
    """Stream a (T,Y,X) TIFF stack into an HDF5 'tyx' uint16 dataset.

    Page-by-page so a multi-GB BigTIFF never lands fully in RAM.
    """
    with tifffile.TiffFile(input_tif) as tf:
        series = tf.series[0]
        shape = series.shape
        if len(shape) != 3:
            raise ValueError(
                f"expected a 3D (T,Y,X) stack, got shape {shape} from {input_tif}"
            )
        T, Ly, Lx = (int(s) for s in shape)
        # Page-per-frame stacks (the roigbiv assembler's layout) stream one IFD
        # per frame — memory-light for multi-GB inputs. A volumetric single-IFD
        # TIFF (page count < T) can't be indexed per frame, so fall back to a
        # full read.
        multipage = len(tf.pages) >= T
        vol = None if multipage else series.asarray()
        with h5py.File(h5_path, "w") as hf:
            dset = hf.create_dataset(
                key, shape=(T, Ly, Lx), dtype="uint16",
                chunks=(1, Ly, Lx),
            )
            for t in range(T):
                frame = tf.pages[t].asarray() if multipage else vol[t]
                dset[t] = np.asarray(frame, dtype=np.uint16)
    return T, Ly, Lx


def _frame_to_2d(frame5d):
    """Collapse a SIMA frame (planes, Y, X, channels) to a 2D (Y, X) array.

    The legacy backend is single-plane / single-channel; assert that so a
    silent multi-plane reduction can never corrupt the output.
    """
    arr = np.asarray(frame5d)
    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[-1] != 1:
        raise ValueError(
            f"expected single plane/channel frame (1,Y,X,1), got {arr.shape}"
        )
    return arr[0, :, :, 0]


def _per_frame_trace(displacements):
    """Reduce SIMA's (T, planes, rows, 2) row displacements to per-frame (x, y).

    SIMA stores non-negative canvas offsets ordered ``(y, x)``; the value is the
    offset applied to *register* the frame (i.e. the negative of the frame's
    motion). We take the median over rows (robust to HMM edge outliers), center
    on the temporal median, and negate so the trace reads as frame motion —
    matching the sign convention of Suite2p ``xoff``/``yoff`` used by the other
    backends. This is a 1-D QC summary only; the actual non-rigid per-row
    correction lives in the pixels of ``{stem}_mc.tif``.
    """
    disp = np.asarray(displacements, dtype=np.float64)        # (T, planes, rows, 2)
    flat = disp.reshape(disp.shape[0], -1, disp.shape[-1])    # (T, planes*rows, 2)
    pf = np.median(flat, axis=1)                              # (T, 2) as (y, x)
    pf = pf - np.median(pf, axis=0, keepdims=True)            # center
    motion_y = (-pf[:, 0]).astype(np.float32)
    motion_x = (-pf[:, 1]).astype(np.float32)
    return motion_x, motion_y


def main(argv=None):
    ap = argparse.ArgumentParser(description="Legacy SIMA HMM2D motion correction")
    ap.add_argument("--input", required=True, help="source .tif stack (T,Y,X)")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--stem", required=True, help="output basename stem")
    ap.add_argument("--max-displacement", type=int, default=50)
    ap.add_argument("--granularity", default="row")
    args = ap.parse_args(argv)

    input_tif = os.path.abspath(args.input)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    stem = args.stem

    mc_tif = os.path.join(outdir, f"{stem}_mc.tif")
    disp_npz = os.path.join(outdir, f"{stem}_mc_disp.npz")
    meta_json = os.path.join(outdir, f"{stem}_mc_meta.json")

    scratch = tempfile.mkdtemp(prefix=f"{stem}_sima_", dir=outdir)
    h5_path = os.path.join(scratch, f"{stem}.h5")
    sima_dir = os.path.join(scratch, f"{stem}.sima")
    mc_sima_dir = os.path.join(scratch, f"{stem}_mc.sima")

    seq = corrected = ds = None
    try:
        _log(f"staging {input_tif} -> {h5_path}")
        T, Ly, Lx = _stage_tif_to_hdf5(input_tif, h5_path)
        _algo = "PlaneTranslation2D" if args.granularity == "frame" else "HiddenMarkov2D"
        _log(f"staged T={T} Ly={Ly} Lx={Lx}; running SIMA "
             f"{_algo}(granularity={args.granularity!r}, "
             f"max_displacement=[{args.max_displacement}]*2)")

        seq = sima.Sequence.create("HDF5", h5_path, "tyx", key="imaging")
        ds = sima.ImagingDataset([seq], sima_dir)
        if args.granularity == "frame":
            # PlaneTranslation2D.__init__ has no `verbose` kwarg (unlike HMM2D).
            mc = PlaneTranslation2D(
                max_displacement=[args.max_displacement, args.max_displacement])
        else:
            mc = HiddenMarkov2D(
                granularity=args.granularity,
                max_displacement=[args.max_displacement, args.max_displacement],
                verbose=False)
        corrected = mc.correct([seq], mc_sima_dir)
        cseq = corrected.sequences[0]

        motion_x, motion_y = _per_frame_trace(cseq.displacements)

        # Probe output canvas dims from the first corrected frame.
        out_shape = _frame_to_2d(cseq._get_frame(0)).shape
        _log(f"exporting corrected movie {T}x{out_shape} -> {mc_tif}")
        # >4 GB output is plausible after padding; always emit BigTIFF.
        with tifffile.TiffWriter(mc_tif, bigtiff=True) as tw:
            for t in range(T):
                fr = _frame_to_2d(cseq._get_frame(t))
                fr = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
                tw.write(np.clip(fr, 0, 65535).astype(np.uint16),
                         contiguous=True)

        np.savez(disp_npz, motion_x=motion_x, motion_y=motion_y)
        with open(meta_json, "w") as f:
            json.dump({
                "stem": stem,
                "T": int(T),
                "in_Ly": int(Ly), "in_Lx": int(Lx),
                "out_Ly": int(out_shape[0]), "out_Lx": int(out_shape[1]),
                "granularity": args.granularity,
                "max_displacement": int(args.max_displacement),
                "sima_version": sima.__version__,
                "displacement_shape": list(np.asarray(cseq.displacements).shape),
            }, f, indent=2)
        _log(f"done: {mc_tif} (+ _disp.npz, _meta.json)")
        return 0
    finally:
        # Drop SIMA's open HDF5 handle before unlinking scratch.
        seq = corrected = ds = None
        gc.collect()
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — surface a clean error to the parent
        _log(f"FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
