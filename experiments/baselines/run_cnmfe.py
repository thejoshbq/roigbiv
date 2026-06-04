"""CNMF-E baseline runner for roigbiv eval harness (Phase 3).

Runs CaImAn CNMF-E on a motion-corrected (T, H, W) TIFF and emits a uint16
label TIFF in the same format as roigbiv merged_masks.tif.

This script is designed to run in the caiman conda env (isolated from roigbiv):
    conda run -n caiman python experiments/baselines/run_cnmfe.py \\
        --input /abs/path/to/{stem}_mc.tif \\
        --gSig 6 --fs 7.5 \\
        --output /abs/path/to/experiments/runs/{stem}_cnmfe_masks.tif

All outputs go to experiments/runs/; nothing is written to inference/.

Parameter choices:
- gSig=6: half-soma diameter (~12px somata in GRIN lens 2p PrL recordings)
- fs=7.5: effective frame rate (4x-averaged 30 Hz acquisitions)
- tau_d=1.0: GCaMP6s decay constant (spec §18)
- All other CNMF-E params: CaImAn defaults for 1-photon-style microendoscopy data
  (CNMF-E was validated on GRIN lens 1p data; parameters are documented comments below)
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile


log = logging.getLogger(__name__)


def run_cnmfe(
    input_tif: Path,
    output_mask: Path,
    gSig: int = 6,
    fs: float = 7.5,
    tau_d: float = 1.0,
    patch_size: int = 100,
    overlap: int = 20,
    min_corr: float = 0.8,
    min_pnr: float = 10.0,
) -> None:
    """Run CNMF-E on input_tif and write uint16 label mask to output_mask."""
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import cnmf as cnmf_module
    from caiman.source_extraction.cnmf.params import CNMFParams

    input_tif = Path(input_tif).resolve()
    output_mask = Path(output_mask).resolve()

    # Load movie dimensions first (no full load)
    with tifffile.TiffFile(str(input_tif)) as tf:
        shape = tf.series[0].shape
    T, H, W = shape
    log.info("Input: %s — shape %s", input_tif.name, shape)

    # CaImAn needs a Caiman-format mmap; write to a temp dir
    with tempfile.TemporaryDirectory(prefix="cnmfe_") as tmpdir:
        # Convert TIFF → caiman mmap
        movie = tifffile.imread(str(input_tif)).astype(np.float32)
        mmap_path = os.path.join(tmpdir, "movie_C_order.mmap")
        # CaImAn uses Fortran-order mmap internally; we pass the numpy array directly
        # via create_memmap_from_numpy
        mmap_file = cm.save_memmap([str(input_tif)],
                                   base_name=os.path.join(tmpdir, "movie"),
                                   order="C")
        del movie

        Yr, dims, T_actual = cm.load_memmap(mmap_file)
        images = np.reshape(Yr.T, [T_actual] + list(dims), order="F")
        log.info("Loaded mmap: T=%d, dims=%s", T_actual, dims)

        # CNMF-E parameters
        # K=None (automatic component count), gSiz = 4*gSig+1 (kernel support)
        # use_cnn=False must be in the initial dict — change_params after construction
        # does not propagate into evaluate_components (CaImAn 1.13 limitation).
        # Quality gating via rval_thr (spatial correlation) and min_SNR only.
        opts = CNMFParams(params_dict={
            "fr": fs,
            "decay_time": tau_d,
            "method_init": "corr_pnr",    # CNMF-E initialisation (ring background)
            "gSig": (gSig, gSig),
            "gSiz": (4 * gSig + 1, 4 * gSig + 1),
            "ring_size_factor": 1.5,      # ring model radius / soma radius
            "min_corr": min_corr,
            "min_pnr": min_pnr,
            "K": None,                    # no upper bound on cell count
            "nb": 0,                      # ring background (no global)
            "nb_patch": 0,
            "center_psf": True,
            "ssub": 1,                    # no spatial downsampling
            "tsub": 1,                    # no temporal downsampling
            "rf": patch_size // 2,        # half-size of each patch
            "stride": patch_size - overlap,
            "low_rank_background": None,
            "update_background_components": True,
            "normalize_init": False,
            "del_duplicates": True,
            "n_processes": 8,
            "p": 1,                       # AR order 1 for GCaMP6s
            "merge_thr": 0.7,
            "use_cnn": False,
            "min_SNR": 2.0,
            "rval_thr": 0.8,
        })

        cnm = cnmf_module.CNMF(n_processes=8, params=opts)
        cnm.fit(images)

        # Quality checks: spatial correlation + SNR, no CNN (model file unavailable)
        cnm.estimates.evaluate_components(images, cnm.params)
        idx_accepted = cnm.estimates.idx_components

        if len(idx_accepted) == 0:
            log.warning("No components accepted by quality check — writing empty mask.")
            label_img = np.zeros((H, W), dtype=np.uint16)
        else:
            # Convert spatial components to label image
            A = cnm.estimates.A[:, idx_accepted]  # (H*W, n_acc) sparse
            label_img = np.zeros(H * W, dtype=np.uint16)
            # Greedy assignment: each pixel gets the label of the component with
            # highest spatial weight (matching roigbiv convention)
            for comp_idx in range(A.shape[1]):
                col = np.asarray(A[:, comp_idx].todense()).ravel()
                threshold = col.max() * 0.1  # 10% of peak weight
                pixels = np.where(col > threshold)[0]
                for px in pixels:
                    if col[px] > (A[:, label_img[px] - 1].max() if label_img[px] > 0 else 0):
                        label_img[px] = comp_idx + 1  # 1-indexed labels

            label_img = label_img.reshape(H, W)
            log.info("Accepted components: %d", A.shape[1])

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_mask), label_img)
    log.info("Wrote label mask: %s  (%d ROIs)", output_mask, label_img.max())


def main() -> None:
    ap = argparse.ArgumentParser(description="CNMF-E baseline runner for roigbiv eval")
    ap.add_argument("--input", required=True, type=Path,
                    help="Motion-corrected _mc.tif stack (T, H, W) float32 or uint16")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output uint16 label TIFF (H, W)")
    ap.add_argument("--gSig", type=int, default=6,
                    help="Half-soma diameter in pixels (default 6 for ~12px somata)")
    ap.add_argument("--fs", type=float, default=7.5,
                    help="Effective frame rate Hz (default 7.5)")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="GCaMP decay constant (default 1.0 for GCaMP6s)")
    ap.add_argument("--min-corr", type=float, default=0.8,
                    help="Min spatial correlation for component acceptance (default 0.8)")
    ap.add_argument("--min-pnr", type=float, default=10.0,
                    help="Min peak-to-noise ratio for component acceptance (default 10.0)")
    ap.add_argument("--patch-size", type=int, default=100,
                    help="Patch half-size for parallel processing (default 100)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_cnmfe(
        input_tif=args.input,
        output_mask=args.output,
        gSig=args.gSig,
        fs=args.fs,
        tau_d=args.tau,
        min_corr=args.min_corr,
        min_pnr=args.min_pnr,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
