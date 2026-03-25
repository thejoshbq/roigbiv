#!/usr/bin/env python3
"""
ROI G. Biv — Suite2p spatiotemporal detection on raw stacks.
Usage:
  python run_suite2p.py --input_dir ~/roigbiv/data/raw --fs 30.0
  python run_suite2p.py --single_file ~/roigbiv/data/raw/FOV1_mc.tif --fs 30.0
  python run_suite2p.py --config configs/pipeline.yaml --input_dir ~/roigbiv/data/raw
"""

import argparse, os, shutil, tempfile
from pathlib import Path
import numpy as np
import tifffile
from suite2p import run_s2p, default_ops

from config import BASE_DIR, load_config

S2P_OUT  = BASE_DIR / 'suite2p_workspace' / 'output'

def build_ops(input_dir, fs, tau=1.0, cfg=None, anatomical_only=0):
    """Build Suite2p ops dict, optionally merging values from pipeline config."""
    ops = default_ops()
    s2p_cfg = (cfg or {}).get('suite2p', {})

    ops.update({
        # ---- Data ----
        'data_path':       [str(input_dir)],
        'save_path0':      str(S2P_OUT),
        'save_folder':     'suite2p',
        'nplanes':         s2p_cfg.get('nplanes', 1),
        'nchannels':       s2p_cfg.get('nchannels', 1),
        'functional_chan':  s2p_cfg.get('functional_chan', 1),
        'tau':             tau,
        'fs':              fs,

        # ---- Registration ----
        'do_registration': s2p_cfg.get('do_registration', 0),
        'nimg_init':       s2p_cfg.get('nimg_init', 300),
        'batch_size':      s2p_cfg.get('batch_size', 250),
        'smooth_sigma':    s2p_cfg.get('smooth_sigma', 1.15),
        'maxregshift':     s2p_cfg.get('maxregshift', 0.1),
        'nonrigid':        s2p_cfg.get('nonrigid', True),
        'block_size':      s2p_cfg.get('block_size', [128, 128]),

        # ---- Detection ----
        'spatial_scale':       s2p_cfg.get('spatial_scale', 0),
        'threshold_scaling':   s2p_cfg.get('threshold_scaling', 1.0),
        'max_iterations':      s2p_cfg.get('max_iterations', 20),
        'connected':           s2p_cfg.get('connected', True),
        'nbinned':             s2p_cfg.get('nbinned', 5000),
        'allow_overlap':       s2p_cfg.get('allow_overlap', False),

        # ---- Classification ----
        'preclassify':     s2p_cfg.get('preclassify', 0.0),

        # ---- Neuropil ----
        'high_pass':              s2p_cfg.get('high_pass', 100),
        'inner_neuropil_radius':  s2p_cfg.get('inner_neuropil_radius', 2),
        'min_neuropil_pixels':    s2p_cfg.get('min_neuropil_pixels', 350),

        # ---- Spike deconvolution ----
        'spikedetect':     s2p_cfg.get('spikedetect', True),

        # ---- Anatomical detection ----
        # 0=activity-based; 1=mean-image anatomy (safe); 2=uses Cellpose 4.x (DO NOT USE)
        'anatomical_only': anatomical_only,
    })
    return ops

def _run_one_fov(src, fs, tau, cfg, do_registration, extract_vcorr,
                 anatomical_only, s2p_out):
    """Run Suite2p on a single TIF file in a correctly stem-named temp dir.

    Suite2p names its output subdir after basename(data_path[0]).  By placing
    the TIF inside a temp dir named {stem} (without _mc suffix) we get:
        s2p_out/{stem}/suite2p/plane0/
    which matches the existing pipeline naming convention.
    """
    stem = src.stem.replace('_mc', '')
    tmp_base = tempfile.mkdtemp()
    named_dir = Path(tmp_base) / stem
    named_dir.mkdir()
    os.symlink(src.resolve(), named_dir / src.name)
    try:
        ops = build_ops(named_dir, fs, tau, cfg, anatomical_only=anatomical_only)
        if s2p_out:
            ops['save_path0'] = str(s2p_out / stem)
        ops['do_registration'] = 1 if do_registration else 0
        output_ops = run_s2p(ops=ops)
        n_rois = output_ops.get('nROIs', '?')
        print(f'  {stem}: {n_rois} ROIs')
        if extract_vcorr and 'Vcorr' in output_ops:
            annotated_dir = BASE_DIR / 'data' / 'annotated'
            annotated_dir.mkdir(parents=True, exist_ok=True)
            vcorr = output_ops['Vcorr'].astype(np.float32)
            tifffile.imwrite(annotated_dir / f'{stem}_mc_vcorr.tif', vcorr)
        # Delete data.bin — only needed during processing, not for downstream use
        if s2p_out:
            data_bin = s2p_out / stem / 'suite2p' / 'plane0' / 'data.bin'
            if data_bin.exists():
                data_bin.unlink()
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir',
                    help='Directory containing raw .tif stacks')
    ap.add_argument('--single_file',
                    help='Path to a single .tif stack (creates a temp dir for Suite2p)')
    ap.add_argument('--fs', type=float, default=None,
                    help='Frame rate in Hz (overrides config)')
    ap.add_argument('--tau', type=float, default=None,
                    help='Sensor decay time constant in seconds (overrides config)')
    ap.add_argument('--skip_registration', action='store_true', default=True,
                    help='Skip registration for pre-motion-corrected data (default: True)')
    ap.add_argument('--do_registration', action='store_true',
                    help='Force registration even for _mc.tif files')
    ap.add_argument('--batch', action='store_true',
                    help='Process each .tif in --input_dir individually. '
                         'Required when FOVs have different image dimensions '
                         '(the normal case for data/raw/).')
    ap.add_argument('--extract_vcorr', action='store_true',
                    help='Extract Vcorr temporal map after Suite2p completes')
    ap.add_argument('--anatomical_only', type=int, default=0, choices=[0, 1],
                    help='0=activity-based detection (default); '
                         '1=mean-image anatomy. Do NOT use 2 — imports Cellpose 4.x '
                         'which conflicts with our Cellpose 3.x env.')
    ap.add_argument('--s2p_out', default=None,
                    help='Override Suite2p output directory '
                         '(default: suite2p_workspace/output). '
                         'Use a separate dir for anatomy runs to avoid overwriting.')
    ap.add_argument('--config', default=None,
                    help='Path to pipeline YAML config')
    args = ap.parse_args()

    cfg = load_config(args.config)
    s2p_cfg = cfg.get('suite2p', {})

    # Resolve fs and tau: CLI > config > defaults
    fs  = args.fs  or s2p_cfg.get('fs',  30.0)
    tau = args.tau or s2p_cfg.get('tau', 1.0)

    s2p_out = Path(args.s2p_out) if args.s2p_out else S2P_OUT
    s2p_out.mkdir(parents=True, exist_ok=True)

    # --batch: process each TIF in --input_dir individually
    if args.batch:
        if not args.input_dir:
            ap.error('--batch requires --input_dir')
        tif_files = sorted(Path(args.input_dir).glob('*.tif'))
        if not tif_files:
            ap.error(f'No .tif files found in {args.input_dir}')
        print(f'Batch mode: {len(tif_files)} TIFs → {s2p_out}')
        print(f'  Frame rate: {fs} Hz  Tau: {tau} s  anatomical_only: {args.anatomical_only}')
        for tif in tif_files:
            print(f'Processing {tif.name} ...')
            try:
                _run_one_fov(tif, fs, tau, cfg,
                             do_registration=args.do_registration,
                             extract_vcorr=args.extract_vcorr,
                             anatomical_only=args.anatomical_only,
                             s2p_out=s2p_out)
            except Exception as e:
                print(f'  ERROR: {tif.name}: {e}, skipping')
        return

    # --single_file: one TIF in a stem-named temp dir
    if args.single_file:
        src = Path(args.single_file)
        if not src.exists():
            raise FileNotFoundError(f'Stack not found: {src}')
        print(f'Single-file mode: {src.name}')
        _run_one_fov(src, fs, tau, cfg,
                     do_registration=args.do_registration,
                     extract_vcorr=args.extract_vcorr,
                     anatomical_only=args.anatomical_only,
                     s2p_out=s2p_out)
        return

    # --input_dir (legacy: all TIFs in one directory as a single recording)
    if not args.input_dir:
        ap.error('Provide --input_dir [--batch], --single_file, or --batch')
    input_dir = Path(args.input_dir)

    ops = build_ops(input_dir, fs, tau, cfg, anatomical_only=args.anatomical_only)
    ops['save_path0'] = str(s2p_out)

    if args.do_registration:
        ops['do_registration'] = 1
    elif args.skip_registration:
        ops['do_registration'] = 0

    print(f'Running Suite2p on {input_dir}')
    print(f'  Frame rate:    {fs} Hz')
    print(f'  Tau:           {tau} s')
    print(f'  Registration:  {"ON" if ops["do_registration"] else "OFF (pre-corrected)"}')
    print(f'  Output:        {s2p_out}')

    output_ops = run_s2p(ops=ops)

    n_rois = output_ops.get('nROIs', '?')
    print(f'Suite2p complete. {n_rois} ROIs detected.')
    print(f'Results saved to: {s2p_out}')

    # Extract Vcorr temporal map if requested
    # Vcorr extraction for legacy --input_dir path (batch/single modes handle it in _run_one_fov)
    if args.extract_vcorr and 'Vcorr' in output_ops:
        annotated_dir = BASE_DIR / 'data' / 'annotated'
        annotated_dir.mkdir(parents=True, exist_ok=True)
        vcorr = output_ops['Vcorr'].astype(np.float32)
        stem = input_dir.name.replace('_mc', '')
        out_path = annotated_dir / f'{stem}_mc_vcorr.tif'
        tifffile.imwrite(out_path, vcorr)
        print(f'Vcorr saved to: {out_path}')

if __name__ == '__main__':
    main()
