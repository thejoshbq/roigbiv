#!/usr/bin/env python3
"""
ROI G. Biv — Joint Suite2p + Cellpose consensus pipeline orchestrator.
Runs both pipelines, converts formats, matches ROIs, and outputs consensus masks.

Usage:
  python run_joint_pipeline.py --fov T1_221209_..._mc --fs 30.0
  python run_joint_pipeline.py --fov "T1_221209*" --config configs/pipeline.yaml
  python run_joint_pipeline.py --all --config configs/pipeline.yaml
"""

import argparse, glob, logging, os, sys, time
from pathlib import Path
import numpy as np
import tifffile
import yaml
import torch

BASE_DIR = Path.home() / 'Otis-Lab' / 'Projects' / 'roigbiv'

def load_config(config_path=None):
    default = BASE_DIR / 'configs' / 'pipeline.yaml'
    path = Path(config_path) if config_path else default
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def resolve_fovs(fov_arg, raw_dir):
    """Resolve FOV argument to list of stem names."""
    if fov_arg == '__all__':
        return sorted({p.stem.replace('_mc', '') for p in raw_dir.glob('*_mc.tif')})

    stems = []
    for part in fov_arg.split(','):
        part = part.strip()
        # Check if it's a glob pattern
        if '*' in part or '?' in part:
            matches = sorted(raw_dir.glob(f'{part}*_mc.tif'))
            stems.extend(m.stem.replace('_mc', '') for m in matches)
        else:
            # Exact stem (with or without _mc suffix)
            stem = part.replace('_mc', '')
            if (raw_dir / f'{stem}_mc.tif').exists():
                stems.append(stem)
            else:
                print(f'WARNING: No raw stack for {stem}')
    return sorted(set(stems))


def run_cellpose_if_needed(stem, annotated_dir, out_dir, cfg, log):
    """Run Cellpose inference if masks don't already exist."""
    mask_path = out_dir / f'{stem}_mc_masks.tif'
    if mask_path.exists():
        log.info(f'Cellpose mask exists: {mask_path.name}')
        return mask_path

    # Also check without _mc suffix
    alt_path = out_dir / f'{stem}_masks.tif'
    if alt_path.exists():
        log.info(f'Cellpose mask exists: {alt_path.name}')
        return alt_path

    from cellpose import models
    cp_cfg = cfg.get('cellpose', {})

    model_path = Path(cp_cfg.get('model_path', 'models/deployed/current_model'))
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    # Find mean projection
    mean_path = annotated_dir / f'{stem}_mc_mean.tif'
    if not mean_path.exists():
        mean_path = annotated_dir / f'{stem}_mean.tif'
    if not mean_path.exists():
        log.warning(f'No mean projection for {stem}, skipping Cellpose')
        return None

    log.info(f'Running Cellpose on {mean_path.name}')
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    img = tifffile.imread(mean_path).astype(np.float32)
    masks, _, _ = model.eval(
        img,
        diameter=cp_cfg.get('diameter', 30),
        channels=cp_cfg.get('channels', [0, 0]),
        flow_threshold=cp_cfg.get('flow_threshold', 0.4),
        cellprob_threshold=cp_cfg.get('cellprob_threshold', 0.0),
        normalize=cp_cfg.get('normalize', True),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(mask_path, masks.astype(np.uint16))
    log.info(f'  Cellpose: {masks.max()} ROIs → {mask_path.name}')

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    return mask_path


def run_suite2p_if_needed(stem, raw_dir, s2p_base_dir, cfg, log):
    """Run Suite2p if outputs don't already exist."""
    # Suite2p saves to: s2p_base_dir/suite2p/plane0/
    plane_dir = s2p_base_dir / 'suite2p' / 'plane0'
    stat_path = plane_dir / 'stat.npy'

    # For per-FOV separation, check a stem-specific subdir
    fov_out = s2p_base_dir / stem
    fov_plane = fov_out / 'suite2p' / 'plane0'
    fov_stat  = fov_plane / 'stat.npy'

    if fov_stat.exists():
        log.info(f'Suite2p output exists: {fov_plane}')
        return fov_plane

    from suite2p import run_s2p, default_ops
    import shutil, tempfile

    s2p_cfg = cfg.get('suite2p', {})
    raw_path = raw_dir / f'{stem}_mc.tif'
    if not raw_path.exists():
        log.warning(f'No raw stack for {stem}, skipping Suite2p')
        return None

    log.info(f'Running Suite2p on {raw_path.name}')

    # Create temp dir with symlink (not copy) to avoid doubling disk usage
    tmp_dir = tempfile.mkdtemp(prefix='s2p_')
    os.symlink(raw_path.resolve(), Path(tmp_dir) / raw_path.name)

    ops = default_ops()
    ops.update({
        'data_path':       [tmp_dir],
        'save_path0':      str(fov_out),
        'save_folder':     'suite2p',
        'nplanes':         s2p_cfg.get('nplanes', 1),
        'nchannels':       s2p_cfg.get('nchannels', 1),
        'functional_chan':  s2p_cfg.get('functional_chan', 1),
        'tau':             s2p_cfg.get('tau', 1.0),
        'fs':              s2p_cfg.get('fs', 30.0),
        'do_registration': s2p_cfg.get('do_registration', 0),
        'nimg_init':       s2p_cfg.get('nimg_init', 300),
        'batch_size':      s2p_cfg.get('batch_size', 250),
        'smooth_sigma':    s2p_cfg.get('smooth_sigma', 1.15),
        'maxregshift':     s2p_cfg.get('maxregshift', 0.1),
        'nonrigid':        s2p_cfg.get('nonrigid', True),
        'block_size':      s2p_cfg.get('block_size', [128, 128]),
        'spatial_scale':       s2p_cfg.get('spatial_scale', 0),
        'threshold_scaling':   s2p_cfg.get('threshold_scaling', 1.0),
        'max_iterations':      s2p_cfg.get('max_iterations', 20),
        'connected':           s2p_cfg.get('connected', True),
        'nbinned':             s2p_cfg.get('nbinned', 5000),
        'allow_overlap':       s2p_cfg.get('allow_overlap', False),
        'preclassify':     s2p_cfg.get('preclassify', 0.0),
        'high_pass':              s2p_cfg.get('high_pass', 100),
        'inner_neuropil_radius':  s2p_cfg.get('inner_neuropil_radius', 2),
        'min_neuropil_pixels':    s2p_cfg.get('min_neuropil_pixels', 350),
        'spikedetect':     s2p_cfg.get('spikedetect', True),
        # Must be 0 to avoid importing cellpose 4.x (we use cellpose 3.x)
        'anatomical_only': 0,
    })

    fov_out.mkdir(parents=True, exist_ok=True)
    output_ops = run_s2p(ops=ops)
    n_rois = output_ops.get('nROIs', '?')
    log.info(f'  Suite2p: {n_rois} ROIs detected')

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Delete Suite2p binary files (multi-GB, only needed during processing)
    for binf in fov_out.rglob('data*.bin'):
        binf.unlink(missing_ok=True)
    for binf in fov_out.rglob('data_raw.bin'):
        binf.unlink(missing_ok=True)

    torch.cuda.empty_cache()

    return fov_plane


def run_s2p_to_masks(s2p_plane_dir, ref_mask_path, cfg, log):
    """Convert Suite2p stat to uint16 mask, aligned to Cellpose dims."""
    from s2p_to_masks import s2p_stat_to_mask

    cons_cfg = cfg.get('consensus', {})
    min_prob = cons_cfg.get('s2p_min_iscell_prob', 0.5)

    ref_shape = None
    if ref_mask_path and Path(ref_mask_path).exists():
        ref = tifffile.imread(ref_mask_path)
        ref_shape = ref.shape[:2]

    mask, n_rois = s2p_stat_to_mask(
        s2p_plane_dir / 'stat.npy',
        s2p_plane_dir / 'iscell.npy',
        s2p_plane_dir / 'ops.npy',
        min_prob=min_prob,
        ref_shape=ref_shape,
    )

    out_path = s2p_plane_dir / 's2p_masks.tif'
    tifffile.imwrite(out_path, mask)
    log.info(f'  Suite2p mask: {n_rois} ROIs → {out_path.name}')
    return out_path


def run_consensus(cp_mask_path, s2p_mask_path, s2p_plane_dir, stem,
                  annotated_dir, out_dir, cfg, log):
    """Run consensus matching and save outputs."""
    from match_rois import match_and_tier, build_consensus_mask, save_diagnostics
    import csv

    cons_cfg = cfg.get('consensus', {})
    iou_threshold = cons_cfg.get('iou_threshold', 0.3)
    default_tiers = [t.upper() for t in cons_cfg.get('default_tiers', ['gold', 'silver'])]

    cp_mask  = tifffile.imread(cp_mask_path).astype(np.uint16)
    s2p_mask = tifffile.imread(s2p_mask_path).astype(np.uint16)

    s2p_iscell = None
    iscell_path = s2p_plane_dir / 'iscell.npy'
    if iscell_path.exists():
        s2p_iscell = np.load(iscell_path)

    records = match_and_tier(cp_mask, s2p_mask, iou_threshold, s2p_iscell)

    n_gold   = sum(1 for r in records if r['tier'] == 'GOLD')
    n_silver = sum(1 for r in records if r['tier'] == 'SILVER')
    n_bronze = sum(1 for r in records if r['tier'] == 'BRONZE')
    log.info(f'  Consensus: GOLD={n_gold} SILVER={n_silver} BRONZE={n_bronze}')

    out_dir.mkdir(parents=True, exist_ok=True)

    # Default consensus mask
    consensus = build_consensus_mask(cp_mask, s2p_mask, records,
                                     tiers=tuple(default_tiers))
    tifffile.imwrite(out_dir / f'{stem}_consensus_masks.tif', consensus)

    # All-tiers mask
    all_tiers = build_consensus_mask(cp_mask, s2p_mask, records,
                                     tiers=('GOLD', 'SILVER', 'BRONZE'))
    tifffile.imwrite(out_dir / f'{stem}_all_tiers_masks.tif', all_tiers)

    # Per-ROI CSV
    csv_path = out_dir / f'{stem}_consensus.csv'
    fields = ['roi_id', 'tier', 'iou_score', 'cellpose_label', 's2p_label',
              'centroid_y', 'centroid_x', 'area_px', 's2p_iscell_prob']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    # Diagnostic overlay
    mean_path = annotated_dir / f'{stem}_mc_mean.tif'
    if not mean_path.exists():
        mean_path = annotated_dir / f'{stem}_mean.tif'
    if mean_path.exists():
        diag_dir = out_dir / 'diagnostics' / stem
        save_diagnostics(str(mean_path), cp_mask, s2p_mask, records, diag_dir)

    return records


def main():
    ap = argparse.ArgumentParser(
        description='Joint Suite2p + Cellpose consensus pipeline')
    ap.add_argument('--fov', default=None,
                    help='FOV stem(s): comma-separated or glob pattern')
    ap.add_argument('--all', action='store_true',
                    help='Process all FOVs in raw_dir')
    ap.add_argument('--config', default=None,
                    help='Path to pipeline YAML config')
    ap.add_argument('--skip_traces', action='store_true',
                    help='Skip trace extraction step')
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get('paths', {})

    raw_dir       = BASE_DIR / paths.get('raw_dir', 'data/raw')
    annotated_dir = BASE_DIR / paths.get('annotated_dir', 'data/annotated')
    s2p_out       = BASE_DIR / paths.get('s2p_output', 'suite2p_workspace/output')
    cp_out        = BASE_DIR / paths.get('inference_output', 'inference/output')
    cons_out      = BASE_DIR / paths.get('consensus_output', 'inference/consensus')

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()])
    log = logging.getLogger('roigbiv.joint')

    # Add scripts dir to path for local imports
    sys.path.insert(0, str(BASE_DIR / 'scripts'))

    # Resolve FOVs
    if args.all:
        fov_stems = resolve_fovs('__all__', raw_dir)
    elif args.fov:
        fov_stems = resolve_fovs(args.fov, raw_dir)
    else:
        ap.error('Provide --fov or --all')

    log.info(f'Processing {len(fov_stems)} FOV(s)')

    for i, stem in enumerate(fov_stems):
        log.info(f'\n[{i+1}/{len(fov_stems)}] {stem}')
        t0 = time.time()

        # Step 1: Cellpose inference
        cp_mask_path = run_cellpose_if_needed(
            stem, annotated_dir, cp_out, cfg, log)
        if cp_mask_path is None:
            log.warning(f'Skipping {stem}: no Cellpose output')
            continue

        # Step 2: Suite2p detection
        s2p_plane_dir = run_suite2p_if_needed(
            stem, raw_dir, s2p_out, cfg, log)
        if s2p_plane_dir is None:
            log.warning(f'Skipping {stem}: no Suite2p output')
            continue

        # Step 3: Convert Suite2p stat → mask
        s2p_mask_path = run_s2p_to_masks(
            s2p_plane_dir, cp_mask_path, cfg, log)

        # Step 4: Consensus matching
        records = run_consensus(
            cp_mask_path, s2p_mask_path, s2p_plane_dir, stem,
            annotated_dir, cons_out, cfg, log)

        elapsed = time.time() - t0
        log.info(f'  Completed in {elapsed:.1f}s')

    log.info(f'\nPipeline complete. Results: {cons_out}')

if __name__ == '__main__':
    main()
