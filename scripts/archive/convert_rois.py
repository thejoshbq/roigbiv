#!/usr/bin/env python3
"""
Convert ImageJ ROI .zip files to Cellpose uint16 label masks.
Expects: FOVNAME.zip alongside FOVNAME_mean.tif in the annotated directory.
Outputs: FOVNAME_masks.tif in the masks directory.
"""
import roifile
import tifffile
import numpy as np
from pathlib import Path
from skimage.draw import polygon
import argparse

def roi_zip_to_mask(roi_zip_path, reference_tif_path, out_mask_path):
    ref = tifffile.imread(reference_tif_path)
    H, W = ref.shape[-2], ref.shape[-1]
    mask = np.zeros((H, W), dtype=np.uint16)

    rois = roifile.roiread(str(roi_zip_path))
    for idx, roi in enumerate(rois, start=1):
        coords = roi.coordinates()
        if coords is None or len(coords) < 3:
            continue
        rr, cc = polygon(coords[:, 1], coords[:, 0], shape=(H, W))
        mask[rr, cc] = idx

    tifffile.imwrite(out_mask_path, mask)
    print(f'  {len(rois)} ROIs -> {out_mask_path}')

def batch_convert(annotated_dir, masks_dir):
    annotated_dir = Path(annotated_dir)
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for zip_file in sorted(annotated_dir.glob('*.zip')):
        stem = zip_file.stem
        ref_tif = annotated_dir / f'{stem}_mean.tif'
        if not ref_tif.exists():
            print(f'WARNING: No mean projection found for {stem}, skipping')
            continue
        out_mask = masks_dir / f'{stem}_masks.tif'
        print(f'Converting {zip_file.name} ...')
        roi_zip_to_mask(zip_file, ref_tif, out_mask)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('annotated_dir')
    ap.add_argument('masks_dir')
    args = ap.parse_args()
    batch_convert(args.annotated_dir, args.masks_dir)
