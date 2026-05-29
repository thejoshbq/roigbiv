#!/usr/bin/env python3
"""
ROI G. Biv — Model comparison: evaluate two Cellpose models on the same
held-out validation set and produce a structured report.

Usage:
  python compare_models.py \
      --old_model models/deployed/current_model \
      --new_model models/checkpoints/models/run015_epoch_XXXX \
      --extra_data_dir /mnt/external/ROIGBIV-DATA/cellpose_ready/annotated \
      --extra_masks_dir /mnt/external/ROIGBIV-DATA/cellpose_ready/masks \
      --diameter 12 \
      --output_dir /mnt/external/ROIGBIV-DATA/cellpose_ready/evaluation
"""

import argparse
import logging
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import tifffile
from cellpose import models, metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_DIR
from train import load_dataset

MODELS_DIR = BASE_DIR / "models" / "checkpoints" / "models"
DATA_DIR = BASE_DIR / "data" / "annotated"
MASKS_DIR = BASE_DIR / "data" / "masks"

log = logging.getLogger("compare")

# IoU thresholds for AP@0.5:0.95
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05).tolist()
# Named thresholds for the report
NAMED_THRESHOLDS = [0.5, 0.75]


def evaluate_model(
    model_path: Path,
    val_imgs: list,
    val_masks: list,
    diameter: float,
    channels: list[int],
    batch_size: int = 8,
) -> dict:
    """Run AP evaluation for a single model.

    Returns dict with AP scores, TP/FP/FN, and predicted masks.
    """
    model_eval = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    pred_masks, _, _ = model_eval.eval(
        val_imgs, diameter=diameter, channels=channels,
        batch_size=batch_size,
    )

    # AP at all thresholds for AP@0.5:0.95
    ap_all, tp_all, fp_all, fn_all = metrics.average_precision(
        val_masks, pred_masks, threshold=IOU_THRESHOLDS,
    )

    # AP at named thresholds
    ap_named, tp_named, fp_named, fn_named = metrics.average_precision(
        val_masks, pred_masks, threshold=NAMED_THRESHOLDS,
    )

    return {
        "path": str(model_path),
        "ap_50": float(ap_named[:, 0].mean()),
        "ap_75": float(ap_named[:, 1].mean()),
        "ap_50_95": float(ap_all.mean()),
        "tp": int(tp_named[:, 0].sum()),
        "fp": int(fp_named[:, 0].sum()),
        "fn": int(fn_named[:, 0].sum()),
        "per_image_ap_50": ap_named[:, 0].tolist(),
        "pred_masks": pred_masks,
    }


def generate_overlays(
    val_imgs: list,
    val_masks: list,
    old_preds: list,
    new_preds: list,
    output_dir: Path,
    n_samples: int = 5,
) -> list[Path]:
    """Generate side-by-side overlay images for qualitative comparison.

    Each overlay shows: mean projection | ground truth | old model | new model
    Saves as uint8 RGB TIFFs.
    """
    from skimage import segmentation

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    indices = np.linspace(0, len(val_imgs) - 1, min(n_samples, len(val_imgs)),
                          dtype=int)

    for idx in indices:
        img = val_imgs[idx]
        # Extract first channel if multi-channel
        if img.ndim == 3:
            img = img[:, :, 0]
        # Normalize to 0-255 for visualization
        lo, hi = np.percentile(img, (1, 99))
        img_norm = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
        img_u8 = (img_norm * 255).astype(np.uint8)
        base = np.stack([img_u8, img_u8, img_u8], axis=-1)  # grayscale to RGB

        gt_mask = val_masks[idx]
        old_pred = old_preds[idx]
        new_pred = new_preds[idx]

        panels = []
        # Panel 1: Mean projection
        panels.append(base.copy())

        # Panel 2: Ground truth contours (green)
        gt_overlay = base.copy()
        if gt_mask.max() > 0:
            boundaries = segmentation.find_boundaries(gt_mask, mode="thick")
            gt_overlay[boundaries] = [0, 255, 0]
        panels.append(gt_overlay)

        # Panel 3: Old model contours (red)
        old_overlay = base.copy()
        if old_pred.max() > 0:
            boundaries = segmentation.find_boundaries(old_pred, mode="thick")
            old_overlay[boundaries] = [255, 0, 0]
        panels.append(old_overlay)

        # Panel 4: New model contours (cyan)
        new_overlay = base.copy()
        if new_pred.max() > 0:
            boundaries = segmentation.find_boundaries(new_pred, mode="thick")
            new_overlay[boundaries] = [0, 255, 255]
        panels.append(new_overlay)

        composite = np.concatenate(panels, axis=1)
        out_path = output_dir / f"comparison_{idx:03d}.tif"
        tifffile.imwrite(str(out_path), composite)
        paths.append(out_path)

    return paths


def write_report(
    old_result: dict,
    new_result: dict,
    new_epoch: str,
    n_val: int,
    val_source: str,
    overlay_paths: list[Path],
    output_path: Path,
) -> None:
    """Write structured evaluation report."""

    def delta_str(new_val, old_val):
        diff = new_val - old_val
        if old_val > 0:
            pct = diff / old_val * 100
            return f"{diff:+.4f} ({pct:+.1f}%)"
        return f"{diff:+.4f}"

    ap50_d = delta_str(new_result["ap_50"], old_result["ap_50"])
    ap75_d = delta_str(new_result["ap_75"], old_result["ap_75"])
    ap5095_d = delta_str(new_result["ap_50_95"], old_result["ap_50_95"])

    # Determine verdict
    improved = new_result["ap_50_95"] > old_result["ap_50_95"]
    regressed = new_result["ap_50_95"] < old_result["ap_50_95"] - 0.01
    if improved:
        verdict = "IMPROVEMENT"
        verdict_msg = (f"New model improves AP@0.5:0.95 by "
                       f"{new_result['ap_50_95'] - old_result['ap_50_95']:.4f}")
        recommendation = "Promote new model to production"
    elif regressed:
        verdict = "REGRESSION"
        verdict_msg = (f"New model regresses AP@0.5:0.95 by "
                       f"{old_result['ap_50_95'] - new_result['ap_50_95']:.4f}")
        recommendation = "Retain current model"
    else:
        verdict = "EQUIVALENT"
        verdict_msg = "No significant difference in AP@0.5:0.95"
        recommendation = "Requires further investigation"

    overlay_list = "\n".join(f"    {p}" for p in overlay_paths) if overlay_paths else "    (none)"

    report = dedent(f"""\
    ================================================================================
    CELLPOSE MODEL COMPARISON REPORT
    ================================================================================

    Validation Set
    --------------
      Source:   {val_source}
      Images:   {n_val}

    Current Production Model
    ------------------------
      Path:     {old_result['path']}
      AP@0.5:   {old_result['ap_50']:.4f}
      AP@0.75:  {old_result['ap_75']:.4f}
      AP@0.5:0.95: {old_result['ap_50_95']:.4f}
      TP: {old_result['tp']}  FP: {old_result['fp']}  FN: {old_result['fn']}

    New Model (Best Epoch)
    ----------------------
      Path:     {new_result['path']}
      Epoch:    {new_epoch}
      AP@0.5:   {new_result['ap_50']:.4f}
      AP@0.75:  {new_result['ap_75']:.4f}
      AP@0.5:0.95: {new_result['ap_50_95']:.4f}
      TP: {new_result['tp']}  FP: {new_result['fp']}  FN: {new_result['fn']}

    Delta (New - Old)
    -----------------
      AP@0.5 change:     {ap50_d}
      AP@0.75 change:    {ap75_d}
      AP@0.5:0.95 change: {ap5095_d}

    Verdict
    -------
      {verdict} — {verdict_msg}

    Recommendation
    --------------
      {recommendation}

    Qualitative Samples
    -------------------
    {overlay_list}

    Caveats
    -------
      - Old model may have trained on some validation images (conservative bias).
      - Vcorr for new data computed via Suite2p with nbinned=2000 (vs. default 5000).
      - AP@0.5:0.95 averaged over IoU thresholds 0.50, 0.55, ..., 0.95.
    """)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    log.info("Report written: %s", output_path)
    print(report)


def main():
    ap = argparse.ArgumentParser(
        description="Compare two Cellpose models on the same validation set.")

    ap.add_argument("--old_model", required=True,
                    help="Path to current deployed model")
    ap.add_argument("--new_model", required=True,
                    help="Path to new model checkpoint")
    ap.add_argument("--new_epoch", default="unknown",
                    help="Epoch number of new model (for report)")
    ap.add_argument("--diameter", type=float, default=12)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_vcorr", action="store_true", default=True)
    ap.add_argument("--no_vcorr", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--masks_dir", default=None)
    ap.add_argument("--extra_data_dir", nargs="+", default=None)
    ap.add_argument("--extra_masks_dir", nargs="+", default=None)
    ap.add_argument("--output_dir", default=None,
                    help="Directory for report and overlays")
    ap.add_argument("--n_samples", type=int, default=5,
                    help="Number of qualitative overlay samples")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    use_vcorr = args.use_vcorr and not args.no_vcorr
    channels = [1, 2] if use_vcorr else [0, 0]

    # Resolve model paths
    old_model = Path(args.old_model)
    if not old_model.is_absolute():
        old_model = BASE_DIR / old_model
    new_model = Path(args.new_model)
    if not new_model.is_absolute():
        candidate = MODELS_DIR / args.new_model
        new_model = candidate if candidate.exists() else BASE_DIR / new_model

    # Resolve data directories
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    masks_dir = Path(args.masks_dir) if args.masks_dir else MASKS_DIR

    extra_dirs = None
    if args.extra_data_dir and args.extra_masks_dir:
        if len(args.extra_data_dir) != len(args.extra_masks_dir):
            log.error("--extra_data_dir and --extra_masks_dir must have same count")
            sys.exit(1)
        extra_dirs = [(Path(d), Path(m))
                      for d, m in zip(args.extra_data_dir, args.extra_masks_dir)]

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "evaluation"

    # Load dataset and split
    log.info("Loading dataset ...")
    images, masks_list = load_dataset(
        data_dir, masks_dir, seed=args.seed,
        use_vcorr=use_vcorr, extra_dirs=extra_dirs,
    )
    n_val = max(1, int(len(images) * args.val_frac))
    val_imgs, val_masks = images[:n_val], masks_list[:n_val]
    log.info("Val set: %d images (total %d, val_frac=%.2f)",
             n_val, len(images), args.val_frac)

    # Evaluate old model
    log.info("Evaluating old model: %s", old_model)
    old_result = evaluate_model(
        old_model, val_imgs, val_masks,
        args.diameter, channels, args.batch_size,
    )
    log.info("Old: AP@0.5=%.4f, AP@0.75=%.4f, AP@0.5:0.95=%.4f",
             old_result["ap_50"], old_result["ap_75"], old_result["ap_50_95"])

    # Evaluate new model
    log.info("Evaluating new model: %s", new_model)
    new_result = evaluate_model(
        new_model, val_imgs, val_masks,
        args.diameter, channels, args.batch_size,
    )
    log.info("New: AP@0.5=%.4f, AP@0.75=%.4f, AP@0.5:0.95=%.4f",
             new_result["ap_50"], new_result["ap_75"], new_result["ap_50_95"])

    # Generate overlay images
    log.info("Generating qualitative overlays ...")
    overlay_dir = output_dir / "overlays"
    overlay_paths = generate_overlays(
        val_imgs, val_masks,
        old_result["pred_masks"], new_result["pred_masks"],
        overlay_dir, n_samples=args.n_samples,
    )

    # Write report
    val_source = f"{data_dir}"
    if extra_dirs:
        val_source += f" + {len(extra_dirs)} extra dir(s)"
    write_report(
        old_result, new_result,
        new_epoch=args.new_epoch,
        n_val=n_val,
        val_source=val_source,
        overlay_paths=overlay_paths,
        output_path=output_dir / "report.txt",
    )


if __name__ == "__main__":
    main()
