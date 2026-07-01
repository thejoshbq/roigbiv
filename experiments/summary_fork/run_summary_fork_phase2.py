"""Summary-image fork experiment — Phase 2: denoising (DeepCAD-RT).

One variable vs Phase 1: the motion-corrected movie is self-supervised-denoised with
DeepCAD-RT, then the Phase-1 winning summary (``mean_M``) is recomputed from the
DENOISED movie and the held-constant detector re-run. Everything else (detector cfg,
ch2 channel, Gate 1 on the ORIGINAL mean_M, IoU matcher) is fixed to Phase 1.

DeepCAD-RT is vendored from source at experiments/summary_fork/_vendor/DeepCAD-RT
(the PyPI ``deepcad`` wheel is a non-functional stub). Trains per-movie.

Scoped output: experiments/summary_fork/run_<NNN>/ with denoise_input/, pth/,
results/, the recomputed denoised mean_M, masks, manifest, metrics.

Usage:
    python .../run_summary_fork_phase2.py --fov grin  --run-id 003 [--smoke]
    python .../run_summary_fork_phase2.py --fov prism --run-id 004
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path("/home/thejoshbq/Otis-Lab/Projects/Phoxel-Workbench/roigbiv")
SFORK = PROJECT / "experiments/summary_fork"
DEEPCAD = SFORK / "_vendor/DeepCAD-RT/DeepCAD_RT_pytorch"
sys.path.insert(0, str(SFORK))
sys.path.insert(0, str(DEEPCAD))          # vendored deepcad source package

import run_summary_fork as P1              # noqa: E402  (Phase-1 helpers/registry)
from roigbiv.pipeline.foundation import _open_data_bin      # noqa: E402
from roigbiv.pipeline.gate1 import evaluate_gate1           # noqa: E402
from roigbiv.pipeline.stage1 import run_cellpose_detection  # noqa: E402


def export_movie_to_tiff(data_bin: Path, Ly: int, Lx: int, out_tif: Path,
                         max_frames: int | None = None) -> int:
    """data.bin (int16, T,H,W) -> a single tiff stack DeepCAD can read."""
    T = data_bin.stat().st_size // (Ly * Lx * 2)
    if max_frames:
        T = min(T, max_frames)
    mov = _open_data_bin(data_bin, Ly, Lx)[:T]
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    # DeepCAD normalizes internally; keep int16 dynamic range as uint16>=0.
    tifffile.imwrite(str(out_tif), np.asarray(mov), bigtiff=True)
    return int(T)


def deepcad_train(input_dir: Path, pth_dir: Path, *, n_epochs: int,
                  train_size: int, patch_xy: int, patch_t: int) -> str:
    # DeepCAD's save_model() does an ONNX export (for its realtime path) that torch
    # 2.12 routes through onnxscript (absent). The .pth is torch.save'd BEFORE the
    # export, so no-op the export — we only need the state_dict for denoising.
    import torch
    torch.onnx.export = lambda *a, **k: None
    from deepcad.train_collection import training_class
    train_dict = {
        "patch_x": patch_xy, "patch_y": patch_xy, "patch_t": patch_t,
        "overlap_factor": 0.25, "scale_factor": 1, "select_img_num": 100000,
        "train_datasets_size": train_size, "datasets_path": str(input_dir),
        "pth_dir": str(pth_dir), "n_epochs": n_epochs, "lr": 5e-5,
        "b1": 0.5, "b2": 0.999, "fmap": 16, "GPU": "0", "num_workers": 0,
        "visualize_images_per_epoch": False, "save_test_images_per_epoch": False,
    }
    pre = set(p.name for p in pth_dir.glob("*")) if pth_dir.exists() else set()
    training_class(train_dict).run()
    post = set(p.name for p in pth_dir.glob("*"))
    new = sorted(post - pre)
    if not new:
        raise SystemExit("DeepCAD training produced no model folder")
    return new[-1]   # <datasets_name>_<timestamp>


def deepcad_denoise(input_dir: Path, pth_dir: Path, model_folder: str,
                    output_dir: Path, *, patch_xy: int, patch_t: int) -> Path:
    """Prune to the final-epoch .pth, run testing, return the denoised tiff path."""
    model_dir = pth_dir / model_folder
    pths = sorted(model_dir.glob("*.pth"))
    if not pths:
        raise SystemExit(f"no .pth in {model_dir}")
    keep = pths[-1]                                    # final epoch
    for p in pths:
        if p != keep:
            p.unlink()

    from deepcad.test_collection import testing_class
    test_dict = {
        "patch_x": patch_xy, "patch_y": patch_xy, "patch_t": patch_t,
        "overlap_factor": 0.6, "scale_factor": 1, "test_datasize": 100000,
        "datasets_path": str(input_dir), "pth_dir": str(pth_dir),
        "denoise_model": model_folder, "output_dir": str(output_dir),
        "fmap": 16, "GPU": "0", "num_workers": 0,
        "visualize_images_per_epoch": False,
    }
    testing_class(test_dict).run()
    outs = sorted(output_dir.rglob("*_output.tif"))
    if not outs:
        raise SystemExit(f"no denoised *_output.tif under {output_dir}")
    return outs[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", required=True, choices=sorted(P1.FOVS))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--train-size", type=int, default=3000)
    ap.add_argument("--patch-xy", type=int, default=150)
    ap.add_argument("--patch-t", type=int, default=150)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run (few frames/epochs) to validate the pipeline")
    args = ap.parse_args()

    fov = P1.FOVS[args.fov]
    run_dir = SFORK / f"run_{args.run_id}"
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    n_epochs, train_size, max_frames = args.n_epochs, args.train_size, None
    patch_xy, patch_t = args.patch_xy, args.patch_t
    if args.smoke:
        n_epochs, train_size, max_frames, patch_xy, patch_t = 1, 300, 200, 96, 96

    print(f"=== summary-fork PHASE 2 (denoise) | fov={args.fov} | run_{args.run_id} "
          f"| epochs={n_epochs} train_size={train_size} smoke={args.smoke} ===", flush=True)

    cfg, _ = P1.load_cfg(fov["manifest"], fov["expected_cfg"])

    # Original (Phase-1) summaries — held constant except the denoised morph channel.
    summ = fov["summary"]
    mean_M_orig = tifffile.imread(str(summ / "mean_M.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(summ / "vcorr_S.tif")).astype(np.float32)
    dog_map = tifffile.imread(str(summ / "dog_map.tif")).astype(np.float32)
    max_S = tifffile.imread(str(summ / "max_S.tif")).astype(np.float32)
    gt = tifffile.imread(str(fov["gt"])).astype(np.uint16)
    Ly, Lx = mean_M_orig.shape

    # 1) Export MC movie to tiff for DeepCAD.
    data_bin = fov["s2p_plane"] / "data.bin"
    in_dir = run_dir / "denoise_input"
    in_tif = in_dir / f"{fov['stem']}.tif"
    print(f"  exporting movie -> {in_tif.name} ...", flush=True)
    T = export_movie_to_tiff(data_bin, Ly, Lx, in_tif, max_frames=max_frames)
    print(f"  exported T={T} frames ({Ly}x{Lx})", flush=True)

    # 2) DeepCAD-RT self-supervised training on this movie.
    pth_dir = run_dir / "pth"
    pth_dir.mkdir(exist_ok=True)
    print(f"  DeepCAD training ({n_epochs} epochs, {train_size} patches) ...", flush=True)
    t0 = time.time()
    model_folder = deepcad_train(in_dir, pth_dir, n_epochs=n_epochs,
                                 train_size=train_size, patch_xy=patch_xy, patch_t=patch_t)
    print(f"  trained model folder: {model_folder} ({time.time()-t0:.0f}s)", flush=True)

    # 3) Denoise the movie.
    out_dir = run_dir / "results"
    print(f"  DeepCAD denoising ...", flush=True)
    t0 = time.time()
    denoised_tif = deepcad_denoise(in_dir, pth_dir, model_folder, out_dir,
                                   patch_xy=patch_xy, patch_t=patch_t)
    print(f"  denoised -> {denoised_tif.name} ({time.time()-t0:.0f}s)", flush=True)

    # 4) Recompute mean_M from the DENOISED movie.
    den = tifffile.imread(str(denoised_tif))
    if den.ndim != 3:
        raise SystemExit(f"unexpected denoised ndim {den.ndim}")
    mean_M_den = den.astype(np.float64).mean(axis=0).astype(np.float32)
    if mean_M_den.shape != mean_M_orig.shape:
        raise SystemExit(f"denoised mean_M shape {mean_M_den.shape} != {mean_M_orig.shape}")
    tifffile.imwrite(str(run_dir / "mean_M_denoised.tif"), mean_M_den)

    # 5) Re-detect (denoised morph; ch2 + Gate-1 mean_M held to ORIGINAL).
    gt_ids = np.unique(gt); gt_ids = gt_ids[gt_ids != 0]
    results = {}
    for name, morph in [("mean_M_denoised", mean_M_den), ("mean_M_baseline", mean_M_orig)]:
        masks_list, probs_list, label_image, _ = run_cellpose_detection(
            morph.astype(np.float32), vcorr_S, cfg, max_S=max_S)
        rois = evaluate_gate1(masks_list, probs_list, mean_M_orig, vcorr_S, dog_map, cfg,
                              starting_label_id=1)
        n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
        raw_lab = P1.label_from_masks(masks_list, mean_M_orig.shape)
        acc_lab = P1.accept_labels_from_rois(rois, mean_M_orig.shape)
        tifffile.imwrite(str(run_dir / "masks" / f"{name}_raw_labels.tif"), raw_lab)
        tifffile.imwrite(str(run_dir / "masks" / f"{name}_accept_labels.tif"), acc_lab)
        results[name] = {"n_detected": len(masks_list), "n_accept": n_accept,
                         "raw": P1.score(gt, raw_lab), "accept": P1.score(gt, acc_lab)}
        print(f"  {name}: det={len(masks_list)} acc={n_accept} | "
              f"raw@0.5={results[name]['raw']['iou_0.5']['recall']:.3f} "
              f"@0.3={results[name]['raw']['iou_0.3']['recall']:.3f} | "
              f"acc@0.5={results[name]['accept']['iou_0.5']['recall']:.3f} "
              f"@0.3={results[name]['accept']['iou_0.3']['recall']:.3f}", flush=True)

    manifest = {
        "experiment": "summary_image_fork", "phase": 2, "variable": "denoising (DeepCAD-RT)",
        "fov": args.fov, "lens": fov["lens"], "stem": fov["stem"],
        "gt_independent": fov["gt_independent"], "n_gt": int(len(gt_ids)),
        "smoke": args.smoke, "T_frames": T,
        "deepcad": {"n_epochs": n_epochs, "train_datasets_size": train_size,
                    "patch_xy": patch_xy, "patch_t": patch_t, "model_folder": model_folder,
                    "source": "vendored cabooster/DeepCAD-RT (pypi wheel is a stub)"},
        "held_constant": "detector cfg, ch2 (vcorr_S/max_S), Gate-1 mean_M, IoU matcher = Phase 1",
        "pinned_cfg": fov["expected_cfg"],
        "runtime_s": round(time.time() - t_all, 1),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({"conditions": results}, indent=2))
    # keep the run lean: drop the big exported input movie (reproducible from data.bin)
    if not args.smoke:
        shutil.rmtree(in_dir, ignore_errors=True)
    print(f"\n  wrote {run_dir}/ (manifest, metrics, mean_M_denoised, masks, pth)", flush=True)


if __name__ == "__main__":
    main()
