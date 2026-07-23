#!/usr/bin/env python
"""Benchmark motion-correction backends on a raw two-photon stack.

Diagnostic harness for the ``rowwise-pcc`` quality regression (see
``~/.claude/plans/mode-audit-escalte-context-temporal-cloud.md``). It runs the
same stack subset through each available backend, builds the temporal-mean
image of the registered movie, and scores sharpness / contrast / horizontal
banding so we can verify "≥ legacy quality" by measurement rather than eyeball.

A well-registered movie has a *sharp* temporal mean; residual motion blurs it.
Per-row jitter (the rowwise-pcc failure mode) shows up as horizontal banding,
which the banding metric isolates.

Backends
--------
  raw              no correction — the motion-blur floor
  rowwise-pcc      current GPU backend, as shipped
  rowwise-pcc-fixed current backend with Phase-B knobs (skipped if unsupported)
  phasecorr        Suite2p rigid+nonrigid registration
  sima             legacy SIMA HiddenMarkov2D (skipped if not importable)

The legacy SIMA render (``--legacy-ref``, e.g. logan_fov2.png) is folded into
the montage + metrics as the visual quality bar when SIMA itself won't install.

Usage
-----
    conda run -n roigbiv python scripts/bench_motion_correction.py \
        --stack data/logan_cousa_trial/_stacks/052126_DS-Prism-3_VI15_D2_FOV2_pre-005.tif \
        --fs 7.5 --max-frames 1200 \
        --backends raw,rowwise-pcc,phasecorr \
        --legacy-ref /home/thejoshbq/Downloads/logan_fov2.png
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.mc_metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────

def make_subset(src: Path, dst: Path, n: int, start: int = 0) -> Path:
    """Write frames ``[start, start+n)`` of ``src`` to ``dst`` (page-selective read)."""
    if dst.exists():
        return dst
    with tifffile.TiffFile(str(src)) as tf:
        T = len(tf.pages)
        start = max(0, min(start, max(0, T - 1)))
        end = min(start + n, T)
        arr = tf.asarray(key=range(start, end))
    if arr.ndim == 2:
        arr = arr[None]
    tifffile.imwrite(str(dst), arr, bigtiff=True)
    print(f"  subset: frames [{start},{end}) {arr.shape[1:]} -> {dst.name}", flush=True)
    return dst


def temporal_mean_tif(path: Path, chunk: int = 256) -> np.ndarray:
    with tifffile.TiffFile(str(path)) as tf:
        T = len(tf.pages)
        Ly, Lx = tf.pages[0].shape
        acc = np.zeros((Ly, Lx), dtype=np.float64)
        for b0 in range(0, T, chunk):
            b1 = min(b0 + chunk, T)
            arr = tf.asarray(key=range(b0, b1)).astype(np.float64)
            if arr.ndim == 2:
                arr = arr[None]
            acc += arr.sum(axis=0)
    return (acc / T).astype(np.float32)


def save_png(img: np.ndarray, path: Path, label: str) -> None:
    from PIL import Image, ImageDraw
    from roigbiv.overlay import _stretch_to_uint8
    u8 = _stretch_to_uint8(img, 0.5, 99.5)
    im = Image.fromarray(u8).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, max(140, 8 * len(label)), 16], fill=(0, 0, 0))
    d.text((3, 3), label, fill=(255, 255, 0))
    im.save(str(path))


def montage(images: list[tuple[str, np.ndarray]], path: Path, cols: int = 3) -> None:
    from PIL import Image, ImageDraw
    from roigbiv.overlay import _stretch_to_uint8
    tiles = []
    H = W = 0
    for label, img in images:
        u8 = _stretch_to_uint8(img, 0.5, 99.5)
        im = Image.fromarray(u8).convert("RGB")
        H, W = u8.shape
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, max(160, 8 * len(label)), 18], fill=(0, 0, 0))
        d.text((3, 4), label, fill=(255, 255, 0))
        tiles.append(im)
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * W, rows * H), (30, 30, 30))
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * W, r * H))
    canvas.save(str(path))
    print(f"  montage -> {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# Backends — each returns (mean_image, seconds)
# ─────────────────────────────────────────────────────────────────────────

def run_raw(subset: Path, outdir: Path, fs: float) -> np.ndarray:
    return temporal_mean_tif(subset)


def run_rowwise(subset: Path, outdir: Path, fs: float,
                extra: dict | None = None, work_name: str = "rowwise") -> np.ndarray:
    from roigbiv.pipeline.registration import run_rowwise_pcc_register
    extra = extra or {}
    sig = inspect.signature(run_rowwise_pcc_register)
    accepted = {k: v for k, v in extra.items() if k in sig.parameters}
    rejected = set(extra) - set(accepted)
    if rejected:
        raise RuntimeError(
            f"rowwise backend got params not in run_rowwise_pcc_register: "
            f"{sorted(rejected)}")
    work = outdir / work_name
    work.mkdir(parents=True, exist_ok=True)
    mc_path, _mx, _my = run_rowwise_pcc_register(
        subset, work, fs=fs, do_registration=True, **accepted)
    return temporal_mean_tif(mc_path)


def run_phasecorr(subset: Path, outdir: Path, fs: float) -> np.ndarray:
    from roigbiv.suite2p import run_suite2p_fov
    work = outdir / "phasecorr"
    work.mkdir(parents=True, exist_ok=True)
    stem = Path(subset).stem.replace("_mc", "")
    run_suite2p_fov(subset, work, fs=fs, anatomical_only=0, tau=1.0,
                    do_registration=True, cfg=None)
    ops_path = work / stem / "suite2p" / "plane0" / "ops.npy"
    # allow_pickle: ops.npy is Suite2p's own output written moments ago in this
    # same process tree — trusted (same load pattern as foundation.py).
    ops = np.load(ops_path, allow_pickle=True).item()
    return np.asarray(ops["meanImg"], dtype=np.float32)


def run_sima(subset: Path, outdir: Path, fs: float) -> np.ndarray:
    import sima
    import sima.motion
    work = outdir / "sima"
    work.mkdir(parents=True, exist_ok=True)
    seq = sima.Sequence.create("TIFF", str(subset))
    mc = sima.motion.HiddenMarkov2D(
        granularity="row", max_displacement=[50, 50], verbose=False)
    ds = mc.correct([seq], str(work / "dataset.sima"))
    avg_path = work / "avg.tif"
    ds.export_averages([str(avg_path)], fmt="TIFF16", projection_type="average")
    return np.asarray(tifffile.imread(str(avg_path)), dtype=np.float32)


# "rowwise-pcc" is pinned to the ORIGINAL (unregularized) defaults so the bench
# keeps measuring the regression baseline; "rowwise-pcc-fixed" carries the
# Option-B parity defaults (band-pass + confidence-weighted strip regularization
# + larger strips + stronger smoothing) that now ship as run_rowwise_pcc_register
# defaults. The A/B is the whole point — do not let them collapse to the same call.
_ROWWISE_LEGACY = {
    "prefilter": False, "strip_height": 8, "smooth_sigma_rows": 3.0,
    "strip_confidence_weight": False,
}
# Strip regularization (taller strips + median/confidence + smoothing) is what
# closes the gap on synthetic; prefilter is left OFF (it degraded white-noise
# frames in ablation) but is an available knob — flip it on here to A/B the
# band-pass on real, structured-background data.
_ROWWISE_FIXED = {
    "prefilter": False, "strip_height": 32, "smooth_sigma_rows": 6.0,
    "strip_confidence_weight": True,
}
BACKENDS = {
    "raw": run_raw,
    "rowwise-pcc": lambda s, o, f: run_rowwise(
        s, o, f, extra=_ROWWISE_LEGACY, work_name="rowwise"),
    "rowwise-pcc-fixed": lambda s, o, f: run_rowwise(
        s, o, f, extra=_ROWWISE_FIXED, work_name="rowwise_fixed"),
    "phasecorr": run_phasecorr,
    "sima": run_sima,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=7.5)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--start-frame", type=int, default=0,
                    help="first frame of the benchmark window")
    ap.add_argument("--backends",
                    default="raw,rowwise-pcc,rowwise-pcc-fixed,phasecorr,sima")
    ap.add_argument("--legacy-ref", type=Path, default=None,
                    help="PNG render of the legacy SIMA mean, as a visual bar")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    stack = args.stack.resolve()
    stem = stack.stem
    # Absolute paths throughout: Suite2p's stager resolves tiff_list relative to
    # data_path, so a relative subset path yields a doubled, nonexistent path.
    outdir = (args.outdir or (Path("experiments/runs") / f"mc_bench_{stem}")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"== MC bench: {stem} | fs={args.fs} | <= {args.max_frames} frames ==", flush=True)
    print(f"   outdir: {outdir}", flush=True)

    subset = make_subset(
        stack, outdir / f"{stem}_s{args.start_frame}_n{args.max_frames}.tif",
        args.max_frames, start=args.start_frame)

    requested = [b.strip() for b in args.backends.split(",") if b.strip()]
    results: dict[str, dict] = {}
    means: list[tuple[str, np.ndarray]] = []

    for name in requested:
        if name not in BACKENDS:
            print(f"  ! unknown backend {name!r}, skipping", flush=True)
            continue
        print(f"\n-- {name} --", flush=True)
        t0 = time.time()
        try:
            mean = BACKENDS[name](subset, outdir, args.fs)
        except Exception as exc:  # noqa: BLE001 — diagnostic: log + continue
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        dt = time.time() - t0
        m = compute_metrics(mean)
        m["seconds"] = round(dt, 1)
        results[name] = m
        tifffile.imwrite(str(outdir / f"mean_{name}.tif"), mean.astype(np.float32))
        save_png(mean, outdir / f"mean_{name}.png", name)
        means.append((name, mean))
        print(f"  {dt:.1f}s | lap_smooth={m['lap_var_smooth']:.4f} "
              f"lap_var={m['lap_var']:.3f} grad_e={m['grad_energy']:.4f} "
              f"band={m['banding_score']:.4f} anis_xy={m['grad_anisotropy_xy']:.3f}", flush=True)

    if args.legacy_ref and args.legacy_ref.exists():
        from PIL import Image
        ref = np.asarray(Image.open(str(args.legacy_ref)).convert("L"),
                         dtype=np.float32)
        results["legacy_ref(png)"] = compute_metrics(ref)
        means.append(("legacy_ref", ref))
        print(f"\n-- legacy_ref(png) -- lap_var={results['legacy_ref(png)']['lap_var']:.3f}",
              flush=True)

    montage(means, outdir / "montage.png")
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2))

    # Summary table
    print("\n=== SUMMARY (higher lap_var/grad/tenengrad/contrast = sharper; "
          "lower banding = better; anisotropy_xy ~1.0 = isotropic) ===", flush=True)
    cols = ["lap_var_smooth", "lap_var", "grad_energy", "grad_anisotropy_xy",
            "banding_score", "contrast_rms", "seconds"]
    hdr = f"{'backend':<20}" + "".join(f"{c:>20}" for c in cols)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<20}{m['error']:>20}", flush=True)
            continue
        row = f"{name:<20}" + "".join(
            f"{m.get(c, float('nan')):>20.4f}" if isinstance(m.get(c), (int, float))
            else f"{'-':>20}" for c in cols)
        print(row, flush=True)
    print(f"\nArtifacts in {outdir} (montage.png, mean_*.png, metrics.json)", flush=True)


if __name__ == "__main__":
    main()
