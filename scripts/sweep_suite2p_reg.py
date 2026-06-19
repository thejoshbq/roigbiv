#!/usr/bin/env python
"""Sweep Suite2p registration params to match / beat legacy SIMA MC.

One-off tuning harness (loose-rigor ``scripts/``), runs in the ``roigbiv`` env.
The shipped ``phasecorr`` backend is ~17-24% softer than legacy SIMA on
``lap_var_smooth``; the prior a+b investigation showed legacy's edge is *robust
between-frame shift estimation on dim Prism frames*, not row-granularity. So this
sweep leans on the Suite2p knobs that raise SNR for shift estimation
(``smooth_sigma_time``, the 1P high-pass family) rather than block granularity.

Two correctness choices vs. ``bench_motion_correction.py``:
  * **Full ops control** — builds ops straight from ``default_ops()`` so it can set
    keys ``roigbiv/suite2p.py::_build_ops`` does not expose (``smooth_sigma_time``,
    ``1Preg``, ``spatial_hp_reg``, ``pre_smooth``, ``spatial_taper``,
    ``two_step_registration``, ``maxregshiftNR``). Unknown keys are a hard error,
    not silently dropped.
  * **True temporal mean from ``data.bin``** — NOT ``ops['meanImg']`` (the prior
    contrast_rms 355-vs-733 gap was a meanImg scaling artifact). Registration-only
    (``roidetect=False``) for speed.

The legacy bar is the temporal mean of ``--legacy-mean`` (the SIMA-corrected
movie, e.g. pre005_sub400_mc.tif). Acceptance = ``lap_var_smooth`` >= that bar
with no banding/haze on the per-panel-stretched montage.

    conda run -n roigbiv python scripts/sweep_suite2p_reg.py \
        --stack experiments/runs/mc_legacy_val/pre005_sub400.tif --fs 7.5 \
        --legacy-mean experiments/runs/mc_legacy_val/pre005_sub400_mc.tif \
        --outdir experiments/runs/mc_s2p_tune --configs tier1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import tifffile

# Shared metric fns — imported the way mc_legacy_val/score_replacements.py does.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_motion_correction import (  # noqa: E402
    compute_metrics, temporal_mean_tif, make_subset, montage,
)


# ─────────────────────────────────────────────────────────────────────────
# Config grid — each entry is a label -> dict of Suite2p ops overrides.
# Baseline reproduces the shipped _build_ops registration defaults so the sweep
# establishes its OWN data.bin baseline (don't trust the stale meanImg figure).
# ─────────────────────────────────────────────────────────────────────────

_BASELINE = {
    "smooth_sigma": 1.15, "maxregshift": 0.1, "nonrigid": True,
    "block_size": [128, 128], "nimg_init": 300,
}


def _cfg(**over):
    d = dict(_BASELINE)
    d.update(over)
    return d


# Tier 1 — dim-frame shift-estimation robustness (the hypothesis).
TIER1 = {
    "baseline":        _cfg(),
    "sst1":            _cfg(smooth_sigma_time=1.0),
    "sst2":            _cfg(smooth_sigma_time=2.0),
    "sst4":            _cfg(smooth_sigma_time=4.0),
    "1preg":           _cfg(**{"1Preg": True, "spatial_hp_reg": 42,
                              "pre_smooth": 0, "spatial_taper": 40}),
    "1preg_presm":     _cfg(**{"1Preg": True, "spatial_hp_reg": 42,
                              "pre_smooth": 2, "spatial_taper": 40}),
    "1preg_sst2":      _cfg(smooth_sigma_time=2.0,
                            **{"1Preg": True, "spatial_hp_reg": 42,
                               "pre_smooth": 0, "spatial_taper": 40}),
}

# Tier 2/3 — registration ACCURACY levers, the real gap (Tier-1 showed legacy is
# genuinely ~15% sharper on cells; robustness knobs barely moved it, sst HURT).
# All keep the 1P high-pass family (Tier-1's only positive lever) and drop
# smooth_sigma_time. Vary nonrigid granularity / block-shift / two-step.
_1P = {"1Preg": True, "spatial_hp_reg": 42, "pre_smooth": 2, "spatial_taper": 40}
TIER2: dict[str, dict] = {
    "fb64_1p":       _cfg(block_size=[64, 64], **_1P),
    "fb64_plain":    _cfg(block_size=[64, 64]),   # finer blocks, NO 1P high-pass
    "fb32_1p":       _cfg(block_size=[32, 32], **_1P),
    "fb32x256_1p":   _cfg(block_size=[32, 256], **_1P),
    "fb64_nr12":     _cfg(block_size=[64, 64], maxregshiftNR=12, **_1P),
    "fb64_2step":    _cfg(block_size=[64, 64], maxregshiftNR=12,
                          two_step_registration=True, **_1P),
    "fb64_sharpref": _cfg(block_size=[64, 64], smooth_sigma=0.8,
                          maxregshiftNR=12, **_1P),
    "combo":         _cfg(block_size=[32, 32], maxregshiftNR=12,
                          two_step_registration=True, nimg_init=1000, **_1P),
}

GRIDS = {"tier1": TIER1, "tier2": TIER2, "all": {**TIER1, **TIER2}}


# ─────────────────────────────────────────────────────────────────────────
# Suite2p registration-only run + true-mean reconstruction
# ─────────────────────────────────────────────────────────────────────────

def _validate_keys(params: dict, default_ops: dict, label: str) -> None:
    """Hard-error on ops keys absent from this Suite2p version (no silent drop)."""
    unknown = [k for k in params if k not in default_ops]
    if unknown:
        raise KeyError(
            f"config {label!r}: ops keys not in this Suite2p's default_ops: "
            f"{unknown}. Check version-specific names (e.g. spatial_hp vs "
            f"spatial_hp_reg).")


def register_mean(subset: Path, work: Path, fs: float, params: dict,
                  label: str) -> np.ndarray:
    """Run registration-only Suite2p with *params*; return true temporal mean
    of the registered movie reconstructed from data.bin."""
    from suite2p.default_ops import default_ops
    from suite2p.run_s2p import run_s2p

    base = default_ops()
    _validate_keys(params, base, label)

    stem = subset.stem.replace("_mc", "")
    run_root = work / label
    plane = run_root / stem / "suite2p" / "plane0"
    if (plane / "data.bin").exists() and (plane / "ops.npy").exists():
        return _mean_from_bin(plane)

    # Stage the tif into a dir named by stem (Suite2p names output after
    # basename(data_path)); hardlink to avoid copying GBs.
    stage = run_root / "_stage" / stem
    stage.mkdir(parents=True, exist_ok=True)
    local = stage / subset.name
    if local.exists():
        local.unlink()
    try:
        os.link(str(subset), str(local))
    except OSError:
        shutil.copy2(str(subset), str(local))

    ops = base
    ops.update({
        "data_path": [str(stage)],
        "tiff_list": [str(local)],
        "save_path0": str(run_root / stem),
        "save_folder": "suite2p",
        "nplanes": 1, "nchannels": 1, "functional_chan": 1,
        "fs": fs, "tau": 1.0,
        "do_registration": 1,
        "roidetect": False,       # registration only — we only need data.bin
        "reg_tif": False,
        "delete_bin": False,      # keep data.bin for mean reconstruction
        "keep_movie_raw": bool(params.get("two_step_registration", False)),
    })
    ops.update(params)
    try:
        run_s2p(ops=ops)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return _mean_from_bin(plane)


def _mean_from_bin(plane: Path, chunk: int = 256) -> np.ndarray:
    # allow_pickle: ops.npy is Suite2p's own output written moments ago in this
    # same process tree — trusted (same load pattern as bench_motion_correction.py).
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    n = int(ops.get("nframes", 0))
    mov = np.memmap(plane / "data.bin", dtype=np.int16, mode="r").reshape(-1, Ly, Lx)
    if n and n != mov.shape[0]:
        mov = mov[:n]
    T = mov.shape[0]
    acc = np.zeros((Ly, Lx), dtype=np.float64)
    for b0 in range(0, T, chunk):
        acc += mov[b0:b0 + chunk].astype(np.float64).sum(axis=0)
    return (acc / T).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=7.5)
    ap.add_argument("--legacy-mean", type=Path, required=True,
                    help="SIMA-corrected movie; its temporal mean is the bar")
    ap.add_argument("--configs", default="tier1",
                    help="grid name (tier1|tier2|all) or comma-list of labels")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    stack = args.stack.resolve()
    stem = stack.stem
    outdir = (args.outdir or (Path("experiments/runs") / f"mc_s2p_tune_{stem}")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "_runs"
    work.mkdir(exist_ok=True)

    # Resolve which configs to run.
    if args.configs in GRIDS:
        grid = GRIDS[args.configs]
    else:
        want = [c.strip() for c in args.configs.split(",") if c.strip()]
        allcfg = {**TIER1, **TIER2}
        missing = [c for c in want if c not in allcfg]
        if missing:
            raise SystemExit(f"unknown config labels: {missing}")
        grid = {c: allcfg[c] for c in want}
    if not grid:
        raise SystemExit(f"no configs selected for {args.configs!r}")

    print(f"== s2p reg sweep: {stem} | fs={args.fs} | {len(grid)} configs ==", flush=True)
    print(f"   outdir: {outdir}", flush=True)

    subset = make_subset(
        stack, outdir / f"{stem}_s{args.start_frame}_n{args.max_frames}.tif",
        args.max_frames, start=args.start_frame)

    # Quality bar: temporal mean of the SIMA-corrected movie.
    legacy_mean = temporal_mean_tif(args.legacy_mean.resolve())
    bar = compute_metrics(legacy_mean)
    bar_lvs = bar["lap_var_smooth"]
    print(f"\n-- legacy bar -- lap_var_smooth={bar_lvs:.5f} "
          f"band={bar['banding_score']:.5f}", flush=True)

    results: dict[str, dict] = {"legacy_SIMA": {**bar, "seconds": 0.0}}
    means: list[tuple[str, np.ndarray]] = [("legacy_SIMA", legacy_mean)]

    for label, params in grid.items():
        print(f"\n-- {label} -- {params}", flush=True)
        t0 = time.time()
        try:
            mean = register_mean(subset, work, args.fs, params, label)
        except Exception as exc:  # noqa: BLE001 — diagnostic: log + continue
            print(f"  SKIP {label}: {type(exc).__name__}: {exc}", flush=True)
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        dt = time.time() - t0
        m = compute_metrics(mean)
        m["seconds"] = round(dt, 1)
        m["delta_lvs"] = m["lap_var_smooth"] - bar_lvs
        m["pct_of_bar"] = 100.0 * m["lap_var_smooth"] / (bar_lvs + 1e-12)
        results[label] = m
        tifffile.imwrite(str(outdir / f"mean_{label}.tif"), mean.astype(np.float32))
        means.append((label, mean))
        flag = "PASS" if m["lap_var_smooth"] >= bar_lvs else "    "
        print(f"  {flag} {dt:.1f}s | lap_smooth={m['lap_var_smooth']:.5f} "
              f"({m['pct_of_bar']:.1f}% of bar) band={m['banding_score']:.5f} "
              f"anis={m['grad_anisotropy_xy']:.3f}", flush=True)

    montage(means, outdir / "montage_sweep.png")
    (outdir / "sweep_metrics.json").write_text(json.dumps(results, indent=2))

    # Ranked summary
    print(f"\n=== RANKED (bar lap_var_smooth = {bar_lvs:.5f}) ===", flush=True)
    ranked = sorted(
        ((k, v) for k, v in results.items() if "error" not in v),
        key=lambda kv: kv[1]["lap_var_smooth"], reverse=True)
    hdr = f"{'config':<16}{'lap_var_smooth':>16}{'%bar':>8}{'banding':>12}{'anis_xy':>10}{'sec':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for k, v in ranked:
        pct = v.get("pct_of_bar", 100.0)
        print(f"{k:<16}{v['lap_var_smooth']:>16.5f}{pct:>8.1f}"
              f"{v['banding_score']:>12.5f}{v['grad_anisotropy_xy']:>10.3f}"
              f"{v.get('seconds', 0):>8.1f}", flush=True)
    print(f"\nArtifacts in {outdir} (montage_sweep.png, mean_*.tif, sweep_metrics.json)",
          flush=True)


if __name__ == "__main__":
    main()
