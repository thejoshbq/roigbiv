"""
Algebra / contract tests for the rowwise-pcc motion-correction backend.

Run via:
    conda run -n roigbiv python -m roigbiv.pipeline.tests.test_registration

All tests force CPU (force_cpu / device="cpu") so they pass without CUDA.

Tests catch:
  - Rigid phase-correlation recovers a known global translation (subpixel)
  - Row-wise non-rigid step recovers a per-row shear (registered ≈ template)
  - max_displacement clamps oversized shifts
  - data.bin / ops.npy / _mc.tif output contract consumed by Foundation
  - A motion-free stack registers to ≈ zero shift (no spurious motion)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Synthetic helpers
# ─────────────────────────────────────────────────────────────────────────

def _synthetic_frame(Ly=64, Lx=64, n_blobs=8, seed=0) -> np.ndarray:
    """A few Gaussian blobs on a low noise floor — a stand-in FOV."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[:Ly, :Lx]
    img = np.zeros((Ly, Lx), dtype=np.float32)
    for _ in range(n_blobs):
        cy = rng.uniform(0.2 * Ly, 0.8 * Ly)
        cx = rng.uniform(0.2 * Lx, 0.8 * Lx)
        sig = rng.uniform(2.0, 4.0)
        amp = rng.uniform(400.0, 1200.0)
        img += amp * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig ** 2))
    img += rng.normal(0, 5.0, size=(Ly, Lx)).astype(np.float32)
    return np.clip(img, 0, None).astype(np.float32)


def _textured_frame(Ly=128, Lx=128, seed=0) -> np.ndarray:
    """A FOV with structure in *every* horizontal strip — sinusoidal gratings +
    blobs. Sparse-blob frames leave most strips empty, so strip-wise phase
    correlation just aliases on noise there; the gratings give each strip
    correlatable content, the regime where strip registration is actually tested.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[:Ly, :Lx]
    img = np.zeros((Ly, Lx), np.float32)
    for _ in range(6):
        fx = rng.uniform(0.05, 0.25); fy = rng.uniform(0.05, 0.25)
        ph = rng.uniform(0, 2 * np.pi)
        img += np.cos(2 * np.pi * (fx * xs + fy * ys) + ph)
    img = (img - img.min()) / (np.ptp(img) + 1e-9) * 120.0
    for _ in range(25):
        cy = rng.uniform(0, Ly); cx = rng.uniform(0, Lx)
        s = rng.uniform(2, 4); a = rng.uniform(150, 400)
        img += a * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * s ** 2))
    return img.astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────
# Helper-level tests
# ─────────────────────────────────────────────────────────────────────────

def test_rigid_shift_recovery():
    """_pcc_shifts recovers known global translations (integer + subpixel)."""
    import torch
    from scipy.ndimage import shift as ndi_shift
    from roigbiv.pipeline.registration import _pcc_shifts

    Ly = Lx = 64
    tmpl = _synthetic_frame(Ly, Lx, seed=1)
    applied = [(0.0, 0.0), (3.0, -2.0), (-5.0, 4.0), (2.3, 1.7)]
    frames = np.stack([ndi_shift(tmpl, (sy, sx), order=1, mode="reflect")
                       for (sy, sx) in applied]).astype(np.float32)

    t_tmpl = torch.as_tensor(tmpl)
    t_frames = torch.as_tensor(frames)
    tconj = torch.conj(torch.fft.rfft2(t_tmpl))
    fy, fx = _pcc_shifts(t_frames, tconj, Ly, Lx, 50, torch)
    fy = fy.numpy(); fx = fx.numpy()

    max_err = 0.0
    for i, (sy, sx) in enumerate(applied):
        ey = abs(fy[i] - sy); ex = abs(fx[i] - sx)
        max_err = max(max_err, ey, ex)
        tol = 0.25 if (sy, sx) == (2.3, 1.7) else 0.5
        assert ey < tol and ex < tol, (
            f"shift {(sy, sx)} recovered as {(fy[i], fx[i])} (tol {tol})")
    print(f"  [PASS] test_rigid_shift_recovery (max |err| = {max_err:.3f} px)")


def test_rowwise_warp_recovery():
    """Full rigid+rowwise registration undoes a per-row horizontal shear."""
    import torch
    from roigbiv.pipeline.registration import _register_batch

    Ly = Lx = 96
    tmpl = _synthetic_frame(Ly, Lx, n_blobs=14, seed=2)

    # Apply a smooth sinusoidal per-row x-shear (amplitude 4 px) by sampling.
    rows = np.arange(Ly)
    dx = 4.0 * np.sin(2 * np.pi * rows / Ly)          # per-row x displacement
    xs = np.clip(np.arange(Lx)[None, :] - dx[:, None], 0, Lx - 1)
    x0 = np.floor(xs).astype(int); x1 = np.clip(x0 + 1, 0, Lx - 1)
    w = (xs - x0).astype(np.float32)
    warped = ((1 - w) * tmpl[rows[:, None], x0] + w * tmpl[rows[:, None], x1]).astype(np.float32)

    t_tmpl = torch.as_tensor(tmpl)
    t_frames = torch.as_tensor(warped[None])          # (1, Ly, Lx)
    reg, _fy, _fx = _register_batch(
        t_frames, t_tmpl, strip_h=8, max_disp=50,
        smooth_rows=3.0, smooth_time=0.0, torch=torch,
        Fnn=torch.nn.functional)
    reg = reg[0].numpy()

    c_before = _corr(warped, tmpl)
    c_after = _corr(reg, tmpl)
    assert c_after > c_before, (
        f"registration did not improve match: {c_before:.3f} → {c_after:.3f}")
    assert c_after > 0.9, f"registered correlation {c_after:.3f} < 0.9"
    print(f"  [PASS] test_rowwise_warp_recovery "
          f"(corr {c_before:.3f} → {c_after:.3f})")


def test_max_displacement_clamp():
    """Shifts beyond max_displacement are clamped, never returned oversized."""
    import torch
    from scipy.ndimage import shift as ndi_shift
    from roigbiv.pipeline.registration import _pcc_shifts

    Ly = Lx = 128
    tmpl = _synthetic_frame(Ly, Lx, seed=3)
    frame = ndi_shift(tmpl, (80.0, -75.0), order=1, mode="reflect").astype(np.float32)

    tconj = torch.conj(torch.fft.rfft2(torch.as_tensor(tmpl)))
    fy, fx = _pcc_shifts(torch.as_tensor(frame[None]), tconj, Ly, Lx, 50, torch)
    assert abs(float(fy)) <= 50.0 and abs(float(fx)) <= 50.0, (
        f"clamp failed: ({float(fy)}, {float(fx)}) exceeds ±50")
    print(f"  [PASS] test_max_displacement_clamp "
          f"(clamped to ({float(fy):.1f}, {float(fx):.1f}))")


# ─────────────────────────────────────────────────────────────────────────
# Output-contract tests
# ─────────────────────────────────────────────────────────────────────────

def test_mc_tif_export_contract():
    """run_rowwise_pcc_register writes a uint16 {stem}_mc.tif + motion traces."""
    import tifffile
    from scipy.ndimage import shift as ndi_shift
    from roigbiv.pipeline.registration import run_rowwise_pcc_register

    T, Ly, Lx = 20, 32, 32
    rng = np.random.default_rng(4)
    base = _synthetic_frame(Ly, Lx, n_blobs=5, seed=4)
    frames = np.stack([
        ndi_shift(base, (rng.uniform(-2, 2), rng.uniform(-2, 2)),
                  order=1, mode="reflect")
        for _ in range(T)
    ]).astype(np.uint16)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tif_path = td / "fov_test.tif"
        tifffile.imwrite(str(tif_path), frames)
        out_dir = td / "out"
        out_dir.mkdir()

        # frame_batch=4 << T=20 forces ≥5 GPU batches → exercises the
        # multi-batch write path (a single-batch write hides 4D corruption).
        mc_tif_path, mx, my = run_rowwise_pcc_register(
            tif_path, out_dir, fs=7.5, force_cpu=True, frame_batch=4)

        # corrected movie lands in the output dir as {stem}_mc.tif
        assert mc_tif_path == out_dir / "fov_test_mc.tif"
        assert mc_tif_path.exists(), "missing {stem}_mc.tif export"
        mc_arr = tifffile.imread(str(mc_tif_path))
        assert mc_arr.ndim == 3, f"expected 3D (T,Ly,Lx), got {mc_arr.ndim}D {mc_arr.shape}"
        assert mc_arr.shape == (T, Ly, Lx), f"shape {mc_arr.shape}"
        assert mc_arr.dtype == np.uint16, f"dtype {mc_arr.dtype}"
        # motion traces, one shift per frame
        assert mx.shape == (T,) and my.shape == (T,)
        assert mx.dtype == np.float32 and my.dtype == np.float32
    print("  [PASS] test_mc_tif_export_contract (3D uint16 _mc.tif across 5 batches)")


def test_passthrough_preserves_3d():
    """do_registration=False over multiple batches stays a flat (T,Ly,Lx) stack.

    Regression guard: tw.write of a 3D block per batch folded the batch axis into
    a hyperstack dimension, producing a 4D file (and dropping a short tail batch).
    """
    import tifffile
    from roigbiv.pipeline.registration import run_rowwise_pcc_register

    T, Ly, Lx = 23, 16, 16   # 23 is prime → guarantees a short final batch
    frame = _synthetic_frame(Ly, Lx, n_blobs=4, seed=6).astype(np.uint16)
    frames = np.repeat(frame[None], T, axis=0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tif_path = td / "pass.tif"
        tifffile.imwrite(str(tif_path), frames)
        out_dir = td / "out"; out_dir.mkdir()

        mc_tif_path, mx, my = run_rowwise_pcc_register(
            tif_path, out_dir, fs=7.5, do_registration=False,
            force_cpu=True, frame_batch=4)

        mc_arr = tifffile.imread(str(mc_tif_path))
        assert mc_arr.ndim == 3, f"expected 3D, got {mc_arr.ndim}D {mc_arr.shape}"
        assert mc_arr.shape == (T, Ly, Lx), f"shape {mc_arr.shape} (tail batch dropped?)"
        assert np.array_equal(mc_arr, frames), "passthrough altered pixels"
        assert np.all(mx == 0) and np.all(my == 0), "passthrough should report zero motion"
    print("  [PASS] test_passthrough_preserves_3d "
          f"(T={T} over {-(-T // 4)} batches, no tail-drop)")


def test_write_mc_tif_multibatch_3d():
    """_write_mc_tif over a data.bin with chunk < T yields a 3D roundtrip-equal stack."""
    import tifffile
    from roigbiv.pipeline.registration import _write_mc_tif

    T, Ly, Lx = 25, 12, 12   # chunk=4 → 7 chunks incl. a 1-frame tail
    rng = np.random.default_rng(7)
    movie = rng.integers(0, 5000, size=(T, Ly, Lx), dtype=np.int16)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_bin = td / "data.bin"
        movie.tofile(str(data_bin))
        mc_tif_path = td / "out_mc.tif"

        _write_mc_tif(data_bin, mc_tif_path, Ly, Lx, chunk=4)

        mc_arr = tifffile.imread(str(mc_tif_path))
        assert mc_arr.ndim == 3, f"expected 3D, got {mc_arr.ndim}D {mc_arr.shape}"
        assert mc_arr.shape == (T, Ly, Lx), f"shape {mc_arr.shape}"
        assert mc_arr.dtype == np.uint16, f"dtype {mc_arr.dtype}"
        # _write_mc_tif clips negatives to 0; the rng draws are all ≥ 0 here.
        assert np.array_equal(mc_arr, movie.astype(np.uint16)), "roundtrip mismatch"
    print("  [PASS] test_write_mc_tif_multibatch_3d (7 chunks → flat 3D stack)")


def test_identity_movie_zero_shift():
    """A motion-free stack registers to ≈ zero shift and preserves the movie."""
    import tifffile
    from roigbiv.pipeline.registration import run_rowwise_pcc_register

    T, Ly, Lx = 12, 48, 48
    frame = _synthetic_frame(Ly, Lx, n_blobs=10, seed=5).astype(np.uint16)
    frames = np.repeat(frame[None], T, axis=0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tif_path = td / "still.tif"
        tifffile.imwrite(str(tif_path), frames)
        out_dir = td / "out"; out_dir.mkdir()

        mc_tif_path, mx, my = run_rowwise_pcc_register(
            tif_path, out_dir, fs=7.5, force_cpu=True)

        assert np.max(np.abs(mx)) < 0.5 and np.max(np.abs(my)) < 0.5, (
            f"spurious motion on a still movie: "
            f"max|x|={np.max(np.abs(mx)):.2f} max|y|={np.max(np.abs(my)):.2f}")
        reg = tifffile.imread(str(mc_tif_path))
        c = _corr(reg[0].astype(np.float32), frame.astype(np.float32))
        assert c > 0.99, f"registered still frame correlation {c:.4f} < 0.99"
    print("  [PASS] test_identity_movie_zero_shift "
          f"(max|x|={np.max(np.abs(mx)):.2f} max|y|={np.max(np.abs(my)):.2f})")


def test_strip_regularization_suppresses_noise_warp():
    """The shipped strip regularization barely hallucinates motion on a still
    noisy frame, where the original unregularized strips warp wildly.

    This is the GPU-free proxy for the Option-B fix and the documented
    regression mechanism: with NO real motion (only additive noise), any per-row
    displacement is spurious. The original algorithm (strip_h=8, no median/
    confidence) lets noise drive large per-strip shifts → the banding/anisotropy
    smear; the regularized config (taller strips + median/confidence + stronger
    smoothing) holds the field near zero. Averaged over 12 seeds the regularized
    peak warp is sub-pixel while the original's tail is many pixels.

    NOT a parity claim on real data — absolute phasecorr parity needs the dim
    prism stack on GPU via scripts/bench_motion_correction.py. This only proves
    the direction: regularization suppresses noise-driven warps.
    """
    import torch
    from roigbiv.pipeline.registration import _rowwise_residual
    Fnn = torch.nn.functional

    LEGACY = dict(strip_h=8, smooth_rows=3.0, prefilter=False,
                  confidence_weight=False)            # original algorithm
    SHIPPED = dict(strip_h=16, smooth_rows=6.0, prefilter=False,
                   confidence_weight=True)            # regularized fix

    def peak_warp(cfg, seed):
        base = _textured_frame(seed=seed)
        rng = np.random.default_rng(400 + seed)
        noisy = (base + rng.normal(0, 45.0, base.shape)).astype(np.float32)   # no motion
        disp = _rowwise_residual(
            torch.as_tensor(noisy[None]), torch.as_tensor(base),
            max_disp=50, smooth_time=0.0, torch=torch, Fnn=Fnn, **cfg)
        return float(np.max(np.abs(disp[0, :, 0, 0].numpy())))

    leg = np.array([peak_warp(LEGACY, s) for s in range(1, 13)])
    ship = np.array([peak_warp(SHIPPED, s) for s in range(1, 13)])
    # regularized config: spurious warp stays sub-pixel; original blows past it.
    assert ship.max() < 1.0, f"regularized still-frame warp too large: {ship.max():.2f}px"
    assert np.median(ship) < np.median(leg), (
        f"regularization did not suppress noise warp: median ship "
        f"{np.median(ship):.3f} >= legacy {np.median(leg):.3f}")
    assert leg.max() > 5.0 * ship.max(), (
        f"expected original tail >> regularized; legacy max {leg.max():.2f}, "
        f"ship max {ship.max():.2f}")
    print(f"  [PASS] test_strip_regularization_suppresses_noise_warp "
          f"(still-frame peak warp: legacy max {leg.max():.2f}px → "
          f"regularized max {ship.max():.2f}px)")


def test_strip_regularization_recovers_clean_shear():
    """The regularized config still recovers a genuine smooth per-row shear.

    The stiffer field trades some warp resolution for noise rejection, so this
    guards the floor: registration must still meaningfully improve a clean shear
    (not flatten it away).
    """
    import torch
    from roigbiv.pipeline.registration import _register_batch

    Ly = Lx = 96
    tmpl = _textured_frame(Ly, Lx, seed=3)
    rows = np.arange(Ly)
    dx = 4.0 * np.sin(2 * np.pi * rows / Ly)
    xs = np.clip(np.arange(Lx)[None, :] - dx[:, None], 0, Lx - 1)
    x0 = np.floor(xs).astype(int); x1 = np.clip(x0 + 1, 0, Lx - 1)
    w = (xs - x0).astype(np.float32)
    warped = ((1 - w) * tmpl[rows[:, None], x0]
              + w * tmpl[rows[:, None], x1]).astype(np.float32)

    reg, _fy, _fx = _register_batch(
        torch.as_tensor(warped[None]), torch.as_tensor(tmpl),
        strip_h=16, max_disp=50, smooth_rows=6.0, smooth_time=0.0,
        torch=torch, Fnn=torch.nn.functional,
        prefilter=False, confidence_weight=True)
    c_before = _corr(warped, tmpl)
    c_after = _corr(reg[0].numpy(), tmpl)
    assert c_after > c_before, (
        f"registration did not improve match: {c_before:.3f} → {c_after:.3f}")
    assert c_after > 0.6, f"regularized clean-shear corr {c_after:.3f} < 0.6"
    print(f"  [PASS] test_strip_regularization_recovers_clean_shear "
          f"(corr {c_before:.3f} → {c_after:.3f})")


def test_mc_config_fields_forwarded():
    """New mc_* knobs exist on PipelineConfig with the shipped defaults."""
    from roigbiv.pipeline.types import PipelineConfig
    cfg = PipelineConfig(fs=7.5)
    assert cfg.mc_strip_height == 32, cfg.mc_strip_height
    assert cfg.mc_smooth_sigma_rows == 6.0, cfg.mc_smooth_sigma_rows
    assert cfg.mc_smooth_sigma_time == 1.0, cfg.mc_smooth_sigma_time
    assert cfg.mc_strip_confidence_weight is True
    assert cfg.mc_prefilter is False        # band-pass off by default (ablation)
    assert cfg.mc_prefilter_sigma_low == 1.0
    assert cfg.mc_prefilter_sigma_high == 8.0
    assert cfg.foundation_only is False
    print("  [PASS] test_mc_config_fields_forwarded (parity knobs wired)")


def test_default_backend_is_phasecorr():
    """Guard the MC-regression fix: the default backend must stay ``phasecorr``.

    ``rowwise-pcc`` injects noise-driven per-row warps on dim/low-SNR FOVs
    (hazy, horizontally-banded mean — the Logan/Prism regression). It is opt-in
    only; a silent revert of this default would reintroduce the regression.
    """
    from roigbiv.pipeline.types import PipelineConfig
    backend = PipelineConfig(fs=7.5).motion_correction_backend
    assert backend == "phasecorr", (
        f"default motion_correction_backend regressed to {backend!r}; "
        f"rowwise-pcc must remain opt-in (see MC bench audit)")
    print("  [PASS] test_default_backend_is_phasecorr (default = phasecorr)")


def test_s2p_reg_default_is_tuned_config():
    """phasecorr registration defaults to the validated fb64_1p config.

    Full-session validation (2271-frame mean vs a grid-aligned legacy SIMA mean
    on the Logan Prism FOV) showed the old [128,128]/no-1Preg default reached only
    ~58% of legacy cell-sharpness; block_size=[64,64] + 1Preg=True reach ~103%
    (at/above legacy) with no over-fit banding. Guard against a silent revert of
    that tuning (which would reintroduce the regression). Bright high-SNR 2P data
    can opt out with --mc-block-size 128 128 --no-mc-1preg.
    """
    from roigbiv.pipeline.types import PipelineConfig
    cfg = PipelineConfig(fs=7.5)
    assert cfg.mc_s2p_block_size == [64, 64], (
        f"mc_s2p_block_size regressed to {cfg.mc_s2p_block_size}; tuned default "
        f"is [64,64] (full-session MC validation)")
    assert cfg.mc_s2p_one_photon_reg is True, (
        "mc_s2p_one_photon_reg (1Preg) regressed to False; it is load-bearing "
        "for legacy parity (91% → 103% cell-sharp)")
    # remaining knobs keep Suite2p's own defaults
    assert cfg.mc_s2p_smooth_sigma == 1.15, cfg.mc_s2p_smooth_sigma
    assert cfg.mc_s2p_smooth_sigma_time == 0.0, cfg.mc_s2p_smooth_sigma_time
    assert cfg.mc_s2p_maxregshift == 0.1, cfg.mc_s2p_maxregshift
    assert cfg.mc_s2p_nonrigid is True
    assert cfg.mc_s2p_maxregshift_nr == 5, cfg.mc_s2p_maxregshift_nr
    assert cfg.mc_s2p_nimg_init == 300, cfg.mc_s2p_nimg_init
    assert cfg.mc_s2p_two_step_registration is False
    assert cfg.mc_s2p_spatial_hp_reg == 42, cfg.mc_s2p_spatial_hp_reg
    assert cfg.mc_s2p_pre_smooth == 0.0, cfg.mc_s2p_pre_smooth
    assert cfg.mc_s2p_spatial_taper == 40.0, cfg.mc_s2p_spatial_taper
    # must not collide with the rowwise-pcc temporal-smoothing knob (different
    # semantics, different default)
    assert cfg.mc_smooth_sigma_time == 1.0, cfg.mc_smooth_sigma_time
    print("  [PASS] test_s2p_reg_default_is_tuned_config (block [64,64] + 1Preg)")


def test_build_ops_injects_reg_keys():
    """_build_ops sets the new registration ops keys; cfg=None == Suite2p defaults.

    The cfg=None branch is the regression guard: the registration ops for a stock
    run must match Suite2p's own default_ops (no silent behavior change from the
    cfg-flow fix). A populated cfg must override them.
    """
    try:
        from suite2p.default_ops import default_ops
    except ImportError:
        print("  [SKIP] test_build_ops_injects_reg_keys (suite2p unavailable)")
        return
    from roigbiv.suite2p import _build_ops

    base = default_ops()
    none_ops = _build_ops("/tmp/x", fs=7.5, do_registration=True, cfg=None)
    # cfg=None registration keys equal Suite2p's own defaults (byte-identical path)
    for k in ("smooth_sigma_time", "1Preg", "spatial_hp_reg", "pre_smooth",
              "spatial_taper", "maxregshiftNR", "two_step_registration",
              "smooth_sigma", "maxregshift", "block_size"):
        assert none_ops[k] == base[k], (k, none_ops[k], base[k])

    tuned = _build_ops("/tmp/x", fs=7.5, do_registration=True, cfg={"suite2p": {
        "block_size": [64, 64], "smooth_sigma_time": 2.0, "1Preg": True,
        "spatial_hp_reg": 50, "maxregshiftNR": 12, "two_step_registration": True,
    }})
    assert tuned["block_size"] == [64, 64], tuned["block_size"]
    assert tuned["smooth_sigma_time"] == 2.0
    assert tuned["1Preg"] is True
    assert tuned["spatial_hp_reg"] == 50
    assert tuned["maxregshiftNR"] == 12
    assert tuned["two_step_registration"] is True
    # two-step needs the raw movie kept; _build_ops infers it
    assert tuned["keep_movie_raw"] is True, tuned["keep_movie_raw"]
    print("  [PASS] test_build_ops_injects_reg_keys (cfg=None==defaults, cfg overrides)")


def test_phasecorr_forwards_reg_cfg():
    """Foundation's phasecorr path passes a real cfg (not None) carrying mc_s2p_*.

    Regression guard for the cfg=None bug: registration params were unreachable.
    Stubs run_suite2p_fov to capture the cfg it receives.
    """
    import roigbiv.suite2p as s2p
    from roigbiv.pipeline.foundation import run_motion_correction
    from roigbiv.pipeline.types import PipelineConfig

    captured = {}

    class _Captured(Exception):
        pass

    orig = s2p.run_suite2p_fov

    def _stub(tif_path, output_dir, **k):
        captured["cfg"] = k.get("cfg")
        raise _Captured

    s2p.run_suite2p_fov = _stub
    try:
        cfg = PipelineConfig(fs=7.5)  # default backend = phasecorr
        cfg.mc_s2p_block_size = [64, 64]
        cfg.mc_s2p_one_photon_reg = True
        with tempfile.TemporaryDirectory() as d:
            tif = Path(d) / "fov.tif"
            tif.write_bytes(b"")  # never read; stub raises first
            try:
                run_motion_correction(tif, cfg, Path(d))
            except _Captured:
                pass
        c = captured.get("cfg")
        assert c is not None, "phasecorr passed cfg=None (the bug is back)"
        s2pc = c.get("suite2p", {})
        assert s2pc.get("block_size") == [64, 64], s2pc
        assert s2pc.get("1Preg") is True, s2pc
        # only registration keys forwarded — detection keys stay at _build_ops defaults
        assert "threshold_scaling" not in s2pc, s2pc
        print("  [PASS] test_phasecorr_forwards_reg_cfg (cfg flows, reg-only)")
    finally:
        s2p.run_suite2p_fov = orig


def test_suppress_tifffile_uic_divide_is_scoped():
    """The Suite2p call-site filter swallows tifffile's benign UIC divide warning
    but leaves a same-message warning from another module visible.

    Guards the fix for the upstream MetaMorph/UIC RATIONAL tag (0 denominator)
    that emits ``invalid value encountered in divide`` during Suite2p's TIFF
    load — the warning must be silenced without masking genuine numerical
    warnings from Suite2p's own math (same message, different origin module).
    """
    import warnings
    from roigbiv.suite2p import _suppress_tifffile_uic_divide

    # 1) tifffile-origin warning is swallowed
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with _suppress_tifffile_uic_divide():
            warnings.warn_explicit(
                "invalid value encountered in divide", RuntimeWarning,
                filename="tifffile.py", lineno=20158, module="tifffile")
        leaked = [str(w.message) for w in rec]
        assert not leaked, f"tifffile UIC divide warning leaked: {leaked}"

    # 2) same message from a different module still surfaces (narrow scope)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with _suppress_tifffile_uic_divide():
            warnings.warn_explicit(
                "invalid value encountered in divide", RuntimeWarning,
                filename="suite2p/detection.py", lineno=1,
                module="suite2p.detection")
        assert len(rec) == 1 and issubclass(rec[0].category, RuntimeWarning), (
            f"non-tifffile divide warning was wrongly suppressed: {rec}")
    print("  [PASS] test_suppress_tifffile_uic_divide_is_scoped "
          "(tifffile silenced, other modules surface)")


# ─────────────────────────────────────────────────────────────────────────
# Legacy (SIMA) backend
# ─────────────────────────────────────────────────────────────────────────

def _write_synth_tif(path, T=3, Ly=48, Lx=40, seed=1):
    """A small moving-blob uint16 stack, written page-per-frame like real data.

    The roigbiv assembler writes one IFD per frame; mirror that so reads behave
    identically to production stacks (a volumetric single-IFD TIFF reads back
    differently). SIMA's cross-correlation needs the frame comfortably larger
    than 2*max_displacement, so keep Ly/Lx ≥ ~128 for the SIMA-running tests.
    """
    import tifffile
    rng = np.random.default_rng(seed)
    base = np.zeros((Ly, Lx), np.float32)
    for cy, cx in [(Ly // 3, Lx // 3), (2 * Ly // 3, 2 * Lx // 3), (Ly // 2, Lx // 4)]:
        base[max(cy - 5, 0):cy + 5, max(cx - 5, 0):cx + 5] = 1200.0
    with tifffile.TiffWriter(str(path)) as tw:
        for t in range(T):
            dy, dx = int(round(2 * np.sin(t))), int(round(2 * np.cos(t)))
            f = np.roll(np.roll(base, dy, 0), dx, 1) + rng.poisson(40, (Ly, Lx))
            tw.write(np.clip(f, 0, 65535).astype(np.uint16), contiguous=True)
    return T, Ly, Lx


def _sima_env_available(env="sima-legacy"):
    import subprocess
    try:
        r = subprocess.run(
            ["conda", "run", "-n", env, "python", "-c", "import sima"],
            capture_output=True, timeout=180)
        return r.returncode == 0
    except Exception:
        return False


def test_legacy_backend_in_validated_set():
    """Foundation routes ``backend='legacy'`` to the SIMA driver, not ValueError.

    Stubs the driver so the dispatcher's validation + branch selection is
    exercised without launching SIMA or Suite2p.
    """
    import roigbiv.pipeline.legacy_mc as lm
    from roigbiv.pipeline.foundation import run_motion_correction
    from roigbiv.pipeline.types import PipelineConfig

    class _Routed(Exception):
        pass

    orig = lm.run_sima_legacy_register

    def _stub(*a, **k):
        raise _Routed

    lm.run_sima_legacy_register = _stub
    try:
        cfg = PipelineConfig(fs=7.5)
        cfg.motion_correction_backend = "legacy"
        with tempfile.TemporaryDirectory() as d:
            tif = Path(d) / "fov.tif"
            tif.write_bytes(b"")  # never read; stub raises first
            try:
                run_motion_correction(tif, cfg, Path(d))
            except _Routed:
                print("  [PASS] test_legacy_backend_in_validated_set (routed to SIMA)")
                return
        raise AssertionError("backend='legacy' did not route to the SIMA driver")
    finally:
        lm.run_sima_legacy_register = orig


def test_legacy_missing_env_raises_actionable():
    """A missing/broken sidecar env yields a clear, build-pointing RuntimeError."""
    from roigbiv.pipeline.legacy_mc import run_sima_legacy_register
    with tempfile.TemporaryDirectory() as d:
        tif = Path(d) / "fov.tif"
        _write_synth_tif(tif, Ly=128, Lx=128)  # large enough to pass the size guard
        try:
            run_sima_legacy_register(
                tif, Path(d), fs=7.5, do_registration=True,
                sima_env="does-not-exist-xyz-roigbiv")
        except RuntimeError as e:
            msg = str(e)
            assert "does-not-exist-xyz-roigbiv" in msg, msg
            assert "build_sima_legacy.sh" in msg, msg
            print("  [PASS] test_legacy_missing_env_raises_actionable")
            return
    raise AssertionError("missing sidecar env did not raise RuntimeError")


def test_legacy_config_fields_forwarded():
    """Legacy backend knobs exist on PipelineConfig with shipped defaults."""
    from roigbiv.pipeline.types import PipelineConfig
    cfg = PipelineConfig(fs=7.5)
    assert cfg.mc_sima_env == "sima-legacy", cfg.mc_sima_env
    assert cfg.mc_granularity == "row", cfg.mc_granularity
    assert cfg.mc_max_displacement == 50, cfg.mc_max_displacement  # shared knob
    print("  [PASS] test_legacy_config_fields_forwarded")


def test_legacy_passthrough_when_precorrected():
    """``do_registration=False`` copies to {stem}_mc.tif with zero traces, no SIMA."""
    import tifffile
    from roigbiv.pipeline.legacy_mc import run_sima_legacy_register
    with tempfile.TemporaryDirectory() as d:
        tif = Path(d) / "fov_mc.tif"
        T, Ly, Lx = _write_synth_tif(tif)
        mc_path, mx, my = run_sima_legacy_register(
            tif, Path(d), fs=7.5, do_registration=False,
            sima_env="does-not-exist-xyz")  # bogus env must NOT matter (no SIMA call)
        out = tifffile.imread(str(mc_path))
        assert out.shape == (T, Ly, Lx), out.shape
        assert out.dtype == np.uint16, out.dtype
        assert mx.shape == (T,) and my.shape == (T,), (mx.shape, my.shape)
        assert not np.any(mx) and not np.any(my), "passthrough traces must be zero"
        print("  [PASS] test_legacy_passthrough_when_precorrected")


def test_legacy_sima_roundtrip_smoke():
    """End-to-end SIMA correction via the sidecar env (skips if env absent)."""
    import tifffile
    if not _sima_env_available():
        print("  [SKIP] test_legacy_sima_roundtrip_smoke (sima-legacy env unavailable)")
        return
    from roigbiv.pipeline.legacy_mc import run_sima_legacy_register
    with tempfile.TemporaryDirectory() as d:
        tif = Path(d) / "fov.tif"
        T, Ly, Lx = _write_synth_tif(tif, T=6, Ly=128, Lx=128)
        mc_path, mx, my = run_sima_legacy_register(
            tif, Path(d), fs=7.5, do_registration=True, max_displacement=50)
        assert Path(mc_path).exists(), mc_path
        out = tifffile.imread(str(mc_path))
        assert out.ndim == 3 and out.shape[0] == T, out.shape
        assert out.dtype == np.uint16, out.dtype
        assert mx.shape == (T,) and my.shape == (T,), (mx.shape, my.shape)
        # SIMA pads/crops to its own canvas; dims need not equal the input.
        print(f"  [PASS] test_legacy_sima_roundtrip_smoke (out {out.shape})")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_rigid_shift_recovery,
        test_rowwise_warp_recovery,
        test_max_displacement_clamp,
        test_mc_tif_export_contract,
        test_passthrough_preserves_3d,
        test_write_mc_tif_multibatch_3d,
        test_identity_movie_zero_shift,
        test_strip_regularization_suppresses_noise_warp,
        test_strip_regularization_recovers_clean_shear,
        test_mc_config_fields_forwarded,
        test_default_backend_is_phasecorr,
        test_s2p_reg_default_is_tuned_config,
        test_build_ops_injects_reg_keys,
        test_phasecorr_forwards_reg_cfg,
        test_suppress_tifffile_uic_divide_is_scoped,
        test_legacy_backend_in_validated_set,
        test_legacy_missing_env_raises_actionable,
        test_legacy_config_fields_forwarded,
        test_legacy_passthrough_when_precorrected,
        test_legacy_sima_roundtrip_smoke,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            traceback.print_exc()
            failed.append(test.__name__)
    print()
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")
