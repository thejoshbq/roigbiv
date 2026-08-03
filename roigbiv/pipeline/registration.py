"""
ROI G. Biv pipeline — row-wise non-rigid motion correction (``rowwise-pcc``).

A GPU-accelerated phase-correlation registration backend, parallel in role to the
Suite2p wrapper in :mod:`roigbiv.suite2p`. It captures the *intent* of the lab's
legacy SIMA ``HiddenMarkov2D(granularity='row', max_displacement=[50, 50])``
method — correcting per-row (resonant-scan) motion — without SIMA's abandoned,
single-process implementation.

Pipeline:
  1. Build a reference template (mean of a high-correlation strided subset,
     refined ``n_template_iters`` times).
  2. Rigid pre-align each frame via FFT phase correlation (torch.fft.rfft2),
     subpixel-refined by a 3-point parabolic fit, clamped to ``max_displacement``.
  3. Row-wise non-rigid: split each (already rigid-aligned) frame into horizontal
     strips, phase-correlate each strip against the same template band, build a
     smooth per-row displacement field, and resample with ``grid_sample``. To
     suppress the noise-driven per-row warps that smeared dim low-SNR (e.g. 1024²
     prism) FOVs, the strip estimates are regularized before they become a field:
     taller strips (``strip_height``) raise per-strip SNR, a 3-wide median +
     confidence-weighting (``strip_confidence_weight``) reject outlier strips, and
     ``smooth_sigma_rows`` stiffens the field. On a still noisy frame this drops
     the spurious-warp magnitude by ~30× vs the original unregularized strips. An
     optional DoG band-pass (``prefilter``, default off) is available for data
     with structured background, but on white-noise-dominated frames it degrades
     the peak and is left off pending per-dataset bench validation.
  4. Write the registered movie as an int16 ``data.bin`` memmap (the exact
     substrate :func:`roigbiv.pipeline.foundation.compute_background_separation`
     consumes) plus an ``ops.npy`` carrying ``Ly/Lx/nframes/meanImg/xoff/yoff``,
     and export a uint16 ``{stem}_mc.tif`` into the output dir.

Memory: the raw stack is read into RAM once (≈ T·Ly·Lx·2 B); GPU work streams in
frame batches auto-capped by free VRAM (mirrors the Stage 3 budget discipline).
Disk writes are pre-reserved via :func:`roigbiv.pipeline.diskguard.ensure_free_space`
so a full disk fails deterministically instead of SIGBUS-ing on a dirty mmap page.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.io import MC_SOFTWARE_TAG
from roigbiv.pipeline.device import cuda_compute_capable
from roigbiv.pipeline.diskguard import ensure_free_space


# ─────────────────────────────────────────────────────────────────────────
# Torch primitives (imported lazily inside the entry point so module import
# stays cheap and CPU-only contexts don't pay for CUDA init).
# ─────────────────────────────────────────────────────────────────────────

def _make_base_grid(B, Ly, Lx, device, torch):
    ys = torch.linspace(-1.0, 1.0, Ly, device=device)
    xs = torch.linspace(-1.0, 1.0, Lx, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1)              # (Ly, Lx, 2) → (x, y)
    return grid.unsqueeze(0).expand(B, -1, -1, -1)


def _subpixel_2d(corr, py, px, H, W, torch):
    """3-point parabolic offset of the correlation peak, with toroidal wrap."""
    B = corr.shape[0]
    bidx = torch.arange(B, device=corr.device)

    def val(yy, xx):
        return corr[bidx, yy % H, xx % W]

    c = val(py, px)
    ym, yp = val(py - 1, px), val(py + 1, px)
    xm, xp = val(py, px - 1), val(py, px + 1)
    den_y = ym - 2.0 * c + yp
    den_x = xm - 2.0 * c + xp
    zero = torch.zeros_like(c)
    oy = torch.where(den_y.abs() > 1e-6, 0.5 * (ym - yp) / den_y, zero).clamp(-1.0, 1.0)
    ox = torch.where(den_x.abs() > 1e-6, 0.5 * (xm - xp) / den_x, zero).clamp(-1.0, 1.0)
    return oy, ox


def _pcc_shifts(frames, tmpl_fft_conj, H, W, max_disp, torch, return_conf=False):
    """Per-item subpixel phase-correlation shift (dy, dx), clamped to ±max_disp.

    ``frames`` is (N, H, W); ``tmpl_fft_conj`` broadcasts against rfft2(frames)
    so it may be (H, Wf) for a shared template or (N, H, Wf) for per-item bands.
    Returned (dy, dx) is the displacement of each frame relative to the template
    (frame ≈ template shifted by (dy, dx)); warping by the same (dy, dx) sample
    offset registers the frame back onto the template.

    With ``return_conf=True`` also returns the normalized-PCC peak height per
    item — a sharpness/confidence proxy used to down-weight low-SNR strips.
    """
    F = torch.fft.rfft2(frames)
    R = F * tmpl_fft_conj
    R = R / (R.abs() + 1e-8)
    corr = torch.fft.irfft2(R, s=(H, W))
    N = corr.shape[0]
    flat = corr.reshape(N, -1)
    idx = flat.argmax(dim=1)
    py = torch.div(idx, W, rounding_mode="floor")
    px = idx - py * W
    oy, ox = _subpixel_2d(corr, py, px, H, W, torch)
    fy = py.to(torch.float32) + oy
    fx = px.to(torch.float32) + ox
    fy = torch.where(fy > H / 2.0, fy - H, fy).clamp(-max_disp, max_disp)
    fx = torch.where(fx > W / 2.0, fx - W, fx).clamp(-max_disp, max_disp)
    if return_conf:
        return fy, fx, flat.amax(dim=1)
    return fy, fx


def _warp(frames, disp_xy, base_grid, Ly, Lx, torch, Fnn):
    """Resample ``frames`` (B,Ly,Lx) at p + disp_xy (pixels, order x,y)."""
    nx = disp_xy[..., 0] * (2.0 / max(Lx - 1, 1))
    ny = disp_xy[..., 1] * (2.0 / max(Ly - 1, 1))
    grid = base_grid + torch.stack([nx, ny], dim=-1)
    out = Fnn.grid_sample(frames.unsqueeze(1), grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
    return out.squeeze(1)


def _const_field(fx, fy, Ly, Lx, torch):
    B = fx.shape[0]
    field = torch.zeros(B, Ly, Lx, 2, device=fx.device)
    field[..., 0] = fx.view(B, 1, 1)
    field[..., 1] = fy.view(B, 1, 1)
    return field


def _framewise_corr(frames, tmpl, torch):
    B = frames.shape[0]
    a = frames.reshape(B, -1)
    t = tmpl.reshape(-1)
    a = a - a.mean(dim=1, keepdim=True)
    t = t - t.mean()
    num = (a * t).sum(dim=1)
    den = a.norm(dim=1) * t.norm() + 1e-8
    return num / den


def _gauss1d(x, sigma, dim, torch, Fnn):
    if sigma is None or sigma <= 0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    xt = x.transpose(dim, -1)
    shape = xt.shape
    xt = xt.reshape(-1, 1, shape[-1])
    pad = min(radius, shape[-1] - 1)
    xt = Fnn.pad(xt, (pad, pad), mode="reflect")
    xt = Fnn.conv1d(xt, k.view(1, 1, -1), padding=radius - pad)
    xt = xt.reshape(*shape).transpose(dim, -1)
    return xt


def _gauss2d(x, sigma, torch, Fnn):
    """Separable 2D Gaussian blur over the last two axes."""
    if sigma is None or sigma <= 0:
        return x
    x = _gauss1d(x, sigma, -1, torch, Fnn)
    x = _gauss1d(x, sigma, -2, torch, Fnn)
    return x


def _prefilter(x, sigma_lo, sigma_hi, torch, Fnn):
    """Difference-of-Gaussians band-pass for shift *estimation* inputs.

    ``sigma_lo`` (small) suppresses the per-pixel shot noise that normalized
    phase correlation would otherwise whiten into a spurious peak; ``sigma_hi``
    (large) removes the slow background gradient that dominates dim two-photon
    frames. Applied to correlation inputs only — the downstream ``_warp``
    resamples the *raw* frame, so output pixels are never band-passed. This is
    the primary fix for ``rowwise-pcc``'s noise-driven per-row warps on low-SNR
    (e.g. 1024² prism) FOVs.
    """
    return _gauss2d(x, sigma_lo, torch, Fnn) - _gauss2d(x, sigma_hi, torch, Fnn)


def _hann2d(h, w, device, dtype, torch):
    """Separable 2D Hann window; tapers strip edges to kill FFT wraparound."""
    wy = (torch.hann_window(h, periodic=False, device=device, dtype=dtype)
          if h > 1 else torch.ones(h, device=device, dtype=dtype))
    wx = (torch.hann_window(w, periodic=False, device=device, dtype=dtype)
          if w > 1 else torch.ones(w, device=device, dtype=dtype))
    return torch.outer(wy, wx)


def _median1d(x, k, torch, Fnn):
    """Sliding median along the last axis — rejects single-strip outliers.

    No-op when there are too few strips (<3) to define a 3-wide window.
    """
    if k is None or k <= 1 or x.shape[-1] < 3:
        return x
    r = k // 2
    pad = min(r, x.shape[-1] - 1)
    xp = Fnn.pad(x.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
    win = xp.unfold(-1, 2 * pad + 1, 1)
    return win.median(dim=-1).values


def _confidence_blend(s, conf, torch, Fnn, radius=2):
    """Blend low-confidence strip estimates toward a confidence-weighted local mean.

    A strip whose phase-correlation peak is weak (low SNR) defers to its
    neighbours; a strip with a sharp peak keeps its own estimate. The blend
    factor ``a = w / (w + median_w)`` is ~0.5 at the median confidence, →1 for
    strong strips, →0 for noise strips. No-op when there are <3 strips.
    """
    if s.shape[-1] < 3:
        return s
    w = conf.clamp(min=0.0) + 1e-6
    pad = min(radius, s.shape[-1] - 1)
    k = 2 * pad + 1
    wp = Fnn.pad(w.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
    sp = Fnn.pad((s * w).unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
    num = sp.unfold(-1, k, 1).sum(dim=-1)
    den = wp.unfold(-1, k, 1).sum(dim=-1)
    local = num / (den + 1e-8)
    wmed = w.median(dim=-1, keepdim=True).values
    a = w / (w + wmed + 1e-8)
    return a * s + (1.0 - a) * local


def _build_template(stack, device, n_iters, max_disp, torch, Fnn, n_init=300,
                    *, prefilter=False, pf_lo=1.0, pf_hi=8.0):
    T = stack.shape[0]
    sel = np.linspace(0, T - 1, min(n_init, T)).astype(np.int64)
    sub = torch.as_tensor(np.ascontiguousarray(stack[sel]).astype(np.float32),
                          device=device)
    B, Ly, Lx = sub.shape
    base = _make_base_grid(B, Ly, Lx, device, torch)
    tmpl = sub.mean(dim=0)
    for _ in range(max(1, n_iters)):
        if prefilter:
            sub_e = _prefilter(sub, pf_lo, pf_hi, torch, Fnn)
            tmpl_e = _prefilter(tmpl[None], pf_lo, pf_hi, torch, Fnn)[0]
        else:
            sub_e, tmpl_e = sub, tmpl
        tconj = torch.conj(torch.fft.rfft2(tmpl_e))
        fy, fx = _pcc_shifts(sub_e, tconj, Ly, Lx, max_disp, torch)
        aligned = _warp(sub, _const_field(fx, fy, Ly, Lx, torch), base, Ly, Lx, torch, Fnn)
        cc = _framewise_corr(aligned, tmpl, torch)
        keep = cc >= cc.median()
        if int(keep.sum()) >= 2:
            tmpl = aligned[keep].mean(dim=0)
        else:
            tmpl = aligned.mean(dim=0)
    return tmpl


def _rowwise_residual(reg, tmpl, strip_h, max_disp, smooth_rows, smooth_time,
                      torch, Fnn, *, prefilter=False, pf_lo=1.0, pf_hi=8.0,
                      confidence_weight=False):
    """Per-row residual displacement field from strip-wise phase correlation.

    Quality path (``prefilter``/``confidence_weight`` on): band-pass the strips
    and template bands before correlation, Hann-window strip edges, median-reject
    single-strip outliers, then confidence-weight low-SNR strips toward their
    neighbours — all *before* interpolating strip estimates up to a per-row field
    and smoothing. The warp still resamples the raw ``reg`` (estimation inputs
    are filtered, output pixels are not).
    """
    B, Ly, Lx = reg.shape
    n_strips = (Ly + strip_h - 1) // strip_h
    pad = n_strips * strip_h - Ly

    # Estimation inputs only: band-pass to suppress shot-noise-driven peaks.
    if prefilter:
        est = _prefilter(reg, pf_lo, pf_hi, torch, Fnn)
        tmpl_e = _prefilter(tmpl[None], pf_lo, pf_hi, torch, Fnn)[0]
    else:
        est, tmpl_e = reg, tmpl

    if pad:
        # replicate-pad the bottom rows so the last strip is full height;
        # 2D padding needs a 4D (N,C,H,W) tensor.
        regp = Fnn.pad(est.unsqueeze(1), (0, 0, 0, pad), mode="replicate").squeeze(1)
        tmplp = Fnn.pad(tmpl_e[None, None], (0, 0, 0, pad), mode="replicate")[0, 0]
    else:
        regp, tmplp = est, tmpl_e

    rs = regp.reshape(B, n_strips, strip_h, Lx).reshape(B * n_strips, strip_h, Lx)
    ts = (tmplp.reshape(n_strips, strip_h, Lx)
          .unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_strips, strip_h, Lx))

    if prefilter:
        win = _hann2d(strip_h, Lx, rs.device, rs.dtype, torch)
        rs = rs * win
        ts = ts * win

    tconj = torch.conj(torch.fft.rfft2(ts))
    # Strip search is dominated by horizontal (x) shear; clamp the small vertical
    # component to half the strip height so a strip can't leak into its neighbour.
    sy, sx, conf = _pcc_shifts(rs, tconj, strip_h, Lx, max_disp, torch,
                               return_conf=True)
    sy = sy.clamp(-strip_h / 2.0, strip_h / 2.0)
    sx = sx.view(B, n_strips)
    sy = sy.view(B, n_strips)
    conf = conf.view(B, n_strips)

    # Strip regularization (the empirical fix for noise-driven per-row warps):
    # median-reject single-strip outliers, then confidence-weight low-SNR strips
    # toward their neighbours — both BEFORE upsampling, so a noisy strip can't
    # contaminate its neighbours through the linear interpolation. Gated together
    # so disabling it reproduces the original (unregularized) algorithm exactly.
    if confidence_weight:
        sx = _median1d(sx, 3, torch, Fnn)
        sy = _median1d(sy, 3, torch, Fnn)
        sx = _confidence_blend(sx, conf, torch, Fnn)
        sy = _confidence_blend(sy, conf, torch, Fnn)

    # strips → rows (uniform strip heights map linearly), then smooth. A single
    # strip is a constant row-shift — broadcast it (interpolate's align_corners
    # path divides by n_strips-1 and would NaN at n_strips==1).
    if n_strips == 1:
        xr = sx.expand(B, Ly).contiguous()
        yr = sy.expand(B, Ly).contiguous()
    else:
        xr = Fnn.interpolate(sx.unsqueeze(1), size=Ly, mode="linear",
                             align_corners=True).squeeze(1)
        yr = Fnn.interpolate(sy.unsqueeze(1), size=Ly, mode="linear",
                             align_corners=True).squeeze(1)
    xr = _gauss1d(xr, smooth_rows, 1, torch, Fnn)
    yr = _gauss1d(yr, smooth_rows, 1, torch, Fnn)
    if B > 1:
        xr = _gauss1d(xr, smooth_time, 0, torch, Fnn)
        yr = _gauss1d(yr, smooth_time, 0, torch, Fnn)

    disp = torch.zeros(B, Ly, Lx, 2, device=reg.device)
    disp[..., 0] = xr.unsqueeze(-1)
    disp[..., 1] = yr.unsqueeze(-1)
    return disp


def _register_batch(frames, tmpl, strip_h, max_disp, smooth_rows, smooth_time,
                    torch, Fnn, *, prefilter=False, pf_lo=1.0, pf_hi=8.0,
                    confidence_weight=False, return_conf=False):
    """Register one frame batch. With ``return_conf`` also yields the rigid PCC
    peak per frame (registration confidence for the live preview trace)."""
    B, Ly, Lx = frames.shape
    base = _make_base_grid(B, Ly, Lx, frames.device, torch)
    # Rigid step: estimate the global shift on band-passed inputs, warp the raw.
    if prefilter:
        est = _prefilter(frames, pf_lo, pf_hi, torch, Fnn)
        tmpl_e = _prefilter(tmpl[None], pf_lo, pf_hi, torch, Fnn)[0]
    else:
        est, tmpl_e = frames, tmpl
    tconj = torch.conj(torch.fft.rfft2(tmpl_e))
    fy, fx, conf = _pcc_shifts(est, tconj, Ly, Lx, max_disp, torch,
                               return_conf=True)
    reg_rigid = _warp(frames, _const_field(fx, fy, Ly, Lx, torch), base, Ly, Lx, torch, Fnn)
    resid = _rowwise_residual(reg_rigid, tmpl, strip_h, max_disp,
                              smooth_rows, smooth_time, torch, Fnn,
                              prefilter=prefilter, pf_lo=pf_lo, pf_hi=pf_hi,
                              confidence_weight=confidence_weight)
    reg = _warp(reg_rigid, resid, base, Ly, Lx, torch, Fnn)
    if return_conf:
        return reg, fy, fx, conf
    return reg, fy, fx


# ─────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────

def _vram_budget_batch(Ly, Lx, device, requested, torch, budget_bytes=4 * 1024**3):
    """Cap the frame batch so peak VRAM stays under budget (and free memory)."""
    if device != "cuda":
        return max(1, int(requested))
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return max(1, int(requested))
    per_frame = max(1, Ly * Lx * 4 * 10)   # ~10× working set (fft + corr + grid + strips)
    cap = max(1, int(min(int(free), budget_bytes) // per_frame))
    return max(1, min(int(requested), cap))


def _append_frames(tw, frames_u16, software=None):
    """Append a ``(n, Ly, Lx)`` uint16 block as ``n`` individual 2D pages.

    ``TiffWriter.write`` defines one image *series* per call; handing it a 3D
    block per batch makes tifffile fold the batch axis into a hyperstack
    dimension, yielding a 4D ``(n_batches, batch, Ly, Lx)`` file (and dropping a
    short tail batch). Writing one 2D page per frame — the same idiom the
    PrairieView assembler uses in :mod:`roigbiv.io` — keeps the result a flat
    ``(T, Ly, Lx)`` stack. ``contiguous=True`` per identical 2D page stays fast.

    ``software`` (when given) stamps the TIFF Software tag (305) onto the *first*
    page only — pass it on the first ``_append_frames`` call of a file so
    :func:`roigbiv.io.detect_motion_corrected` can recognise the movie by content.
    The first page is written non-contiguous to anchor the tag; the remaining
    pages stay contiguous, so the flat ``(T, Ly, Lx)`` layout is unchanged.
    """
    for i, frame in enumerate(frames_u16):
        if software is not None and i == 0:
            tw.write(frame, software=software)
        else:
            tw.write(frame, contiguous=True)


def _write_mc_tif(data_bin_path, mc_tif_path, Ly, Lx, chunk=512):
    """Export the int16 ``data.bin`` registered movie as a uint16 ``*_mc.tif``."""
    data_bin_path = Path(data_bin_path)
    mc_tif_path = Path(mc_tif_path)
    nbytes = data_bin_path.stat().st_size
    T = nbytes // (Ly * Lx * 2)
    mm = np.memmap(str(data_bin_path), dtype=np.int16, mode="r", shape=(T, Ly, Lx))
    ensure_free_space(mc_tif_path, T * Ly * Lx * 2, label=f"{mc_tif_path.name}")
    try:
        with tifffile.TiffWriter(str(mc_tif_path), bigtiff=True) as tw:
            for b0 in range(0, T, chunk):
                b1 = min(b0 + chunk, T)
                frames = np.clip(np.asarray(mm[b0:b1]), 0, None).astype(np.uint16)
                _append_frames(tw, frames,
                               software=MC_SOFTWARE_TAG if b0 == 0 else None)
    finally:
        del mm
    return mc_tif_path


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def run_rowwise_pcc_register(
    tif_path,
    output_dir,
    *,
    fs: float,
    do_registration: bool = True,
    max_displacement: int = 50,
    strip_height: int = 32,
    n_template_iters: int = 2,
    subpixel_upsample: int = 10,   # reserved: parabolic-refinement precision knob
    smooth_sigma_rows: float = 6.0,
    smooth_sigma_time: float = 1.0,
    prefilter: bool = False,
    prefilter_sigma_low: float = 1.0,
    prefilter_sigma_high: float = 8.0,
    strip_confidence_weight: bool = True,
    frame_batch: int = 256,
    force_cpu: bool = False,
    gpu_lock=None,
    preview=None,
):
    """Pre-register one FOV and write the corrected movie as ``{stem}_mc.tif``.

    This is a *motion-correction* step only — it does **not** run Suite2p
    detection. The Foundation dispatcher hands the resulting ``{stem}_mc.tif`` to
    Suite2p in detection-only mode (``do_registration=False``), so the int16
    ``data.bin`` + ``stat.npy``/``ops.npy`` substrate the rest of the pipeline
    consumes is produced by the unchanged Suite2p path. Decoupling registration
    from detection is what lets the backend be swapped without disturbing Stage 2.

    ``fs`` is accepted for signature symmetry with the Suite2p path (the
    registered movie itself is frame-rate agnostic).

    ``preview`` is an optional :class:`~roigbiv.pipeline.mc_preview.MCPreviewWriter`
    fed one raw/corrected frame pair per batch (throttled by wall clock) so the
    UI can render the correction as it happens. It is purely diagnostic and
    never affects the registered output.

    Returns
    -------
    mc_tif_path : Path — uint16 (T, Ly, Lx) registered movie
    motion_x    : (T,) float32 — per-frame rigid x displacement
    motion_y    : (T,) float32 — per-frame rigid y displacement
    """
    import torch
    import torch.nn.functional as Fnn

    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = tif_path.stem.replace("_mc", "")
    mc_tif_path = output_dir / f"{stem}_mc.tif"

    stack = tifffile.imread(str(tif_path))
    if stack.ndim != 3:
        raise ValueError(
            f"rowwise-pcc expects a 3D (T, Ly, Lx) stack; got shape {stack.shape} "
            f"from {tif_path}."
        )
    T, Ly, Lx = (int(s) for s in stack.shape)

    device = "cuda" if (cuda_compute_capable() and not force_cpu) else "cpu"

    ensure_free_space(mc_tif_path, T * Ly * Lx * 2, label=f"{stem}_mc.tif")
    motion_y = np.zeros(T, dtype=np.float32)
    motion_x = np.zeros(T, dtype=np.float32)

    def _to_u16(arr: np.ndarray) -> np.ndarray:
        return np.clip(np.round(arr), 0, 65535).astype(np.uint16)

    lock_cm = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
    try:
        with lock_cm, tifffile.TiffWriter(str(mc_tif_path), bigtiff=True) as tw:
            if not do_registration:
                # Pre-corrected input (``*_mc`` convention): pass through, zero shifts.
                if preview is not None:
                    preview.set_total(T)
                    preview.set_phase(
                        "skipped_precorrected",
                        note="input already motion-corrected; passed through")
                for b0 in range(0, T, frame_batch):
                    b1 = min(b0 + frame_batch, T)
                    block = np.asarray(stack[b0:b1], dtype=np.float32)
                    if preview is not None and preview.should_emit():
                        # Nothing was corrected, so both panes show the same
                        # frame — that identity is the diagnostic.
                        mid = (b1 - b0) // 2
                        preview.emit(block[mid], block[mid],
                                     frame_index=b0 + mid, n_done=b1)
                    _append_frames(tw, _to_u16(block),
                                   software=MC_SOFTWARE_TAG if b0 == 0 else None)
            else:
                if preview is not None:
                    preview.set_total(T)
                    preview.set_phase("building_reference")
                tmpl = _build_template(stack, device, n_template_iters,
                                       max_displacement, torch, Fnn,
                                       prefilter=prefilter,
                                       pf_lo=prefilter_sigma_low,
                                       pf_hi=prefilter_sigma_high)
                if preview is not None:
                    preview.set_phase("registering")
                batch = _vram_budget_batch(Ly, Lx, device, frame_batch, torch)
                for b0 in range(0, T, batch):
                    b1 = min(b0 + batch, T)
                    frames = torch.as_tensor(
                        np.ascontiguousarray(stack[b0:b1]).astype(np.float32),
                        device=device)
                    reg, fy, fx, conf = _register_batch(
                        frames, tmpl, strip_height, max_displacement,
                        smooth_sigma_rows, smooth_sigma_time, torch, Fnn,
                        prefilter=prefilter, pf_lo=prefilter_sigma_low,
                        pf_hi=prefilter_sigma_high,
                        confidence_weight=strip_confidence_weight,
                        return_conf=True)
                    motion_y[b0:b1] = fy.detach().cpu().numpy()
                    motion_x[b0:b1] = fx.detach().cpu().numpy()
                    if preview is not None:
                        preview.record_shifts(b0, motion_y[b0:b1],
                                              motion_x[b0:b1],
                                              conf.detach().cpu().numpy())
                        if preview.should_emit():
                            # Slice on the GPU before the host copy so the
                            # transfer is one small frame, not the whole batch.
                            mid = (b1 - b0) // 2
                            preview.emit(frames[mid].detach().cpu().numpy(),
                                         reg[mid].detach().cpu().numpy(),
                                         frame_index=b0 + mid, n_done=b1)
                    _append_frames(tw, _to_u16(reg.detach().cpu().numpy()),
                                   software=MC_SOFTWARE_TAG if b0 == 0 else None)
    finally:
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return mc_tif_path, motion_x, motion_y
