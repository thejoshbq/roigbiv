"""Legacy SIMA motion-correction backend (sidecar-subprocess driver).

This is the roigbiv-side (Python 3.10) half of the ``"legacy"`` motion-correction
backend. It runs the *genuine* legacy notebook algorithm — SIMA
``HiddenMarkov2D(granularity='row', max_displacement=[50,50])`` — by shelling out
to the quarantined ``sima-legacy`` conda env (Python 3.8), because SIMA 1.3.2
(2017) cannot be installed alongside torch/suite2p on Python 3.10.

The actual SIMA call lives in ``scripts/sima_mc_worker.py``, executed by the
sidecar interpreter (it must never import roigbiv). This module stages the call,
ingests ``{stem}_mc.tif`` + the per-frame displacement traces, and returns the
same ``(mc_tif_path, motion_x, motion_y)`` contract as
:func:`roigbiv.pipeline.registration.run_rowwise_pcc_register`, so the Foundation
dispatcher then runs Suite2p in detection-only mode on the corrected movie.

Build the sidecar env once with ``bash envs/build_sima_legacy.sh``.

Notes
-----
* SIMA is **CPU-only** and single-process (Viterbi over rows): expect tens of
  minutes to hours per FOV. ``gpu_lock`` is accepted for signature symmetry and
  ignored. Do not run two of these concurrently under ``batch.py`` — they would
  oversubscribe cores/RAM (the lock here is the wrong semantics for that).
* SIMA pads/crops to a common canvas, so the corrected ``{stem}_mc.tif`` dims
  differ from the input dims. That is faithful to the legacy output and Suite2p
  detection-only handles whatever dims it receives.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.io import MC_SOFTWARE_TAG
from roigbiv.pipeline.diskguard import ensure_free_space

_BUILD_HINT = (
    "create it with `bash envs/build_sima_legacy.sh` and verify with "
    "`conda run -n {env} python -c \"import sima.motion\"`. "
    "Or point ROIGBIV_SIMA_PYTHON at a Python interpreter that has SIMA 1.3.2."
)


def _worker_path() -> Path:
    # roigbiv/pipeline/legacy_mc.py -> repo root is parents[2].
    return Path(__file__).resolve().parents[2] / "scripts" / "sima_mc_worker.py"


def _interp_cmd(sima_env: str) -> list[str]:
    """Command prefix that runs ``python`` inside the SIMA env.

    ``ROIGBIV_SIMA_PYTHON`` overrides with an absolute interpreter path (skips
    conda). Otherwise use ``conda run -n <env>``.
    """
    override = os.environ.get("ROIGBIV_SIMA_PYTHON")
    if override:
        return [override]
    return ["conda", "run", "--no-capture-output", "-n", sima_env, "python"]


def _probe_env(sima_env: str) -> None:
    """Raise an actionable error if the sidecar interpreter can't import SIMA."""
    cmd = _interp_cmd(sima_env)
    # Drop --no-capture-output for the probe so we can read stderr on failure.
    probe_cmd = [c for c in cmd if c != "--no-capture-output"] + ["-c", "import sima"]
    try:
        res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"legacy SIMA backend: could not launch the sidecar interpreter "
            f"({exc}). The '{sima_env}' conda env is required — "
            + _BUILD_HINT.format(env=sima_env)
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"legacy SIMA backend: probing the '{sima_env}' env timed out."
        ) from exc
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"legacy SIMA backend: SIMA is not importable in the '{sima_env}' "
            f"conda env — " + _BUILD_HINT.format(env=sima_env)
            + ("\n  detail: " + " | ".join(tail) if tail else "")
        )


def _passthrough(tif_path: Path, mc_tif_path: Path, T: int, Ly: int, Lx: int):
    """Pre-corrected ``*_mc`` input: copy to ``{stem}_mc.tif`` with zero traces.

    Mirrors the ``do_registration=False`` branch of the rowwise-pcc backend so
    the contract is identical across backends and SIMA is never invoked on data
    that is already corrected.
    """
    ensure_free_space(mc_tif_path, T * Ly * Lx * 2, label=mc_tif_path.name)
    # imread handles both page-per-frame and volumetric layouts (matches the
    # rowwise-pcc passthrough in registration.py); the stack is already corrected.
    stack = tifffile.imread(str(tif_path))
    with tifffile.TiffWriter(str(mc_tif_path), bigtiff=True) as tw:
        for i, frame in enumerate(stack):
            page = np.clip(np.asarray(frame), 0, 65535).astype(np.uint16)
            # Stamp the Software tag on the first page only (anchors the series);
            # the rest stay contiguous for the flat (T, Ly, Lx) layout.
            if i == 0:
                tw.write(page, software=MC_SOFTWARE_TAG)
            else:
                tw.write(page, contiguous=True)
    zeros = np.zeros(T, dtype=np.float32)
    return mc_tif_path, zeros, zeros.copy()


def run_sima_legacy_register(
    tif_path,
    output_dir,
    *,
    fs: float,
    do_registration: bool = True,
    max_displacement: int = 50,
    granularity: str = "row",
    sima_env: str = "sima-legacy",
    gpu_lock=None,
):
    """Motion-correct one FOV with genuine SIMA HMM2D via the sidecar env.

    ``fs`` is accepted for signature symmetry with the other backends (the
    corrected movie is frame-rate agnostic). ``gpu_lock`` is accepted and
    ignored (SIMA is CPU-only).

    Returns
    -------
    mc_tif_path : Path — uint16 (T, Ly', Lx') corrected movie (SIMA canvas dims)
    motion_x    : (T,) float32 — per-frame x displacement trace (QC summary)
    motion_y    : (T,) float32 — per-frame y displacement trace (QC summary)
    """
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = tif_path.stem.replace("_mc", "")
    mc_tif_path = output_dir / f"{stem}_mc.tif"

    with tifffile.TiffFile(str(tif_path)) as tf:
        shape = tf.series[0].shape
    if len(shape) != 3:
        raise ValueError(
            f"legacy backend expects a 3D (T, Ly, Lx) stack; got shape {shape} "
            f"from {tif_path}."
        )
    T, Ly, Lx = (int(s) for s in shape)

    if not do_registration:
        return _passthrough(tif_path, mc_tif_path, T, Ly, Lx)

    # SIMA's cross-correlation bounds degenerate (fractional slice indices, a
    # cryptic TypeError deep in sima.misc.align) when the search window approaches
    # the frame size. Real 512²/1024² FOVs are far larger than any sane
    # max_displacement; guard the degenerate case with a clear message.
    if 2 * int(max_displacement) >= min(Ly, Lx):
        raise ValueError(
            f"legacy backend: max_displacement={max_displacement} is too large "
            f"for a {Ly}x{Lx} FOV (needs 2*max_displacement < min(Ly,Lx)). "
            f"Lower --mc-max-displacement."
        )

    # SIMA stages a temp HDF5 copy and writes a (padded) output TIFF in output_dir.
    # Reserve ~3× the raw stack: temp .h5 + the corrected movie + headroom.
    ensure_free_space(mc_tif_path, 3 * T * Ly * Lx * 2,
                      label=f"{stem}_mc.tif (SIMA scratch + output)")

    _probe_env(sima_env)

    worker = _worker_path()
    if not worker.exists():
        raise RuntimeError(f"legacy SIMA backend: worker not found at {worker}")

    cmd = _interp_cmd(sima_env) + [
        str(worker),
        "--input", str(tif_path.resolve()),
        "--outdir", str(output_dir.resolve()),
        "--stem", stem,
        "--max-displacement", str(int(max_displacement)),
        "--granularity", str(granularity),
    ]
    print(f"  [legacy/SIMA] correcting {stem} ({T} frames, {Ly}x{Lx}) — "
          f"CPU HMM2D, this is slow…", flush=True)
    # Stream the worker's progress live (no capture); it can run for hours.
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(
            f"legacy SIMA backend: worker failed (exit {res.returncode}) for "
            f"{stem}. See the [sima_mc_worker] log lines above for the cause."
        )

    disp_npz = output_dir / f"{stem}_mc_disp.npz"
    if not mc_tif_path.exists() or not disp_npz.exists():
        raise RuntimeError(
            f"legacy SIMA backend: expected outputs missing for {stem} "
            f"({mc_tif_path.name}, {disp_npz.name}) despite a clean worker exit."
        )
    with np.load(disp_npz) as d:
        motion_x = np.asarray(d["motion_x"], dtype=np.float32)
        motion_y = np.asarray(d["motion_y"], dtype=np.float32)
    if motion_x.shape[0] != T:
        # Non-fatal: trace length mismatch shouldn't sink the run, but flag it.
        print(f"  WARN: legacy trace length {motion_x.shape[0]} != T={T} for {stem}",
              file=sys.stderr, flush=True)
    return mc_tif_path, motion_x, motion_y
