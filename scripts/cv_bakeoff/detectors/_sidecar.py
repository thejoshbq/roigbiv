"""Shared sidecar-subprocess machinery for env-incompatible detectors.

CP4 / Cellpose-SAM (``cellpose>=4``) and StarDist (TensorFlow) cannot live in
the ``roigbiv`` env (the former violates the ``cellpose<4.0.0`` pin, the latter
fights the torch/CUDA stack). They run in quarantined conda envs invoked as
subprocesses — the same pattern as ``roigbiv/pipeline/legacy_mc.py`` +
``scripts/sima_mc_worker.py``.

Data hand-off: the driver writes the needed summary channels to a scratch
``inputs.npz``; the worker (which must NOT import roigbiv) writes
``{stem}_{method}_masks.tif`` (uint16 label mask) + ``{stem}_{method}_meta.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile

from cv_bakeoff.detector import DetectorResult


def _interp_cmd(env: str, override_var: str) -> list[str]:
    """Command prefix running ``python`` in the sidecar env.

    ``<override_var>`` (e.g. ``ROIGBIV_CPSAM_PYTHON``) overrides with an absolute
    interpreter path; otherwise ``conda run -n <env>``.
    """
    override = os.environ.get(override_var)
    if override:
        return [override]
    return ["conda", "run", "--no-capture-output", "-n", env, "python"]


def probe_env(env: str, override_var: str, import_stmt: str, build_hint: str) -> None:
    """Raise an actionable error if the sidecar can't import its package."""
    cmd = _interp_cmd(env, override_var)
    probe = [c for c in cmd if c != "--no-capture-output"] + ["-c", import_stmt]
    try:
        res = subprocess.run(probe, capture_output=True, text=True, timeout=180)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"sidecar: could not launch interpreter for env '{env}' ({exc}). "
            + build_hint
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"sidecar: probing env '{env}' timed out.") from exc
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"sidecar: '{import_stmt}' failed in env '{env}' — " + build_hint
            + ("\n  detail: " + " | ".join(tail) if tail else "")
        )


def run_sidecar(
    *,
    env: str,
    override_var: str,
    import_stmt: str,
    build_hint: str,
    worker_path: Path,
    method: str,
    channels: dict[str, np.ndarray],
    stem: str,
    extra_args: list[str],
) -> DetectorResult:
    """Run a sidecar worker and ingest its uint16 label mask."""
    if not worker_path.exists():
        raise RuntimeError(f"sidecar: worker not found at {worker_path}")
    probe_env(env, override_var, import_stmt, build_hint)

    with tempfile.TemporaryDirectory(prefix=f"cvbk_{method}_") as tmp:
        tmpdir = Path(tmp)
        np.savez(
            tmpdir / "inputs.npz",
            **{k: np.asarray(v, dtype=np.float32) for k, v in channels.items()},
        )
        cmd = _interp_cmd(env, override_var) + [
            str(worker_path),
            "--in", str(tmpdir / "inputs.npz"),
            "--out", str(tmpdir),
            "--stem", stem,
        ] + extra_args
        t0 = time.time()
        res = subprocess.run(cmd)
        elapsed = time.time() - t0
        if res.returncode != 0:
            raise RuntimeError(
                f"sidecar {method}: worker failed (exit {res.returncode}) for "
                f"{stem}. See [{method}_worker] log lines above."
            )

        mask_path = tmpdir / f"{stem}_{method}_masks.tif"
        if not mask_path.exists():
            raise RuntimeError(
                f"sidecar {method}: expected {mask_path.name} missing despite "
                f"a clean worker exit."
            )
        label_mask = tifffile.imread(str(mask_path)).astype(np.uint16)

        meta = {"method": method, "env": env, "runtime_s": round(elapsed, 2)}
        meta_path = tmpdir / f"{stem}_{method}_meta.json"
        if meta_path.exists():
            try:
                meta.update(json.loads(meta_path.read_text()))
            except (ValueError, OSError):
                pass
        meta["n_rois"] = int(label_mask.max())

    return DetectorResult(label_mask=label_mask, meta=meta)
