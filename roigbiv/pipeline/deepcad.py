"""DeepCAD-RT denoising backend (sidecar-subprocess driver).

Runs DeepCAD-RT out-of-process, shelling out to a dedicated ``deepcad`` conda
env, so ROIGBIV never depends on DeepCAD-RT's Python environment (CUDA/torch
version pins that would otherwise collide with this interpreter's own torch).

The actual DeepCAD-RT call lives in ``scripts/deepcad_sidecar.py``, executed by
the sidecar interpreter (it must never import roigbiv). This module stages the
call via a JSON manifest, validates the denoised output, and returns its path.

Trust boundary
--------------
``cfg.deepcad_python``, ``cfg.deepcad_script``, ``cfg.deepcad_env``, and
``cfg.deepcad_model`` come from ``PipelineConfig`` (repo-authored YAML / CLI
flags), not untrusted external input. Subprocess args are always list-form
(never ``shell=True``, never string interpolation) so shell metacharacters in
paths cannot be interpreted — but the interpreter/script/model path themselves
are still "run whatever is named there"; that is inherent to a config-driven
sidecar and is why these fields are pipeline-config, not per-request input.

Scope
-----
This module only produces ``{stem}_deepcad.tif`` + provenance and records the
path on ``FOVData.denoised_path``. It does not decide which stages consume the
denoised branch vs. the raw branch — that routing is a separate concern
(BranchView).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.diskguard import ensure_free_space

_BUILD_HINT = (
    "install DeepCAD-RT in the '{env}' conda env and verify with "
    "`conda run -n {env} python -c \"import deepcad\"`. "
    "Or point ROIGBIV_DEEPCAD_PYTHON at a Python interpreter that has DeepCAD-RT."
)

# conda env names are passed as a `-n <env>` argv to `conda run`; guard against
# a value starting with `-` being parsed as a flag (argument-injection, not
# neutralized by list-form subprocess.run). The leading character is
# restricted to alnum specifically so a value like "-n" or "--help" cannot
# match (a bare `[...-]+` class would accept a leading '-' since '-' is a
# valid class member, not just a range operator, at the end of the bracket).
_ENV_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Generous ceiling for a single-FOV denoise pass. DeepCAD-RT is GPU-bound but
# a hung/deadlocked process must not hold gpu_lock forever (it would stall
# every other FOV in a batch run indefinitely).
_PROBE_TIMEOUT_S = 120
_RUN_TIMEOUT_S = 3600


def _validate_env_name(env: str) -> None:
    if not _ENV_NAME_RE.match(env):
        raise RuntimeError(
            f"deepcad sidecar: invalid conda env name {env!r} — must match "
            f"{_ENV_NAME_RE.pattern} (no leading '-', no shell metacharacters)."
        )


def _worker_path(cfg) -> Path:
    script = getattr(cfg, "deepcad_script", "") or ""
    if script:
        return Path(script)
    # roigbiv/pipeline/deepcad.py -> repo root is parents[2].
    return Path(__file__).resolve().parents[2] / "scripts" / "deepcad_sidecar.py"


def _interp_cmd(cfg) -> list[str]:
    """Command prefix that runs ``python`` inside the DeepCAD env.

    ``ROIGBIV_DEEPCAD_PYTHON`` (or ``cfg.deepcad_python``) overrides with an
    absolute interpreter path (skips conda). Otherwise use ``conda run -n <env>``.
    """
    override = os.environ.get("ROIGBIV_DEEPCAD_PYTHON") or getattr(cfg, "deepcad_python", "") or ""
    if override:
        return [override]
    env = getattr(cfg, "deepcad_env", "deepcad") or "deepcad"
    _validate_env_name(env)
    return ["conda", "run", "--no-capture-output", "-n", env, "python"]


def _probe_env(cfg) -> None:
    """Raise an actionable error if the sidecar interpreter can't import DeepCAD-RT.

    Skipped only when BOTH ``cfg.deepcad_python`` and ``cfg.deepcad_script`` are
    explicitly set (the override path used by tests/dev with a fake stub) — a
    single real override still gets the pre-flight check, so partial overrides
    in production don't silently lose the actionable setup error.
    """
    if getattr(cfg, "deepcad_python", "") and getattr(cfg, "deepcad_script", ""):
        return
    env = getattr(cfg, "deepcad_env", "deepcad") or "deepcad"
    cmd = _interp_cmd(cfg)
    # Drop --no-capture-output for the probe so we can read stderr on failure.
    probe_cmd = [c for c in cmd if c != "--no-capture-output"] + ["-c", "import deepcad"]
    try:
        res = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"deepcad sidecar: could not launch the sidecar interpreter ({exc}). "
            f"The '{env}' conda env is required — " + _BUILD_HINT.format(env=env)
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"deepcad sidecar: probing the '{env}' env timed out after "
            f"{_PROBE_TIMEOUT_S}s."
        ) from exc
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"deepcad sidecar: DeepCAD-RT is not importable in the '{env}' conda "
            f"env — " + _BUILD_HINT.format(env=env)
            + ("\n  detail: " + " | ".join(tail) if tail else "")
        )


def _read_shape_dtype(tif_path: Path):
    with tifffile.TiffFile(str(tif_path)) as tf:
        series = tf.series[0]
        return tuple(int(s) for s in series.shape), series.dtype


def _run_with_timeout(cmd: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    """Like ``subprocess.run(cmd, timeout=...)`` but kills the whole process
    group on timeout, not just the immediate child.

    ``conda run`` execs (or forks) the real interpreter under itself; a plain
    ``subprocess.run(..., timeout=...)`` only terminates the direct child on
    ``TimeoutExpired`` and is not guaranteed to reach a grandchild process,
    which would leave a hung GPU-bound DeepCAD-RT process alive after
    ``gpu_lock`` has already been released — defeating the point of the
    timeout under concurrent batch runs.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # own process group, so we can kill descendants too
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()  # reap the now-dead process group leader
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def run_deepcad_denoise(
    tif_path,
    output_dir,
    cfg,
    *,
    gpu_lock=None,
) -> Path:
    """Denoise one FOV with DeepCAD-RT via the out-of-process sidecar.

    Parameters
    ----------
    tif_path   : input TIFF stack (any dtype, shape (T, Ly, Lx)); the RAW movie —
                 this backend never operates on motion-corrected/residual data.
                 The worker must return output in the same dtype (validated).
    output_dir : directory to write ``{stem}_deepcad.tif`` + provenance JSON.
    cfg        : PipelineConfig with the deepcad_* fields.
    gpu_lock   : multiprocessing.Manager().Lock() from the batch runner, or
                 None for a single-FOV run (no-op).

    Returns
    -------
    Path to the denoised output TIFF.

    Raises
    ------
    RuntimeError
        On subprocess failure, output validation failure, or disk-space checks.
        The message includes actionable setup instructions — this IS the
        "fails gracefully with setup instructions" behavior; callers should let
        it propagate rather than swallow it.
    """
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # .stem only has one suffix stripped by pathlib; strip a trailing "_mc"
    # marker specifically (not a substring replace, which would also mangle a
    # stem containing "_mc" anywhere else, e.g. "fov_mc_test_mc").
    raw_stem = tif_path.stem
    stem = raw_stem[: -len("_mc")] if raw_stem.endswith("_mc") else raw_stem
    output_path = output_dir / f"{stem}_deepcad.tif"

    _probe_env(cfg)

    worker = _worker_path(cfg)
    if not worker.exists():
        raise RuntimeError(f"deepcad sidecar: worker not found at {worker}")

    input_shape, input_dtype = _read_shape_dtype(tif_path)
    if len(input_shape) != 3:
        raise RuntimeError(
            f"deepcad sidecar: expected a 3D (T, Ly, Lx) input stack; got shape "
            f"{input_shape} from {tif_path}."
        )
    T, Ly, Lx = input_shape

    # A stale output from a prior crashed/partial run must never be mistaken
    # for this run's result. ensure_free_space below pre-creates/fallocates
    # output_path regardless of whether the subprocess ever writes to it, so
    # freshness has to be guaranteed here, before that call, not inferred from
    # "does the file exist" after the subprocess exits.
    output_path.unlink(missing_ok=True)
    ensure_free_space(
        output_path,
        T * Ly * Lx * np.dtype(input_dtype).itemsize,
        label=f"{stem}_deepcad.tif (denoised output)",
    )

    manifest = {
        "input": str(tif_path.resolve()),
        "output": str(output_path.resolve()),
        "model": getattr(cfg, "deepcad_model", "") or "",
        "gpu": True,
    }
    # Manifest lives in output_dir (already unique per FOV) but uses a unique
    # suffix so concurrent/retried runs on the same stem never clobber each
    # other's in-flight manifest.
    manifest_fd, manifest_name = tempfile.mkstemp(
        suffix=".json", prefix=f"{stem}_deepcad_manifest_", dir=str(output_dir)
    )
    manifest_path = Path(manifest_name)
    try:
        with os.fdopen(manifest_fd, "w") as mf:
            json.dump(manifest, mf)

        cmd = _interp_cmd(cfg) + [str(worker), str(manifest_path)]

        lock_cm = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        env = getattr(cfg, "deepcad_env", "deepcad") or "deepcad"
        try:
            with lock_cm:
                result = _run_with_timeout(cmd, _RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"deepcad sidecar: worker timed out after {_RUN_TIMEOUT_S}s for "
                f"{stem} — a hung DeepCAD process (and its process group) was "
                f"killed rather than holding the GPU lock indefinitely."
            ) from exc
        except FileNotFoundError as exc:
            # Most likely a typo'd cfg.deepcad_python override (the _probe_env
            # pre-flight is skipped on that exact path when deepcad_script is
            # also set) — surface the same actionable hint rather than a bare
            # traceback.
            raise RuntimeError(
                f"deepcad sidecar: could not launch the sidecar interpreter "
                f"({exc}) — " + _BUILD_HINT.format(env=env)
            ) from exc
    finally:
        try:
            manifest_path.unlink()
        except OSError:
            pass

    if result.returncode != 0:
        error_tail = (result.stderr or "").strip()[-500:]
        # A failed run must not leave a stale/partial output behind for a
        # later call to mistake as valid.
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"deepcad sidecar: {error_tail} — " + _BUILD_HINT.format(env=env)
        )

    _validate_output(output_path, input_shape, tif_path)

    metadata = {
        "input_path": str(tif_path.resolve()),
        "output_path": str(output_path.resolve()),
        "exit_code": result.returncode,
        "cmd_line": cmd,
    }
    metadata_path = output_dir / f"{stem}_deepcad_metadata.json"
    with open(metadata_path, "w") as mf:
        json.dump(metadata, mf, indent=2)

    return output_path


def _validate_output(output_path: Path, input_shape: tuple[int, ...], tif_path: Path) -> None:
    """Validate the denoised TIFF before it is trusted as pipeline input.

    Always re-validates, even if ``output_path`` already existed before this
    call (a stale file from a prior crashed run must never be assumed valid).
    On any failure the invalid file is removed so it cannot be mistaken for a
    successful run later.
    """
    if not output_path.exists():
        raise RuntimeError(
            f"deepcad sidecar: output validation failed — {output_path.name} "
            f"was not created despite a clean worker exit."
        )
    try:
        with tifffile.TiffFile(str(output_path)) as tf:
            series = tf.series[0]
            output_shape = tuple(int(s) for s in series.shape)
            dtype = series.dtype
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"deepcad sidecar: output validation failed — could not read "
            f"{output_path.name}: {exc}"
        ) from exc

    if len(output_shape) != 3:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"deepcad sidecar: output validation failed — output has "
            f"{len(output_shape)} dims, expected 3 (T, Ly, Lx)."
        )
    if output_shape != input_shape:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"deepcad sidecar: output validation failed — shape mismatch: "
            f"input {input_shape}, output {output_shape}."
        )

    with tifffile.TiffFile(str(tif_path)) as tf:
        input_dtype = tf.series[0].dtype
    if dtype != input_dtype:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"deepcad sidecar: output validation failed — dtype mismatch: "
            f"input {input_dtype}, output {dtype}. DeepCAD-RT commonly emits "
            f"float32; the worker must cast back to the input dtype before "
            f"writing, or this driver's contract needs an explicit rescale step."
        )

    # Sample-based finiteness check (avoid materializing huge stacks fully in
    # RAM twice): read a bounded number of frames and scan for NaN/Inf.
    with tifffile.TiffFile(str(output_path)) as tf:
        n_frames = len(tf.pages)
        sample_idx = range(0, n_frames, max(1, n_frames // 32))
        for i in sample_idx:
            frame = np.asarray(tf.pages[i].asarray())
            if not np.isfinite(frame).all():
                output_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"deepcad sidecar: output validation failed — non-finite "
                    f"(NaN/Inf) values detected in frame {i} of {output_path.name}."
                )
