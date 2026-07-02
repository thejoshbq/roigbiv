"""Benchmark run orchestrator — runs the current ROIGBIV pipeline over every
entry in a validated manifest, writing per-FOV outputs + logs + a top-level
benchmark_run.json. Issue #28 (Milestone A / roadmap A4).

Deliberately does NOT call roigbiv.pipeline.workspace.run_with_workspace or
_process_one: this must never write to the FOV registry (registry.db) —
benchmark runs are throwaway harness executions, not production FOV identity
tracking. It calls roigbiv.pipeline.run.run_pipeline directly instead.

Out of scope (later roadmap items — do not implement here): metrics
computation against manual masks / synthetic ground truth (#30), external
baseline runners, report generation (#32).
"""
from __future__ import annotations

import contextlib
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FovRunResult:
    """Outcome of running the pipeline on one manifest entry."""
    fov_id: str
    dataset_id: str
    status: str                       # "success" | "error"
    duration_s: Optional[float] = None
    output_dir: Optional[str] = None
    log_path: Optional[str] = None
    config_used: Optional[dict] = None        # cfg.summary_for_log()
    config_fingerprint: Optional[str] = None  # compute_cfg_fingerprint(...)
    roi_counts: dict = field(default_factory=dict)  # {"accept":N,"flag":N,"reject":N}
    error: Optional[str] = None


@dataclass
class BenchmarkRunReport:
    """Full report for one `roigbiv-bench run` invocation — the in-memory
    twin of benchmark_run.json."""
    manifest_path: str
    output_dir: str
    started_at: str            # ISO 8601 UTC
    finished_at: str
    total_runtime_s: float
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    hardware: dict
    roigbiv_version: Optional[str]
    fov_results: list          # list[FovRunResult]

    def to_json_dict(self) -> dict:
        return asdict(self)


def _git_commit_hash(repo_dir: Path) -> Optional[str]:
    """Best-effort: current commit hash. Never raises. None on any failure
    (not a git repo, git not installed, detached weirdness, etc.)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def _git_dirty(repo_dir: Path) -> Optional[bool]:
    """Best-effort: True if the working tree has uncommitted changes. Never raises."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        return bool(result.stdout.strip())
    except Exception:
        return None


def _hardware_info() -> dict:
    """Best-effort hardware/runtime snapshot. Never raises — any field that
    can't be determined is omitted or set None, and the whole thing degrades
    to {"platform": ..., "cpu_count": ...} in the worst case."""
    info: dict = {
        "platform": None,
        "cpu_count": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_total_mem_gb": None,
    }
    try:
        info["platform"] = platform.platform()
    except Exception:
        pass
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                info["gpu_total_mem_gb"] = round(total_b / (1024 ** 3), 2)
            except Exception:
                pass
    except Exception:
        pass
    return info


def _roigbiv_version() -> Optional[str]:
    """Best-effort: installed roigbiv package version. Never raises."""
    try:
        from importlib.metadata import version
        return version("roigbiv")
    except Exception:
        return None


def _roi_counts(fov) -> dict:
    """Tally ROI gate outcomes. Reimplemented (not imported) — mirrors the
    private roigbiv.pipeline.workspace._roi_counts helper 1:1, kept local
    since it's an internal symbol of a different module."""
    if fov is None:
        return {}
    out = {"accept": 0, "flag": 0, "reject": 0}
    for r in fov.rois:
        out[r.gate_outcome] = out.get(r.gate_outcome, 0) + 1
    return out


def _resolve_entry_tif(entry, source_path: Path) -> Path:
    """Resolve a manifest entry's path to exactly one input TIF.

    Raises FileNotFoundError / ValueError with a message intended to be
    caught by the caller and recorded as a per-FOV error — never lets a
    discovery failure abort the whole benchmark run.
    """
    from roigbiv.io import discover_tifs

    resolved = (source_path / entry.path).resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"entry path does not exist: {resolved}")

    tifs = discover_tifs(resolved)
    if len(tifs) == 0:
        raise ValueError(f"no TIF files discovered under {resolved}")
    if len(tifs) > 1:
        raise ValueError(
            f"expected exactly one TIF under {resolved} for a single-FOV "
            f"benchmark entry, found {len(tifs)}: "
            f"{', '.join(t.name for t in tifs[:5])}"
            f"{'...' if len(tifs) > 5 else ''}"
        )
    return tifs[0]


def _run_one_fov(entry, source_path: Path, fov_output_dir: Path, log_path: Path) -> FovRunResult:
    """Run the pipeline on one manifest entry, capturing stdout/stderr to
    log_path. Returns a fully populated FovRunResult. Never raises —
    every failure mode (discovery, optics confirmation, abort, generic
    exception) is caught and recorded on the result.
    """
    result = FovRunResult(
        fov_id=entry.fov_id, dataset_id=entry.dataset_id, status="error",
        output_dir=str(fov_output_dir),
    )

    # Everything below is wrapped in a single top-level handler so a setup
    # failure (mkdir permissions, bad config, log file unwritable) is
    # recorded on this FOV's result and the benchmark run continues to the
    # next entry, rather than crashing the whole run. Failure modes with
    # more specific messages (discovery, optics confirmation, abort,
    # run_pipeline errors) are still handled by their own inner except
    # blocks below, which set duration_s themselves before returning.
    try:
        fov_output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            tif_path = _resolve_entry_tif(entry, source_path)
        except (FileNotFoundError, ValueError) as exc:
            result.error = f"discovery: {exc}"
            return result

        from roigbiv.pipeline.profiles import merged_overrides
        from roigbiv.pipeline.types import PipelineConfig
        from roigbiv.pipeline.resume import compute_cfg_fingerprint

        overrides = merged_overrides(
            entry.lens_type,
            base={
                "fs": entry.fs,
                "frame_averaging": entry.frame_averaging,
                "output_dir": fov_output_dir,
                "no_viewer": True,
            },
            explicit_dicts=[],
        )
        cfg = PipelineConfig(**overrides)
        result.config_used = cfg.summary_for_log()
        try:
            result.config_fingerprint = compute_cfg_fingerprint(cfg, tif_path)
        except Exception:
            pass  # best-effort; never block the run over a fingerprint failure

        from roigbiv.pipeline.run import (
            OpticsConfirmationRequired,
            PipelineAborted,
            run_pipeline,
        )

        t0 = time.perf_counter()
        with open(log_path, "w") as logf, \
             contextlib.redirect_stdout(logf), \
             contextlib.redirect_stderr(logf):
            # Set only once the file is actually open, so log_path never
            # points at a file that was never created (e.g. open() itself
            # raising PermissionError falls to the outer except below,
            # where log_path correctly stays None).
            result.log_path = str(log_path)
            try:
                fov = run_pipeline(tif_path, cfg)
            except OpticsConfirmationRequired as need:
                result.duration_s = time.perf_counter() - t0
                result.error = (
                    f"optics_confirmation_required: candidate="
                    f"{need.payload.get('candidate_profile')!r}"
                )
                return result
            except PipelineAborted:
                result.duration_s = time.perf_counter() - t0
                result.error = "aborted"
                return result
            except BaseException as exc:  # noqa: BLE001
                traceback.print_exc()  # goes into the redirected log file
                result.duration_s = time.perf_counter() - t0
                result.error = f"{type(exc).__name__}: {exc}"
                return result

        result.duration_s = time.perf_counter() - t0
        result.status = "success"
        result.error = None
        result.roi_counts = _roi_counts(fov)
        return result
    except BaseException as exc:  # noqa: BLE001 — setup-phase safety net
        result.error = f"setup: {type(exc).__name__}: {exc}"
        return result


def run_benchmark(manifest_path: Path, output_dir: Path) -> BenchmarkRunReport:
    """Load + validate the manifest, run the current ROIGBIV pipeline on
    every entry, and return a BenchmarkRunReport. Writes per-FOV output
    dirs and per-FOV log files under output_dir as a side effect (log
    files are written incrementally, per FOV, as each entry completes —
    NOT deferred to the end).

    Raises ManifestError / FileNotFoundError if the manifest itself cannot
    be loaded/parsed — the caller (CLI) is responsible for turning that
    into exit code 2 before any FOV has run. Per-FOV failures never raise
    out of this function; they are recorded on the corresponding
    FovRunResult instead.
    """
    from datetime import datetime, timezone
    from roigbiv.benchmark.schema import load_manifest, validate_manifest, ManifestError

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    raw = load_manifest(manifest_path)  # raises ManifestError/FileNotFoundError — caller's problem
    manifest, errors = validate_manifest(raw, base_dir=manifest_path.parent)
    if errors:
        raise ManifestError(
            f"manifest failed validation ({len(errors)} error(s)): "
            + "; ".join(str(e) for e in errors)
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"

    repo_dir = Path(__file__).resolve().parents[2]  # roigbiv/benchmark/runner.py -> repo root

    fov_results: list = []
    for entry in manifest.entries:
        fov_output_dir = output_dir / entry.fov_id
        log_path = logs_dir / f"{entry.fov_id}.log"
        t_fov = time.perf_counter()
        result = _run_one_fov(entry, manifest.source_path, fov_output_dir, log_path)
        elapsed = time.perf_counter() - t_fov
        if result.status == "success":
            counts = result.roi_counts
            print(
                f"[{entry.fov_id}] OK ({elapsed:.1f}s) — "
                f"accept={counts.get('accept', 0)} flag={counts.get('flag', 0)} "
                f"reject={counts.get('reject', 0)}",
                flush=True,
            )
        else:
            print(f"[{entry.fov_id}] ERROR ({elapsed:.1f}s) — {result.error}",
                  file=sys.stderr, flush=True)
        fov_results.append(result)

    finished = datetime.now(timezone.utc)
    report = BenchmarkRunReport(
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        total_runtime_s=time.perf_counter() - t0,
        git_commit=_git_commit_hash(repo_dir),
        git_dirty=_git_dirty(repo_dir),
        hardware=_hardware_info(),
        roigbiv_version=_roigbiv_version(),
        fov_results=fov_results,
    )
    return report
