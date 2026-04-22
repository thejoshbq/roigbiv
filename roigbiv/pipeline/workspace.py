"""Input-rooted workspace runner for the ROIGBIV pipeline.

Conventions
-----------
A *workspace* lives entirely under one user-chosen ``input_root`` directory:

    input_root/
        *.tif                      raw / motion-corrected stacks
        output/                    pipeline outputs (one subdir per FOV stem)
        registry.db                cross-session SQLite registry
        registry_blobs/            per-FOV fingerprint blobs
        registry_calibration.json  optional calibration model (auto-fallback)

Calling :func:`run_with_workspace` does, in order:

    1. Set the ``ROIGBIV_REGISTRY_*`` env vars so every downstream
       ``RegistryConfig.from_env()`` and ``build_store()`` resolves to the
       in-workspace SQLite + blob root (the registry config re-reads env on
       every call — see :mod:`roigbiv.registry.config`).
    2. ``store.ensure_schema()`` — runs ``alembic upgrade head`` idempotently
       (see :func:`roigbiv.registry.migrate.ensure_alembic_head`).
    3. For each TIF in the workspace, runs :func:`roigbiv.pipeline.run.run_pipeline`
       with ``output_dir = input_root/output/{stem}/`` and immediately
       registers the just-written session against the registry.
    4. Runs :func:`roigbiv.registry.backfill.run_backfill` once over
       ``input_root/output`` as an idempotent safety net (catches outputs from
       prior runs that never made it into the registry).

The user therefore never has to think about the output directory, run
``alembic upgrade head``, or call ``backfill`` themselves.
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from roigbiv.io import discover_tifs, validate_tif
from roigbiv.pipeline.types import FOVData, PipelineConfig

LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved on-disk locations for one workspace.

    ``input_root`` is the directory the user pointed at (the parent directory
    if they passed a single file). All other paths are derived from it.
    """

    input_root: Path
    tifs: tuple[Path, ...]
    output_root: Path
    db_path: Path
    blob_root: Path
    calibration_path: Path

    @property
    def db_dsn(self) -> str:
        return f"sqlite:///{self.db_path}"


def resolve_workspace(input_path: Path) -> WorkspacePaths:
    """Resolve a file or directory into a :class:`WorkspacePaths`.

    Discovers TIFs (using the same archive-aware scan as the rest of the
    pipeline), but excludes anything that lives under ``input_root/output``
    so that pipeline-produced TIFFs (masks, summaries) are never picked up
    as inputs on a re-run.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    if input_path.is_file():
        input_root = input_path.parent
        tif_list: list[Path] = [input_path]
    else:
        input_root = input_path
        all_tifs = discover_tifs(input_root)
        output_root = input_root / "output"
        tif_list = [
            t for t in all_tifs
            if output_root not in t.resolve().parents
        ]

    if not tif_list:
        raise FileNotFoundError(
            f"no TIF stacks discovered under {input_root} (excluding output/)"
        )

    return WorkspacePaths(
        input_root=input_root,
        tifs=tuple(tif_list),
        output_root=input_root / "output",
        db_path=input_root / "registry.db",
        blob_root=input_root / "registry_blobs",
        calibration_path=input_root / "registry_calibration.json",
    )


def configure_registry_env(workspace: WorkspacePaths) -> None:
    """Point the registry at this workspace's SQLite + blob root.

    Idempotent. Safe to call repeatedly. Sets only the variables this module
    owns; other ``ROIGBIV_ROICAT_*`` knobs are left to the user.
    """
    workspace.input_root.mkdir(parents=True, exist_ok=True)
    workspace.output_root.mkdir(parents=True, exist_ok=True)
    workspace.blob_root.mkdir(parents=True, exist_ok=True)

    os.environ["ROIGBIV_REGISTRY_DSN"] = workspace.db_dsn
    os.environ["ROIGBIV_BLOB_ROOT"] = str(workspace.blob_root)
    os.environ["ROIGBIV_CALIBRATION_PATH"] = str(workspace.calibration_path)


@dataclass
class FOVRunResult:
    """One FOV's outcome from :func:`run_with_workspace`."""

    tif: Path
    output_dir: Path
    duration_s: float = 0.0
    fov: Optional[FOVData] = None
    error: Optional[str] = None
    registry: Optional[dict] = None
    roi_counts: dict = field(default_factory=dict)


def run_with_workspace(
    workspace: WorkspacePaths,
    cfg_overrides: Optional[dict] = None,
    *,
    log_cb: Optional[LogCallback] = None,
    skip_registry: bool = False,
    skip_backfill: bool = False,
) -> list[FOVRunResult]:
    """Run the pipeline + registry over every TIF in ``workspace``.

    Sequential by design — :func:`roigbiv.pipeline.batch.run_batch` exists
    for parallel execution, but for the workspace flow we want predictable
    per-FOV log streaming and don't want to ship a multiprocessing GPU lock
    through the UI's background thread.

    Returns one :class:`FOVRunResult` per TIF, in the same order as
    ``workspace.tifs``. Failed FOVs have ``error`` populated; successful ones
    have ``fov`` and ``registry``.
    """
    configure_registry_env(workspace)
    log = log_cb or (lambda _msg: None)
    cfg_overrides = dict(cfg_overrides or {})

    log(f"Workspace: {workspace.input_root}")
    log(f"Output:    {workspace.output_root}")
    log(f"Registry:  {workspace.db_path}")
    log(f"Found {len(workspace.tifs)} TIF stack(s) to process.")

    _ensure_registry_schema(log)

    results: list[FOVRunResult] = []
    for idx, tif in enumerate(workspace.tifs, start=1):
        log(f"\n[FOV {idx}/{len(workspace.tifs)}] {tif.name}")
        results.append(_process_one(tif, workspace, cfg_overrides, log,
                                    skip_registry=skip_registry))

    if not skip_backfill:
        _safety_backfill(workspace, log)

    return results


# ── internals ──────────────────────────────────────────────────────────────


def _ensure_registry_schema(log: LogCallback) -> None:
    """Open the store once so its ``ensure_schema`` runs ``alembic upgrade head``."""
    from roigbiv.registry import build_store

    try:
        store = build_store()
        store.ensure_schema()
        log("Registry schema verified (alembic head).")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: registry schema check failed — {type(exc).__name__}: {exc}")


def _process_one(
    tif: Path,
    workspace: WorkspacePaths,
    cfg_overrides: dict,
    log: LogCallback,
    *,
    skip_registry: bool,
) -> FOVRunResult:
    from roigbiv.pipeline.run import run_pipeline

    stem = tif.stem.replace("_mc", "")
    out_dir = workspace.output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        validate_tif(tif)
    except ValueError as exc:
        log(f"  invalid TIF: {exc}")
        return FOVRunResult(tif=tif, output_dir=out_dir,
                            error=f"invalid_tif: {exc}")

    cfg = _build_config(out_dir, cfg_overrides)
    t0 = time.perf_counter()
    try:
        fov = run_pipeline(tif, cfg)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        return FOVRunResult(
            tif=tif,
            output_dir=out_dir,
            duration_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )
    duration = time.perf_counter() - t0
    counts = _roi_counts(fov)
    log(f"  pipeline OK ({duration:.1f}s) — "
        f"accept={counts.get('accept', 0)} flag={counts.get('flag', 0)} "
        f"reject={counts.get('reject', 0)}")

    registry: Optional[dict] = None
    if not skip_registry:
        try:
            registry = _register_session(stem, fov, log)
        except Exception as exc:  # noqa: BLE001
            log(f"  WARNING: registry call failed — "
                f"{type(exc).__name__}: {exc}")

    return FOVRunResult(
        tif=tif, output_dir=out_dir,
        duration_s=duration, fov=fov,
        registry=registry, roi_counts=counts,
    )


def _build_config(output_dir: Path, overrides: dict) -> PipelineConfig:
    """Build a PipelineConfig with user overrides applied on top of defaults."""
    base = {"output_dir": output_dir, "no_viewer": True}
    base.update(overrides)
    base["output_dir"] = output_dir   # always force per-FOV path
    return PipelineConfig(**base)


def _roi_counts(fov: Optional[FOVData]) -> dict:
    if fov is None:
        return {}
    out = {"accept": 0, "flag": 0, "reject": 0}
    for r in fov.rois:
        out[r.gate_outcome] = out.get(r.gate_outcome, 0) + 1
    return out


def _build_merged_masks(fov: FOVData) -> Optional[np.ndarray]:
    if fov.mean_M is None or not fov.rois:
        return None
    Ly, Lx = fov.mean_M.shape
    label_image = np.zeros((Ly, Lx), dtype=np.uint16)
    for roi in fov.rois:
        if getattr(roi, "gate_outcome", None) == "reject":
            continue
        if roi.mask is None or not roi.mask.any():
            continue
        label_image[roi.mask] = int(roi.label_id)
    return label_image


def _register_session(stem: str, fov: FOVData, log: LogCallback) -> Optional[dict]:
    """Mirror of ``roigbiv.pipeline.run._register_fov_after_pipeline``.

    Re-implemented here (rather than calling the underscore-prefixed helper)
    so the workspace runner does not depend on a private symbol of run.py.
    """
    from roigbiv.registry import (
        RegistryConfig,
        build_adapter_config,
        build_blob_store,
        build_store,
        load_calibration,
        register_or_match,
    )
    from roigbiv.registry.roicat_adapter import SessionInput

    if fov.mean_M is None:
        log("  registry: skipped (fov.mean_M is None)")
        return None

    merged_masks = _build_merged_masks(fov)
    if merged_masks is None or not (merged_masks > 0).any():
        log("  registry: skipped (no non-rejected ROIs)")
        return None

    cfg = RegistryConfig.from_env()
    store = build_store(cfg)
    blob_store = build_blob_store(cfg)
    adapter_cfg = build_adapter_config(cfg)
    calibration = load_calibration(cfg)

    query = SessionInput(
        session_key=stem,
        mean_m=np.asarray(fov.mean_M, dtype=np.float32),
        merged_masks=np.asarray(merged_masks, dtype=np.uint16),
    )

    report = register_or_match(
        fov_stem=stem,
        query=query,
        output_dir=fov.output_dir,
        store=store,
        blob_store=blob_store,
        adapter_config=adapter_cfg,
        calibration=calibration,
        accept_threshold=cfg.fov_accept_threshold,
        review_threshold=cfg.fov_review_threshold,
    )
    decision = report.get("decision", "unknown")
    posterior = report.get("fov_posterior") or report.get("fov_sim")
    log(_format_registry_decision(decision, report, posterior))
    return report


def _format_registry_decision(decision: str, report: dict,
                              posterior: Optional[float]) -> str:
    if decision == "new_fov":
        return (f"  registry: new_fov fov_id={report.get('fov_id')} "
                f"({report.get('n_new_cells', 0)} cells)")
    if decision in ("auto_match", "hash_match"):
        post = f"{posterior:.3f}" if posterior is not None else "n/a"
        return (f"  registry: {decision} fov_id={report.get('fov_id')} "
                f"posterior={post} matched={report.get('n_matched', 0)} "
                f"new={report.get('n_new', 0)} missing={report.get('n_missing', 0)}")
    if decision == "review":
        post = f"{posterior:.3f}" if posterior is not None else "n/a"
        return (f"  registry: review band (posterior={post}) — "
                "resolve in the UI's Registry tab.")
    return f"  registry: {decision} ({report})"


def _safety_backfill(workspace: WorkspacePaths, log: LogCallback) -> None:
    """Idempotent sweep: register any FOV outputs not yet linked to the DB."""
    from roigbiv.registry.backfill import run_backfill
    from roigbiv.registry.config import RegistryConfig

    if not workspace.output_root.exists():
        return
    log("\nBackfill sweep over output/ (idempotent safety net)")
    try:
        cfg = RegistryConfig.from_env()
        reports = run_backfill(workspace.output_root, cfg=cfg)
    except Exception as exc:  # noqa: BLE001
        log(f"  WARNING: backfill failed — {type(exc).__name__}: {exc}")
        return
    if not reports:
        log("  backfill: nothing to do.")
        return
    decisions: dict[str, int] = {}
    errors = 0
    for r in reports:
        if "error" in r:
            errors += 1
            continue
        d = r.get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(decisions.items()))
    if errors:
        summary += f", errors={errors}"
    log(f"  backfill: {summary}")
