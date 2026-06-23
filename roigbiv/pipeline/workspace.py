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
from typing import Callable, Optional, Sequence

import numpy as np

from roigbiv.io import discover_tifs, validate_tif
from roigbiv.pipeline import fmt
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
        tif_list = _dedup_mc_pairs(tif_list)

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


def _dedup_mc_pairs(tifs: list[Path]) -> list[Path]:
    """Collapse ``foo.tif`` + ``foo_mc.tif`` pairs to a single entry.

    The pipeline maps both variants to the same output stem
    (``_process_one`` strips ``_mc`` — see stem derivation below), so
    processing both doubles the registry session rows for the same FOV.
    When both exist we keep the ``_mc`` file because the pipeline is
    designed around motion-corrected input (see CLAUDE.md §Conventions).
    """
    by_stem: dict[str, Path] = {}
    for t in tifs:
        stem = t.stem.replace("_mc", "")
        incumbent = by_stem.get(stem)
        if incumbent is None:
            by_stem[stem] = t
            continue
        # Prefer the _mc variant when both raw and motion-corrected exist.
        incumbent_is_mc = incumbent.stem.endswith("_mc")
        candidate_is_mc = t.stem.endswith("_mc")
        if candidate_is_mc and not incumbent_is_mc:
            by_stem[stem] = t
    return sorted(by_stem.values())


def configure_registry_env(workspace: WorkspacePaths) -> None:
    """Point the registry at this workspace's SQLite + blob root.

    Idempotent. Safe to call repeatedly. Sets only the variables this module
    owns; other ``ROIGBIV_ROICAT_*`` knobs are left to the user.

    CLI use only. The Dash UI path uses :func:`_registry_config_from_workspace`
    instead so it never mutates the process environment.
    """
    workspace.input_root.mkdir(parents=True, exist_ok=True)
    workspace.output_root.mkdir(parents=True, exist_ok=True)
    workspace.blob_root.mkdir(parents=True, exist_ok=True)

    os.environ["ROIGBIV_REGISTRY_DSN"] = workspace.db_dsn
    os.environ["ROIGBIV_BLOB_ROOT"] = str(workspace.blob_root)
    os.environ["ROIGBIV_CALIBRATION_PATH"] = str(workspace.calibration_path)


def _registry_config_from_workspace(workspace: WorkspacePaths) -> "RegistryConfig":
    """Build a RegistryConfig from workspace paths without touching os.environ.

    Creates workspace directories as a side effect (same as
    :func:`configure_registry_env`). Use this from the UI path so concurrent
    sessions never overwrite each other's registry env vars.
    """
    from roigbiv.registry.config import RegistryConfig

    workspace.input_root.mkdir(parents=True, exist_ok=True)
    workspace.output_root.mkdir(parents=True, exist_ok=True)
    workspace.blob_root.mkdir(parents=True, exist_ok=True)
    return RegistryConfig(
        dsn=workspace.db_dsn,
        blob_backend="local",
        blob_root=workspace.blob_root,
        endpoint=None,
        api_key=None,
        calibration_path=workspace.calibration_path,
    )


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
    png_path: Optional[Path] = None
    # Set when the run paused after foundation for optics confirmation (the
    # foundation outputs are on disk; a --resume run with a confirmed profile
    # continues from Stage 1). Carries the confirmation payload for the UI.
    awaiting_confirmation: Optional[dict] = None


def run_with_workspace(
    workspace: WorkspacePaths,
    cfg_overrides: Optional[dict] = None,
    *,
    log_cb: Optional[LogCallback] = None,
    skip_registry: bool = False,
    skip_backfill: bool = False,
    resume: bool = False,
    n_workers: int = 1,
    registry_config: Optional["RegistryConfig"] = None,
    selected_tifs: Optional[Sequence[Path]] = None,
    abort_event: Optional["threading.Event"] = None,
) -> list[FOVRunResult]:
    """Run the pipeline + registry over every TIF in ``workspace``.

    Sequential by default — pass ``n_workers > 1`` to fan out the heavy
    pipeline calls through :func:`roigbiv.pipeline.batch.run_batch` (capped
    at 2 workers per the GPU constraint). The light post-pipeline steps
    (registry registration, traces bundle write, backfill) always run in
    the parent process so SQLite writes stay serialized.

    When ``registry_config`` is supplied (UI path), it is used directly and
    ``configure_registry_env`` is **not** called — the process environment is
    never mutated, enabling safe concurrent sessions. When omitted (CLI path),
    ``configure_registry_env`` sets the env vars as before.

    When ``resume=True``, each FOV's run consults its existing output
    directory and skips stages whose artifacts are already present. Refuses
    a per-FOV resume if the config or input has changed since those
    artifacts were written. See ``roigbiv.pipeline.resume`` for details.

    When ``selected_tifs`` is supplied (UI subset selection), only those TIFs
    are run; the frozen ``workspace`` is never mutated. ``None`` (the default,
    and the CLI path) runs every TIF.

    Returns one :class:`FOVRunResult` per *run* TIF, in the same order as the
    selected subset of ``workspace.tifs``. Failed FOVs have ``error`` populated;
    successful ones have ``fov`` and ``registry``.
    """
    if registry_config is not None:
        cfg = registry_config
    else:
        configure_registry_env(workspace)
        cfg = _registry_config_from_workspace(workspace)
    log = log_cb or (lambda _msg: None)
    cfg_overrides = dict(cfg_overrides or {})
    if resume:
        cfg_overrides.setdefault("resume", True)
    # ``override`` is a registry directive, not a PipelineConfig field — pop it
    # here so it never reaches ``_build_config``/``PipelineConfig``.
    override = bool(cfg_overrides.pop("override", False))

    # Honor a UI-selected subset without mutating the frozen workspace. Resolve
    # both sides so a non-resolved selection still matches workspace.tifs.
    if selected_tifs is None:
        tifs_to_run = workspace.tifs
    else:
        selected_set = {Path(t).resolve() for t in selected_tifs}
        tifs_to_run = tuple(t for t in workspace.tifs
                            if Path(t).resolve() in selected_set)

    log(f"Workspace: {workspace.input_root}")
    log(f"Output:    {workspace.output_root}")
    log(f"Registry:  {workspace.db_path}")
    if resume:
        log("Resume:    enabled (skipping completed stages per FOV)")
    log(f"Found {len(tifs_to_run)} TIF stack(s) to process.")

    _ensure_registry_schema(log, cfg)

    parallel = n_workers > 1 and len(tifs_to_run) > 1
    if parallel:
        log(f"Parallel:  n_workers={n_workers} (capped at 2)")
        results = _run_parallel(workspace, tifs_to_run, cfg_overrides, log,
                                skip_registry=skip_registry,
                                n_workers=n_workers,
                                registry_cfg=cfg,
                                override=override,
                                abort_event=abort_event)
    else:
        results = []
        for idx, tif in enumerate(tifs_to_run, start=1):
            # Cooperative stop between FOVs: honor a stop requested while a
            # prior FOV was running (an in-FOV stop is handled by run_pipeline).
            if abort_event is not None and abort_event.is_set():
                log(f"Stop requested — halting before FOV {idx}/"
                    f"{len(tifs_to_run)} ({len(tifs_to_run) - idx + 1} not run).")
                break
            if idx > 1:
                log(fmt.fov_separator())
            log(fmt.fov_banner(tif.name, idx, len(tifs_to_run)))
            results.append(_process_one(tif, workspace, cfg_overrides, log,
                                        skip_registry=skip_registry,
                                        registry_cfg=cfg,
                                        override=override,
                                        abort_event=abort_event))

    if not skip_backfill:
        _safety_backfill(workspace, log, cfg)

    return results


def _run_parallel(
    workspace: WorkspacePaths,
    tifs: tuple[Path, ...],
    cfg_overrides: dict,
    log: LogCallback,
    *,
    skip_registry: bool,
    n_workers: int,
    registry_cfg: "RegistryConfig",
    override: bool = False,
    abort_event: Optional["threading.Event"] = None,
) -> list[FOVRunResult]:
    """Fan pipeline calls out to ``pipeline/batch.run_batch``; do the light
    post-pipeline work (registry + traces bundle) sequentially in the parent.

    The in-process ``abort_event`` cannot reach the separate batch worker
    processes, so it only short-circuits *before* the pool launches; a full
    mid-batch stop (future cancel + pool terminate) is a CLI-only follow-up.
    """
    from roigbiv.pipeline.batch import run_batch

    # Honor a stop requested before the pool launches — the threading.Event
    # does not propagate into the worker processes once they're running.
    if abort_event is not None and abort_event.is_set():
        log("Stop requested — halting before the parallel batch launches.")
        return [FOVRunResult(tif=t, output_dir=workspace.output_root /
                             t.stem.replace("_mc", ""), error="aborted")
                for t in tifs]

    jobs: list[tuple[Path, PipelineConfig]] = []
    out_dirs: list[Path] = []
    valid_indices: list[int] = []
    results: list[Optional[FOVRunResult]] = [None] * len(tifs)

    for idx, tif in enumerate(tifs):
        stem = tif.stem.replace("_mc", "")
        out_dir = workspace.output_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(out_dir)

        try:
            validate_tif(tif)
        except ValueError as exc:
            log(f"  invalid TIF ({tif.name}): {exc}")
            results[idx] = FOVRunResult(tif=tif, output_dir=out_dir,
                                        error=f"invalid_tif: {exc}")
            continue

        cfg = _build_config(out_dir, cfg_overrides)
        jobs.append((tif, cfg))
        valid_indices.append(idx)

    if not jobs:
        return [r for r in results if r is not None]

    start_times: dict[int, float] = {}
    fovs: dict[int, Optional[FOVData]] = {}
    errors: dict[int, Optional[str]] = {}
    durations: dict[int, float] = {}

    for slot, idx in enumerate(valid_indices):
        start_times[slot] = time.perf_counter()

    def _log_cb(slot: int, line: str) -> None:
        idx = valid_indices[slot]
        log(f"[FOV {idx + 1}/{len(tifs)}] {line}")

    def _on_complete(slot: int, fov: Optional[FOVData],
                     exc: Optional[BaseException]) -> None:
        durations[slot] = time.perf_counter() - start_times.get(
            slot, time.perf_counter())
        if exc is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            fovs[slot] = None
            errors[slot] = f"{type(exc).__name__}: {exc}"
        else:
            fovs[slot] = fov
            errors[slot] = None

    log(f"\n=== Running {len(jobs)} FOV(s) in parallel ===")
    run_batch(
        jobs=jobs,
        n_workers=n_workers,
        log_callback=_log_cb,
        on_complete=_on_complete,
    )

    for slot, idx in enumerate(valid_indices):
        tif, cfg = jobs[slot]
        out_dir = out_dirs[idx]
        duration = durations.get(slot, 0.0)
        err = errors.get(slot)
        fov = fovs.get(slot)
        if err is not None or fov is None:
            results[idx] = FOVRunResult(tif=tif, output_dir=out_dir,
                                        duration_s=duration, error=err)
            continue

        counts = _roi_counts(fov)
        log(f"[FOV {idx + 1}/{len(tifs)}] "
            f"pipeline OK ({duration:.1f}s) — "
            f"accept={counts.get('accept', 0)} flag={counts.get('flag', 0)} "
            f"reject={counts.get('reject', 0)}")

        stem = tif.stem.replace("_mc", "")
        registry: Optional[dict] = None
        if (skip_registry or getattr(cfg, "scout_mode", False)
                or getattr(cfg, "foundation_only", False)):
            if getattr(cfg, "scout_mode", False):
                log("  registry: skipped (scout run — triage only)")
            elif getattr(cfg, "foundation_only", False):
                log("  registry: skipped (foundation-only dry run)")
        else:
            try:
                registry = _register_session(stem, fov, log, registry_cfg,
                                             override=override)
            except Exception as exc:  # noqa: BLE001
                log(f"  WARNING: registry call failed — "
                    f"{type(exc).__name__}: {exc}")

        try:
            _write_traces_bundle(fov, cfg, workspace, registry, log)
        except Exception as exc:  # noqa: BLE001
            log(f"  WARNING: traces bundle write failed — "
                f"{type(exc).__name__}: {exc}")

        results[idx] = FOVRunResult(
            tif=tif, output_dir=out_dir,
            duration_s=duration, fov=fov,
            registry=registry, roi_counts=counts,
        )

    return [r for r in results if r is not None]


# ── internals ──────────────────────────────────────────────────────────────


def _ensure_registry_schema(log: LogCallback, cfg: "RegistryConfig") -> None:
    """Open the store once so its ``ensure_schema`` runs ``alembic upgrade head``."""
    from roigbiv.registry import build_store

    try:
        store = build_store(cfg)
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
    registry_cfg: "RegistryConfig",
    override: bool = False,
    abort_event: Optional["threading.Event"] = None,
) -> FOVRunResult:
    from roigbiv.pipeline.run import (
        OpticsConfirmationRequired,
        PipelineAborted,
        run_pipeline,
    )

    stem = tif.stem.replace("_mc", "")
    out_dir = workspace.output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        validate_tif(tif)
    except ValueError as exc:
        log(f"  invalid TIF: {exc}")
        return FOVRunResult(tif=tif, output_dir=out_dir,
                            error=f"invalid_tif: {exc}")

    if not skip_registry:
        cfg_overrides = _apply_registry_memory(tif, cfg_overrides, registry_cfg, log)
    cfg = _build_config(out_dir, cfg_overrides)
    t0 = time.perf_counter()
    try:
        fov = run_pipeline(tif, cfg, abort_event=abort_event)
    except OpticsConfirmationRequired as need:
        # Foundation is on disk; this FOV awaits an optics decision. Skip
        # registry/traces — there are no ROIs yet. The UI surfaces this and
        # resumes with a confirmed profile.
        log(f"  paused: optics confirmation needed "
            f"(candidate '{need.payload.get('candidate_profile')}')")
        return FOVRunResult(
            tif=tif,
            output_dir=out_dir,
            duration_s=time.perf_counter() - t0,
            awaiting_confirmation=need.payload,
        )
    except PipelineAborted:
        # Cooperative stop at a stage boundary. The last completed stage's
        # outputs + manifest entry are on disk (a --resume run continues from
        # there); skip the registry write — these are partial detections.
        log("  aborted (stop requested) — registry write skipped.")
        return FOVRunResult(
            tif=tif,
            output_dir=out_dir,
            duration_s=time.perf_counter() - t0,
            error="aborted",
        )
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
    if (skip_registry or getattr(cfg, "scout_mode", False)
            or getattr(cfg, "foundation_only", False)):
        if getattr(cfg, "scout_mode", False):
            log("  registry: skipped (scout run — triage only)")
        elif getattr(cfg, "foundation_only", False):
            log("  registry: skipped (foundation-only dry run)")
    else:
        try:
            from roigbiv.pipeline.optics import resolved_config_payload
            registry = _register_session(
                stem, fov, log, registry_cfg,
                resolved_config=resolved_config_payload(cfg),
                override=override)
        except Exception as exc:  # noqa: BLE001
            log(f"  WARNING: registry call failed — "
                f"{type(exc).__name__}: {exc}")

    # Single-shot traces/ bundle write — after registry so the sidecar can
    # carry session_id / fov_id / global_cell_id in one deterministic pass.
    try:
        _write_traces_bundle(fov, cfg, workspace, registry, log)
    except Exception as exc:  # noqa: BLE001
        log(f"  WARNING: traces bundle write failed — "
            f"{type(exc).__name__}: {exc}")

    return FOVRunResult(
        tif=tif, output_dir=out_dir,
        duration_s=duration, fov=fov,
        registry=registry, roi_counts=counts,
    )


def _write_traces_bundle(
    fov: FOVData,
    cfg: PipelineConfig,
    workspace: "WorkspacePaths",
    registry_report: Optional[dict],
    log: "LogCallback",
) -> None:
    """Write ``{out_dir}/traces/`` for the pynapse handoff. See traces_io."""
    from roigbiv.pipeline.traces_io import finalize_fov_bundle

    if fov.F_raw is None or fov.F_neu is None or fov.F_corrected is None:
        log("  traces: skipped (no trace matrices on FOVData)")
        return
    rois_sorted = sorted(fov.rois, key=lambda r: int(r.label_id))
    finalize_fov_bundle(
        rois_sorted,
        fov.F_raw,
        fov.F_neu,
        fov.F_corrected,
        fov.output_dir,
        cfg,
        registry_report=registry_report,
        data_bin_path=fov.data_bin_path,
        fov_shape=tuple(fov.shape),
        workspace_root=workspace.input_root,
    )
    log(f"  traces: wrote {fov.output_dir}/traces/")


def _build_config(output_dir: Path, overrides: dict) -> PipelineConfig:
    """Build a PipelineConfig with user overrides applied on top of defaults."""
    base = {"output_dir": output_dir, "no_viewer": True}
    base.update(overrides)
    base["output_dir"] = output_dir   # always force per-FOV path
    base.pop("override", None)        # registry directive, not a config field
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


def _apply_registry_memory(
    tif: Path, cfg_overrides: dict, registry_cfg: "RegistryConfig",
    log: LogCallback,
) -> dict:
    """Reuse a known-good optics config for a repeat FOV (propose, don't impose).

    For an AUTO run (cfg carries an auto-adapt prior), look up prior FOVs in the
    same ``(animal_id, region)`` with a stored ``resolved_config``. If found,
    overlay that profile's categorical bundle and suppress the pause-to-confirm
    — we've processed this region before. The per-FOV scale measurement still
    re-runs, so numerics stay FOV-specific; only the categorical profile +
    confidence are seeded. Total: any failure returns the overrides unchanged.
    """
    aa = cfg_overrides.get("auto_adapt") or {}
    if "prior" not in aa:                       # not an auto run → nothing to reuse
        return cfg_overrides
    try:
        import json as _json

        from roigbiv.pipeline.profiles import get_profile
        from roigbiv.registry import build_blob_store, build_store
        from roigbiv.registry.filename import parse_filename_metadata

        meta = parse_filename_metadata(tif.stem.replace("_mc", ""))
        store = build_store(registry_cfg)
        store.ensure_schema()
        hit = next((c for c in store.find_candidates(meta.animal_id, meta.region)
                    if getattr(c, "resolved_config_uri", None)), None)
        if hit is None:
            return cfg_overrides
        stored = _json.loads(
            build_blob_store(registry_cfg).get(hit.resolved_config_uri).decode())
        sp = stored.get("profile") or cfg_overrides.get("profile", "grin")
        overlay = get_profile(sp)
        # auto_scale derives the diameter post-foundation, so force per-image
        # diameter_auto off even when the stored profile (e.g. generic) sets it.
        overlay["diameter_auto"] = False
        log(f"  registry memory: reusing '{sp}' from FOV {hit.fov_id[:8]} "
            f"({meta.animal_id}/{meta.region}) — skipping optics confirmation")
        return {
            **cfg_overrides, **overlay,
            "profile": sp,
            "assume_optics": True,
            "auto_adapt": {**aa, "registry_prior": {
                "fov_id": hit.fov_id, "profile": sp}},
        }
    except Exception as exc:  # noqa: BLE001
        log(f"  registry memory lookup skipped — {type(exc).__name__}: {exc}")
        return cfg_overrides


def _register_session(
    stem: str, fov: FOVData, log: LogCallback, cfg: "RegistryConfig",
    *, resolved_config: Optional[dict] = None, override: bool = False,
) -> Optional[dict]:
    """Mirror of ``roigbiv.pipeline.run._register_fov_after_pipeline``.

    Re-implemented here (rather than calling the underscore-prefixed helper)
    so the workspace runner does not depend on a private symbol of run.py.
    """
    from roigbiv.registry import (
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
        resolved_config=resolved_config,
        override=override,
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
    if decision == "already_registered":
        return (f"  registry: already_registered fov_id={report.get('fov_id')} "
                f"(no-op)")
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


def _safety_backfill(
    workspace: WorkspacePaths, log: LogCallback, cfg: "RegistryConfig"
) -> None:
    """Idempotent sweep: register any FOV outputs not yet linked to the DB."""
    from roigbiv.registry.backfill import run_backfill

    if not workspace.output_root.exists():
        return
    log("\nBackfill sweep over output/ (idempotent safety net)")
    try:
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
