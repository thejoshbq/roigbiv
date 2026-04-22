"""
ROIGBIV — Streamlit interface for the sequential four-stage pipeline.

Launch with:  streamlit run app.py
LAN access:   streamlit run app.py --server.address 0.0.0.0
"""
import contextlib
import errno
import io
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "deployed" / "current_model"


PHASE_MARKERS = (
    "Foundation:", "Stage 1:", "Source subtraction:",
    "Stage 2:", "Stage 3:", "Stage 4:",
    "Trace extraction", "Overlap correction",
    "Computing unified QC features",
)


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class _QueueStreamer(io.TextIOBase):
    """stdout/stderr shim: pushes each completed line onto a queue and
    tees to the original terminal. Safe to call from a worker thread
    because it never touches Streamlit widgets."""

    def __init__(self, q: "queue.Queue[str]"):
        self._q = q
        self._buf = io.StringIO()
        self._line = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf.write(s)
        try:
            sys.__stdout__.write(s)
            sys.__stdout__.flush()
        except Exception:
            pass
        self._line += s
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                self._q.put(line)
        return len(s)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._buf.getvalue()


def _run_pipeline_worker(tif, cfg, log_q, result):
    """Run the pipeline with stdout/stderr captured into `log_q`.
    Store result or exception in the shared `result` dict. Performs no
    Streamlit calls — all UI updates happen on the main thread."""
    from roigbiv.pipeline.run import run_pipeline

    streamer = _QueueStreamer(log_q)
    try:
        with contextlib.redirect_stdout(streamer), contextlib.redirect_stderr(streamer):
            result["fov"] = run_pipeline(tif, cfg)
    except BaseException as exc:
        result["exc"] = exc
    finally:
        result["tail"] = streamer.getvalue().splitlines()[-30:]


def _run_parallel_batch(
    *,
    valid_3d,
    out_path,
    fs,
    tau,
    k_background,
    model_path,
    n_workers,
    progress,
    errors,
    fov_summaries,
):
    """Run multiple FOVs concurrently via the subprocess-pool batch runner.

    Each FOV gets its own `st.status` container (created up-front). Logs
    flow from worker processes through a shared multiprocessing queue;
    the main thread drains them into the matching container, while a side
    thread blocks on `batch.run_batch`. Progress + summary + error lists
    are populated in-place so the surrounding render path stays identical
    to the sequential case.
    """
    from roigbiv.pipeline.types import PipelineConfig
    from roigbiv.pipeline import batch as _batch

    n_valid = len(valid_3d)
    stems = [tif.stem.replace("_mc", "") for tif, _ in valid_3d]
    label_prefixes = [f"[{i+1}/{n_valid}] {stems[i]}" for i in range(n_valid)]

    # Build per-FOV configs + status containers up front so the UI renders
    # all FOVs as "running" from the first paint.
    cfgs: list = []
    statuses: list = []
    start_times: list = []
    for i, (_tif, _shape) in enumerate(valid_3d):
        cfgs.append(PipelineConfig(
            fs=float(fs),
            tau=float(tau),
            k_background=int(k_background),
            cellpose_model=str(model_path),
            output_dir=out_path / stems[i],
            no_viewer=True,
            batch_n_workers=int(n_workers),
        ))
        statuses.append(st.status(label_prefixes[i], expanded=True))
        start_times.append(time.time())

    jobs = [(tif, cfg) for (tif, _shape), cfg in zip(valid_3d, cfgs)]

    # Main-thread-local queue of (fov_index, line) tuples — drained here
    # because only the main thread may touch Streamlit widgets safely.
    main_log_q: "queue.Queue[tuple[int, str]]" = queue.Queue()
    completions: dict = {}
    completions_lock = threading.Lock()
    phases: dict = {}

    def _log_cb(idx, line):
        main_log_q.put((idx, line))

    def _complete_cb(idx, fov, exc):
        with completions_lock:
            completions[idx] = ("err", exc) if exc is not None else ("ok", fov)

    batch_done = threading.Event()

    def _run_batch():
        try:
            _batch.run_batch(jobs, int(n_workers), _log_cb, _complete_cb)
        finally:
            batch_done.set()

    runner = threading.Thread(target=_run_batch, daemon=True)
    runner.start()

    finalized: set = set()
    while not (batch_done.is_set() and main_log_q.empty() and len(finalized) == n_valid):
        # Drain any buffered log lines into their matching status container.
        while True:
            try:
                idx, line = main_log_q.get_nowait()
            except queue.Empty:
                break
            statuses[idx].write(line)
            stripped = line.lstrip()
            for marker in PHASE_MARKERS:
                if stripped.startswith(marker):
                    phases[idx] = marker.rstrip(":")
                    break

        # Tick elapsed labels for FOVs still running.
        now = time.time()
        for idx in range(n_valid):
            if idx in finalized:
                continue
            elapsed_str = _fmt_elapsed(now - start_times[idx])
            bits = [label_prefixes[idx], elapsed_str]
            if phases.get(idx):
                bits.append(phases[idx])
            statuses[idx].update(label=" — ".join(bits))

        # Finalize FOVs the workers have reported completion for.
        with completions_lock:
            new_done = [idx for idx in completions if idx not in finalized]
        for idx in new_done:
            kind, payload = completions[idx]
            elapsed = time.time() - start_times[idx]
            if kind == "err":
                exc = payload
                if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
                    msg = (f"Disk full while writing to {cfgs[idx].output_dir}. "
                           f"Free space and retry.")
                elif isinstance(exc, OSError):
                    msg = f"OSError: {exc}"
                else:
                    msg = f"ERROR: {exc}"
                statuses[idx].write(msg)
                errors.append((stems[idx], msg))
                statuses[idx].update(
                    label=f"{label_prefixes[idx]} — FAILED after {_fmt_elapsed(elapsed)}",
                    state="error",
                )
            else:
                fov = payload
                fov_summaries.append({
                    "stem": stems[idx],
                    "n_rois": len(fov.rois),
                    "elapsed_s": elapsed,
                })
                statuses[idx].update(
                    label=f"{label_prefixes[idx]} — {len(fov.rois)} ROIs "
                          f"({_fmt_elapsed(elapsed)})",
                    state="complete",
                )
            finalized.add(idx)
            progress.progress(
                len(finalized) / n_valid,
                text=f"Processed {len(finalized)}/{n_valid} FOVs",
            )

        time.sleep(0.5)

    runner.join(timeout=5.0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="ROIGBIV", layout="wide")
    st.title("ROIGBIV")
    st.caption("Sequential subtractive ROI detection for two-photon calcium imaging")

    # GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.success(f"GPU: {gpu_name} ({vram:.1f} GB)")
    else:
        st.warning("No GPU detected — Cellpose will run on CPU (slower).")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Pipeline Parameters")

        tif_dir = st.text_input(
            "TIF directory",
            placeholder="/path/to/tif/stacks",
            help="Directory with pre-motion-corrected `*_mc.tif` stacks "
                 "(searched recursively).",
        )
        output_dir = st.text_input(
            "Output directory",
            placeholder="/path/to/output",
            help="Root directory. One subdirectory per FOV will be created.",
        )

        st.divider()
        fs = st.number_input("Frame rate (Hz)", value=30.0, min_value=1.0, step=1.0)
        tau = st.number_input(
            "GCaMP tau (s)", value=1.0, min_value=0.1, step=0.1,
            help="GCaMP6s=1.0, GCaMP6f=0.4, GCaMP7f=0.7")
        model_path = st.text_input(
            "Cellpose model", value=str(DEFAULT_MODEL),
            help="Path to the fine-tuned Cellpose checkpoint used in Stage 1.")

        with st.expander("Advanced"):
            k_background = st.number_input(
                "k_background (L+S separation)", value=30, min_value=1, step=1,
                help="Top-k SVD components reconstructed as the background L. "
                     "Higher = more of the slow/structured signal absorbed into L.")
            batch_n_workers = st.number_input(
                "Parallel FOVs (max 2)", value=1, min_value=1, max_value=2, step=1,
                help="Process multiple FOVs concurrently. GPU phases (Cellpose, "
                     "Suite2p, Stage 3 FFT, subtraction) serialize across workers; "
                     "CPU phases (Foundation, Stage 4 bandpass, traces) overlap. "
                     "Requires ≥ 2 FOVs queued; GPU VRAM caps parallelism at 2.")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_instr, tab_run, tab_results, tab_registry = st.tabs(
        ["Instructions", "Run Pipeline", "View Results", "Registry"])

    with tab_instr:
        _instructions_tab()

    with tab_run:
        _run_tab(
            tif_dir=tif_dir, output_dir=output_dir,
            fs=fs, tau=tau, k_background=int(k_background),
            model_path=model_path,
            batch_n_workers=int(batch_n_workers),
        )

    with tab_results:
        _results_tab(output_dir)

    with tab_registry:
        _registry_tab(output_dir)


# ── Instructions tab ─────────────────────────────────────────────────────────

def _instructions_tab():
    st.markdown("""
## What is ROIGBIV?

ROIGBIV is a sequential subtractive ROI detection pipeline for two-photon
calcium imaging. It runs four complementary detection stages with a
validation gate between each, then produces traces (F, dF/F, spikes),
activity-type classifications, and a prioritized human-in-the-loop (HITL)
review queue.

---

## Required Input

| What | Format | Notes |
|------|--------|-------|
| **TIF stacks** | Multi-frame TIFF (T × H × W) | One per FOV. Pre-motion-corrected files should end in `_mc.tif` (registration auto-skipped). |
| **Cellpose model** | Cellpose checkpoint | The fine-tuned model used in Stage 1. Uses the deployed default if available. |

All TIFs must share a frame rate. Place them in one directory
(searched recursively).

---

## Pipeline Overview

| Phase | What it does |
|-------|-------------|
| **Foundation** | Motion correction (or re-use), SVD-based L+S decomposition (background vs signal), summary images (mean/max/std/Vcorr on S, nuclear-shadow DoG). |
| **Stage 1 — Cellpose** | Spatial morphology detection on denoised S. Gate 1 validates area / solidity / eccentricity / nuclear-shadow / soma-surround contrast. |
| **Stage 2 — Suite2p** | Temporal re-detection on residual S₁ to recover activity-only neurons missed by Cellpose. Gate 2 cross-validates via IoU + temporal correlation. |
| **Stage 3 — Template sweep** | Custom FFT template matching on S₂ for sparse-firing neurons. Gate 3 validates waveform R², rise/decay, anti-correlation. |
| **Stage 4 — Tonic search** | Multi-scale bandpass + correlation contrast on S₃ for tonic / slow-modulation neurons. Gate 4 validates corr-contrast, motion correlation, intensity. All survivors are flagged for HITL. |
| **Post** | Trace extraction, overlap correction, QC features, activity-type classification, dF/F, deconvolution, HITL review queue. |

---

## How to Use

1. Enter the **TIF directory** and **output directory** in the sidebar.
2. Set the **frame rate** and **tau** for your indicator.
3. Switch to **Run Pipeline** and click **Run Pipeline**.
4. When it finishes, use **View Results** to inspect per-FOV outputs and
   open layers in napari.
5. For manual curation, open `{output_dir}/{stem}/hitl_staging/` in the
   Cellpose GUI — it contains mean image + `_seg.tif` mask pairs ready
   for review.

---

## Output Layout (per FOV)

```
output_dir/{stem}/
  summary/              mean_M, mean_S, mean_L, max_S, std_S, vcorr_S, dog_map
  stage1/ … stage4/     per-stage masks + reports + diagnostics
  suite2p/plane0/       motion-corrected movie + Suite2p ops
  merged_masks.tif      uint16 labels — row K of trace arrays = label_id K
  F.npy, Fneu.npy, F_corrected.npy, dFF.npy, spks.npy
  F_bandpass.npy + F_bandpass_index.npy   (tonic ROIs only)
  roi_metadata.json     per-ROI dict: gate_outcome, activity_type, features…
  pipeline_log.json     shape, k_background, stage_counts, timings, warnings
  review_queue.json     prioritized HITL review list
  hitl/                 per-ROI evidence (Stage 3 single-event, Stage 4 tonic)
  hitl_staging/         Cellpose-GUI layout: images/{stem}.tif + masks/{stem}_seg.tif
```
""")


# ── Run tab ──────────────────────────────────────────────────────────────────

def _run_tab(*, tif_dir, output_dir, fs, tau, k_background, model_path,
             batch_n_workers=1):
    if not tif_dir or not output_dir:
        st.info("Enter a TIF directory and output directory in the sidebar to begin.")
        return

    tif_path = Path(tif_dir)
    out_path = Path(output_dir)

    if not tif_path.exists():
        st.error(f"TIF directory not found: {tif_path}")
        return

    from roigbiv.io import discover_tifs, validate_tif

    tif_files = discover_tifs(tif_path)
    if not tif_files:
        st.error(f"No TIF files found under {tif_path}")
        return

    # ── Validate + split into 3D stacks vs. skipped ───────────────────────
    valid_3d: list[tuple[Path, tuple]] = []   # [(path, (T, Ly, Lx)), ...]
    skipped: list[tuple[str, str]] = []       # [(name, reason), ...]
    for tif in tif_files:
        try:
            _, shape = validate_tif(tif)
            valid_3d.append((tif, shape))
        except ValueError as e:
            skipped.append((tif.name, str(e)))

    st.subheader(
        f"Found {len(tif_files)} TIF file(s) "
        f"— {len(valid_3d)} valid 3D stack(s), {len(skipped)} skipped"
    )
    if valid_3d:
        with st.expander("3D stacks to process", expanded=False):
            for tif, shape in valid_3d:
                st.text(
                    f"  {tif.name}  —  {shape[0]} frames, "
                    f"{shape[1]}x{shape[2]} px"
                )
    if skipped:
        with st.expander(f"{len(skipped)} file(s) skipped", expanded=False):
            for name, reason in skipped:
                st.text(f"  {name}: {reason}")

    if not valid_3d:
        st.error("No valid 3D TIF stacks found — nothing to run.")
        return

    # ── Disk-space preflight ──────────────────────────────────────────────
    # Per-FOV footprint: data.bin (int16, 2 B/pix) + residual_S.dat (float32,
    # 4 B/pix) = 6 * T * Ly * Lx bytes. Later stages transiently write S1/S2/S3
    # residuals of the same size but delete as they go, so the maximum live
    # footprint per FOV is ~2x the steady-state. 1.5x headroom below is a
    # minimum safety margin. SIGBUS from mmap writes is uncatchable in Python
    # — once you see "Bus error" the Streamlit process is dead.
    def _bytes_per_fov(shape):
        T, Ly, Lx = shape[0], shape[1], shape[2]
        return int(T) * int(Ly) * int(Lx) * 6

    max_fov = max(_bytes_per_fov(s) for _, s in valid_3d)
    total_fov = sum(_bytes_per_fov(s) for _, s in valid_3d)

    probe = out_path if out_path.exists() else out_path.parent
    try:
        free_bytes = shutil.disk_usage(probe).free
    except FileNotFoundError:
        st.error(
            f"Cannot check free space — neither {out_path} nor its parent exists."
        )
        return

    def _human(nb):
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if nb < 1024:
                return f"{nb:.1f} {unit}"
            nb /= 1024
        return f"{nb:.1f} PB"

    # Effective parallelism is hard-capped by batch.MAX_BATCH_WORKERS and by
    # the number of queued FOVs. The guard scales the min-safety threshold
    # with concurrent workers: parallel FOVs each hold their own memmap set
    # in flight at the same time.
    from roigbiv.pipeline.batch import MAX_BATCH_WORKERS
    effective_workers = max(1, min(int(batch_n_workers), MAX_BATCH_WORKERS, len(valid_3d)))
    safety = int(1.5 * max_fov * effective_workers)
    c1, c2, c3 = st.columns(3)
    c1.metric("Free at output", _human(free_bytes))
    c2.metric("Need per FOV (min)", _human(max_fov))
    c3.metric("Batch total (min)", _human(total_fov))

    if free_bytes < safety:
        worker_note = (
            f" across {effective_workers} parallel worker(s)"
            if effective_workers > 1 else ""
        )
        st.error(
            f"Insufficient free space at `{probe}`: "
            f"{_human(free_bytes)} available, "
            f"need ≥ {_human(safety)} for the largest FOV{worker_note} "
            f"(and ideally ≥ {_human(int(1.5 * total_fov))} for the whole batch). "
            f"Foundation writes multi-GB memmaps (`data.bin` + `residual_S.dat`); "
            f"on a full filesystem these trigger SIGBUS (‘Bus error’) which "
            f"cannot be caught and will kill the Streamlit process. "
            f"Free disk and re-run."
        )
        return

    if not st.button("Run Pipeline", type="primary", use_container_width=True):
        return

    from roigbiv.pipeline.types import PipelineConfig

    progress = st.progress(0.0, text="Starting pipeline...")
    errors: list[tuple[str, str]] = []
    fov_summaries: list[dict] = []
    t_batch = time.time()

    n_valid = len(valid_3d)

    if effective_workers >= 2 and n_valid >= 2:
        # ── Parallel batch path (Phase B) ────────────────────────────────
        _run_parallel_batch(
            valid_3d=valid_3d, out_path=out_path,
            fs=fs, tau=tau, k_background=k_background, model_path=model_path,
            n_workers=effective_workers,
            progress=progress, errors=errors, fov_summaries=fov_summaries,
        )
    else:
        # ── Sequential path (unchanged from pre-Phase-B) ─────────────────
        for i, (tif, _shape) in enumerate(valid_3d):
            stem = tif.stem.replace("_mc", "")
            fov_out = out_path / stem

            cfg = PipelineConfig(
                fs=float(fs),
                tau=float(tau),
                k_background=int(k_background),
                cellpose_model=str(model_path),
                output_dir=fov_out,
                no_viewer=True,
            )

            label_prefix = f"[{i+1}/{n_valid}] {stem}"
            with st.status(label_prefix, expanded=True) as status:
                elapsed_ph = st.empty()
                log_q: "queue.Queue[str]" = queue.Queue()
                result: dict = {}
                current_phase = ""
                start = time.time()

                worker = threading.Thread(
                    target=_run_pipeline_worker,
                    args=(tif, cfg, log_q, result),
                    daemon=True,
                )
                worker.start()

                while worker.is_alive():
                    while True:
                        try:
                            line = log_q.get_nowait()
                        except queue.Empty:
                            break
                        status.write(line)
                        stripped = line.lstrip()
                        for marker in PHASE_MARKERS:
                            if stripped.startswith(marker):
                                current_phase = marker.rstrip(":")
                                break
                    elapsed = time.time() - start
                    elapsed_str = _fmt_elapsed(elapsed)
                    elapsed_ph.markdown(f"**Elapsed:** {elapsed_str}")
                    label_bits = [label_prefix, elapsed_str]
                    if current_phase:
                        label_bits.append(current_phase)
                    status.update(label=" — ".join(label_bits))
                    time.sleep(0.5)

                worker.join()
                while not log_q.empty():
                    status.write(log_q.get_nowait())

                elapsed = time.time() - start
                elapsed_ph.markdown(f"**Elapsed:** {_fmt_elapsed(elapsed)}")

                if "exc" in result:
                    exc = result["exc"]
                    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
                        msg = (f"Disk full while writing to {fov_out}. "
                               f"Free space and retry. (Note: any SIGBUS "
                               f"from a partially-written memmap would have "
                               f"killed Streamlit; if you're seeing this "
                               f"caught error instead, you got lucky.)")
                    elif isinstance(exc, OSError):
                        msg = f"OSError: {exc}"
                    else:
                        msg = f"ERROR: {exc}"
                    status.write(msg)
                    errors.append((stem, msg))
                    status.update(
                        label=f"{label_prefix} — FAILED after {_fmt_elapsed(elapsed)}",
                        state="error",
                    )
                else:
                    fov = result["fov"]
                    fov_summaries.append({
                        "stem": stem,
                        "n_rois": len(fov.rois),
                        "elapsed_s": elapsed,
                    })
                    status.update(
                        label=f"{label_prefix} — {len(fov.rois)} ROIs "
                              f"({_fmt_elapsed(elapsed)})",
                        state="complete",
                    )
            progress.progress(
                (i + 1) / n_valid,
                text=f"Processed {i+1}/{n_valid} FOVs",
            )

    elapsed_batch = time.time() - t_batch
    progress.progress(
        1.0, text=f"Pipeline complete — {n_valid} FOV(s) in "
                  f"{elapsed_batch:.0f}s")

    # ── Summary ──────────────────────────────────────────────────────────
    if fov_summaries:
        st.balloons()
        total_rois = sum(f["n_rois"] for f in fov_summaries)
        c1, c2, c3 = st.columns(3)
        c1.metric("FOVs processed", len(fov_summaries))
        c2.metric("Total ROIs", total_rois)
        c3.metric("Batch time (s)", f"{elapsed_batch:.0f}")

    if errors:
        with st.expander(f"{len(errors)} error(s) occurred", expanded=False):
            for stem, err in errors:
                st.warning(f"**{stem}**: {err}")


# ── Napari launcher ──────────────────────────────────────────────────────────

def _launch_napari(fov_out_dir: Path) -> None:
    """Spawn a napari process that reads the FOV output dir and opens the full layer set."""
    script = "\n".join([
        "import sys",
        f"sys.path.insert(0, r'{PROJECT_ROOT}')",
        "from pathlib import Path",
        "from roigbiv.pipeline.loaders import load_fov_from_output_dir",
        "from roigbiv.pipeline.napari_viewer import display_pipeline_results",
        f"fov, review_queue = load_fov_from_output_dir(Path(r'{fov_out_dir}'))",
        "display_pipeline_results(fov, review_queue=review_queue)",
    ])

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    subprocess.Popen([sys.executable, tmp_path])


def _launch_cross_session_napari(fov_id: str) -> None:
    """Spawn a napari process that opens the cross-session viewer for a FOV.

    The subprocess re-resolves the registry from the inherited
    ``ROIGBIV_REGISTRY_DSN`` env, so only ``fov_id`` needs to be passed.
    """
    script = "\n".join([
        "import sys",
        f"sys.path.insert(0, r'{PROJECT_ROOT}')",
        "from roigbiv.pipeline.cross_session_viewer import display_cross_session_fov",
        f"display_cross_session_fov({fov_id!r})",
    ])

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    subprocess.Popen([sys.executable, tmp_path])


# ── Results tab ──────────────────────────────────────────────────────────────

def _iter_fov_dirs(out_path: Path) -> list[Path]:
    """Return subdirectories of out_path that look like pipeline output dirs."""
    if not out_path.exists():
        return []
    return sorted(
        d for d in out_path.iterdir()
        if d.is_dir() and (d / "pipeline_log.json").exists()
    )


def _aggregate_logs(fov_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in fov_dirs:
        try:
            log = json.loads((d / "pipeline_log.json").read_text())
        except Exception:
            continue
        sc = log.get("stage_counts", {})
        row = {
            "fov": d.name,
            "total_rois": log.get("total_rois", 0),
            "shape": "x".join(str(x) for x in log.get("shape", [])),
        }
        for key in ("stage1", "stage2", "stage3", "stage4"):
            s = sc.get(key, {})
            row[f"{key}_detected"] = s.get("detected", 0)
            row[f"{key}_accepted"] = s.get("accepted", 0)
            row[f"{key}_flagged"]  = s.get("flagged", 0)
            row[f"{key}_rejected"] = s.get("rejected", 0)
        act = log.get("activity_type_counts", {})
        for k in ("phasic", "sparse", "tonic", "silent", "ambiguous"):
            row[k] = act.get(k, 0)
        rq = log.get("review_queue_summary", {})
        row["review_total"] = rq.get("total", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _results_tab(output_dir):
    if not output_dir:
        st.info("Enter an output directory in the sidebar.")
        return

    out_path = Path(output_dir)
    fov_dirs = _iter_fov_dirs(out_path)
    if not fov_dirs:
        st.info("No pipeline outputs found under this directory. Run the pipeline first.")
        return

    df = _aggregate_logs(fov_dirs)

    # ── Top-line metrics ──────────────────────────────────────────────────
    if not df.empty:
        total_rois = int(df["total_rois"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FOVs", len(df))
        c2.metric("Total ROIs", total_rois)
        c3.metric("Phasic + Sparse", int(df["phasic"].sum() + df["sparse"].sum()))
        c4.metric("Tonic", int(df["tonic"].sum()))

        st.subheader("Per-stage breakdown")
        stage_cols = [c for c in df.columns if c.startswith(("stage1_", "stage2_", "stage3_", "stage4_"))]
        st.dataframe(
            df[["fov", "total_rois", *stage_cols]],
            use_container_width=True, hide_index=True,
        )

        st.subheader("Activity types")
        act_cols = ["phasic", "sparse", "tonic", "silent", "ambiguous"]
        st.dataframe(
            df[["fov", *act_cols, "review_total"]],
            use_container_width=True, hide_index=True,
        )

    # ── Per-FOV detail + napari launcher ──────────────────────────────────
    st.divider()
    st.subheader("Inspect a FOV")
    stems = [d.name for d in fov_dirs]
    stem = st.selectbox("FOV", stems, key="viz_fov")
    fov_out = out_path / stem

    meta_path = fov_out / "roi_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta_df = pd.DataFrame(meta)
        keep = [c for c in [
            "label_id", "source_stage", "confidence", "gate_outcome",
            "activity_type", "area", "solidity", "eccentricity",
            "cellpose_prob", "iscell_prob", "event_count", "corr_contrast",
            "review_priority",
        ] if c in meta_df.columns]
        st.dataframe(meta_df[keep], use_container_width=True,
                     hide_index=True, height=320)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Open in Napari", type="primary", key="viz_open"):
            try:
                _launch_napari(fov_out)
                st.success("Napari launched — check for a new window.")
            except Exception as e:
                st.error(f"Failed to launch Napari: {e}")
    with col2:
        hitl_staging = fov_out / "hitl_staging"
        if hitl_staging.exists():
            st.info(
                f"**HITL curation**: open `{hitl_staging}` in the Cellpose GUI. "
                f"It contains `images/{stem}.tif` + `masks/{stem}_seg.tif` "
                f"ready for review / editing."
            )

    # ── Downloads ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Downloads")
    dl_cols = st.columns(3)
    for col, name in zip(dl_cols, ("pipeline_log.json", "roi_metadata.json", "review_queue.json")):
        p = fov_out / name
        if p.exists():
            col.download_button(
                name, data=p.read_bytes(),
                file_name=f"{stem}_{name}", mime="application/json",
                key=f"dl_{name}",
            )

    if st.button(f"Download all artifacts for {stem} as ZIP"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in fov_out.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(fov_out))
        st.download_button(
            "Save ZIP", data=buf.getvalue(),
            file_name=f"{stem}_pipeline.zip", mime="application/zip",
        )


# ── Registry tab ─────────────────────────────────────────────────────────────


def _registry_tab(output_dir: str) -> None:
    """Cross-session FOV + cell tracking.

    Three panels:
      1. Overview — every FOV in the registry with session counts.
      2. FOV detail — per-session match quality and cell deltas.
      3. Longitudinal cell browser — select global_cell_id, see every session
         it appears in and the local_label_id used each time.
    """
    from roigbiv.registry import build_blob_store, build_store, register_or_match
    from roigbiv.registry.backfill import run_backfill

    st.header("Cross-Session Registry")
    st.caption(
        "Tracks which FOVs have been seen before and maps local ROI labels "
        "across sessions. Populated by running the pipeline with `--registry` "
        "or the Backfill button below."
    )

    try:
        store = build_store()
        store.ensure_schema()
    except Exception as exc:
        st.error(f"Could not open registry: {exc}")
        return

    with st.expander("Registry maintenance"):
        st.caption(
            "Run pending alembic migrations against the active DSN. Safe to "
            "re-run — the migration is idempotent."
        )
        if st.button(
            "Run database migrations",
            key="registry_migrate",
            help="Applies any pending alembic migrations to the active "
                 "registry DSN.",
        ):
            try:
                from roigbiv.registry.migrate import ensure_alembic_head

                result = ensure_alembic_head()
                st.success(f"Migrations: {result}")
            except Exception as exc:
                st.error(f"Migration failed: {exc}")

    with st.expander("Backfill existing runs"):
        backfill_root = st.text_input(
            "Root directory to scan",
            value=output_dir or "inference/pipeline",
            key="registry_backfill_root",
            help="Walks this directory, registers/matches every FOV in "
                 "chronological order.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Dry run", key="registry_backfill_dryrun"):
                reports = run_backfill(Path(backfill_root), dry_run=True)
                st.write(f"{len(reports)} candidate(s) found")
                st.dataframe(pd.DataFrame(reports))
        with col_b:
            if st.button("Backfill now", type="primary", key="registry_backfill_run"):
                with st.spinner("Registering…"):
                    reports = run_backfill(Path(backfill_root), dry_run=False)
                st.success(f"Processed {len(reports)} FOV(s)")
                st.dataframe(pd.DataFrame(reports))

    fovs = store.list_fovs()
    if not fovs:
        st.info(
            "Registry is empty. Run the pipeline with `--registry` or use the "
            "Backfill button above to ingest existing outputs."
        )
        return

    fov_rows = pd.DataFrame([
        {
            "fov_id": f.fov_id,
            "animal_id": f.animal_id,
            "region": f.region,
            "latest_session": (
                f.latest_session_date.isoformat() if f.latest_session_date else "-"
            ),
            "sessions": len(store.list_sessions(f.fov_id)),
            "cells": len(store.list_cells(f.fov_id)),
            "fingerprint": f.fingerprint_hash[:12],
        }
        for f in fovs
    ])
    st.subheader(f"{len(fovs)} FOV(s) in registry")
    st.dataframe(fov_rows, use_container_width=True)

    st.divider()
    st.subheader("FOV detail")
    fov_choice = st.selectbox(
        "Select a FOV",
        options=[f.fov_id for f in fovs],
        format_func=lambda fid: (
            f"{fid[:8]} — "
            f"{next((f.animal_id for f in fovs if f.fov_id == fid), '?')} / "
            f"{next((f.region for f in fovs if f.fov_id == fid), '?')}"
        ),
        key="registry_fov_select",
    )
    if fov_choice:
        sessions = store.list_sessions(fov_choice)
        cells = store.list_cells(fov_choice)
        st.write(f"**{len(cells)} cell(s), {len(sessions)} session(s)**")
        if sessions:
            st.dataframe(pd.DataFrame([
                {
                    "session_date": s.session_date.isoformat(),
                    "fov_posterior": s.fov_posterior if s.fov_posterior is not None else s.fov_sim,
                    "n_matched": s.n_matched,
                    "n_new": s.n_new,
                    "n_missing": s.n_missing,
                    "output_dir": s.output_dir,
                }
                for s in sessions
            ]), use_container_width=True)

            if st.button(
                "Open cross-session viewer in Napari",
                type="primary",
                key=f"xsess_napari_{fov_choice}",
                help="Per-session mean projections, ROI overlays colored by "
                     "global_cell_id (same color across sessions = same cell), "
                     "and a hidden cell-ID text layer.",
            ):
                try:
                    _launch_cross_session_napari(fov_choice)
                    st.success("Napari launched — check for a new window.")
                except Exception as exc:
                    st.error(f"Launch failed: {exc}")

        with st.expander("Cells in this FOV"):
            st.dataframe(pd.DataFrame([
                {
                    "global_cell_id": c.global_cell_id,
                    "first_session_id": c.first_seen_session_id,
                    "first_local_label_id": c.morphology_summary.get(
                        "first_local_label_id"
                    ),
                }
                for c in cells
            ]), use_container_width=True)

    st.divider()
    st.subheader("Longitudinal cell browser")
    cell_id = st.text_input(
        "global_cell_id",
        key="registry_track_cell",
        help="Paste a global_cell_id from the table above to see every "
             "session it was observed in.",
    )
    if cell_id:
        obs = store.list_observations_for_cell(cell_id.strip())
        if not obs:
            st.info("No observations for that global_cell_id.")
        else:
            sessions_by_id = {}
            for f in fovs:
                for s in store.list_sessions(f.fov_id):
                    sessions_by_id[s.session_id] = s
            rows = []
            for o in obs:
                s = sessions_by_id.get(o.session_id)
                rows.append({
                    "session_date": s.session_date.isoformat() if s else "-",
                    "local_label_id": o.local_label_id,
                    "match_score": o.match_score,
                    "output_dir": s.output_dir if s else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
