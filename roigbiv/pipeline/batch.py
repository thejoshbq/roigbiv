"""
Multi-FOV batch runner for the ROIGBIV pipeline (Phase B of Tier-1 perf plan).

Runs ≥ 2 FOVs concurrently via a ProcessPoolExecutor with the `spawn` start
method. A shared `multiprocessing.Manager().Lock()` serializes GPU-heavy
phases (Cellpose, Suite2p, Stage 3 FFT, source subtraction) across workers
so the 8 GiB RTX-4060 is never double-booked, while CPU-bound phases
(Foundation summary images, Stage 4 bandpass, traces, QC) overlap freely.

Why a subprocess pool:
 - `app.py` replaces the process-global `sys.stdout` via
   `contextlib.redirect_stdout` during sequential runs. Two concurrent
   writers in the same process cross-contaminate each other's log queue,
   so threads are ruled out for FOV-level parallelism.
 - Each subprocess has its own `sys.stdout`, so stdout redirection is safe.

Why `spawn`, not `fork`:
 - Forking a process that has already initialized a CUDA context (torch,
   Cellpose) deadlocks on the first CUDA call in the child. `spawn` starts
   each worker from a fresh Python interpreter. The trade-off is a one-time
   re-import + Cellpose-model-load per worker; `ProcessPoolExecutor` keeps
   workers alive across jobs so the cost is amortized per batch, not per
   FOV.

Hard cap: 2 workers. The GPU lock serializes GPU phases, so adding more
than two workers can't reduce wall-time further (the CPU overlap budget is
saturated at 2) and only risks host-RAM pressure.
"""
from __future__ import annotations

import io
import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from roigbiv.pipeline.types import FOVData, PipelineConfig

# Hard cap: 2 concurrent FOVs per 8 GiB GPU.
MAX_BATCH_WORKERS = 2


# ── Worker-side state (set by the pool initializer) ──────────────────────────

_LOG_Q = None       # type: ignore[assignment]
_GPU_LOCK = None    # type: ignore[assignment]


def _worker_init(log_q, gpu_lock):
    """Pool initializer — runs once per worker process after spawn.

    Stashes the shared log queue and GPU lock into module globals so
    `_run_one_fov` can reach them without them being passed through the
    call payload (and repeatedly re-pickled).
    """
    global _LOG_Q, _GPU_LOCK
    _LOG_Q = log_q
    _GPU_LOCK = gpu_lock


class _QueuedStdout(io.TextIOBase):
    """Worker-side stdout shim — buffers partial writes, pushes full lines
    onto the shared log queue tagged with this job's fov_index.
    """

    def __init__(self, fov_index: int):
        self._idx = fov_index
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip() and _LOG_Q is not None:
                _LOG_Q.put((self._idx, line))
        return len(s)

    def flush(self) -> None:
        if self._buf and _LOG_Q is not None:
            _LOG_Q.put((self._idx, self._buf))
            self._buf = ""


def _run_one_fov(fov_index: int, tif_path: Path, cfg: PipelineConfig) -> FOVData:
    """Worker entry point — runs the full pipeline on one FOV with stdout
    redirected into the shared log queue and the GPU lock wired through.

    Exceptions propagate up to the ProcessPoolExecutor future; the batch
    runner catches them and forwards to the `on_complete` callback.
    """
    from roigbiv.pipeline.run import run_pipeline

    streamer = _QueuedStdout(fov_index)
    sys.stdout = streamer
    sys.stderr = streamer
    try:
        return run_pipeline(tif_path, cfg, gpu_lock=_GPU_LOCK)
    finally:
        streamer.flush()


# ── Main-process orchestration ───────────────────────────────────────────────

def run_batch(
    jobs: list[tuple[Path, PipelineConfig]],
    n_workers: int,
    log_callback: Callable[[int, str], None],
    on_complete: Callable[[int, Optional[FOVData], Optional[BaseException]], None],
    poll_interval: float = 0.25,
) -> None:
    """Run a list of FOV jobs concurrently in a spawn-based process pool.

    Args:
        jobs: list of (tif_path, per-FOV PipelineConfig) tuples, in the
            order they should appear in the UI. Each FOV's cfg.output_dir
            must be pre-set by the caller.
        n_workers: requested parallelism; hard-capped at MAX_BATCH_WORKERS
            and at len(jobs).
        log_callback: invoked as log_callback(fov_index, line) for every
            completed log line from any worker. Called on the main thread
            from the pump loop — safe to update Streamlit widgets.
        on_complete: invoked as on_complete(fov_index, result_or_None,
            exc_or_None) when each job finishes, exactly once per job.
        poll_interval: seconds between queue-drain ticks.

    Returns when every submitted job has completed (success or failure).
    """
    if not jobs:
        return

    workers = max(1, min(int(n_workers), MAX_BATCH_WORKERS, len(jobs)))

    # `spawn` avoids CUDA-context-in-fork deadlocks on Linux. Guard against
    # an outer process having pinned a different default start method.
    spawn_ctx = mp.get_context("spawn")

    manager = spawn_ctx.Manager()
    try:
        log_q = manager.Queue()
        gpu_lock = manager.Lock()

        # The pump thread drains log_q and forwards lines via log_callback.
        # It exits when the `stop_pump` event is set AND the queue is empty.
        stop_pump = threading.Event()

        def _pump():
            while not (stop_pump.is_set() and log_q.empty()):
                try:
                    idx, line = log_q.get(timeout=poll_interval)
                except Exception:
                    continue
                try:
                    log_callback(idx, line)
                except Exception:
                    # A broken callback must not take down the pool.
                    pass

        pump_thread = threading.Thread(target=_pump, daemon=True)
        pump_thread.start()

        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn_ctx,
                initializer=_worker_init,
                initargs=(log_q, gpu_lock),
            ) as pool:
                futures = {
                    pool.submit(_run_one_fov, idx, tif, cfg): idx
                    for idx, (tif, cfg) in enumerate(jobs)
                }
                pending = set(futures.keys())
                while pending:
                    # Poll completed futures with a short timeout so the
                    # pump thread keeps draining logs while workers run.
                    done = {f for f in pending if f.done()}
                    for fut in done:
                        idx = futures[fut]
                        try:
                            fov = fut.result()
                            on_complete(idx, fov, None)
                        except BaseException as exc:  # noqa: BLE001
                            on_complete(idx, None, exc)
                    pending -= done
                    if pending:
                        time.sleep(poll_interval)
        finally:
            stop_pump.set()
            pump_thread.join(timeout=2.0)
    finally:
        try:
            manager.shutdown()
        except Exception:
            pass
