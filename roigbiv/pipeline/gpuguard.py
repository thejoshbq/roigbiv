"""GPU VRAM guard — free the card before a GPU-heavy pipeline run.

On this lab's single-GPU box (RTX 5080, 16 GB) the local-Qwen MCP server shares
the card: ``ollama`` keeps a large model (e.g. ``qwen3-coder:30b`` ≈ 18 GB)
resident for several minutes after each call. While it is loaded the pipeline
gets almost no free VRAM, so Cellpose / Foundation SVD OOM and silently fall
back to CPU — the GPU looks "broken" when it is merely occupied.

``free_gpu_for_run`` is a best-effort preflight: if free VRAM is below a
threshold and ``ollama`` is the one holding it, ask ``ollama`` to unload its
models (``keep_alive=0``) and wait briefly for the memory to come back. It
**never raises** — any failure (no CUDA, ``ollama`` unreachable, the memory held
by some other process) degrades gracefully to "leave things as they are," and
the existing per-stage CPU fallback still applies.

The model ollama unloads is a reloadable cache; ollama reloads it on its next
request, so this is non-destructive.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
_GB = 1024 ** 3


def _cuda_device_present() -> bool:
    """True if the driver sees a CUDA device — cheap, no context allocation.

    Distinct from :func:`_free_gpu_bytes` returning ``None``: a device can be
    *present* yet so full that probing its free memory fails (initializing the
    primary CUDA context itself needs VRAM). That is precisely the case we must
    evict for, so presence and free-bytes are detected separately.
    """
    try:
        import torch

        return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except Exception:
        return False


def _free_gpu_bytes() -> Optional[int]:
    """Free bytes on the active CUDA device, or ``None`` if it can't be probed.

    ``None`` means "couldn't read free memory" — which on a present device
    usually means the card is *so* full that even the context init OOMs. Callers
    must treat ``None`` as "contended," not as "no GPU."
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def _ollama_loaded_models(timeout_s: float) -> list[str]:
    """Names of models ollama currently has resident (``GET /api/ps``)."""
    try:
        req = urllib.request.Request(f"{_OLLAMA_URL}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


def _ollama_unload(model: str, timeout_s: float) -> bool:
    """Ask ollama to evict ``model`` now (``keep_alive=0``). Best-effort bool."""
    payload = json.dumps(
        {"model": model, "prompt": "", "keep_alive": 0}
    ).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{_OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def free_gpu_for_run(
    min_free_gb: float = 8.0,
    *,
    enabled: bool = True,
    timeout_s: float = 20.0,
    log_cb=print,
) -> None:
    """Best-effort: ensure ``min_free_gb`` of VRAM before a GPU run. Never raises.

    Parameters
    ----------
    min_free_gb:
        Headroom target. If at least this much VRAM is already free, this is a
        no-op. 8 GB comfortably covers Foundation SVD + Cellpose on this lab's FOVs.
    enabled:
        Pass ``False`` (e.g. under ``--cpu`` or ``--no-free-gpu``) to skip entirely.
    timeout_s:
        Overall budget for the unload + reclaim wait.
    log_cb:
        Where progress lines go (defaults to ``print``; workspace runs pass their
        own logger).
    """
    if not enabled:
        return
    try:
        if not _cuda_device_present():
            return  # no CUDA device at all — nothing to free

        need = int(min_free_gb * _GB)
        free = _free_gpu_bytes()
        # free is None ⇒ device present but too full to even probe ⇒ contended.
        if free is not None and free >= need:
            return  # already enough headroom

        loaded = _ollama_loaded_models(timeout_s=3.0)
        if not loaded:
            # VRAM is tight but ollama isn't the culprit (or is unreachable).
            free_str = "unprobeable (GPU full)" if free is None else f"{free / _GB:.1f} GB"
            log_cb(
                f"  NOTE: GPU low on memory ({free_str} free, want {min_free_gb:.0f} GB) "
                f"and no ollama model is loaded; another process may hold VRAM. "
                f"Proceeding — stages will fall back to CPU if they OOM.",
            )
            return

        free_str = "unprobeable (GPU full)" if free is None else f"{free / _GB:.1f} GB free"
        log_cb(
            f"  Freeing GPU for run: {free_str}, need {min_free_gb:.0f} GB; "
            f"unloading ollama model(s): {', '.join(loaded)}",
        )
        for model in loaded:
            _ollama_unload(model, timeout_s=5.0)

        # Poll for the memory to actually come back (unload is async-ish). A
        # None reading here means still-too-full-to-probe → keep waiting.
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.5)
            free = _free_gpu_bytes()
            if free is not None and free >= need:
                break

        if free is not None and free >= need:
            log_cb(f"  GPU freed: {free / _GB:.1f} GB now available.")
        else:
            got = "still unprobeable" if free is None else f"{free / _GB:.1f} GB"
            log_cb(
                f"  WARN: GPU still tight after unload ({got} free); "
                f"stages will fall back to CPU if they OOM.",
            )
    except BaseException:  # noqa: BLE001 — preflight is best-effort, never fatal-by-bug
        return
