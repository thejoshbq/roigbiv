"""GPU device capability helpers."""
from __future__ import annotations

from typing import Optional


def cuda_unavailable_reason() -> Optional[str]:
    """Classify *why* CUDA can't be used, or ``None`` if it can.

    ``torch.cuda.is_available()`` returns True whenever the driver sees the
    device, even if the GPU is out of memory or the installed PyTorch build
    lacks kernel images for that compute capability. This probe forces a
    synchronous kernel launch and maps the failure to a stable reason code so
    callers can give an accurate diagnostic instead of always blaming the build:

    * ``None``          — CUDA is usable.
    * ``"no_cuda"``     — no CUDA device the driver can see.
    * ``"oom"``         — device present but out of memory (e.g. another process
                          holds the VRAM); transient, not a build problem.
    * ``"sm_mismatch"`` — PyTorch build has no kernel image for this GPU's
                          compute capability (e.g. sm_120 on a cu126 wheel).
    * ``"error: ..."``  — any other launch failure (short message).
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 — torch missing/broken ⇒ no CUDA
        return "no_cuda"

    try:
        if not torch.cuda.is_available():
            return "no_cuda"
    except Exception:  # noqa: BLE001
        return "no_cuda"

    try:
        x = torch.zeros(4, device="cuda")
        _ = (x + 1).sum().item()  # kernel launch + sync
        return None
    except Exception as exc:  # noqa: BLE001
        # OutOfMemoryError is a subclass of RuntimeError; check it first.
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        msg = str(exc).lower()
        if (oom_type and isinstance(exc, oom_type)) or "out of memory" in msg:
            return "oom"
        if "no kernel image" in msg or "cuda capability" in msg or "kernel image is available" in msg:
            return "sm_mismatch"
        short = str(exc).splitlines()[0][:120] if str(exc) else type(exc).__name__
        return f"error: {short}"


def cuda_compute_capable() -> bool:
    """Return True only if CUDA can actually launch a compute kernel.

    Thin bool wrapper over :func:`cuda_unavailable_reason` — kept for the many
    call sites that only need a yes/no. Use ``cuda_unavailable_reason()`` when
    you want to report *why* the GPU is unusable.
    """
    return cuda_unavailable_reason() is None
