"""GPU device capability helpers."""
from __future__ import annotations


def cuda_compute_capable() -> bool:
    """Return True only if CUDA can actually launch a compute kernel.

    torch.cuda.is_available() returns True whenever the driver sees the device,
    even if the installed PyTorch build lacks kernel images for that GPU's
    compute capability (e.g. sm_120 on RTX 5080 vs a cu126 build compiled for
    sm_50–sm_90). This probe forces a synchronous kernel launch and treats any
    exception as unusable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        x = torch.zeros(4, device="cuda")
        _ = (x + 1).sum().item()  # kernel launch + sync
        return True
    except Exception:
        return False
