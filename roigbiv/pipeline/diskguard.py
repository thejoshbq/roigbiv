"""Disk-safety guard for large on-disk allocations.

The pipeline writes multi-gigabyte memmaps (Stage 4 detrend/bandpass temps,
svd_factors). Writing into a ``mode="w+"`` memmap whose backing file cannot be
extended (filesystem full) faults with **SIGBUS** on the dirty page — an
uncatchable signal that kills the process with no Python traceback, exactly the
silent-exit failure this module exists to prevent.

``ensure_free_space`` converts that into a deterministic, catchable failure:
  1. An ``os.statvfs`` pre-check that raises a clear ``RuntimeError`` naming the
     bytes needed vs. available, and
  2. ``os.posix_fallocate`` to physically reserve the blocks up front, so a
     later out-of-space condition surfaces as ``OSError(ENOSPC)`` here (where it
     is catchable) instead of a SIGBUS on a subsequent mmap store.

Open the memmap with ``mode="r+"`` over the pre-allocated file (not ``"w+"``)
so every page already has backing blocks.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Optional

# Require a little headroom beyond the exact byte count so a write that lands
# right at the filesystem boundary (plus FS metadata) still succeeds.
_SAFETY_FACTOR = 1.05


def ensure_free_space(path: Path, nbytes: int, label: str = "allocation") -> None:
    """Reserve ``nbytes`` for a file about to be written at ``path``.

    Parameters
    ----------
    path   : target file path (its parent directory is the filesystem probed).
    nbytes : number of bytes the file will occupy.
    label  : human-readable description used in error messages.

    Raises
    ------
    RuntimeError
        If the filesystem has less than ``nbytes * 1.05`` free.
    OSError
        If ``posix_fallocate`` fails to reserve the space (e.g. ENOSPC) — the
        target file is removed before re-raising so a half-reserved stub is not
        left on disk.
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    needed = int(nbytes * _SAFETY_FACTOR)
    st = os.statvfs(str(parent))
    free = st.f_bavail * st.f_frsize
    if free < needed:
        raise RuntimeError(
            f"Insufficient disk space for {label}: need "
            f"{nbytes / 1e9:.1f} GB (+5% margin = {needed / 1e9:.1f} GB), "
            f"only {free / 1e9:.1f} GB free at {parent}. "
            f"Free disk space and re-run."
        )

    # Physically reserve the blocks so a later mmap store cannot SIGBUS.
    # posix_fallocate is Linux-only; on platforms without it the statvfs
    # pre-check above is the only guard (still better than nothing).
    if not hasattr(os, "posix_fallocate"):
        return

    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.posix_fallocate(fd, 0, int(nbytes))
    except OSError as exc:
        os.close(fd)
        # Don't leave a partially-reserved stub behind.
        try:
            path.unlink()
        except OSError:
            pass
        if exc.errno == errno.ENOSPC:
            raise OSError(
                errno.ENOSPC,
                f"Out of disk space reserving {nbytes / 1e9:.1f} GB for "
                f"{label} at {path}",
            ) from exc
        raise
    else:
        os.close(fd)


def free_bytes(path: Path) -> int:
    """Bytes available to an unprivileged writer at ``path``'s filesystem."""
    parent = Path(path)
    if not parent.is_dir():
        parent = parent.parent
    parent.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(str(parent))
    return st.f_bavail * st.f_frsize


def preflight_disk_budget(
    output_dir: Path,
    *,
    data_bin_bytes: int,
    stage4_temp_bytes: int = 0,
    label: str = "pipeline run",
) -> Optional[str]:
    """Fail-fast disk check at pipeline entry, before any minutes-long work.

    Two thresholds, chosen to minimise false refusals (stages free their
    transients between steps, so the naive sum-of-all-stages over-estimates):

    * **Hard floor** — ``data_bin_bytes`` (the int16 Suite2p substrate that
      *must* persist for the whole run). Below this the run cannot even lay
      down its foundation; raise ``RuntimeError`` now instead of SIGBUS-ing or
      half-writing 20 minutes in.
    * **Soft high-water** — ``data_bin_bytes + stage4_temp_bytes`` (data.bin
      held resident while Stage 4 writes its detrend/bandpass float32 temps).
      Below this the run *probably* fits but could fail late; return a warning
      string so the caller can surface it without aborting.

    Returns ``None`` if there is ample space, or a human-readable warning
    string for the soft case. Raises ``RuntimeError`` for the hard case.
    """
    output_dir = Path(output_dir)
    free = free_bytes(output_dir)
    hard = int(data_bin_bytes * _SAFETY_FACTOR)
    soft = int((data_bin_bytes + stage4_temp_bytes) * _SAFETY_FACTOR)

    if free < hard:
        raise RuntimeError(
            f"Insufficient disk for {label}: data.bin alone needs "
            f"{data_bin_bytes / 1e9:.1f} GB (+5% = {hard / 1e9:.1f} GB), "
            f"only {free / 1e9:.1f} GB free at {output_dir}. "
            f"Free disk space and re-run."
        )
    if free < soft:
        return (
            f"Low disk for {label}: {free / 1e9:.1f} GB free at {output_dir}; "
            f"peak transient (data.bin + Stage 4 temps) is ~{soft / 1e9:.1f} GB. "
            f"The run may fail late in Stage 4 — free space or pass "
            f"--no-stage-4 if you hit ENOSPC."
        )
    return None
