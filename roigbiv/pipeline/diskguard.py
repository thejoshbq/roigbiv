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
