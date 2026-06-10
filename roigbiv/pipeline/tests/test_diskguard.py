"""Tests for :func:`roigbiv.pipeline.diskguard.ensure_free_space`.

The guard converts a would-be SIGBUS (mmap store into a full filesystem) into a
deterministic, catchable failure: a clear ``RuntimeError`` from the statvfs
pre-check, and ``posix_fallocate`` reservation so a later out-of-space surfaces
as ``OSError`` here rather than an uncatchable signal later.
"""
from __future__ import annotations

import os
import types

import pytest

from roigbiv.pipeline.diskguard import ensure_free_space


def test_raises_when_insufficient_space(tmp_path, monkeypatch):
    target = tmp_path / "big.dat"

    # Pretend the filesystem has only 1000 bytes free.
    fake = types.SimpleNamespace(f_bavail=1000, f_frsize=1, f_bfree=1000)
    monkeypatch.setattr(os, "statvfs", lambda p: fake)

    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        ensure_free_space(target, nbytes=10_000, label="unit-test")
    # No stub file left behind (we never opened it).
    assert not target.exists()


def test_reserves_space_when_available(tmp_path):
    target = tmp_path / "reserved.dat"
    nbytes = 4096
    ensure_free_space(target, nbytes=nbytes, label="unit-test")
    # posix_fallocate should have physically reserved the file at full size.
    if hasattr(os, "posix_fallocate"):
        assert target.exists()
        assert target.stat().st_size == nbytes


def test_message_names_label_and_sizes(tmp_path, monkeypatch):
    target = tmp_path / "x.dat"
    fake = types.SimpleNamespace(f_bavail=0, f_frsize=4096, f_bfree=0)
    monkeypatch.setattr(os, "statvfs", lambda p: fake)
    with pytest.raises(RuntimeError) as exc:
        ensure_free_space(target, nbytes=2_000_000_000, label="residual_S")
    msg = str(exc.value)
    assert "residual_S" in msg
    assert "GB" in msg
