"""Tests for roigbiv.benchmark.provenance — git + config hash helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path

from roigbiv.benchmark.provenance import compute_config_hash, get_git_commit, is_git_dirty


def test_get_git_commit_returns_full_sha_in_real_repo():
    commit = get_git_commit(cwd=Path(__file__).parent)
    assert commit is not None
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_get_git_commit_returns_none_outside_repo(tmp_path: Path):
    assert get_git_commit(cwd=tmp_path) is None


def test_get_git_commit_returns_none_when_git_unavailable(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert get_git_commit() is None


def test_is_git_dirty_returns_bool_in_real_repo():
    result = is_git_dirty(cwd=Path(__file__).parent)
    assert result in (True, False)


def test_is_git_dirty_returns_none_outside_repo(tmp_path: Path):
    assert is_git_dirty(cwd=tmp_path) is None


def test_compute_config_hash_deterministic():
    cfg = {"b": 2, "a": 1}
    assert compute_config_hash(cfg) == compute_config_hash({"a": 1, "b": 2})


def test_compute_config_hash_sensitive_to_value_changes():
    assert compute_config_hash({"a": 1}) != compute_config_hash({"a": 2})


def test_compute_config_hash_has_sha256_prefix():
    h = compute_config_hash({"a": 1})
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64
