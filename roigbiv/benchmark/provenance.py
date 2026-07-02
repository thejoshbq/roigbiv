"""Git and config provenance helpers for benchmark run records (issue #32).

No git-commit-hash helper exists elsewhere in the repo; the only precedent
(experiments/summary_fork/run_summary_fork.py) captures branch name, not SHA.
`compute_config_hash` mirrors pipeline/resume.py::compute_cfg_fingerprint's
canonicalize -> json.dumps(sort_keys=True) -> sha256 pattern, but is generic
over any JSON-able dict (not PipelineConfig-specific) to keep roigbiv/benchmark/
decoupled from roigbiv/pipeline/ internals — callers (e.g. the future #28
runner) pass cfg.summary_for_log() in directly.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Full 40-char HEAD SHA, or None if not in a git repo / git unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def is_git_dirty(cwd: Optional[Path] = None) -> Optional[bool]:
    """True if the working tree has uncommitted changes, None if undetermined."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def compute_config_hash(cfg_summary: dict) -> str:
    """SHA-256 over a canonicalized JSON dict.

    Mirrors resume.py's fingerprint pattern (sort_keys=True -> sha256 ->
    "sha256:" prefix) but is config-only — no input-file stat component,
    since that's resume.py's resume-specific concern.
    """
    cfg_json = json.dumps(cfg_summary, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
