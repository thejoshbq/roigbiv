"""Regression tests for Cellpose model-path resolution.

The previous behavior (`_resolve_model_path` returning the bare relative
string when the path didn't exist from cwd) silently fell through to stock
cyto3 because Cellpose 3.x doesn't raise on missing pretrained_model paths.
These tests pin the new contract: built-ins pass through, valid paths
resolve, and unresolvable paths raise FileNotFoundError.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from roigbiv.pipeline.stage1 import _resolve_model_path, _CELLPOSE_BUILTINS
from roigbiv.pipeline.types import PipelineConfig


def test_builtin_names_pass_through():
    for name in ("cyto3", "cpsam", "nuclei"):
        assert _resolve_model_path(name) == name


def test_default_config_path_is_absolute_and_exists():
    cfg = PipelineConfig()
    p = Path(cfg.cellpose_model)
    assert p.is_absolute(), f"default cellpose_model is not absolute: {cfg.cellpose_model}"
    assert p.exists(), f"default cellpose_model does not exist: {cfg.cellpose_model}"


def test_default_config_resolves_from_arbitrary_cwd(tmp_path):
    """Pipeline run from any cwd must still load the fine-tuned model."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        cfg = PipelineConfig()
        resolved = _resolve_model_path(cfg.cellpose_model)
        assert Path(resolved).exists(), (
            f"PipelineConfig default did not resolve from cwd={tmp_path}: {resolved}"
        )
    finally:
        os.chdir(original_cwd)


def test_relative_default_string_resolves_to_repo_root(tmp_path):
    """A user passing the legacy relative string from any cwd should still
    resolve to the same fine-tuned model rather than silently falling back."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        resolved = _resolve_model_path("models/deployed/current_model")
        assert Path(resolved).exists(), resolved
        assert resolved.endswith("models/deployed/current_model")
    finally:
        os.chdir(original_cwd)


def test_unresolvable_path_raises():
    with pytest.raises(FileNotFoundError, match="not resolvable"):
        _resolve_model_path("definitely/not/a/real/path")


def test_unresolvable_path_error_lists_candidates(tmp_path):
    """The error message should help the user diagnose the failure."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        with pytest.raises(FileNotFoundError) as excinfo:
            _resolve_model_path("nope/nope")
        msg = str(excinfo.value)
        # Should reference both candidate locations
        assert "nope/nope" in msg
        assert "built-in" in msg.lower()
    finally:
        os.chdir(original_cwd)
