"""Tests for :mod:`roigbiv.pipeline.workspace`."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.workspace import (
    WorkspacePaths,
    configure_registry_env,
    resolve_workspace,
)


def _write_fake_tif(path: Path, T: int = 5, H: int = 8, W: int = 8) -> None:
    """Write a minimum-viable 3-D TIF (validate_tif only needs 3-D)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.zeros((T, H, W), dtype=np.int16)
    tifffile.imwrite(str(path), stack)


def test_resolve_single_file(tmp_path: Path) -> None:
    tif = tmp_path / "sampleA_mc.tif"
    _write_fake_tif(tif)
    ws = resolve_workspace(tif)

    assert ws.input_root == tmp_path.resolve()
    assert ws.tifs == (tif.resolve(),)
    assert ws.output_root == tmp_path.resolve() / "output"
    assert ws.db_path == tmp_path.resolve() / "registry.db"
    assert ws.blob_root == tmp_path.resolve() / "registry_blobs"
    assert ws.db_dsn == f"sqlite:///{ws.db_path}"


def test_resolve_directory_discovers_tifs(tmp_path: Path) -> None:
    for name in ("foo_mc.tif", "bar.tif"):
        _write_fake_tif(tmp_path / name)
    ws = resolve_workspace(tmp_path)
    discovered = {t.name for t in ws.tifs}
    assert discovered == {"foo_mc.tif", "bar.tif"}


def test_resolve_directory_excludes_output_subtree(tmp_path: Path) -> None:
    _write_fake_tif(tmp_path / "real.tif")
    _write_fake_tif(tmp_path / "output" / "stale.tif")
    ws = resolve_workspace(tmp_path)
    names = {t.name for t in ws.tifs}
    assert names == {"real.tif"}, (
        "output/ subtree contents should be filtered so pipeline-produced "
        "TIFFs are never re-ingested as inputs on a second run."
    )


def test_resolve_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_workspace(tmp_path / "does-not-exist")


def test_resolve_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_workspace(tmp_path)


def test_configure_registry_env_sets_expected_vars(tmp_path: Path, monkeypatch) -> None:
    _write_fake_tif(tmp_path / "only.tif")
    ws = resolve_workspace(tmp_path)

    # Pre-populate to prove configure_ rewrites, not appends.
    monkeypatch.setenv("ROIGBIV_REGISTRY_DSN", "sqlite:///somewhere-else.db")
    monkeypatch.setenv("ROIGBIV_BLOB_ROOT", "/nope")
    monkeypatch.setenv("ROIGBIV_CALIBRATION_PATH", "/nope.json")

    configure_registry_env(ws)

    assert os.environ["ROIGBIV_REGISTRY_DSN"] == ws.db_dsn
    assert os.environ["ROIGBIV_BLOB_ROOT"] == str(ws.blob_root)
    assert os.environ["ROIGBIV_CALIBRATION_PATH"] == str(ws.calibration_path)

    assert ws.output_root.exists()
    assert ws.blob_root.exists()


def test_configure_registry_env_is_idempotent(tmp_path: Path) -> None:
    _write_fake_tif(tmp_path / "only.tif")
    ws = resolve_workspace(tmp_path)
    configure_registry_env(ws)
    configure_registry_env(ws)
    assert os.environ["ROIGBIV_REGISTRY_DSN"] == ws.db_dsn
