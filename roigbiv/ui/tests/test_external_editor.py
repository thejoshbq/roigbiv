"""Tests for :mod:`roigbiv.ui.services.external_editor`."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from roigbiv.ui.services import external_editor as ee


# ── find_tiff_editor ───────────────────────────────────────────────────────


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_env_override_absolute_path_wins(tmp_path, monkeypatch):
    fake = _make_executable(tmp_path / "my-editor")
    monkeypatch.setenv("ROIGBIV_TIFF_EDITOR", str(fake))
    monkeypatch.setattr(ee, "which", lambda _name: None)
    monkeypatch.setattr(ee, "_candidate_install_paths", list)
    assert ee.find_tiff_editor() == fake


def test_env_override_resolved_via_path(tmp_path, monkeypatch):
    fake = _make_executable(tmp_path / "fancy-editor")
    monkeypatch.setenv("ROIGBIV_TIFF_EDITOR", "fancy-editor")
    monkeypatch.setattr(
        ee, "which",
        lambda name: str(fake) if name == "fancy-editor" else None,
    )
    monkeypatch.setattr(ee, "_candidate_install_paths", list)
    assert ee.find_tiff_editor() == fake


def test_env_override_bad_path_errors(monkeypatch):
    monkeypatch.setenv("ROIGBIV_TIFF_EDITOR", "/nonexistent/binary-xyz")
    monkeypatch.setattr(ee, "which", lambda _name: None)
    monkeypatch.setattr(ee, "_candidate_install_paths", list)
    with pytest.raises(ee.EditorNotFoundError):
        ee.find_tiff_editor()


def test_path_lookup_finds_fiji(monkeypatch):
    monkeypatch.delenv("ROIGBIV_TIFF_EDITOR", raising=False)
    fake_fiji = "/opt/Fiji.app/ImageJ-linux64"
    monkeypatch.setattr(
        ee, "which",
        lambda name: fake_fiji if name == "fiji" else None,
    )
    monkeypatch.setattr(ee, "_candidate_install_paths", list)
    assert ee.find_tiff_editor() == Path(fake_fiji)


def test_path_lookup_finds_gimp_after_no_fiji(monkeypatch):
    monkeypatch.delenv("ROIGBIV_TIFF_EDITOR", raising=False)
    monkeypatch.setattr(
        ee, "which",
        lambda name: "/usr/bin/gimp" if name == "gimp" else None,
    )
    monkeypatch.setattr(ee, "_candidate_install_paths", list)
    assert ee.find_tiff_editor() == Path("/usr/bin/gimp")


def test_install_path_lookup(tmp_path, monkeypatch):
    monkeypatch.delenv("ROIGBIV_TIFF_EDITOR", raising=False)
    candidate = _make_executable(tmp_path / "ImageJ-linux64")
    monkeypatch.setattr(ee, "which", lambda _name: None)
    monkeypatch.setattr(ee, "_candidate_install_paths", lambda: [candidate])
    assert ee.find_tiff_editor() == candidate


def test_no_editor_raises(monkeypatch):
    monkeypatch.delenv("ROIGBIV_TIFF_EDITOR", raising=False)
    monkeypatch.setattr(ee, "which", lambda _name: None)
    monkeypatch.setattr(ee, "_candidate_install_paths", list)
    with pytest.raises(ee.EditorNotFoundError):
        ee.find_tiff_editor()


# ── resolve_mask_target ────────────────────────────────────────────────────


def test_resolve_prefers_corrections(tmp_path):
    (tmp_path / "merged_masks.tif").write_bytes(b"x")
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    corrected = corrections / "corrected_masks.tif"
    corrected.write_bytes(b"x")
    assert ee.resolve_mask_target(tmp_path) == corrected


def test_resolve_falls_back_to_merged(tmp_path):
    merged = tmp_path / "merged_masks.tif"
    merged.write_bytes(b"x")
    assert ee.resolve_mask_target(tmp_path) == merged


def test_resolve_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ee.resolve_mask_target(tmp_path)


# ── launch_editor ──────────────────────────────────────────────────────────


def test_launch_editor_invokes_popen_detached(tmp_path, monkeypatch):
    captured: dict = {}

    class _FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    fake_editor = _make_executable(tmp_path / "fake-editor")
    target = tmp_path / "merged_masks.tif"
    target.write_bytes(b"x")

    monkeypatch.setenv("ROIGBIV_TIFF_EDITOR", str(fake_editor))
    monkeypatch.setattr(ee.subprocess, "Popen", _FakePopen)

    returned = ee.launch_editor(target)
    assert returned == fake_editor
    assert captured["args"] == [str(fake_editor), str(target)]
    # Stdio must be discarded so the Dash worker isn't blocked.
    assert captured["kwargs"]["stdin"] is ee.subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is ee.subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is ee.subprocess.DEVNULL
    # Detachment flag varies by platform; one of the two must be set.
    assert (
        captured["kwargs"].get("start_new_session") is True
        or captured["kwargs"].get("creationflags", 0) != 0
    )


def test_launch_editor_propagates_not_found(monkeypatch, tmp_path):
    monkeypatch.delenv("ROIGBIV_TIFF_EDITOR", raising=False)
    monkeypatch.setattr(ee, "which", lambda _name: None)
    monkeypatch.setattr(ee, "_candidate_install_paths", list)
    with pytest.raises(ee.EditorNotFoundError):
        ee.launch_editor(tmp_path / "merged_masks.tif")
