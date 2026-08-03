"""Tests for single-frame TIF series detection and assembly in `roigbiv.io`.

Synthetic fixtures only — tiny 4x4 frames, no pipeline run. Covers the three
things that made the previous PrairieView-only implementation miss real data:
pointing the input at the session directory itself, non-PrairieView numbering,
and multi-cycle ordering. Plus the guards that keep the generic pattern from
swallowing directories that are not a series at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.io import assemble_frame_series, discover_tifs, validate_tif


_H = _W = 4


def _frame(path: Path, value: int, shape=(_H, _W), dtype=np.uint16) -> Path:
    """One single-page TIF whose pixels all equal *value* — the value doubles as
    a position marker, so assembled frame order is directly assertable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.full(shape, value, dtype=dtype))
    return path


def _stack(path: Path, n: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((n, _H, _W), dtype=np.uint16))
    return path


def _prairie(directory: Path, n: int, channel: str = "Ch2",
             cycle: int = 1, start: int = 0) -> None:
    for i in range(n):
        _frame(directory / f"TSeries_Cycle{cycle:05d}_{channel}_{i + 1:06d}.ome.tif",
               start + i)


# ─────────────────────────────────────────────────────────────────────────
# PrairieView — the previously supported layout, still works
# ─────────────────────────────────────────────────────────────────────────

def test_prairie_session_below_root_is_assembled(tmp_path: Path):
    _prairie(tmp_path / "session_a", 6)
    tifs = discover_tifs(tmp_path)
    assert [t.name for t in tifs] == ["session_a.tif"]
    assert tifs[0].parent == tmp_path / "_stacks"
    assert validate_tif(tifs[0])[1] == (6, _H, _W)


def test_real_ome_xml_frames_are_read_one_at_a_time(tmp_path: Path):
    # A genuine PrairieView frame carries OME-XML describing the whole series;
    # tifffile will happily expand that into all N frames per file unless every
    # read passes is_ome=False. If that slips, both detection (which checks for
    # a lone 2D frame) and assembly go wrong.
    session = tmp_path / "session_a"
    session.mkdir()
    for i in range(6):
        tifffile.imwrite(
            str(session / f"TSeries_Cycle00001_Ch2_{i + 1:06d}.ome.tif"),
            np.full((_H, _W), i, dtype=np.uint16), ome=True)
    stack = tifffile.imread(str(discover_tifs(tmp_path)[0]))
    assert stack.shape == (6, _H, _W)
    assert list(stack[:, 0, 0]) == list(range(6))


def test_per_frame_files_are_excluded_from_the_result(tmp_path: Path):
    _prairie(tmp_path / "session_a", 6)
    tifs = discover_tifs(tmp_path)
    assert not any("Cycle" in t.name for t in tifs)


def test_assembly_is_cached_across_calls(tmp_path: Path):
    _prairie(tmp_path / "session_a", 6)
    first = discover_tifs(tmp_path)
    stack = first[0]
    mtime = stack.stat().st_mtime_ns
    assert discover_tifs(tmp_path) == first
    assert stack.stat().st_mtime_ns == mtime


def test_dominant_channel_wins(tmp_path: Path):
    session = tmp_path / "session_a"
    _prairie(session, 6, channel="Ch2")
    _prairie(session, 3, channel="Ch1", start=100)
    tifs = discover_tifs(tmp_path)
    assert validate_tif(tifs[0])[1][0] == 6


def test_multi_cycle_series_orders_by_cycle_then_index(tmp_path: Path):
    # The old implementation sorted on the trailing index alone, which
    # interleaves cycles: frame 1 of cycle 2 landed between frames 1 and 2 of
    # cycle 1. Values are assigned in true acquisition order, so a scrambled
    # assembly shows up as a non-monotonic first column.
    session = tmp_path / "session_a"
    _prairie(session, 3, cycle=1, start=0)
    _prairie(session, 3, cycle=2, start=3)
    stack = tifffile.imread(str(discover_tifs(tmp_path)[0]))
    assert list(stack[:, 0, 0]) == [0, 1, 2, 3, 4, 5]


# ─────────────────────────────────────────────────────────────────────────
# Series directory as the input root
# ─────────────────────────────────────────────────────────────────────────

def test_root_itself_may_be_the_session_directory(tmp_path: Path):
    # Previously this returned every per-frame file as its own one-page "FOV",
    # each of which then failed validate_tif.
    _prairie(tmp_path, 6)
    tifs = discover_tifs(tmp_path)
    assert [t.name for t in tifs] == [f"{tmp_path.name}.tif"]
    assert validate_tif(tifs[0])[1] == (6, _H, _W)


def test_nested_sessions_get_distinct_stack_names(tmp_path: Path):
    _prairie(tmp_path / "mouse1" / "day1", 6)
    _prairie(tmp_path / "mouse2" / "day1", 6)
    names = sorted(t.name for t in discover_tifs(tmp_path))
    assert names == ["mouse1_day1.tif", "mouse2_day1.tif"]


# ─────────────────────────────────────────────────────────────────────────
# Generic numbering
# ─────────────────────────────────────────────────────────────────────────

def test_generic_numbered_series_is_assembled(tmp_path: Path):
    for i in range(10):
        _frame(tmp_path / "run" / f"movie_{i:04d}.tif", i)
    tifs = discover_tifs(tmp_path)
    assert [t.name for t in tifs] == ["run.tif"]
    stack = tifffile.imread(str(tifs[0]))
    assert list(stack[:, 0, 0]) == list(range(10))


def test_generic_series_below_min_frames_is_left_alone(tmp_path: Path):
    # Seven numbered frames is under the generic threshold: assembling saves
    # nothing and a false positive here is plausible, so they stay as files.
    for i in range(7):
        _frame(tmp_path / "run" / f"movie_{i:04d}.tif", i)
    assert len(discover_tifs(tmp_path)) == 7


def test_numbered_multi_frame_stacks_are_not_treated_as_frames(tmp_path: Path):
    # Chunked acquisitions (foo_0001.tif ... each holding many frames) match the
    # generic *name* pattern but are not single-page, so they must survive as
    # independent inputs rather than being concatenated.
    for i in range(10):
        _stack(tmp_path / "run" / f"movie_{i:04d}.tif")
    assert len(discover_tifs(tmp_path)) == 10


def test_two_prefixes_in_one_directory_are_refused(tmp_path: Path):
    for i in range(10):
        _frame(tmp_path / "run" / f"mouse1_{i:04d}.tif", i)
        _frame(tmp_path / "run" / f"mouse2_{i:04d}.tif", i)
    assert len(discover_tifs(tmp_path)) == 20


def test_duplicate_frame_positions_are_refused(tmp_path: Path):
    # Interleaved channels sharing one prefix: concatenating them would produce
    # a silently scrambled stack, so the directory is left as loose files.
    for i in range(10):
        _frame(tmp_path / "run" / f"mov_ChanA_{i:04d}.tif", i)
        _frame(tmp_path / "run" / f"mov_ChanB_{i:04d}.tif", i)
    assert len(discover_tifs(tmp_path)) == 20


def test_inconsistent_frame_shape_is_refused(tmp_path: Path):
    for i in range(10):
        _frame(tmp_path / "run" / f"movie_{i:04d}.tif", i)
    _frame(tmp_path / "run" / "movie_0009.tif", 9, shape=(8, 8))
    assert len(discover_tifs(tmp_path)) == 10


# ─────────────────────────────────────────────────────────────────────────
# Interaction with ordinary stacks and pipeline outputs
# ─────────────────────────────────────────────────────────────────────────

def test_plain_stacks_alongside_a_session_are_both_returned(tmp_path: Path):
    _prairie(tmp_path / "session_a", 6)
    _stack(tmp_path / "fov1_mc.tif")
    names = sorted(t.name for t in discover_tifs(tmp_path))
    assert names == ["fov1_mc.tif", "session_a.tif"]


def test_a_stack_beside_the_frames_survives(tmp_path: Path):
    # Only the files that matched a series pattern are consumed. Excluding the
    # whole directory instead would silently drop this stack now that the input
    # root itself can be the session directory.
    _prairie(tmp_path, 6)
    _stack(tmp_path / "fov2_mc.tif")
    names = {t.name for t in discover_tifs(tmp_path)}
    assert names == {f"{tmp_path.name}.tif", "fov2_mc.tif"}


def test_secondary_channel_frames_are_still_consumed(tmp_path: Path):
    # The non-dominant channel is not assembled, but it must not come back as
    # loose one-page files either — each would fail validate_tif downstream.
    session = tmp_path / "session_a"
    _prairie(session, 6, channel="Ch2")
    _prairie(session, 3, channel="Ch1", start=100)
    assert [t.name for t in discover_tifs(tmp_path)] == ["session_a.tif"]


def test_symlinked_session_is_followed(tmp_path: Path):
    # Acquisition trees are routinely symlinked into a working directory. A
    # plain os.walk skips them, and pathlib's ** does not descend into them
    # either, so without followlinks the session is invisible in both passes.
    real = tmp_path / "acquisition" / "session_a"
    _prairie(real, 6)
    root = tmp_path / "work"
    root.mkdir()
    (root / "linked").symlink_to(real, target_is_directory=True)
    assert [t.name for t in discover_tifs(root)] == ["linked.tif"]


def test_symlink_loop_terminates(tmp_path: Path):
    # followlinks=True recurses forever on a cycle; the realpath set is what
    # stops it. The link sits in a plain subdirectory, not the session — a
    # detected series prunes its own branch and would mask the loop.
    _prairie(tmp_path / "session_a", 6)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert [t.name for t in discover_tifs(tmp_path)] == ["session_a.tif"]


def test_output_tree_is_not_scanned_for_series(tmp_path: Path):
    _stack(tmp_path / "fov1.tif")
    _prairie(tmp_path / "output" / "fov1" / "stage1", 6)
    assert [t.name for t in discover_tifs(tmp_path)] == ["fov1.tif"]


# ─────────────────────────────────────────────────────────────────────────
# assemble_frame_series
# ─────────────────────────────────────────────────────────────────────────

def test_explicit_channel_selection(tmp_path: Path):
    session = tmp_path / "session_a"
    _prairie(session, 6, channel="Ch2")
    _prairie(session, 3, channel="Ch1", start=100)
    out = assemble_frame_series(session, tmp_path / "ch1.tif", channel="Ch1")
    stack = tifffile.imread(str(out))
    assert list(stack[:, 0, 0]) == [100, 101, 102]


def test_missing_channel_raises(tmp_path: Path):
    session = tmp_path / "session_a"
    _prairie(session, 6, channel="Ch2")
    with pytest.raises(ValueError, match="Ch9"):
        assemble_frame_series(session, tmp_path / "x.tif", channel="Ch9")


def test_no_series_raises(tmp_path: Path):
    session = tmp_path / "session_a"
    _stack(session / "just_a_stack.tif")
    with pytest.raises(ValueError, match="No single-frame TIF series"):
        assemble_frame_series(session, tmp_path / "x.tif")


def test_partial_stack_is_not_left_behind_on_failure(tmp_path: Path, monkeypatch):
    session = tmp_path / "session_a"
    _prairie(session, 6)
    out = tmp_path / "out.tif"

    def _boom(*_a, **_k):
        raise OSError("no space left on device")

    monkeypatch.setattr(tifffile, "imread", _boom)
    with pytest.raises(OSError):
        assemble_frame_series(session, out)
    assert not out.exists()
    assert not out.with_suffix(".tmp.tif").exists()
