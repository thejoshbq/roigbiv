"""End-to-end test for :mod:`roigbiv.pipeline.discovery_extract`.

Builds a tiny synthetic FOV: fake Suite2p ``ops.npy``/``data.bin`` under
``{stem}/suite2p/plane0/`` — the layout ``run_suite2p_fov`` actually writes
(see ``roigbiv/suite2p.py::run_suite2p_fov``, ``foundation.py``'s
``s2p_root = output_dir / stem``) — plus a ``merged_masks.tif`` label image
(no primary ``traces/`` bundle, no HITL corrections — the precondition this
module exists for), then verifies:

  * extraction with no prior bundle writes to the primary ``traces/`` location
  * extraction against a FOV that already has a primary bundle writes a
    ``traces/discovery-{hash}/`` sibling and never touches the primary
  * a second, unchanged invocation is idempotent
  * a missing ``merged_masks.tif`` raises ``FileNotFoundError``
  * a present ``registry_match.json`` populates ``global_cell_id`` per row
  * a flattened ``suite2p/plane0/`` (no ``{stem}/`` nesting) also resolves —
    see ``resume.py::_suite2p_plane_dir`` for the same two-layout ambiguity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.discovery_extract import extract_from_merged_masks

H, W, T = 16, 16, 120


def _make_suite2p(fov_out: Path, fs: float = 7.5, *, flattened: bool = False) -> Path:
    plane0 = ((fov_out / "suite2p" / "plane0") if flattened
              else (fov_out / fov_out.name / "suite2p" / "plane0"))
    plane0.mkdir(parents=True)
    data_bin = plane0 / "data.bin"
    rng = np.random.default_rng(0)
    mm = np.memmap(str(data_bin), dtype=np.int16, mode="w+", shape=(T, H, W))
    mm[:] = rng.integers(-2000, 2000, size=(T, H, W), dtype=np.int16)
    mm.flush()
    del mm
    ops = {"Ly": H, "Lx": W, "fs": fs, "nframes": T}
    np.save(str(plane0 / "ops.npy"), ops)
    return data_bin


def _write_merged_masks(fov_out: Path, extra_label: bool = False) -> Path:
    label_img = np.zeros((H, W), dtype=np.uint16)
    label_img[0:4, 0:4] = 1
    label_img[6:10, 6:10] = 2
    if extra_label:
        label_img[10:14, 10:14] = 3
    path = fov_out / "merged_masks.tif"
    tifffile.imwrite(str(path), label_img)
    return path


def test_extract_writes_primary_when_none_exists(tmp_path: Path):
    fov_out = tmp_path / "fov1"
    fov_out.mkdir()
    _make_suite2p(fov_out)
    _write_merged_masks(fov_out)

    bundle = extract_from_merged_masks(fov_out, stats=("median", "mode"))
    assert bundle == fov_out / "traces"

    for fname in ("traces.npy", "traces_raw.npy", "traces_neuropil.npy",
                  "traces_median.npy", "traces_median_raw.npy",
                  "traces_median_neuropil.npy", "traces_mode.npy",
                  "traces_mode_raw.npy", "traces_mode_neuropil.npy",
                  "traces_meta.json"):
        assert (bundle / fname).exists(), f"missing {fname}"

    meta = json.loads((bundle / "traces_meta.json").read_text())
    assert meta["source"] == "discovery"
    assert meta["n_rois"] == 2
    assert meta["stats"] == ["mean", "median", "mode"]
    assert meta["files"]["median"] == "traces_median.npy"
    assert meta["corrections_rev"] is None
    for r in meta["rois"]:
        assert "global_cell_id" not in r  # no registry_match.json yet

    arr = np.load(bundle / "traces_median.npy")
    assert arr.shape == (2, T)


def test_extract_writes_sibling_and_preserves_primary(tmp_path: Path):
    fov_out = tmp_path / "fov2"
    fov_out.mkdir()
    _make_suite2p(fov_out)
    _write_merged_masks(fov_out)

    primary = extract_from_merged_masks(fov_out)
    assert primary == fov_out / "traces"
    primary_bytes = (primary / "traces.npy").read_bytes()

    # Change merged_masks.tif's content so the ROI-set hash differs.
    _write_merged_masks(fov_out, extra_label=True)

    sibling = extract_from_merged_masks(fov_out, stats=("median",))
    assert sibling.parent == fov_out / "traces"
    assert sibling.name.startswith("discovery-")
    assert (primary / "traces.npy").read_bytes() == primary_bytes  # untouched

    meta = json.loads((sibling / "traces_meta.json").read_text())
    assert meta["n_rois"] == 3
    assert meta["corrections_rev"]


def test_extract_idempotent_sibling(tmp_path: Path):
    fov_out = tmp_path / "fov3"
    fov_out.mkdir()
    _make_suite2p(fov_out)
    _write_merged_masks(fov_out)

    extract_from_merged_masks(fov_out)  # seed a primary bundle

    first = extract_from_merged_masks(fov_out, stats=("mode",))
    first_mtime = (first / "traces_mode.npy").stat().st_mtime_ns

    second = extract_from_merged_masks(fov_out, stats=("mode",))
    assert second == first
    assert (second / "traces_mode.npy").stat().st_mtime_ns == first_mtime


def test_extract_resolves_flattened_suite2p_layout(tmp_path: Path):
    """Defensive fallback: a FOV whose suite2p/plane0 has no {stem}/ nesting
    (see resume.py::_suite2p_plane_dir) must still resolve."""
    fov_out = tmp_path / "fov6"
    fov_out.mkdir()
    _make_suite2p(fov_out, flattened=True)
    _write_merged_masks(fov_out)

    bundle = extract_from_merged_masks(fov_out)
    assert (bundle / "traces.npy").exists()


def test_extract_missing_merged_masks_raises(tmp_path: Path):
    fov_out = tmp_path / "fov4"
    fov_out.mkdir()
    _make_suite2p(fov_out)

    with pytest.raises(FileNotFoundError):
        extract_from_merged_masks(fov_out)


def test_extract_picks_up_registry_match(tmp_path: Path):
    fov_out = tmp_path / "fov5"
    fov_out.mkdir()
    _make_suite2p(fov_out)
    _write_merged_masks(fov_out)

    report = {
        "decision": "auto_match",
        "session_id": "sess-1",
        "fov_id": "fov-1",
        "cell_assignments": [
            {"local_label_id": 1, "global_cell_id": "gid-1", "match_kind": "matched"},
            {"local_label_id": 2, "global_cell_id": "gid-2", "match_kind": "new"},
        ],
    }
    (fov_out / "registry_match.json").write_text(json.dumps(report))

    bundle = extract_from_merged_masks(fov_out)
    meta = json.loads((bundle / "traces_meta.json").read_text())
    assert meta["session_id"] == "sess-1"
    assert meta["fov_id"] == "fov-1"
    by_label = {r["local_label_id"]: r for r in meta["rois"]}
    assert by_label[1]["global_cell_id"] == "gid-1"
    assert by_label[2]["global_cell_id"] == "gid-2"
