"""Tests for BranchView dataclass and FOVData.branches integration.

Tests cover:
- BranchView construction with required and optional args
- FOVData backward compatibility (branches field defaults to [])
- FOVData integration with BranchView
- branches.json manifest schema round-trip
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from roigbiv.pipeline.foundation import _build_branches_manifest
from roigbiv.pipeline.types import BranchView, FOVData


def test_branchview_required_args_only():
    """BranchView with required args only defaults optional fields correctly."""
    branch_name = "raw"
    movie_view = Path("/tmp/test_movie.bin")

    branch = BranchView(branch_name=branch_name, movie_view=movie_view)

    assert branch.branch_name == branch_name
    assert branch.movie_view == movie_view
    assert branch.summary_images == {}, "summary_images should default to {}"
    assert branch.provenance == {}, "provenance should default to {}"
    assert branch.is_denoised is False, "is_denoised should default to False"


def test_branchview_all_args_populated():
    """BranchView with all args stores them correctly."""
    branch_name = "denoised"
    movie_view = Path("/tmp/test_movie_denoised.bin")
    summary_images = {
        "mean_M": np.zeros((256, 256), dtype=np.float32),
        "mean_S": np.ones((256, 256), dtype=np.float32),
        "vcorr_S": np.random.rand(256, 256).astype(np.float32),
    }
    provenance = {
        "k_used": 30,
        "fs": 7.5,
        "motion_correction_backend": "phasecorr",
    }
    is_denoised = True

    branch = BranchView(
        branch_name=branch_name,
        movie_view=movie_view,
        summary_images=summary_images,
        provenance=provenance,
        is_denoised=is_denoised,
    )

    assert branch.branch_name == branch_name
    assert branch.movie_view == movie_view
    assert branch.summary_images is summary_images
    assert branch.provenance is provenance
    assert branch.is_denoised is True


def test_fovdata_backward_compat_no_branches_kwarg():
    """FOVData created without branches kwarg defaults to empty list."""
    raw_path = Path("/tmp/raw.tif")
    output_dir = Path("/tmp/output")
    data_bin_path = Path("/tmp/output/data.bin")
    shape = (100, 512, 512)

    fov = FOVData(
        raw_path=raw_path,
        output_dir=output_dir,
        data_bin_path=data_bin_path,
        shape=shape,
    )

    assert fov.branches == [], "branches field should default to empty list"
    assert fov.raw_path == raw_path
    assert fov.output_dir == output_dir
    assert fov.data_bin_path == data_bin_path
    assert fov.shape == shape


def test_fovdata_with_single_branchview():
    """FOVData stores a single BranchView correctly."""
    raw_path = Path("/tmp/raw.tif")
    output_dir = Path("/tmp/output")
    data_bin_path = Path("/tmp/output/data.bin")
    shape = (100, 512, 512)

    branch = BranchView(
        branch_name="raw",
        movie_view=data_bin_path,
        summary_images={"mean_M": np.zeros((512, 512))},
        provenance={"source": "foundation"},
        is_denoised=False,
    )

    fov = FOVData(
        raw_path=raw_path,
        output_dir=output_dir,
        data_bin_path=data_bin_path,
        shape=shape,
        branches=[branch],
    )

    assert len(fov.branches) == 1
    assert fov.branches[0] is branch
    assert fov.branches[0].branch_name == "raw"
    assert fov.branches[0].movie_view == data_bin_path
    assert fov.branches[0].is_denoised is False


def test_fovdata_with_multiple_branchviews():
    """FOVData stores multiple BranchViews in order."""
    raw_path = Path("/tmp/raw.tif")
    output_dir = Path("/tmp/output")
    data_bin_path = Path("/tmp/output/data.bin")
    shape = (100, 512, 512)

    raw_branch = BranchView(
        branch_name="raw",
        movie_view=data_bin_path,
        is_denoised=False,
    )
    denoised_branch = BranchView(
        branch_name="denoised",
        movie_view=Path("/tmp/output/data_denoised.bin"),
        is_denoised=True,
    )

    fov = FOVData(
        raw_path=raw_path,
        output_dir=output_dir,
        data_bin_path=data_bin_path,
        shape=shape,
        branches=[raw_branch, denoised_branch],
    )

    assert len(fov.branches) == 2
    assert fov.branches[0].branch_name == "raw"
    assert fov.branches[1].branch_name == "denoised"
    assert fov.branches[0].is_denoised is False
    assert fov.branches[1].is_denoised is True


def test_branches_json_manifest_schema():
    """_build_branches_manifest (the function foundation.py calls to write branches.json)
    produces the expected schema and round-trips through JSON cleanly."""
    branch = BranchView(
        branch_name="raw",
        movie_view=Path("/tmp/output/data.bin"),
        summary_images={"mean_M": np.zeros((8, 8), dtype=np.float32)},
        provenance={"k_used": 30, "fs": 7.5, "motion_correction_backend": "phasecorr"},
        is_denoised=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        summary_dir = Path(temp_dir) / "summary"
        branches_manifest = _build_branches_manifest([branch], summary_dir)

        json_path = Path(temp_dir) / "branches.json"
        json_path.write_text(json.dumps(branches_manifest, indent=2))
        loaded_manifest = json.loads(json_path.read_text())

    assert len(loaded_manifest) == 1
    entry = loaded_manifest[0]

    expected_keys = {"branch_name", "is_denoised", "provenance", "summary_image_paths"}
    assert set(entry.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(entry.keys())}"

    assert isinstance(entry["branch_name"], str)
    assert isinstance(entry["is_denoised"], bool)
    assert isinstance(entry["provenance"], dict)
    assert isinstance(entry["summary_image_paths"], dict)

    assert entry["branch_name"] == "raw"
    assert entry["is_denoised"] is False
    assert entry["provenance"]["k_used"] == 30
    assert entry["summary_image_paths"]["mean_M"] == str(summary_dir / "mean_M.tif")


def test_branches_json_sparse_summary_image_paths():
    """_build_branches_manifest only includes non-None summary images in summary_image_paths."""
    branch = BranchView(
        branch_name="raw",
        movie_view=Path("/tmp/output/data.bin"),
        summary_images={
            "mean_M": np.zeros((8, 8)),   # included
            "mean_S": None,                # skipped
            "max_S": np.ones((8, 8)),     # included
            "std_S": None,                 # skipped
            "vcorr_S": np.zeros((8, 8)),  # included
            "mean_L": None,                # skipped
            "dog_map": np.zeros((8, 8)),  # included
        },
    )

    manifest = _build_branches_manifest([branch], Path("/tmp/output/summary"))
    summary_image_paths = manifest[0]["summary_image_paths"]

    assert len(summary_image_paths) == 4, "Only 4 of 7 images are non-None"
    assert "mean_M" in summary_image_paths
    assert "max_S" in summary_image_paths
    assert "vcorr_S" in summary_image_paths
    assert "dog_map" in summary_image_paths
    assert "mean_S" not in summary_image_paths
    assert "std_S" not in summary_image_paths
    assert "mean_L" not in summary_image_paths
