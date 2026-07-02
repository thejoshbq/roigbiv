"""Tests for CandidateROI dataclass.

Tests cover:
- CandidateROI construction with required and optional args
- Defaulted field values (detector_score, validation_status, provenance, seed_frames, summary_features, temporal_features)
- to_serializable() excludes dense arrays (mask, sparse_pixels)
- to_serializable() includes all 13 metadata fields
- to_serializable() coerces numpy scalar types to native Python types
- to_serializable() drops arrays inside feature dicts via _jsonable_features
- to_serializable() output is JSON-serializable
- Defaults serialize cleanly
"""
from __future__ import annotations

import json

import numpy as np

from roigbiv.pipeline.types import CandidateROI


def test_candidateroi_required_args_only():
    """CandidateROI with required args only defaults optional fields correctly."""
    candidate = CandidateROI(
        candidate_id="roi_001",
        source_detector="cellpose",
        source_stage=1,
        branch="raw",
        mask=np.zeros((10, 10), dtype=bool),
        sparse_pixels=(np.array([0, 1, 5]), np.array([1.0, 1.0, 1.0])),
        centroid=(5.0, 5.0),
        bbox=(1, 1, 8, 8),
        area=42,
    )

    assert candidate.detector_score is None, "detector_score should default to None"
    assert candidate.validation_status == "pending", "validation_status should default to 'pending'"
    assert candidate.provenance == {}, "provenance should default to {}"
    assert candidate.seed_frames == [], "seed_frames should default to []"
    assert candidate.summary_features == {}, "summary_features should default to {}"
    assert candidate.temporal_features == {}, "temporal_features should default to {}"


def test_candidateroi_all_args_populated():
    """CandidateROI with all args stores them correctly."""
    mask = np.ones((10, 10), dtype=bool)
    sparse_pixels = (np.array([1, 2, 3]), np.array([0.5, 0.6, 0.7]))
    centroid = (5.0, 5.0)
    bbox = (1, 1, 8, 8)
    detector_score = 0.87
    validation_status = "accepted"
    provenance = {"detector_param": "value"}
    seed_frames = [10, 20, 30]
    summary_features = {"skew": 0.5}
    temporal_features = {"n_transients": 3}

    candidate = CandidateROI(
        candidate_id="roi_002",
        source_detector="suite2p",
        source_stage=2,
        branch="raw",
        mask=mask,
        sparse_pixels=sparse_pixels,
        centroid=centroid,
        bbox=bbox,
        area=50,
        detector_score=detector_score,
        validation_status=validation_status,
        provenance=provenance,
        seed_frames=seed_frames,
        summary_features=summary_features,
        temporal_features=temporal_features,
    )

    assert candidate.candidate_id == "roi_002"
    assert candidate.source_detector == "suite2p"
    assert candidate.source_stage == 2
    assert candidate.branch == "raw"
    assert np.array_equal(candidate.mask, mask)
    assert np.array_equal(candidate.sparse_pixels[0], sparse_pixels[0])
    assert np.array_equal(candidate.sparse_pixels[1], sparse_pixels[1])
    assert candidate.centroid == centroid
    assert candidate.bbox == bbox
    assert candidate.area == 50
    assert candidate.detector_score == detector_score
    assert candidate.validation_status == validation_status
    assert candidate.provenance is provenance
    assert candidate.seed_frames is seed_frames
    assert candidate.summary_features is summary_features
    assert candidate.temporal_features is temporal_features


def test_to_serializable_excludes_dense_arrays():
    """to_serializable() returns a dict without mask or sparse_pixels."""
    candidate = CandidateROI(
        candidate_id="roi_003",
        source_detector="cellpose",
        source_stage=1,
        branch="raw",
        mask=np.zeros((10, 10), dtype=bool),
        sparse_pixels=(np.array([0]), np.array([1.0])),
        centroid=(5.0, 5.0),
        bbox=(1, 1, 8, 8),
        area=42,
    )

    serialized = candidate.to_serializable()

    assert "mask" not in serialized, "mask should not be in serialized dict"
    assert "sparse_pixels" not in serialized, "sparse_pixels should not be in serialized dict"


def test_to_serializable_includes_all_metadata_fields():
    """to_serializable() includes exactly 13 metadata fields."""
    candidate = CandidateROI(
        candidate_id="roi_004",
        source_detector="cellpose",
        source_stage=1,
        branch="raw",
        mask=np.zeros((10, 10), dtype=bool),
        sparse_pixels=(np.array([0]), np.array([1.0])),
        centroid=(5.0, 5.0),
        bbox=(1, 1, 8, 8),
        area=42,
    )

    serialized = candidate.to_serializable()
    expected_keys = {
        "candidate_id",
        "source_detector",
        "source_stage",
        "branch",
        "centroid",
        "bbox",
        "area",
        "detector_score",
        "validation_status",
        "seed_frames",
        "provenance",
        "summary_features",
        "temporal_features",
    }

    assert set(serialized.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(serialized.keys())}"
    )
    assert len(serialized) == 13, f"Expected 13 keys, got {len(serialized)}"


def test_to_serializable_coerces_numpy_scalars_to_native_python():
    """to_serializable() coerces numpy scalar types to native Python types."""
    candidate = CandidateROI(
        candidate_id="roi_005",
        source_detector="cellpose",
        source_stage=np.int64(1),
        branch="raw",
        mask=np.zeros((10, 10), dtype=bool),
        sparse_pixels=(np.array([0], dtype=np.int32), np.array([1.0])),
        centroid=(np.float32(5.0), np.float32(5.0)),
        bbox=(1, 1, 8, 8),
        area=np.int64(42),
        detector_score=np.float64(0.87),
        seed_frames=[np.int32(10), np.int32(20)],
    )

    serialized = candidate.to_serializable()

    # Check that numeric types are native Python, not numpy
    assert type(serialized["source_stage"]) is int, (
        f"source_stage should be native int, got {type(serialized['source_stage'])}"
    )
    assert type(serialized["centroid"][0]) is float, (
        f"centroid[0] should be native float, got {type(serialized['centroid'][0])}"
    )
    assert type(serialized["centroid"][1]) is float, (
        f"centroid[1] should be native float, got {type(serialized['centroid'][1])}"
    )
    assert type(serialized["area"]) is int, (
        f"area should be native int, got {type(serialized['area'])}"
    )
    assert type(serialized["detector_score"]) is float, (
        f"detector_score should be native float, got {type(serialized['detector_score'])}"
    )
    # seed_frames values should be native int
    assert all(type(f) is int for f in serialized["seed_frames"]), (
        "all seed_frames values should be native int"
    )


def test_to_serializable_drops_arrays_inside_feature_dicts():
    """to_serializable() drops np.ndarray values from feature dicts via _jsonable_features."""
    candidate = CandidateROI(
        candidate_id="roi_006",
        source_detector="cellpose",
        source_stage=1,
        branch="raw",
        mask=np.zeros((10, 10), dtype=bool),
        sparse_pixels=(np.array([0]), np.array([1.0])),
        centroid=(5.0, 5.0),
        bbox=(1, 1, 8, 8),
        area=42,
        provenance={
            "k": 5,
            "trace_snippet": np.zeros(10),
            "metadata": "string",
        },
        summary_features={
            "scalar_feature": 1.5,
            "array_feature": np.array([1, 2, 3]),
        },
    )

    serialized = candidate.to_serializable()

    # Scalar entries survive, array entries dropped
    assert "k" in serialized["provenance"], "scalar provenance entry should survive"
    assert serialized["provenance"]["k"] == 5
    assert "trace_snippet" not in serialized["provenance"], (
        "array provenance entry should be dropped"
    )
    assert "metadata" in serialized["provenance"]
    assert serialized["provenance"]["metadata"] == "string"

    assert "scalar_feature" in serialized["summary_features"]
    assert serialized["summary_features"]["scalar_feature"] == 1.5
    assert "array_feature" not in serialized["summary_features"], (
        "array feature should be dropped"
    )


def test_to_serializable_json_dumps_succeeds():
    """to_serializable() output is JSON-serializable."""
    candidate = CandidateROI(
        candidate_id="roi_007",
        source_detector="template_sweep",
        source_stage=3,
        branch="raw",
        mask=np.ones((10, 10), dtype=bool),
        sparse_pixels=(np.array([1, 2, 3]), np.array([0.5, 0.6, 0.7])),
        centroid=(5.0, 5.0),
        bbox=(1, 1, 8, 8),
        area=50,
        detector_score=0.75,
        validation_status="accepted",
        provenance={"method": "sweep", "param": 42},
        seed_frames=[10, 20, 30],
        summary_features={"skew": 0.5, "peak": 100},
        temporal_features={"n_transients": 3, "rise_time": 0.1},
    )

    serialized = candidate.to_serializable()
    json_str = json.dumps(serialized)

    assert isinstance(json_str, str), "json.dumps should return a string"
    # Verify round-trip
    reloaded = json.loads(json_str)
    assert reloaded["candidate_id"] == "roi_007"
    assert reloaded["detector_score"] == 0.75


def test_candidateroi_defaults_serialize_cleanly():
    """Candidate with only required args serializes with clean defaults."""
    candidate = CandidateROI(
        candidate_id="roi_008",
        source_detector="cellpose",
        source_stage=1,
        branch="raw",
        mask=np.zeros((10, 10), dtype=bool),
        sparse_pixels=(np.array([0]), np.array([1.0])),
        centroid=(5.0, 5.0),
        bbox=(1, 1, 8, 8),
        area=42,
    )

    serialized = candidate.to_serializable()

    assert serialized["detector_score"] is None, "detector_score should be None"
    assert serialized["validation_status"] == "pending", "validation_status should be 'pending'"
    assert serialized["seed_frames"] == [], "seed_frames should be empty list"
    assert serialized["provenance"] == {}, "provenance should be empty dict"
    assert serialized["summary_features"] == {}, "summary_features should be empty dict"
    assert serialized["temporal_features"] == {}, "temporal_features should be empty dict"
