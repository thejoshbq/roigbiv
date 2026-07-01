"""Tests for roigbiv.benchmark.schema — manifest loading, validation, and error handling."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from roigbiv.benchmark.schema import (
    LENS_TYPES,
    QUALITY_TIERS,
    BenchmarkManifest,
    ManifestEntry,
    ManifestError,
    ValidationError,
    load_manifest,
    validate_manifest,
)


def _write_yaml(data: dict, path: Path) -> None:
    """Write a dict as YAML to a file."""
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_json(data: dict, path: Path) -> None:
    """Write a dict as JSON to a file."""
    with open(path, "w") as f:
        json.dump(data, f)


def _valid_entry_dict(base_dir: Path, dataset_id: str = "ds1", fov_id: str = "fov1") -> dict:
    """Return a valid manifest entry dict with real path pointing to a created directory."""
    data_dir = base_dir / "data" / dataset_id
    data_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_id": dataset_id,
        "fov_id": fov_id,
        "path": str(data_dir),
        "fs": 30.0,
        "has_manual_masks": True,
        "has_longitudinal_ids": False,
        "has_synthetic_injections": True,
        "quality_tier": "high",
        "frame_averaging": 1,
        "lens_type": "generic",
        "notes": None,
    }


def _valid_manifest_dict(base_dir: Path) -> dict:
    """Return a valid manifest dict with two entries."""
    return {
        "entries": [
            _valid_entry_dict(base_dir, dataset_id="ds1", fov_id="fov1"),
            _valid_entry_dict(base_dir, dataset_id="ds2", fov_id="fov2"),
        ]
    }


# ============================================================================
# Tests for load_manifest()
# ============================================================================


def test_load_manifest_valid_yaml(tmp_path: Path) -> None:
    """Valid YAML manifest round-trips correctly."""
    manifest_dict = _valid_manifest_dict(tmp_path)
    path = tmp_path / "manifest.yaml"
    _write_yaml(manifest_dict, path)
    loaded = load_manifest(path)
    assert loaded == manifest_dict


def test_load_manifest_valid_json(tmp_path: Path) -> None:
    """Valid JSON manifest round-trips correctly."""
    manifest_dict = _valid_manifest_dict(tmp_path)
    path = tmp_path / "manifest.json"
    _write_json(manifest_dict, path)
    loaded = load_manifest(path)
    assert loaded == manifest_dict


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nonexistent.yaml")


def test_load_manifest_malformed_yaml(tmp_path: Path) -> None:
    """Malformed YAML raises ManifestError."""
    path = tmp_path / "malformed.yaml"
    path.write_text("{ invalid: yaml: content::")
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON raises ManifestError."""
    path = tmp_path / "malformed.json"
    path.write_text("{invalid json content")
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_top_level_not_dict(tmp_path: Path) -> None:
    """Top-level value that is not a dict raises ManifestError."""
    path = tmp_path / "manifest.yaml"
    _write_yaml(["not", "a", "dict"], path)
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_missing_entries_key(tmp_path: Path) -> None:
    """Manifest dict without 'entries' key raises ManifestError."""
    path = tmp_path / "manifest.yaml"
    _write_yaml({"data": "something"}, path)
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_entries_not_list(tmp_path: Path) -> None:
    """Manifest with 'entries' not a list raises ManifestError."""
    path = tmp_path / "manifest.yaml"
    _write_yaml({"entries": "not a list"}, path)
    with pytest.raises(ManifestError):
        load_manifest(path)


# ============================================================================
# Tests for validate_manifest()
# ============================================================================


def test_validate_manifest_fully_valid_entry(tmp_path: Path) -> None:
    """Fully valid entry with all 11 fields returns (manifest, []) with correct values."""
    manifest_dict = _valid_manifest_dict(tmp_path)
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert errors == []
    assert manifest is not None
    assert len(manifest.entries) == 2
    assert manifest.entries[0].dataset_id == "ds1"
    assert manifest.entries[0].fov_id == "fov1"
    assert manifest.entries[0].fs == 30.0
    assert manifest.entries[0].has_manual_masks is True
    assert manifest.entries[0].has_longitudinal_ids is False
    assert manifest.entries[0].has_synthetic_injections is True
    assert manifest.entries[0].quality_tier == "high"
    assert manifest.entries[0].frame_averaging == 1
    assert manifest.entries[0].lens_type == "generic"
    assert manifest.entries[0].notes is None


def test_validate_manifest_entry_relying_on_defaults(tmp_path: Path) -> None:
    """Entry omitting optional fields gets correct defaults."""
    entry = _valid_entry_dict(tmp_path)
    del entry["frame_averaging"]
    del entry["lens_type"]
    del entry["notes"]
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert errors == []
    assert manifest is not None
    assert manifest.entries[0].frame_averaging == 1
    assert manifest.entries[0].lens_type == "generic"
    assert manifest.entries[0].notes is None


def test_validate_manifest_missing_fs_field(tmp_path: Path) -> None:
    """Missing 'fs' field returns (None, errors) with field=='fs'."""
    entry = _valid_entry_dict(tmp_path)
    del entry["fs"]
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "fs"
    assert errors[0].entry_index == 0


def test_validate_manifest_missing_path_field(tmp_path: Path) -> None:
    """Missing 'path' field returns (None, errors) with field=='path'."""
    entry = _valid_entry_dict(tmp_path)
    del entry["path"]
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "path"


def test_validate_manifest_duplicate_fov_id(tmp_path: Path) -> None:
    """Duplicate fov_id across entries flags both entries in errors."""
    entry1 = _valid_entry_dict(tmp_path, dataset_id="ds1", fov_id="same")
    entry2 = _valid_entry_dict(tmp_path, dataset_id="ds2", fov_id="same")
    manifest_dict = {"entries": [entry1, entry2]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    # Should have 2 errors, one for each duplicate
    dup_errors = [e for e in errors if e.field == "fov_id"]
    assert len(dup_errors) == 2
    assert dup_errors[0].entry_index == 0
    assert dup_errors[1].entry_index == 1


def test_validate_manifest_invalid_quality_tier(tmp_path: Path) -> None:
    """Invalid quality_tier value returns error."""
    entry = _valid_entry_dict(tmp_path)
    entry["quality_tier"] = "platinum"
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "quality_tier"
    assert "platinum" in errors[0].message


def test_validate_manifest_invalid_lens_type(tmp_path: Path) -> None:
    """Invalid lens_type value returns error."""
    entry = _valid_entry_dict(tmp_path)
    entry["lens_type"] = "unknown_lens"
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "lens_type"


def test_validate_manifest_nonexistent_path_rejected(tmp_path: Path) -> None:
    """Path pointing to nonexistent location returns error unless allow_missing_paths=True."""
    entry = _valid_entry_dict(tmp_path)
    entry["path"] = str(tmp_path / "nonexistent" / "path")
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "path"


def test_validate_manifest_nonexistent_path_allowed(tmp_path: Path) -> None:
    """Path pointing to nonexistent location passes with allow_missing_paths=True."""
    entry = _valid_entry_dict(tmp_path)
    entry["path"] = str(tmp_path / "nonexistent" / "path")
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(
        manifest_dict, base_dir=tmp_path, allow_missing_paths=True
    )
    assert errors == []
    assert manifest is not None


def test_validate_manifest_existing_file_path(tmp_path: Path) -> None:
    """Path pointing to an existing FILE (not directory) passes validation."""
    data_dir = tmp_path / "data" / "ds1"
    data_dir.mkdir(parents=True)
    file_path = data_dir / "file.tif"
    file_path.touch()
    entry = _valid_entry_dict(tmp_path)
    entry["path"] = str(file_path)
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert errors == []
    assert manifest is not None


def test_validate_manifest_unknown_field(tmp_path: Path) -> None:
    """Unknown/extra field on entry returns error naming the unknown key."""
    entry = _valid_entry_dict(tmp_path)
    entry["unknown_key"] = "value"
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "unknown_key"
    assert "unknown field" in errors[0].message


def test_validate_manifest_empty_entries_list(tmp_path: Path) -> None:
    """Empty entries list returns (None, errors) with manifest=None."""
    manifest_dict = {"entries": []}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert "no entries" in errors[0].message


def test_validate_manifest_boolean_as_string(tmp_path: Path) -> None:
    """Boolean field given as string 'true' instead of bool returns error (type strictness)."""
    entry = _valid_entry_dict(tmp_path)
    entry["has_manual_masks"] = "true"
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "has_manual_masks"
    assert "boolean" in errors[0].message


def test_validate_manifest_fs_as_int_coerced_to_float(tmp_path: Path) -> None:
    """Numeric fs field as int is coerced to float successfully."""
    entry = _valid_entry_dict(tmp_path)
    entry["fs"] = 30  # int, not float
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert errors == []
    assert manifest is not None
    assert manifest.entries[0].fs == 30.0
    assert isinstance(manifest.entries[0].fs, float)


def test_validate_manifest_base_dir_resolves_relative_paths(tmp_path: Path) -> None:
    """base_dir parameter resolves relative paths correctly."""
    data_dir = tmp_path / "data" / "ds1"
    data_dir.mkdir(parents=True)
    entry = {
        "dataset_id": "ds1",
        "fov_id": "fov1",
        "path": "ds1",  # relative to base_dir
        "fs": 7.5,
        "has_manual_masks": True,
        "has_longitudinal_ids": False,
        "has_synthetic_injections": False,
        "quality_tier": "medium",
    }
    manifest_dict = {"entries": [entry]}
    # base_dir is tmp_path / "data", so "ds1" resolves to tmp_path / "data" / "ds1"
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path / "data")
    assert errors == []
    assert manifest is not None


# ============================================================================
# Tests for ValidationError.__str__()
# ============================================================================


def test_validation_error_str_with_entry_and_field() -> None:
    """ValidationError.__str__() produces readable message with entry, fov_id, and field."""
    error = ValidationError(
        entry_index=0, fov_id="fov1", field="fs", message="missing required field"
    )
    result = str(error)
    assert "entry[0]" in result
    assert "fov_id=fov1" in result
    assert "field=fs" in result
    assert "missing required field" in result


def test_validation_error_str_without_entry() -> None:
    """ValidationError.__str__() for manifest-level error (no entry_index) uses 'manifest:' prefix."""
    error = ValidationError(message="manifest has no entries")
    result = str(error)
    assert "manifest:" in result
    assert "manifest has no entries" in result


def test_validation_error_str_without_fov_id() -> None:
    """ValidationError.__str__() omits fov_id when not present."""
    error = ValidationError(
        entry_index=2, field="quality_tier", message="invalid value 'platinum'"
    )
    result = str(error)
    assert "entry[2]" in result
    assert "fov_id" not in result
    assert "quality_tier" in result


# ============================================================================
# Tests for edge cases and newly-added validation branches
# ============================================================================


def test_validate_manifest_fs_zero(tmp_path: Path) -> None:
    """Test that fs=0 raises an error with a 'positive' message."""
    entry = _valid_entry_dict(tmp_path)
    entry["fs"] = 0
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "fs"
    assert "positive" in errors[0].message


def test_validate_manifest_fs_negative(tmp_path: Path) -> None:
    """Test that fs < 0 raises an error with a 'positive' message."""
    entry = _valid_entry_dict(tmp_path)
    entry["fs"] = -1
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "fs"
    assert "positive" in errors[0].message


def test_validate_manifest_fs_as_bool(tmp_path: Path) -> None:
    """Test that fs as bool (True) raises an error with 'must be number' message."""
    entry = _valid_entry_dict(tmp_path)
    entry["fs"] = True
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "fs"
    assert "must be number" in errors[0].message


def test_validate_manifest_dataset_id_as_int(tmp_path: Path) -> None:
    """Test that dataset_id as int raises an error."""
    entry = _valid_entry_dict(tmp_path)
    entry["dataset_id"] = 123
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "dataset_id"
    assert "string" in errors[0].message


def test_validate_manifest_fov_id_as_int(tmp_path: Path) -> None:
    """Test that fov_id as int raises an error."""
    entry = _valid_entry_dict(tmp_path)
    entry["fov_id"] = 456
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "fov_id"
    assert "string" in errors[0].message


def test_validate_manifest_notes_as_int(tmp_path: Path) -> None:
    """Test that notes as int (non-None, non-string) raises an error."""
    entry = _valid_entry_dict(tmp_path)
    entry["notes"] = 789
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "notes"
    assert "string or null" in errors[0].message


def test_validate_manifest_frame_averaging_zero(tmp_path: Path) -> None:
    """Test that frame_averaging=0 raises an error with '>= 1' message."""
    entry = _valid_entry_dict(tmp_path)
    entry["frame_averaging"] = 0
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "frame_averaging"
    assert ">= 1" in errors[0].message


def test_validate_manifest_frame_averaging_negative(tmp_path: Path) -> None:
    """Test that frame_averaging=-1 raises an error with '>= 1' message."""
    entry = _valid_entry_dict(tmp_path)
    entry["frame_averaging"] = -1
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "frame_averaging"
    assert ">= 1" in errors[0].message


def test_validate_manifest_frame_averaging_as_string(tmp_path: Path) -> None:
    """Test that frame_averaging as string raises an error."""
    entry = _valid_entry_dict(tmp_path)
    entry["frame_averaging"] = "not-a-number"
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "frame_averaging"
    assert "integer" in errors[0].message


def test_validate_manifest_frame_averaging_as_float(tmp_path: Path) -> None:
    """Test that frame_averaging as float raises an error."""
    entry = _valid_entry_dict(tmp_path)
    entry["frame_averaging"] = 1.5
    manifest_dict = {"entries": [entry]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].field == "frame_averaging"
    assert "integer" in errors[0].message


def test_validate_manifest_entry_not_a_dict_with_valid_entry_after(tmp_path: Path) -> None:
    """Test that a non-dict entry (string) flags error; function continues to next entry, but manifest is None overall."""
    entry1 = "not-a-dict"
    entry2 = _valid_entry_dict(tmp_path, dataset_id="ds2", fov_id="fov2")
    manifest_dict = {"entries": [entry1, entry2]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    # Manifest is None because errors is non-empty
    assert manifest is None
    # Should have exactly one error for the malformed entry
    assert len(errors) == 1
    assert errors[0].entry_index == 0
    assert errors[0].field is None
    assert "must be a mapping" in errors[0].message


def test_validate_manifest_entry_as_list_not_dict(tmp_path: Path) -> None:
    """Test that an entry that is a list (not dict) raises error with 'must be a mapping'."""
    entry1 = [1, 2, 3]
    entry2 = _valid_entry_dict(tmp_path)
    manifest_dict = {"entries": [entry1, entry2]}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].entry_index == 0
    assert "must be a mapping" in errors[0].message


def test_validate_manifest_entries_not_a_list_direct(tmp_path: Path) -> None:
    """Test that calling validate_manifest directly with entries='not-a-list' returns (None, [error])."""
    manifest_dict = {"entries": "not-a-list"}
    manifest, errors = validate_manifest(manifest_dict, base_dir=tmp_path)
    assert manifest is None
    assert len(errors) == 1
    assert errors[0].entry_index is None  # Manifest-level error
    assert "must be a list" in errors[0].message
