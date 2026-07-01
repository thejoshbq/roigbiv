"""Schema and validation module for benchmark FOV manifests (issue #27)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


LENS_TYPES: tuple[str, ...] = ("grin", "prism", "generic")
QUALITY_TIERS: tuple[str, ...] = ("high", "medium", "low")
REQUIRED_FIELDS: tuple[str, ...] = ("dataset_id", "fov_id", "path", "fs", "has_manual_masks",
                                    "has_longitudinal_ids", "has_synthetic_injections", "quality_tier")
ALL_FIELDS: tuple[str, ...] = REQUIRED_FIELDS + ("frame_averaging", "lens_type", "notes")


@dataclass(frozen=True)
class ManifestEntry:
    """A single benchmark FOV entry, as parsed and validated from a manifest."""
    dataset_id: str
    fov_id: str
    path: str  # Relative to the owning BenchmarkManifest.source_path — not resolved/absolute. Use `source_path / path` to get the actual filesystem location.
    fs: float
    has_manual_masks: bool
    has_longitudinal_ids: bool
    has_synthetic_injections: bool
    quality_tier: str
    frame_averaging: int = 1
    lens_type: str = "generic"
    notes: Optional[str] = None


@dataclass
class BenchmarkManifest:
    """A validated benchmark manifest: its entries plus the base directory their `path` fields are relative to."""
    entries: list[ManifestEntry]
    source_path: Path


@dataclass(frozen=True)
class ValidationError:
    """One validation failure, identifying which entry/field it came from (if any)."""
    entry_index: Optional[int] = None
    fov_id: Optional[str] = None
    field: Optional[str] = None
    message: str = ""

    def __str__(self) -> str:
        if self.entry_index is None:
            return f"manifest: {self.message}"

        parts = [f"entry[{self.entry_index}]"]

        if self.fov_id is not None:
            parts.append(f"(fov_id={self.fov_id})")

        if self.field is not None:
            parts.append(f"field={self.field}:")

        result = " ".join(parts)
        result += f" {self.message}"

        return result


class ManifestError(ValueError):
    """Raised when a manifest file cannot be loaded or is structurally malformed."""


def load_manifest(path: str | Path) -> dict:
    """Load and parse a manifest file (YAML or JSON).

    Dispatches by file suffix: .yaml/.yml -> yaml.safe_load, .json -> json.loads,
    other -> try json first, then yaml as fallback.

    Args:
        path: Path to manifest file.

    Returns:
        Raw parsed dict (unvalidated).

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ManifestError: If file cannot be parsed or lacks required structure.
    """
    path = Path(path)

    # Check existence
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")

    # Read text
    text = path.read_text()

    # Parse by suffix
    suffix = path.suffix.lower()
    parsed = None

    try:
        if suffix in (".yaml", ".yml"):
            parsed = yaml.safe_load(text)
        elif suffix == ".json":
            parsed = json.loads(text)
        else:
            # Try json first, fallback to yaml
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ManifestError(f"failed to parse manifest {path}: {e}") from e

    # Verify structure
    if not isinstance(parsed, dict):
        raise ManifestError(f"manifest {path} must be a mapping with a top-level 'entries' list")

    if "entries" not in parsed or not isinstance(parsed.get("entries"), list):
        raise ManifestError(f"manifest {path} must be a mapping with a top-level 'entries' list")

    return parsed


def validate_manifest(
    raw: dict,
    *,
    base_dir: Path | None = None,
    allow_missing_paths: bool = False
) -> tuple[BenchmarkManifest | None, list[ValidationError]]:
    """Validate a raw parsed manifest dict.

    Args:
        raw: Raw parsed manifest dict (from load_manifest).
        base_dir: Optional base directory for resolving relative paths.
        allow_missing_paths: If True, don't error on missing path files.

    Returns:
        Tuple of (manifest, errors). If any errors, manifest is None.
    """
    errors: list[ValidationError] = []

    entries_raw = raw.get("entries", [])

    # Precondition guard: entries_raw must be a list if present
    if entries_raw and not isinstance(entries_raw, list):
        return (None, [ValidationError(message="'entries' must be a list")])

    if not entries_raw:
        errors.append(ValidationError(message="manifest has no entries"))
        return (None, errors)

    built_entries: list[ManifestEntry] = []
    seen_fov_ids: dict[str, list[int]] = {}

    for i, entry in enumerate(entries_raw):
        entry_errors: list[ValidationError] = []

        # Check if dict
        if not isinstance(entry, dict):
            entry_errors.append(ValidationError(entry_index=i, message="must be a mapping"))
            errors.extend(entry_errors)
            continue

        # Unknown fields
        for key in entry:
            if key not in ALL_FIELDS:
                entry_errors.append(ValidationError(entry_index=i, field=key,
                                                     message=f"unknown field '{key}'"))

        # Missing required fields
        for f in REQUIRED_FIELDS:
            if f not in entry:
                entry_errors.append(ValidationError(entry_index=i, field=f,
                                                     message="missing required field"))

        # Type and value checks

        # dataset_id: must be str
        if "dataset_id" in entry and not isinstance(entry["dataset_id"], str):
            entry_errors.append(ValidationError(entry_index=i, field="dataset_id",
                                                message="must be string"))

        # fov_id: must be str
        if "fov_id" in entry:
            if not isinstance(entry["fov_id"], str):
                entry_errors.append(ValidationError(entry_index=i, field="fov_id",
                                                    message="must be string"))
            else:
                # Track for duplicate check
                fov_id = entry["fov_id"]
                if fov_id not in seen_fov_ids:
                    seen_fov_ids[fov_id] = []
                seen_fov_ids[fov_id].append(i)

        # path: must be str
        if "path" in entry:
            if not isinstance(entry["path"], str):
                entry_errors.append(ValidationError(entry_index=i, field="path",
                                                    message="must be string"))
            elif not allow_missing_paths:
                # Check path existence
                resolved = (base_dir / entry["path"]) if base_dir else Path(entry["path"])
                if not resolved.exists():
                    entry_errors.append(ValidationError(entry_index=i,
                                                        fov_id=entry.get("fov_id"),
                                                        field="path",
                                                        message=f"path does not exist: {resolved}"))

        # fs: must be int or float (not bool)
        if "fs" in entry:
            if isinstance(entry["fs"], bool) or not isinstance(entry["fs"], (int, float)):
                entry_errors.append(ValidationError(entry_index=i, field="fs",
                                                    message="must be number"))
            elif entry["fs"] <= 0:
                entry_errors.append(ValidationError(entry_index=i, field="fs",
                                                    message=f"must be positive, got {entry['fs']}"))

        # frame_averaging: must be int (not bool), if present
        if "frame_averaging" in entry:
            if isinstance(entry["frame_averaging"], bool) or not isinstance(entry["frame_averaging"], int):
                entry_errors.append(ValidationError(entry_index=i, field="frame_averaging",
                                                    message="must be integer"))
            elif entry["frame_averaging"] < 1:
                entry_errors.append(ValidationError(entry_index=i, field="frame_averaging",
                                                    message=f"must be >= 1, got {entry['frame_averaging']}"))

        # lens_type: must be in LENS_TYPES, if present
        if "lens_type" in entry:
            if not isinstance(entry["lens_type"], str):
                entry_errors.append(ValidationError(entry_index=i, field="lens_type",
                                                    message="must be string"))
            elif entry["lens_type"] not in LENS_TYPES:
                entry_errors.append(ValidationError(entry_index=i, field="lens_type",
                                                    message=f"invalid value '{entry['lens_type']}' (expected one of {', '.join(LENS_TYPES)})"))

        # quality_tier: must be in QUALITY_TIERS
        if "quality_tier" in entry:
            if not isinstance(entry["quality_tier"], str):
                entry_errors.append(ValidationError(entry_index=i, field="quality_tier",
                                                    message="must be string"))
            elif entry["quality_tier"] not in QUALITY_TIERS:
                entry_errors.append(ValidationError(entry_index=i, field="quality_tier",
                                                    message=f"invalid value '{entry['quality_tier']}' (expected one of {', '.join(QUALITY_TIERS)})"))

        # has_manual_masks / has_longitudinal_ids / has_synthetic_injections: must be actual bool
        for bool_field in ("has_manual_masks", "has_longitudinal_ids", "has_synthetic_injections"):
            if bool_field in entry:
                if not isinstance(entry[bool_field], bool):
                    entry_errors.append(ValidationError(entry_index=i, field=bool_field,
                                                        message="must be boolean"))

        # notes: must be str or None, if present
        if "notes" in entry:
            if entry["notes"] is not None and not isinstance(entry["notes"], str):
                entry_errors.append(ValidationError(entry_index=i, field="notes",
                                                    message="must be string or null"))

        # If no entry-level errors, build ManifestEntry
        if not entry_errors:
            manifest_entry = ManifestEntry(
                dataset_id=entry["dataset_id"],
                fov_id=entry["fov_id"],
                path=entry["path"],
                fs=float(entry["fs"]),
                has_manual_masks=entry["has_manual_masks"],
                has_longitudinal_ids=entry["has_longitudinal_ids"],
                has_synthetic_injections=entry["has_synthetic_injections"],
                quality_tier=entry["quality_tier"],
                frame_averaging=entry.get("frame_averaging", 1),
                lens_type=entry.get("lens_type", "generic"),
                notes=entry.get("notes")
            )
            built_entries.append(manifest_entry)
        else:
            errors.extend(entry_errors)

    # Check for duplicate fov_ids
    for fov_id, indices in seen_fov_ids.items():
        if len(indices) > 1:
            for idx in indices:
                other_indices = [j for j in indices if j != idx]
                errors.append(ValidationError(entry_index=idx, fov_id=fov_id, field="fov_id",
                                             message=f"duplicate fov_id '{fov_id}' (also at entries {other_indices})"))

    # Return
    if errors:
        return (None, errors)
    else:
        source_path = base_dir if base_dir is not None else Path(".")
        return (BenchmarkManifest(entries=built_entries, source_path=source_path), [])
