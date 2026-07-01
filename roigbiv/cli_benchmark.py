"""CLI entry point: validate a benchmark FOV manifest (YAML/JSON). Issue #27."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    """Validate a benchmark FOV manifest.

    Exit codes:
        0 — manifest is valid
        1 — semantic validation errors (missing fs, duplicate fov_id, invalid
            quality_tier, missing path, etc.)
        2 — file not found / malformed YAML-JSON / usage error
    """
    parser = argparse.ArgumentParser(
        prog="roigbiv-benchmark",
        description="Validate a benchmark FOV manifest (YAML or JSON).",
    )
    parser.add_argument("manifest", type=Path, help="Path to the manifest file.")
    parser.add_argument(
        "--allow-missing-paths",
        action="store_true",
        default=False,
        help="Skip path-existence checks (useful for template authoring before data lands).",
    )
    args = parser.parse_args(argv)

    from roigbiv.benchmark.schema import (
        ManifestError,
        load_manifest,
        validate_manifest,
    )

    manifest_path = args.manifest
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        raw = load_manifest(manifest_path)
    except (ManifestError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    manifest, errors = validate_manifest(
        raw,
        base_dir=manifest_path.parent,
        allow_missing_paths=args.allow_missing_paths,
    )

    if errors:
        for err in errors:
            print(str(err), file=sys.stderr)
        print(
            f"validation failed: {len(errors)} error(s) in {manifest_path.name}",
            file=sys.stderr,
        )
        return 1

    print(f"{manifest_path.name}: OK ({len(manifest.entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
