"""CLI entry point for the roigbiv benchmark harness.

Subcommands:
  validate — validate a benchmark FOV manifest (issue #27)
  report   — generate a Markdown+JSON comparison report from benchmark_run.json
             file(s) (issue #32)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roigbiv-benchmark",
        description="Benchmark manifest validation and comparison reporting.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate", help="Validate a benchmark FOV manifest (YAML or JSON).")
    validate.add_argument("manifest", type=Path, help="Path to the manifest file.")
    validate.add_argument(
        "--allow-missing-paths",
        action="store_true",
        default=False,
        help="Skip path-existence checks (useful for template authoring before data lands).",
    )

    report = sub.add_parser(
        "report",
        help="Generate a Markdown+JSON comparison report from benchmark_run.json file(s).",
    )
    report.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        dest="runs",
        help="Path to a benchmark_run.json file. Repeat --run for multiple pipeline modes.",
    )
    report.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write report.md and report.json into.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "validate":
        return _cmd_validate(args.manifest, args.allow_missing_paths)
    if args.cmd == "report":
        return _cmd_report(args.runs, args.output_dir)
    return 2


def _cmd_validate(manifest_path: Path, allow_missing_paths: bool) -> int:
    """Validate a benchmark FOV manifest.

    Exit codes:
        0 — manifest is valid
        1 — semantic validation errors (missing fs, duplicate fov_id, invalid
            quality_tier, missing path, etc.)
        2 — file not found / malformed YAML-JSON / usage error
    """
    from roigbiv.benchmark.schema import (
        ManifestError,
        load_manifest,
        validate_manifest,
    )

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
        allow_missing_paths=allow_missing_paths,
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


def _cmd_report(run_paths: list[Path], output_dir: Path) -> int:
    """Generate report.md + report.json from one or more benchmark_run.json files.

    Exit codes:
        0 — report.md and report.json written successfully (warnings, if
            any, are reported inside report.md, not via exit code)
        2 — a --run file is missing / malformed JSON / fails to deserialize
    """
    import json

    from roigbiv.benchmark.report import build_json_report, build_markdown_report
    from roigbiv.benchmark.results import BenchmarkRun

    runs = []
    for run_path in run_paths:
        if not run_path.is_file():
            print(f"error: run file not found: {run_path}", file=sys.stderr)
            return 2
        try:
            payload = json.loads(run_path.read_text())
            runs.append(BenchmarkRun.from_dict(payload))
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            print(f"error: failed to parse {run_path}: {exc}", file=sys.stderr)
            return 2

    md_path = output_dir / "report.md"
    json_path = output_dir / "report.json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(build_markdown_report(runs))
        json_path.write_text(json.dumps(build_json_report(runs), indent=2, default=str))
    except OSError as exc:
        print(f"error: failed to write report to {output_dir}: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
