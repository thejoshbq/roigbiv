"""Terminal entry point: roigbiv-bench — the benchmark harness runner.

Subcommands:
  run    — run the current ROIGBIV pipeline over every entry in a manifest.

Deliberately subcommand-based (argparse.add_subparsers), distinct from the
flat single-command `roigbiv-benchmark` (manifest validator, issue #27).
`roigbiv-bench report` (issue #32) is a future addition to this same
subparser set — not implemented here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    """roigbiv-bench — CLI entry point.

    `run` subcommand exit codes:
        0 — manifest loaded, all FOVs succeeded, benchmark_run.json written
        1 — manifest loaded, >=1 FOV errored, run completed and
            benchmark_run.json was still written
        2 — manifest not found / failed to parse / failed validation, or a
            usage error — no FOV was run, benchmark_run.json was NOT written
    """
    parser = argparse.ArgumentParser(
        prog="roigbiv-bench",
        description="ROIGBIV benchmark harness — run the pipeline over a manifest.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run the current ROIGBIV pipeline over every manifest entry.",
    )
    run_p.add_argument("--manifest", type=Path, required=True,
                        help="Path to a validated benchmark manifest (YAML/JSON).")
    run_p.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write per-FOV outputs, logs, and benchmark_run.json into.")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args.manifest, args.output_dir)

    parser.print_help(sys.stderr)
    return 2


def _cmd_run(manifest_path: Path, output_dir: Path) -> int:
    from roigbiv.benchmark.runner import run_benchmark
    from roigbiv.benchmark.schema import ManifestError

    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        report = run_benchmark(manifest_path, output_dir)
    except (ManifestError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report_path = Path(report.output_dir) / "benchmark_run.json"
    with open(report_path, "w") as f:
        json.dump(report.to_json_dict(), f, indent=2, default=str)

    n_total = len(report.fov_results)
    n_ok = sum(1 for r in report.fov_results if r.status == "success")
    n_err = n_total - n_ok
    print(
        f"roigbiv-bench run: {n_ok}/{n_total} FOVs OK "
        f"({report.total_runtime_s:.1f}s total). "
        f"Report: {report_path}"
    )

    return 1 if n_err > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
