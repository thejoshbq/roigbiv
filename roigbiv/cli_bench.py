"""Terminal entry point: roigbiv-bench — the benchmark harness runner.

Subcommands:
  run    — run the current ROIGBIV pipeline over every entry in a manifest,
           optionally under one or more named ablation presets (--ablation;
           see roigbiv/benchmark/ablations.py, issue #33).

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
    from roigbiv.benchmark.ablations import list_ablations
    run_p.add_argument(
        "--ablation", type=str, nargs="+", default=None,
        choices=list_ablations(),
        help=(
            "Run one or more named ablation presets (see "
            "roigbiv/benchmark/ablations.py) instead of the plain current "
            "pipeline. Space-separated for multiple, or 'all' to run every "
            "registered preset. Outputs are grouped under "
            "output_dir/<ablation_name>/. Omit entirely for the legacy "
            "single-run behavior (flat output_dir layout, no ablation "
            "overrides applied)."
        ),
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args.manifest, args.output_dir, args.ablation)

    parser.print_help(sys.stderr)
    return 2


def _cmd_run(manifest_path: Path, output_dir: Path,
              ablations: Optional[list] = None) -> int:
    from roigbiv.benchmark.runner import run_benchmark
    from roigbiv.benchmark.schema import ManifestError

    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        report = run_benchmark(manifest_path, output_dir, ablations=ablations)
    except (ManifestError, FileNotFoundError, ValueError) as exc:
        # ValueError: run_benchmark's own ablation-name validation. argparse's
        # `choices=` already catches this for the CLI path (SystemExit(2)
        # before _cmd_run is ever entered) — this is defense-in-depth for
        # run_benchmark's other callers (it's a public library function).
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
    if report.ablations:
        _print_ablation_summary(report)

    return 1 if n_err > 0 else 0


def _print_ablation_summary(report) -> None:
    """Per-ablation pass/fail + aggregate ROI counts, side by side. Minimal
    console table — full comparison-report rendering is issue #32's job, not
    this module's."""
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in report.fov_results:
        groups[r.ablation].append(r)

    print("\nablation summary:")
    print(f"  {'ablation':<28} {'ok/total':>9} {'accept':>7} {'flag':>6} {'reject':>7}")
    for name in report.ablations:  # preserves run order from the report
        results = groups.get(name, [])
        ok = sum(1 for r in results if r.status == "success")
        accept = sum(r.roi_counts.get("accept", 0) for r in results)
        flag = sum(r.roi_counts.get("flag", 0) for r in results)
        reject = sum(r.roi_counts.get("reject", 0) for r in results)
        ok_total = f"{ok}/{len(results)}"
        print(f"  {name:<28} {ok_total:>9} {accept:>7} {flag:>6} {reject:>7}")


if __name__ == "__main__":
    raise SystemExit(main())
