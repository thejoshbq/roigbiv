"""CLI entry point: roigbiv-export

Export fluorescence traces from a pipeline FOV output directory to a
pandas HDF5 file. Neuron identity is encoded directly as column names
(global_cell_id when registered, else ``lcl:<local_label_id>``), so
multiple downloads can be merged with ``pd.concat([df1, df2], axis=1)``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="roigbiv-export",
        description=(
            "Export ROI fluorescence traces to HDF5. "
            "Columns are neuron IDs — no index dict needed."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FOV_OUTPUT_DIR",
        help="Pipeline output directory (the one containing traces/).",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        metavar="PATH.h5",
        help="Output file. Default: <fov_stem>_traces.h5 in the current directory.",
    )
    parser.add_argument(
        "--kind", "-k",
        default="dff,f",
        metavar="KINDS",
        help=(
            "Comma-separated signal types to export. "
            "Available: dff, f, raw, neuropil. Default: dff,f"
        ),
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else Path(f"{input_dir.name}_traces.h5")
    kinds = tuple(k.strip() for k in args.kind.split(",") if k.strip())

    from roigbiv.pipeline.export_io import export_fov_traces

    try:
        export_fov_traces(input_dir, out_path, kinds=kinds)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Exported {len(kinds)} signal type(s) → {out_path}")
    print(f"  Load with: pd.HDFStore('{out_path}')")
    print(f"  Keys: {['/'+k for k in kinds] + ['/meta']}")


if __name__ == "__main__":
    main()
