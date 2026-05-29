"""Terminal entry point for ingesting externally-edited ROI masks.

Researchers edit ROIs in Fiji/ImageJ (the in-app editor was retired),
save a new label TIFF, and run::

    roigbiv-reingest \\
        --output-dir inference/pipeline/<stem> \\
        --new-mask /path/to/edited_masks.tif \\
        --notes "Fiji touch-up 2026-04-27"

The diff is appended to ``corrections/corrections.jsonl`` and
``corrected_masks.tif`` / ``corrected_metadata.json`` are re-materialised.
Use ``--dry-run`` to preview the diff without writing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roigbiv-reingest",
        description=(
            "Ingest an externally-edited ROI mask TIFF into the corrections "
            "log for a single FOV's pipeline output directory."
        ),
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="FOV output dir (e.g. inference/pipeline/<stem>/).",
    )
    parser.add_argument(
        "--new-mask", required=True, type=Path,
        help="Externally-edited label TIFF (uint16; background == 0).",
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Free-text note recorded on every emitted correction op.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the diff without writing the corrections log.",
    )
    parser.add_argument(
        "--preserve-iou", type=float, default=0.95,
        help="IoU at/above which an ROI is treated as unchanged (default: 0.95).",
    )
    parser.add_argument(
        "--edit-iou", type=float, default=0.50,
        help=("IoU at/above which an ROI is treated as edited rather than "
              "delete+add (default: 0.50)."),
    )

    args = parser.parse_args(argv)

    from roigbiv.pipeline.reingest import reingest_mask

    output_dir = args.output_dir.resolve()
    new_mask = args.new_mask.resolve()
    if not output_dir.is_dir():
        print(f"error: --output-dir is not a directory: {output_dir}",
              file=sys.stderr)
        return 2
    if not new_mask.is_file():
        print(f"error: --new-mask not found: {new_mask}", file=sys.stderr)
        return 2

    try:
        result = reingest_mask(
            output_dir,
            new_mask,
            notes=args.notes,
            dry_run=args.dry_run,
            preserve_iou=args.preserve_iou,
            edit_iou=args.edit_iou,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    prefix = "would write" if args.dry_run else "wrote"
    if not result.ops:
        print(f"reingest: no changes detected ({result.summary()}).")
    else:
        print(f"reingest: {result.summary()}; {prefix} {len(result.ops)} op(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
