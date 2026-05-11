#!/usr/bin/env python3
"""
ROI G. Biv — Sort TRAP `_mc.tif` files into a flat input directory.

Walks --root for `*_mc.tif` (the only files the pipeline consumes), then
moves each match into --dest as a flat collection. Bruker sidecars stay
where they are. Re-runs are idempotent because --dest is excluded from
the walk.

Downstream:
    roigbiv-pipeline --workspace --input <dest> --fs 7.5

Usage:
    python sort_trap_data.py                       # dry-run from default root
    python sort_trap_data.py --execute             # move to <root>/data/
    python sort_trap_data.py --execute --copy      # copy instead of move
    python sort_trap_data.py --execute --skip-existing
    python sort_trap_data.py --root /other --dest /other/data --execute
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/external/JOSH/TRAP")

log = logging.getLogger("sort_trap")


def discover_mc_tifs(root: Path, dest: Path) -> list[Path]:
    """Find all `*_mc.tif` files under *root*, excluding anything under *dest*."""
    try:
        dest_resolved = dest.resolve()
    except FileNotFoundError:
        dest_resolved = dest

    files: list[Path] = []
    for p in root.rglob("*_mc.tif"):
        try:
            p.resolve().relative_to(dest_resolved)
        except ValueError:
            files.append(p)
    return sorted(files)


def sort_files(
    files: list[Path],
    dest: Path,
    *,
    copy: bool,
    skip_existing: bool,
) -> dict:
    """Move (or copy) each file into *dest*. Returns a counts dict."""
    dest.mkdir(parents=True, exist_ok=True)
    action = "copy" if copy else "move"
    counts = {"moved": 0, "skipped": 0, "errors": 0}

    for src in files:
        target = dest / src.name
        if target.exists():
            if skip_existing:
                log.info("  skip (exists): %s", src.name)
                counts["skipped"] += 1
                continue
            log.error("  collision: %s already in %s — pass --skip-existing to skip",
                      src.name, dest)
            counts["errors"] += 1
            continue

        try:
            if copy:
                shutil.copy2(str(src), str(target))
            else:
                shutil.move(str(src), str(target))
            log.info("  %s: %s", action, src.name)
            counts["moved"] += 1
        except Exception as exc:
            log.error("  failed (%s): %s — %s", action, src.name, exc)
            counts["errors"] += 1

    return counts


def cohort_tally(files: list[Path], root: Path) -> dict[str, int]:
    """Group counts by the cohort directory name (first path component under *root*)."""
    tally: dict[str, int] = {}
    for p in files:
        try:
            rel = p.relative_to(root)
            cohort = rel.parts[0] if len(rel.parts) > 1 else "<root>"
        except ValueError:
            cohort = "<other>"
        tally[cohort] = tally.get(cohort, 0) + 1
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sort TRAP _mc.tif files into a flat data/ directory.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Search root (default: {DEFAULT_ROOT})")
    ap.add_argument("--dest", type=Path, default=None,
                    help="Destination directory (default: <root>/data)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually move/copy files (default is dry-run)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of move (originals stay in place)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip files whose destination already exists")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    root: Path = args.root
    dest: Path = args.dest if args.dest is not None else root / "data"

    if not root.exists():
        log.error("Root does not exist: %s", root)
        return 1

    log.info("Root:        %s", root)
    log.info("Destination: %s", dest)
    log.info("Action:      %s", "copy" if args.copy else "move")

    log.info("Discovering *_mc.tif ...")
    files = discover_mc_tifs(root, dest)
    log.info("Found %d files", len(files))

    if not files:
        log.info("Nothing to do.")
        return 0

    for cohort, n in sorted(cohort_tally(files, root).items()):
        log.info("  %s: %d", cohort, n)

    if not args.execute:
        log.info("DRY RUN — exiting without action. Use --execute to %s.",
                 "copy" if args.copy else "move")
        return 0

    counts = sort_files(files, dest, copy=args.copy, skip_existing=args.skip_existing)
    log.info("Done: %d %s, %d skipped, %d errors",
             counts["moved"], "copied" if args.copy else "moved",
             counts["skipped"], counts["errors"])
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
