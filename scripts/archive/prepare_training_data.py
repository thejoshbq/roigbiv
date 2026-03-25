#!/usr/bin/env python3
"""
Flatten nested two-photon imaging data into pipeline-ready directories.

Walks the nested tree under data/ROI sets and imaging data for Josh/,
pairs *_mc.tif stacks with *RoiSet.zip annotations, and moves them into:
    data/raw/{stem}_mc.tif
    data/annotated/{stem}_mc.zip   (renamed so stem matches the tif)

Dry-run by default. Pass --execute to actually move files.

Usage:
    python scripts/prepare_training_data.py                # dry-run
    python scripts/prepare_training_data.py --execute      # move files
    python scripts/prepare_training_data.py --verbose      # per-file details
"""

import argparse
import datetime
import logging
import os
import shutil
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(PROJECT_ROOT, "data",
                              "ROI sets and imaging data for Josh")
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DEFAULT_ANNOTATED_DIR = os.path.join(PROJECT_ROOT, "data", "annotated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool) -> logging.Logger:
    """Configure file + console logging."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"prepare_training_data_{ts}.log")

    logger = logging.getLogger("prepare_training_data")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("Log file: %s", log_path)
    return logger


def safe_under_root(path: str, root: str) -> bool:
    """Return True if realpath of *path* is under *root*."""
    real = os.path.realpath(path)
    real_root = os.path.realpath(root)
    return real == real_root or real.startswith(real_root + os.sep)


def tif_stem(filename: str) -> str:
    """Extract the stem from a _mc.tif filename (everything before .tif)."""
    assert filename.endswith("_mc.tif"), filename
    return filename[:-4]  # strip .tif → keeps _mc


def zip_stem(filename: str) -> str:
    """Extract the stem from a *RoiSet.zip filename."""
    assert filename.endswith("RoiSet.zip"), filename
    return filename[:-4]  # strip .zip


# ---------------------------------------------------------------------------
# Phase 1 — Discovery
# ---------------------------------------------------------------------------

def discover(source: str, logger: logging.Logger, verbose: bool):
    """Walk the nested tree and collect _mc.tif and *RoiSet.zip paths.

    Returns a dict: leaf_directory -> {'tifs': [path, ...], 'zips': [path, ...]}
    """
    dirs = defaultdict(lambda: {"tifs": [], "zips": []})

    def scan_leaf(dirpath: str):
        """Collect tifs and zips in a session-level or PRE Files directory."""
        try:
            entries = list(os.scandir(dirpath))
        except PermissionError:
            logger.warning("Permission denied: %s", dirpath)
            return

        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                if entry.name.endswith("_mc.tif"):
                    dirs[dirpath]["tifs"].append(entry.path)
                    if verbose:
                        logger.debug("  FOUND tif: %s", entry.path)
                elif entry.name.endswith("RoiSet.zip"):
                    dirs[dirpath]["zips"].append(entry.path)
                    if verbose:
                        logger.debug("  FOUND zip: %s", entry.path)
            elif entry.is_dir(follow_symlinks=False) and entry.name == "PRE Files":
                scan_leaf(entry.path)

    logger.info("Scanning %s ...", source)
    try:
        animal_entries = sorted(os.scandir(source), key=lambda e: e.name)
    except PermissionError:
        logger.error("Cannot read source directory: %s", source)
        return dirs

    for animal in animal_entries:
        if not animal.is_dir(follow_symlinks=False):
            continue
        if animal.name.startswith("."):
            continue

        logger.info("Animal: %s", animal.name)
        try:
            session_entries = sorted(os.scandir(animal.path),
                                     key=lambda e: e.name)
        except PermissionError:
            logger.warning("  Permission denied: %s", animal.path)
            continue

        for session in session_entries:
            if not session.is_dir(follow_symlinks=False):
                continue
            if verbose:
                logger.debug("  Session: %s", session.name)
            scan_leaf(session.path)

    return dirs


# ---------------------------------------------------------------------------
# Phase 2 — Pairing
# ---------------------------------------------------------------------------

def pair_files(dirs, logger, verbose):
    """Match tifs to zips using three strategies.

    Returns (pairs, unpaired_tifs, unpaired_zips) where pairs is a list of
    (tif_path, zip_path, strategy) tuples.
    """
    pairs = []
    unpaired_tifs = []
    unpaired_zips = []

    for dirpath in sorted(dirs.keys()):
        tifs = list(dirs[dirpath]["tifs"])
        zips = list(dirs[dirpath]["zips"])
        matched_tifs = set()
        matched_zips = set()

        # Strategy A: Exact stem match
        # FOO_mc.tif matches FOO_RoiSet.zip (strip _mc vs _RoiSet)
        tif_by_base = {}
        for t in tifs:
            name = os.path.basename(t)
            base = name.replace("_mc.tif", "")
            tif_by_base[base] = t

        zip_by_base = {}
        for z in zips:
            name = os.path.basename(z)
            base = name.replace("_RoiSet.zip", "")
            zip_by_base[base] = z

        for base in set(tif_by_base.keys()) & set(zip_by_base.keys()):
            t = tif_by_base[base]
            z = zip_by_base[base]
            pairs.append((t, z, "A:stem"))
            matched_tifs.add(t)
            matched_zips.add(z)
            if verbose:
                logger.debug("  PAIR [A] %s <-> %s",
                             os.path.basename(t), os.path.basename(z))

        remaining_tifs = [t for t in tifs if t not in matched_tifs]
        remaining_zips = [z for z in zips if z not in matched_zips]

        # Strategy B: Standalone zip (bare "RoiSet.zip") + single unpaired tif
        bare_zips = [z for z in remaining_zips
                     if os.path.basename(z) == "RoiSet.zip"]
        if len(bare_zips) == 1 and len(remaining_tifs) == 1:
            t = remaining_tifs[0]
            z = bare_zips[0]
            pairs.append((t, z, "B:standalone"))
            matched_tifs.add(t)
            matched_zips.add(z)
            if verbose:
                logger.debug("  PAIR [B] %s <-> %s",
                             os.path.basename(t), os.path.basename(z))
            remaining_tifs = [x for x in remaining_tifs if x not in matched_tifs]
            remaining_zips = [x for x in remaining_zips if x not in matched_zips]

        # Strategy C: Fuzzy match — one unpaired tif + one unpaired zip
        if len(remaining_tifs) == 1 and len(remaining_zips) == 1:
            t = remaining_tifs[0]
            z = remaining_zips[0]
            pairs.append((t, z, "C:fuzzy"))
            matched_tifs.add(t)
            matched_zips.add(z)
            logger.warning("  PAIR [C] fuzzy: %s <-> %s  (stems differ)",
                           os.path.basename(t), os.path.basename(z))
            remaining_tifs = []
            remaining_zips = []

        # Leftovers
        for t in remaining_tifs:
            unpaired_tifs.append(t)
            logger.warning("  UNPAIRED tif: %s", t)
        for z in remaining_zips:
            unpaired_zips.append(z)
            logger.warning("  UNPAIRED zip: %s", z)

    return pairs, unpaired_tifs, unpaired_zips


# ---------------------------------------------------------------------------
# Phase 3 — Collision Detection
# ---------------------------------------------------------------------------

def check_collisions(pairs, logger):
    """Verify all destination filenames are unique. Returns collision list."""
    tif_names = defaultdict(list)
    zip_names = defaultdict(list)

    for tif_path, zip_path, strategy in pairs:
        tif_name = os.path.basename(tif_path)
        tif_names[tif_name].append(tif_path)

        # Destination zip name is derived from the tif stem
        stem = tif_stem(tif_name)  # e.g. "FOO_mc"
        dest_zip_name = stem + ".zip"
        zip_names[dest_zip_name].append(zip_path)

    collisions = []
    for name, paths in tif_names.items():
        if len(paths) > 1:
            collisions.append(("tif", name, paths))
    for name, paths in zip_names.items():
        if len(paths) > 1:
            collisions.append(("zip", name, paths))

    for kind, name, paths in collisions:
        logger.error("COLLISION [%s] %s:", kind, name)
        for p in paths:
            logger.error("  %s", p)

    return collisions


# ---------------------------------------------------------------------------
# Phase 4 — Move
# ---------------------------------------------------------------------------

def execute_moves(pairs, raw_dir, annotated_dir, source, logger, dry_run=True):
    """Move paired files to flat directories.

    _mc.tif  → raw_dir/{original_name}
    *RoiSet.zip → annotated_dir/{tif_stem}.zip
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    moved = 0
    errors = 0
    total = len(pairs)

    for i, (tif_path, zip_path, strategy) in enumerate(pairs, 1):
        tif_name = os.path.basename(tif_path)
        stem = tif_stem(tif_name)  # e.g. "FOO_mc"
        dest_zip_name = stem + ".zip"

        dest_tif = os.path.join(raw_dir, tif_name)
        dest_zip = os.path.join(annotated_dir, dest_zip_name)

        if dry_run:
            logger.debug("[%d/%d] %s -> %s", i, total, tif_name, dest_tif)
            logger.debug("[%d/%d] %s -> %s", i, total,
                         os.path.basename(zip_path), dest_zip)
            moved += 1
            continue

        # Safety check
        if not safe_under_root(tif_path, source):
            logger.error("[%d/%d] SKIPPED (outside source): %s",
                         i, total, tif_path)
            errors += 1
            continue

        try:
            shutil.move(tif_path, dest_tif)
            shutil.move(zip_path, dest_zip)
            logger.debug("[%d/%d] Moved %s [%s]", i, total, tif_name, strategy)
            moved += 1
        except OSError as e:
            logger.error("[%d/%d] Failed to move pair %s: %s",
                         i, total, tif_name, e)
            errors += 1

    return moved, errors


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(pairs, unpaired_tifs, unpaired_zips, collisions, logger):
    """Print summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PAIRING SUMMARY")
    logger.info("=" * 70)
    logger.info("")

    strategy_counts = defaultdict(int)
    for _, _, s in pairs:
        strategy_counts[s] += 1

    logger.info("Paired files: %d", len(pairs))
    for s in sorted(strategy_counts):
        logger.info("  Strategy %s: %d", s, strategy_counts[s])
    logger.info("")

    if unpaired_tifs:
        logger.warning("Unpaired tifs (no matching zip): %d", len(unpaired_tifs))
        for t in unpaired_tifs:
            logger.warning("  %s", t)
        logger.info("")

    if unpaired_zips:
        logger.warning("Unpaired zips (no matching tif): %d", len(unpaired_zips))
        for z in unpaired_zips:
            logger.warning("  %s", z)
        logger.info("")

    if collisions:
        logger.error("COLLISIONS DETECTED: %d — aborting", len(collisions))
        logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Flatten nested imaging data into pipeline-ready directories.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually move files (default is dry-run)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file details during discovery and pairing")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Root of nested data tree")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                        help="Destination for _mc.tif files")
    parser.add_argument("--annotated-dir", default=DEFAULT_ANNOTATED_DIR,
                        help="Destination for renamed .zip files")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    logger.info("Source:        %s", args.source)
    logger.info("Raw dir:       %s", args.raw_dir)
    logger.info("Annotated dir: %s", args.annotated_dir)
    logger.info("Mode:          %s", "EXECUTE" if args.execute else "DRY-RUN")
    logger.info("")

    # Phase 1: Discovery
    dirs = discover(args.source, logger, args.verbose)

    total_tifs = sum(len(d["tifs"]) for d in dirs.values())
    total_zips = sum(len(d["zips"]) for d in dirs.values())
    logger.info("")
    logger.info("Discovered %d tifs and %d zips across %d directories.",
                total_tifs, total_zips, len(dirs))
    logger.info("")

    # Phase 2: Pairing
    pairs, unpaired_tifs, unpaired_zips = pair_files(dirs, logger, args.verbose)

    # Phase 3: Collision detection
    collisions = check_collisions(pairs, logger)

    # Report
    print_report(pairs, unpaired_tifs, unpaired_zips, collisions, logger)

    if collisions:
        logger.error("Fix collisions before proceeding.")
        return 1

    if not args.execute:
        logger.info("This was a DRY RUN. No files were moved.")
        logger.info("Re-run with --execute to move files.")
        return 0

    # Phase 4: Move
    logger.info("Moving %d pairs ...", len(pairs))
    moved, errors = execute_moves(pairs, args.raw_dir, args.annotated_dir,
                                  args.source, logger, dry_run=False)

    logger.info("")
    logger.info("Done. %d pairs moved, %d errors.", moved, errors)
    logger.info("  %s/*.tif", args.raw_dir)
    logger.info("  %s/*.zip", args.annotated_dir)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
