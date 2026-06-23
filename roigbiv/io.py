"""
ROI G. Biv — I/O utilities.

Provides:
  discover_tifs()          — recursively find TIF files; auto-assemble PrairieView series
  assemble_prairie_stack() — stream PrairieView single-frame OME-TIFs into a multi-frame stack
  extract_archive()        — unpack .tar.gz / .zip archives
  validate_tif()           — confirm a TIF is 3D multi-frame (frames × H × W)
  extract_projections()    — pull meanImg + Vcorr from Suite2p ops.npy
  download_model()         — fetch model checkpoint from URL with caching
"""
import logging
import re
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import tifffile

_tifffile_log = logging.getLogger("tifffile")

# TIFF Software tag (305) value stamped onto every motion-corrected ``{stem}_mc.tif``
# the pipeline writes (registration.py / legacy_mc.py). Lets a corrected movie be
# recognised by content rather than filename — see ``detect_motion_corrected``.
MC_SOFTWARE_TAG = "roigbiv-mc"


# ---------------------------------------------------------------------------
# PrairieView / Bruker single-frame OME-TIFF support
# ---------------------------------------------------------------------------

_PRAIRIE_FRAME_PATTERN = re.compile(r'_Cycle\d+_Ch\d+_(\d+)\.ome\.tif$')


def _detect_prairie_sessions(root: Path) -> list:
    """
    Return immediate subdirectories of *root* that look like PrairieView sessions.

    Detection heuristic: the directory contains ≥2 files whose names match
    ``*_CycleNNNNN_ChN_NNNNNN.ome.tif``.
    """
    sessions = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        sample = list(subdir.glob("*.ome.tif"))[:5]
        if len(sample) >= 2 and all(_PRAIRIE_FRAME_PATTERN.search(f.name) for f in sample):
            sessions.append(subdir)
    return sessions


def assemble_prairie_stack(session_dir, output_path, channel: str = None) -> Path:
    """
    Assemble a PrairieView/Bruker single-frame OME-TIFF series into a multi-frame stack.

    Each frame is a separate ``*_CycleNNNNN_ChN_NNNNNN.ome.tif`` file.  Frames
    are sorted by their 6-digit suffix and streamed into a single BigTIFF stack
    at *output_path*.  A ``.tmp.tif`` sidecar is used during writing so a
    partial file is never mistaken for a complete stack on re-run.

    Parameters
    ----------
    session_dir  : path-like — PrairieView session directory
    output_path  : path-like — destination multi-frame TIF
    channel      : str or None — e.g. ``"Ch2"``; None = auto-detect (most-frequent channel)

    Returns
    -------
    Path — output_path
    """
    session_dir = Path(session_dir)
    output_path = Path(output_path)

    if channel is None:
        ch_counts: dict = {}
        for f in session_dir.glob("*.ome.tif"):
            m = re.search(r'_(Ch\d+)_', f.name)
            if m:
                ch_counts[m.group(1)] = ch_counts.get(m.group(1), 0) + 1
        if not ch_counts:
            raise ValueError(f"No PrairieView frames found in {session_dir}")
        channel = max(ch_counts, key=ch_counts.get)

    files = sorted(
        (f for f in session_dir.glob(f"*_{channel}_*.ome.tif")
         if _PRAIRIE_FRAME_PATTERN.search(f.name)),
        key=lambda p: int(_PRAIRIE_FRAME_PATTERN.search(p.name).group(1)),
    )
    if not files:
        raise ValueError(f"No {channel} frames found in {session_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.tif")

    n = len(files)

    # PrairieView OME-TIFs embed OME-XML that references every other frame in
    # the series.  Without is_ome=False, tifffile.imread loads the ENTIRE series
    # (all N frames) on each call — both wrong data and 4×N log warnings.
    # Suppress tifffile's UIC-tag parser chatter (julianday=0 timestamps) for
    # the duration of assembly via setLevel + warnings filter.
    # Pre-filter: drop zero-byte or missing files before opening the writer.
    # Bruker sometimes writes empty placeholder files mid-acquisition.
    valid_files = [f for f in files if f.stat().st_size > 0]
    n_skipped = n - len(valid_files)
    if n_skipped:
        print(f"  WARNING: skipping {n_skipped} empty frame file(s) in {session_dir.name}",
              flush=True)
    if not valid_files:
        raise ValueError(f"All {n} frame files in {session_dir.name} are empty")
    n_valid = len(valid_files)

    _saved_level = _tifffile_log.level
    _tifffile_log.setLevel(logging.ERROR)
    n_corrupt = 0
    try:
        with warnings.catch_warnings(), tifffile.TiffWriter(str(tmp_path), bigtiff=True) as tw:
            warnings.simplefilter("ignore")
            first_frame = tifffile.imread(str(valid_files[0]), is_ome=False)
            size_est_gb = n_valid * first_frame.nbytes / 1e9
            print(f"  Assembling {n_valid} {channel} frames (~{size_est_gb:.1f} GB)"
                  f" → {output_path.name}", flush=True)
            tw.write(first_frame, contiguous=True)
            written = 1
            for f in valid_files[1:]:
                try:
                    tw.write(tifffile.imread(str(f), is_ome=False), contiguous=True)
                    written += 1
                except tifffile.TiffFileError:
                    n_corrupt += 1
                    continue
                if written % 1000 == 0 or written == n_valid:
                    print(f"    {written}/{n_valid} frames written", flush=True)
        if n_corrupt:
            print(f"  WARNING: skipped {n_corrupt} corrupt frame file(s)", flush=True)
        tmp_path.rename(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    finally:
        _tifffile_log.setLevel(_saved_level)

    return output_path


# ---------------------------------------------------------------------------
# TIF discovery
# ---------------------------------------------------------------------------

def discover_tifs(root) -> list:
    """
    Recursively find all TIF files under *root*.

    PrairieView/Bruker sessions
    ---------------------------
    Sub-directories containing single-frame ``*.ome.tif`` files named with the
    PrairieView ``*_CycleNNNNN_ChN_NNNNNN.ome.tif`` convention are auto-detected.
    Their frames are assembled (once, cached) into a single multi-frame stack
    under ``{root}/_stacks/{session_name}.tif``.  The individual per-frame files
    are excluded from the return value.

    Before scanning, automatically extracts any .tar.gz, .tgz, .tar.bz2, or
    .zip archives found anywhere under *root*.

    Returns
    -------
    list of Path — sorted, deduplicated TIF paths.

    Raises
    ------
    FileNotFoundError if *root* does not exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    # Extract archives before scanning
    archive_suffixes = (".tar.gz", ".tgz", ".tar.bz2", ".tar", ".zip")
    for archive in sorted(root.rglob("*")):
        if any(archive.name.lower().endswith(s) for s in archive_suffixes):
            stem = archive.name
            for _sfx in (".tar.gz", ".tar.bz2", ".tgz", ".tar", ".zip"):
                if stem.lower().endswith(_sfx):
                    stem = stem[: -len(_sfx)]
                    break
            dest = archive.parent / stem
            if not dest.exists():
                try:
                    extract_archive(archive, dest)
                    print(f"  Extracted: {archive.name} → {dest.name}/")
                except Exception as e:
                    print(f"  WARNING: could not extract {archive.name}: {e}")

    # Detect and assemble PrairieView session directories
    prairie_sessions = _detect_prairie_sessions(root)
    prairie_frame_dirs = set(prairie_sessions)
    assembled_stacks: list = []
    if prairie_sessions:
        stacks_dir = root / "_stacks"
        for session_dir in prairie_sessions:
            output_path = stacks_dir / f"{session_dir.name}.tif"
            if not output_path.exists():
                print(f"PrairieView session detected: {session_dir.name}")
                assembled_stacks.append(assemble_prairie_stack(session_dir, output_path))
            else:
                assembled_stacks.append(output_path)

    # Collect all TIF files
    tif_files: set = set()
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tif_files.update(root.rglob(pattern))

    # Exclude pipeline outputs, staging copies, stacks dir, and prairie frame dirs.
    # "inference"/"pipeline" guard the default output tree so an exported
    # {stem}_mc.tif is never re-discovered as a pre-corrected input.
    tif_files = {
        p for p in tif_files
        if "output" not in p.parts
        and "inference" not in p.parts
        and "pipeline" not in p.parts
        and "_stage" not in p.parts
        and "_stacks" not in p.parts
        and not any(p.is_relative_to(d) for d in prairie_frame_dirs)
    }

    # Add assembled stacks
    tif_files.update(assembled_stacks)

    return sorted(tif_files)


def extract_archive(archive_path, extract_to=None) -> Path:
    """
    Extract a .tar.gz, .tgz, .tar.bz2, .tar, or .zip archive.

    Parameters
    ----------
    archive_path : path-like
    extract_to   : path-like or None — defaults to archive parent / stem

    Returns
    -------
    Path — the extraction directory.
    """
    archive_path = Path(archive_path)
    if extract_to is None:
        # Strip all extensions: "foo.tar.gz" → "foo"
        stem = archive_path.name
        for suffix in (".tar.gz", ".tar.bz2", ".tgz", ".tar", ".zip"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        extract_to = archive_path.parent / stem

    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        with tarfile.open(archive_path) as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(extract_to, filter="data")
            else:
                tf.extractall(extract_to)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    return extract_to


# ---------------------------------------------------------------------------
# Motion-correction detection
# ---------------------------------------------------------------------------

def detect_motion_corrected(path) -> tuple:
    """Decide whether *path* is an already motion-corrected stack.

    Two-tier, content-first — so the decision is no longer a bare filename
    substring test:

      1. ``"metadata"`` — the TIFF Software tag (305) starts with
         :data:`MC_SOFTWARE_TAG`. Stamped by roigbiv's own MC writers, this
         survives renames and is authoritative for pipeline-produced movies.
      2. ``"filename"`` — the stem ends in the ``_mc`` suffix convention. Covers
         externally pre-corrected inputs (and legacy roigbiv outputs predating the
         tag) that carry the lab convention but no embedded marker. Strict suffix,
         not a substring, so names like ``exp_mcg_001`` / ``foo_mc_raw`` are *not*
         misread as corrected.

    Metadata wins over filename. A tag-read failure (unreadable/odd TIFF) degrades
    silently to the filename tier — never raises.

    Returns
    -------
    (corrected, signal) : (bool, str) — ``signal`` ∈ {"metadata", "filename", "none"}.
    """
    path = Path(path)
    try:
        with tifffile.TiffFile(str(path)) as tif:
            tag = tif.pages[0].tags.get("Software")
            value = tag.value if tag is not None else None
        if isinstance(value, bytes):
            value = value.decode("ascii", "ignore")
        if value and str(value).startswith(MC_SOFTWARE_TAG):
            return True, "metadata"
    except Exception:
        pass  # unreadable Software tag — fall through to the filename tier
    if path.stem.endswith("_mc"):
        return True, "filename"
    return False, "none"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tif(path) -> tuple:
    """
    Verify that a TIF file is a valid multi-frame monochrome stack (frames × H × W).

    Returns
    -------
    (stem, shape) on success.

    Raises
    ------
    ValueError with a descriptive message on failure.
    """
    path = Path(path)
    try:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series
            if not series:
                raise ValueError("TIF contains no image series")
            shape = series[0].shape
            dtype = series[0].dtype
            n_pages = len(tif.pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{path.name}: cannot read TIF — {exc}") from exc

    # Single-page files are reference/display images, not time-series stacks.
    # tifffile may report shape (H, W, C) for a single RGBA page, which passes
    # the len==3 check below — catch it here first.
    if n_pages < 2:
        raise ValueError(
            f"{path.name}: only {n_pages} page(s) — this looks like a static "
            f"reference image (shape {shape}, dtype {dtype}), not a multi-frame "
            f"time-series stack. Ensure you are pointing at the functional "
            f"imaging movie, not a reference/projection TIF."
        )

    if len(shape) != 3:
        raise ValueError(
            f"{path.name}: expected 3D array (frames × H × W), got shape {shape}. "
            f"Ensure this is a multi-frame TIF stack, not a single image."
        )
    return path.stem.replace("_mc", ""), shape


def read_tiff_optics_metadata(path) -> dict:
    """Best-effort optics metadata (pixel size) for FOV auto-classification.

    Reads, in priority order: OME-XML ``PhysicalSizeX``, ScanImage
    ``SI.objectiveResolution`` (µm/deg, not directly µm/px — skipped unless a
    pixel-size is derivable), and the TIFF ``XResolution``/``ResolutionUnit``
    tags. Returns ``{"pixel_size_um": float}`` when a µm/px value is recoverable,
    else ``{}``.

    **Total** — never raises; absence of metadata is the common case. Used only
    as a tiebreaker/confidence booster by ``optics.classify_optics_prior``; the
    frame-size prior never depends on it.
    """
    path = Path(path)
    try:
        with tifffile.TiffFile(str(path)) as tif:
            # 1) OME-XML PhysicalSizeX (µm by default).
            ome = getattr(tif, "ome_metadata", None)
            if ome:
                import re
                m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
                unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome)
                if m:
                    import unicodedata
                    val = float(m.group(1))
                    # OME-XML uses U+03BC (Greek mu); TIFF tags often U+00B5
                    # (micro sign). NFKC folds the micro sign to Greek mu; fold
                    # that to ASCII 'u' so every form of "µm" matches.
                    u = (unicodedata.normalize("NFKC", unit.group(1) if unit else "um")
                         .strip().lower().replace("μ", "u"))
                    if u in ("um", "micron", "microns", "micrometer", "micrometre"):
                        return {"pixel_size_um": val}
                    if u == "nm":
                        return {"pixel_size_um": val / 1000.0}

            # 2) Baseline TIFF resolution tags (pixels per ResolutionUnit).
            page = tif.pages[0]
            tags = page.tags
            xres = tags.get("XResolution")
            runit = tags.get("ResolutionUnit")
            if xres is not None and xres.value:
                num, den = xres.value if isinstance(xres.value, tuple) else (xres.value, 1)
                if num:
                    px_per_unit = float(num) / float(den or 1)
                    unit_um = {2: 25400.0, 3: 10000.0}.get(
                        int(runit.value) if runit is not None else 0
                    )  # 2=inch, 3=cm → µm per unit
                    if unit_um and px_per_unit > 0:
                        return {"pixel_size_um": unit_um / px_per_unit}
    except Exception:
        return {}
    return {}


# ---------------------------------------------------------------------------
# Projection extraction
# ---------------------------------------------------------------------------

def extract_projections(s2p_activity_dir, out_dir, max_proj_dir=None) -> int:
    """
    Extract mean, Vcorr, and max projections from Suite2p output.

    Reads ``ops.npy`` from every FOV in *s2p_activity_dir* and writes:
      ``{out_dir}/{stem}_mean.tif``      — float32 time-averaged projection
      ``{out_dir}/{stem}_vcorr.tif``     — float32 Vcorr map (if available)
      ``{max_proj_dir}/{stem}_max.tif``  — float32 max projection (if available)

    If *max_proj_dir* is None, max projections are written to *out_dir*.

    Parameters
    ----------
    s2p_activity_dir : path-like — Suite2p output directory
    out_dir          : path-like — where to write mean/Vcorr TIF projections
    max_proj_dir     : path-like or None — where to write max projections

    Returns
    -------
    int — number of FOVs processed.
    """
    s2p_activity_dir = Path(s2p_activity_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_proj_dir is not None:
        max_proj_dir = Path(max_proj_dir)
        max_proj_dir.mkdir(parents=True, exist_ok=True)
    else:
        max_proj_dir = out_dir

    fov_dirs = sorted(d for d in s2p_activity_dir.iterdir() if d.is_dir())
    n = 0

    for fov_dir in fov_dirs:
        ops_path = fov_dir / "suite2p" / "plane0" / "ops.npy"
        if not ops_path.exists():
            continue

        stem = fov_dir.name
        ops = np.load(str(ops_path), allow_pickle=True).item()
        parts = []

        if "meanImg" in ops:
            mean = ops["meanImg"].astype(np.float32)
            tifffile.imwrite(str(out_dir / f"{stem}_mean.tif"), mean)
            parts.append("mean")

        if "Vcorr" in ops:
            vcorr = ops["Vcorr"].astype(np.float32)
            tifffile.imwrite(str(out_dir / f"{stem}_vcorr.tif"), vcorr)
            parts.append("vcorr")

        if "max_proj" in ops:
            max_proj = ops["max_proj"].astype(np.float32)
            tifffile.imwrite(str(max_proj_dir / f"{stem}_max.tif"), max_proj)
            parts.append("max")

        if parts:
            n += 1
            print(f"  {stem}: saved {', '.join(parts)}")
        else:
            print(f"  {stem}: WARNING — no meanImg, Vcorr, or max_proj in ops.npy")

    print(f"\nExtracted projections for {n} FOVs → {out_dir}")
    return n


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model(url: str, cache_path) -> Path:
    """
    Download a Cellpose model checkpoint from *url* to *cache_path*.

    Skips the download if *cache_path* already exists (safe to re-run).

    Parameters
    ----------
    url        : str — direct download URL for the checkpoint file
    cache_path : path-like — local destination

    Returns
    -------
    Path — the local model path (suitable for CellposeModel(pretrained_model=...)).
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Model already cached: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from:\n  {url}")

    _progress_state = {"last_pct": -1}

    def _reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, block_num * block_size * 100 // total_size)
        if pct != _progress_state["last_pct"]:
            print(f"\r  {pct:3d}%", end="", flush=True)
            _progress_state["last_pct"] = pct

    try:
        urlretrieve(url, str(cache_path), reporthook=_reporthook)
    except Exception:
        if cache_path.exists():
            cache_path.unlink()
        raise
    print(f"\nModel saved: {cache_path}  ({cache_path.stat().st_size / 1e6:.1f} MB)")
    return cache_path
