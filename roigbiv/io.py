"""
ROI G. Biv — I/O utilities.

Provides:
  discover_tifs()          — recursively find TIF files; auto-assemble frame series
  assemble_frame_series()  — stream single-frame TIFs into a multi-frame stack
  extract_archive()        — unpack .tar.gz / .zip archives
  validate_tif()           — confirm a TIF is 3D multi-frame (frames × H × W)
  extract_projections()    — pull meanImg + Vcorr from Suite2p ops.npy
  download_model()         — fetch model checkpoint from URL with caching
"""
import logging
import os
import re
import sys
import tarfile
import warnings
import zipfile
from dataclasses import dataclass
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
# Single-frame TIF series support (PrairieView / Bruker and generic)
# ---------------------------------------------------------------------------

TIF_SUFFIXES = (".tif", ".tiff")

#: Directory names never descended into when hunting for frame series: the
#: pipeline's own output trees, plus the assembled-stack cache itself.
_SKIP_DIRS = frozenset({"output", "inference", "pipeline", "_stage", "_stacks"})

#: How many files are opened to confirm a candidate directory really holds a
#: one-frame-per-file series. Spread across the sequence rather than taken from
#: the front — a truncated or dtype-shifted tail is a real failure mode that
#: the first five frames would never reveal.
_VERIFY_SAMPLE = 5


@dataclass(frozen=True)
class _SeriesPattern:
    """One filename convention for a single-frame TIF series.

    ``regex`` must expose an ``index`` group and may expose ``cycle`` (an outer
    counter, ordered ahead of ``index``) and ``channel``. ``min_frames`` is the
    per-channel count below which a match is not worth trusting — low for a
    convention specific enough to be self-identifying, high for the generic
    fallback where a false positive is plausible.
    """

    name: str
    regex: "re.Pattern"
    min_frames: int
    require_common_prefix: bool = False


_SERIES_PATTERNS = (
    # PrairieView / Bruker: foo_Cycle00001_Ch2_000001.ome.tif. `cycle` matters —
    # in a multi-cycle T-series the frame index restarts each cycle, so ordering
    # on the trailing number alone interleaves the cycles.
    _SeriesPattern(
        name="PrairieView",
        regex=re.compile(r"_Cycle(?P<cycle>\d+)_(?P<channel>Ch\d+)_(?P<index>\d+)"
                         r"\.ome\.tiff?$", re.IGNORECASE),
        min_frames=2,
    ),
    # Anything ending in a run of digits: ScanImage, Thorlabs, hand-rolled
    # exports. Deliberately last and deliberately strict — see `min_frames` and
    # `require_common_prefix`.
    _SeriesPattern(
        name="numbered",
        regex=re.compile(r"(?:^|[_.\-])(?P<index>\d{3,})\.(?:ome\.)?tiff?$",
                         re.IGNORECASE),
        min_frames=8,
        require_common_prefix=True,
    ),
)


def _parse_series(files: list, pattern: _SeriesPattern) -> dict:
    """Group *files* into ``{channel: [path, ...]}`` ordered by frame position.

    Returns ``{}`` when *files* do not form a clean series under *pattern*. The
    strict part is the duplicate-key check: two files claiming the same position
    means the directory holds interleaved channels or two different movies, and
    concatenating them would silently produce a scrambled stack. Refusing the
    whole directory leaves the files to be reported individually, which is
    wrong but visible.
    """
    by_channel: dict = {}
    prefixes: set = set()
    for path in files:
        m = pattern.regex.search(path.name)
        if m is None:
            continue
        groups = m.groupdict()
        order = tuple(int(groups[g]) for g in ("cycle", "index")
                      if groups.get(g) is not None)
        by_channel.setdefault(groups.get("channel") or "", []).append((order, path))
        prefixes.add(path.name[:m.start()])

    if pattern.require_common_prefix and len(prefixes) > 1:
        return {}

    series: dict = {}
    for channel, items in by_channel.items():
        if len(items) < pattern.min_frames:
            continue
        if len({order for order, _ in items}) != len(items):
            return {}
        series[channel] = [p for _, p in sorted(items, key=lambda it: it[0])]
    return series


def _verify_frame_files(files: list) -> bool:
    """Confirm a sample of *files* really are single-page frames of one movie.

    Rejects a file that is unreadable, that is not a lone 2D frame (a chunked
    multi-frame stack named like a frame — concatenating those is a different
    operation), or whose shape/dtype differs from the rest of the sample.

    The shape test reads ``series[0]``, not ``pages``: tifffile stores a small
    3D array as a single page whose *page* shape is 3D, so a page count of one
    does not imply a single frame.
    """
    n = len(files)
    positions = sorted({round(i * (n - 1) / (_VERIFY_SAMPLE - 1))
                        for i in range(_VERIFY_SAMPLE)})
    reference = None
    for i in positions:
        try:
            with tifffile.TiffFile(str(files[i]), is_ome=False) as tif:
                series = tif.series
                if not series or len(series[0].shape) != 2:
                    return False
                signature = (tuple(series[0].shape), str(series[0].dtype))
        except Exception:
            return False
        if reference is None:
            reference = signature
        elif signature != reference:
            return False
    return True


def _detect_series_dirs(root: Path) -> list:
    """Walk *root* for directories holding a single-frame TIF series.

    Inspects *root* itself as well as every subdirectory, so pointing
    ``--input`` straight at a session directory behaves the same as pointing at
    its parent — the alternative is thousands of one-page "FOVs", each of which
    fails validation individually.

    Symlinked directories are followed — labs routinely symlink acquisition
    trees into a working directory — with a realpath set breaking loops and
    repeat visits.

    Returns ``[(directory, pattern, channel, series), ...]`` where ``series`` is
    the full ``{channel: [path, ...]}`` mapping and ``channel`` is the one to
    assemble.
    """
    found = []
    seen: set = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.startswith(".")]
        directory = Path(dirpath)
        files = [directory / f for f in filenames
                 if f.lower().endswith(TIF_SUFFIXES)]
        for pattern in _SERIES_PATTERNS:
            series = _parse_series(files, pattern)
            if not series:
                continue
            channel = max(series, key=lambda c: len(series[c]))
            if not _verify_frame_files(series[channel]):
                continue
            found.append((directory, pattern, channel, series))
            dirnames[:] = []  # frames are leaves; nothing below to recurse into
            break
    return found


def _series_stem(root: Path, directory: Path) -> str:
    """Flat, collision-free stack name for a series directory.

    An immediate child keeps its bare name so stacks assembled by earlier
    versions are still found in ``_stacks/``; anything deeper joins its path
    components rather than colliding on a shared leaf name.
    """
    parts = directory.relative_to(root).parts
    return "_".join(parts) if parts else root.name


def assemble_frame_series(session_dir, output_path, channel: str = None) -> Path:
    """
    Assemble a single-frame TIF series into one multi-frame stack.

    Recognises the conventions in :data:`_SERIES_PATTERNS` (PrairieView/Bruker
    ``*_CycleNNNNN_ChN_NNNNNN.ome.tif`` first, then any ``*_NNN.tif``-style
    numbering). Frames are ordered by their parsed position and streamed into a
    single BigTIFF at *output_path*. A ``.tmp.tif`` sidecar is used during
    writing so a partial file is never mistaken for a complete stack on re-run.

    Parameters
    ----------
    session_dir  : path-like — directory holding the per-frame TIFs
    output_path  : path-like — destination multi-frame TIF
    channel      : str or None — e.g. ``"Ch2"``; None = most-frequent channel

    Returns
    -------
    Path — output_path

    Raises
    ------
    ValueError if no series is recognised, or if *channel* is absent from one.
    """
    session_dir = Path(session_dir)
    files = [p for p in session_dir.iterdir()
             if p.is_file() and p.name.lower().endswith(TIF_SUFFIXES)]

    for pattern in _SERIES_PATTERNS:
        series = _parse_series(files, pattern)
        if not series:
            continue
        if channel is None:
            picked = max(series, key=lambda c: len(series[c]))
        elif channel in series:
            picked = channel
        else:
            raise ValueError(f"No {channel} frames found in {session_dir}")
        return _assemble_frames(series[picked], Path(output_path))

    raise ValueError(f"No single-frame TIF series found in {session_dir}")


def assemble_prairie_stack(session_dir, output_path, channel: str = None) -> Path:
    """Deprecated alias for :func:`assemble_frame_series`, which handles the
    PrairieView convention along with the rest."""
    return assemble_frame_series(session_dir, output_path, channel=channel)


def _assemble_frames(files: list, output_path: Path) -> Path:
    """Stream an ordered list of single-frame TIFs into one BigTIFF stack."""
    output_path = Path(output_path)
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
    session_name = files[0].parent.name
    if n_skipped:
        print(f"  WARNING: skipping {n_skipped} empty frame file(s) in {session_name}",
              flush=True)
    if not valid_files:
        raise ValueError(f"All {n} frame files in {session_name} are empty")
    n_valid = len(valid_files)

    _saved_level = _tifffile_log.level
    _tifffile_log.setLevel(logging.ERROR)
    n_corrupt = 0
    try:
        with warnings.catch_warnings(), tifffile.TiffWriter(str(tmp_path), bigtiff=True) as tw:
            warnings.simplefilter("ignore")
            first_frame = tifffile.imread(str(valid_files[0]), is_ome=False)
            size_est_gb = n_valid * first_frame.nbytes / 1e9
            print(f"  Assembling {n_valid} frames (~{size_est_gb:.1f} GB)"
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

    Single-frame series
    -------------------
    Directories holding one TIF per frame — *root* itself or any subdirectory —
    are auto-detected (PrairieView/Bruker ``*_CycleNNNNN_ChN_NNNNNN.ome.tif``
    first, then generic ``*_NNN.tif`` numbering; see :data:`_SERIES_PATTERNS`).
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

    # Detect and assemble single-frame TIF series. Only the files that actually
    # matched a series pattern are consumed — a genuine multi-frame stack living
    # alongside the frames stays an input in its own right.
    series_dirs = _detect_series_dirs(root)
    frame_files = {p for _, _, _, series in series_dirs
                   for paths in series.values() for p in paths}
    assembled_stacks: list = []
    stacks_dir = root / "_stacks"
    for directory, pattern, channel, series in series_dirs:
        files = series[channel]
        output_path = stacks_dir / f"{_series_stem(root, directory)}.tif"
        if output_path.exists():
            assembled_stacks.append(output_path)
            continue
        detail = f"{len(files)} frames" + (f", {channel}" if channel else "")
        print(f"{pattern.name} series detected: {directory.name} ({detail})")
        assembled_stacks.append(_assemble_frames(files, output_path))

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
        and p not in frame_files
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
