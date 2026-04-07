"""
ROI G. Biv — I/O utilities.

Provides:
  discover_tifs()       — recursively find TIF files; auto-extract archives
  extract_archive()     — unpack .tar.gz / .zip archives
  validate_tif()        — confirm a TIF is 3D (frames × H × W)
  extract_projections() — pull meanImg + Vcorr from Suite2p ops.npy
  download_model()      — fetch model checkpoint from URL with caching
"""
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# TIF discovery
# ---------------------------------------------------------------------------

def discover_tifs(root) -> list:
    """
    Recursively find all TIF files under *root*.

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

    # Collect all TIF files
    tif_files: set = set()
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tif_files.update(root.rglob(pattern))

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
# Validation
# ---------------------------------------------------------------------------

def validate_tif(path) -> tuple:
    """
    Verify that a TIF file is readable and three-dimensional (frames × H × W).

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
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{path.name}: cannot read TIF — {exc}") from exc

    if len(shape) != 3:
        raise ValueError(
            f"{path.name}: expected 3D array (frames × H × W), got shape {shape}. "
            f"Ensure this is a multi-frame TIF stack, not a single image."
        )
    return path.stem.replace("_mc", ""), shape


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
