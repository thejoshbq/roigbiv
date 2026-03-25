"""
ROI G. Biv — Suite2p runner.

Provides
--------
run_suite2p_fov()   — run Suite2p on a single TIF file
run_suite2p_batch() — run Suite2p sequentially on a list of TIFs with progress

Both functions support resumability: a FOV is skipped if stat.npy already
exists in the expected output location.

Disk management: data.bin (~500 MB per FOV) is deleted after each FOV
completes, keeping peak disk usage bounded to approximately one FOV at a time.

Motion correction: controlled by ``do_registration``. Set False (default) for
pre-corrected (*_mc.tif) stacks. The parameter is scaffolded so it can be
toggled per-dataset without changing any other code.
"""
import os
import shutil
import tempfile
import time
from pathlib import Path


def _build_ops(input_dir, fs: float, tau: float = 1.0,
               anatomical_only: int = 0, do_registration: bool = False,
               cfg: dict = None) -> dict:
    """
    Construct a Suite2p ops dict, merging values from an optional pipeline
    config dict. CLI-supplied ``fs``, ``tau``, ``anatomical_only``, and
    ``do_registration`` always take precedence over config values.
    """
    from suite2p.default_ops import default_ops

    ops = default_ops()
    s2p_cfg = (cfg or {}).get("suite2p", {})

    ops.update({
        # ── Data ──────────────────────────────────────────────────────────
        "data_path":        [str(input_dir)],
        "save_folder":      "suite2p",
        "nplanes":          s2p_cfg.get("nplanes", 1),
        "nchannels":        s2p_cfg.get("nchannels", 1),
        "functional_chan":  s2p_cfg.get("functional_chan", 1),
        "tau":              tau,
        "fs":               fs,

        # ── Registration ──────────────────────────────────────────────────
        "do_registration":  1 if do_registration else 0,
        "nimg_init":        s2p_cfg.get("nimg_init", 300),
        "batch_size":       s2p_cfg.get("batch_size", 250),
        "smooth_sigma":     s2p_cfg.get("smooth_sigma", 1.15),
        "maxregshift":      s2p_cfg.get("maxregshift", 0.1),
        "nonrigid":         s2p_cfg.get("nonrigid", True),
        "block_size":       s2p_cfg.get("block_size", [128, 128]),

        # ── Detection ─────────────────────────────────────────────────────
        "spatial_scale":    s2p_cfg.get("spatial_scale", 0),
        "threshold_scaling": s2p_cfg.get("threshold_scaling", 1.0),
        "max_iterations":   s2p_cfg.get("max_iterations", 20),
        "connected":        s2p_cfg.get("connected", True),
        "nbinned":          s2p_cfg.get("nbinned", 5000),
        "allow_overlap":    s2p_cfg.get("allow_overlap", False),

        # ── Classification ────────────────────────────────────────────────
        "preclassify":      s2p_cfg.get("preclassify", 0.0),

        # ── Neuropil ──────────────────────────────────────────────────────
        "high_pass":             s2p_cfg.get("high_pass", 100),
        "inner_neuropil_radius": s2p_cfg.get("inner_neuropil_radius", 2),
        "min_neuropil_pixels":   s2p_cfg.get("min_neuropil_pixels", 350),

        # ── Spike deconvolution ───────────────────────────────────────────
        "spikedetect":      s2p_cfg.get("spikedetect", True),

        # ── Anatomical mode ───────────────────────────────────────────────
        # 0 = activity-based (temporal correlation)
        # 1 = anatomy-based  (mean-image morphology; safe with any Cellpose version)
        # 2 = Cellpose-backed — NOT USED (would require Suite2p to import Cellpose)
        "anatomical_only":  anatomical_only,
    })
    return ops


def run_suite2p_fov(tif_path, output_dir, fs: float,
                    anatomical_only: int = 0, tau: float = 1.0,
                    do_registration: bool = False, cfg: dict = None) -> bool:
    """
    Run Suite2p on a single TIF file.

    Output path convention
    ----------------------
    Suite2p names its output subdirectory after ``basename(data_path[0])``.
    We symlink the TIF into a temporary directory whose name equals the FOV
    stem (with ``_mc`` stripped) so the output lands at::

        output_dir/{stem}/suite2p/plane0/

    This matches the naming convention used throughout the rest of the pipeline.

    Resumability
    ------------
    If ``output_dir/{stem}/suite2p/plane0/stat.npy`` already exists, the FOV
    is skipped and the function returns ``False``.

    Disk management
    ---------------
    ``data.bin`` (~500 MB) is deleted after Suite2p completes; it is not
    needed for any downstream step.

    Parameters
    ----------
    tif_path        : path-like — path to the raw (or pre-corrected) TIF stack
    output_dir      : path-like — Suite2p output root
    fs              : float     — acquisition frame rate in Hz
    anatomical_only : int       — 0 (activity) or 1 (anatomy); never 2
    tau             : float     — GCaMP decay time constant in seconds
    do_registration : bool      — False = skip (stacks are pre-corrected)
    cfg             : dict      — optional pipeline YAML config (for advanced params)

    Returns
    -------
    bool — True if processed, False if skipped (already done).
    """
    from suite2p.run_s2p import run_s2p

    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    stem = tif_path.stem.replace("_mc", "")

    stat_path = output_dir / stem / "suite2p" / "plane0" / "stat.npy"
    if stat_path.exists():
        return False

    tmp_base = tempfile.mkdtemp()
    named_dir = Path(tmp_base) / stem
    named_dir.mkdir()
    os.symlink(tif_path.resolve(), named_dir / tif_path.name)

    try:
        ops = _build_ops(named_dir, fs, tau, anatomical_only, do_registration, cfg)
        ops["save_path0"] = str(output_dir / stem)
        run_s2p(ops)
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        data_bin = output_dir / stem / "suite2p" / "plane0" / "data.bin"
        if data_bin.exists():
            data_bin.unlink()

    return True


def run_suite2p_batch(tif_files, output_dir, fs: float,
                      anatomical_only: int = 0, tau: float = 1.0,
                      do_registration: bool = False, cfg: dict = None) -> None:
    """
    Run Suite2p on each TIF in *tif_files* sequentially.

    Prints per-FOV progress with wall-clock timing. Skips FOVs whose
    outputs already exist (resumability after Colab disconnects).

    Parameters
    ----------
    tif_files       : list of path-like
    output_dir      : path-like — Suite2p output root
    fs              : float     — acquisition frame rate in Hz
    anatomical_only : int       — 0 = activity pass, 1 = anatomy pass
    tau             : float     — GCaMP decay time constant in seconds
    do_registration : bool      — False = skip (stacks are pre-corrected)
    cfg             : dict      — optional pipeline YAML config
    """
    tif_files = [Path(p) for p in tif_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pass_name = "anatomy" if anatomical_only else "activity"
    n = len(tif_files)
    print(f"\nSuite2p {pass_name} pass — {n} FOV(s)"
          f"  [fs={fs} Hz  tau={tau} s  registration={'ON' if do_registration else 'OFF'}]")

    n_done = n_skip = n_err = 0

    for i, tif in enumerate(tif_files, 1):
        stem = tif.stem.replace("_mc", "")
        print(f"  [{i:>{len(str(n))}}/ {n}] {stem} ... ", end="", flush=True)
        t0 = time.time()
        try:
            processed = run_suite2p_fov(
                tif, output_dir, fs, anatomical_only, tau, do_registration, cfg
            )
            elapsed = time.time() - t0
            if processed:
                print(f"done ({elapsed:.0f}s)")
                n_done += 1
            else:
                print("skipped (already exists)")
                n_skip += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            n_err += 1

    status_parts = [f"{n_done} processed"]
    if n_skip:
        status_parts.append(f"{n_skip} skipped")
    if n_err:
        status_parts.append(f"{n_err} errors")
    print(f"\nSuite2p {pass_name} complete: {', '.join(status_parts)}  →  {output_dir}")
