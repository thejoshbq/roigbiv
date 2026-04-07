"""
ROI G. Biv — Streamlit web interface for the three-branch ROI detection pipeline.

Launch with:  streamlit run app.py
Access from LAN:  streamlit run app.py --server.address 0.0.0.0
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile
import torch
import yaml

# ── Project root & defaults ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "deployed" / "current_model"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


def _load_config():
    if DEFAULT_CONFIG.exists():
        with open(DEFAULT_CONFIG) as f:
            return yaml.safe_load(f)
    return {}


# ── Branch A helper: Cellpose inference ──────────────────────────────────────

def _run_cellpose_branch(
    projections_dir: Path,
    out_dir: Path,
    model_path: str,
    diameter: float,
    flow_threshold: float,
    cellprob_threshold: float,
    tile_norm_blocksize: int,
    use_vcorr: bool,
    denoise: bool,
    status_writer=None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Run Cellpose on mean projections. Returns (results, errors)."""
    from cellpose import models

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    has_denoise = False
    if denoise:
        try:
            from cellpose import denoise as cp_denoise
            model = cp_denoise.CellposeDenoiseModel(
                gpu=torch.cuda.is_available(),
                pretrained_model=str(model_path),
                restore_type="denoise_cyto3",
            )
            has_denoise = True
        except Exception:
            model = models.CellposeModel(
                gpu=torch.cuda.is_available(), pretrained_model=str(model_path))
    else:
        model = models.CellposeModel(
            gpu=torch.cuda.is_available(), pretrained_model=str(model_path))

    channels = [1, 2] if use_vcorr else [0, 0]
    normalize = ({"tile_norm_blocksize": tile_norm_blocksize}
                 if tile_norm_blocksize > 0 else True)

    mean_files = sorted(projections_dir.glob("*_mean.tif"))
    results, errors = [], []

    for i, tif in enumerate(mean_files):
        stem = tif.stem.replace("_mean", "")
        mask_path = out_dir / f"{stem}_masks.tif"

        if mask_path.exists():
            n = int(tifffile.imread(str(mask_path)).max())
            results.append({"stem": stem, "n_rois": n, "skipped": True})
            if status_writer:
                status_writer.write(f"  {stem}: skipped ({n} ROIs)")
            continue

        try:
            img = tifffile.imread(str(tif)).astype(np.float32)
            if use_vcorr:
                vcorr_path = projections_dir / f"{stem}_vcorr.tif"
                if vcorr_path.exists():
                    vcorr = tifffile.imread(str(vcorr_path)).astype(np.float32)
                    img = np.stack([img, vcorr], axis=-1)
                else:
                    img = np.stack([img, np.zeros_like(img)], axis=-1)

            eval_kw = dict(
                diameter=diameter, channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize=normalize,
            )
            if has_denoise:
                masks, flows, styles, _ = model.eval(img, **eval_kw)
            else:
                masks, flows, styles = model.eval(img, **eval_kw)

            tifffile.imwrite(str(mask_path), masks.astype(np.uint16))
            n = int(masks.max())
            results.append({"stem": stem, "n_rois": n, "skipped": False})
            if status_writer:
                status_writer.write(f"  [{i+1}/{len(mean_files)}] {stem}: {n} ROIs")
        except Exception as e:
            errors.append((stem, str(e)))
            if status_writer:
                status_writer.write(f"  {stem}: ERROR — {e}")

    return results, errors


# ── Branch C helper: Tonic detection ─────────────────────────────────────────

def _run_tonic_branch(
    s2p_dir: Path,
    out_dir: Path,
    fs: float,
    n_components: int = 500,
    soma_radius: int = 8,
    corr_threshold: float = 0.25,   # was 0.15
    min_size: int = 80,
    max_size: int = 350,             # was 300
    min_solidity: float = 0.6,
    max_eccentricity: float = 0.85,
    status_writer=None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Run tonic detection on all FOVs with data.bin. Returns (results, errors)."""
    from roigbiv.tonic import run_tonic_detection

    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover FOVs
    fovs = []
    if s2p_dir.exists():
        for fov_dir in sorted(d for d in s2p_dir.iterdir() if d.is_dir()):
            plane = fov_dir / "suite2p" / "plane0"
            bp, op = plane / "data.bin", plane / "ops.npy"
            if bp.exists() and op.exists():
                fovs.append({"stem": fov_dir.name, "bin_path": bp, "ops_path": op})

    results, errors = [], []

    for i, fov in enumerate(fovs):
        stem = fov["stem"]
        mask_path = out_dir / f"{stem}_tonic_masks.tif"

        if mask_path.exists():
            n = int(tifffile.imread(str(mask_path)).max())
            results.append({"stem": stem, "n_rois": n, "skipped": True})
            if status_writer:
                status_writer.write(f"  {stem}: skipped ({n} ROIs)")
            continue

        try:
            masks, corr_map, info = run_tonic_detection(
                bin_path=fov["bin_path"], ops_path=fov["ops_path"], fs=fs,
                n_components=n_components, soma_radius=soma_radius,
                corr_threshold=corr_threshold, min_size=min_size, max_size=max_size,
                min_solidity=min_solidity, max_eccentricity=max_eccentricity,
            )
            tifffile.imwrite(str(mask_path), masks)
            tifffile.imwrite(str(out_dir / f"{stem}_corr_map.tif"), corr_map)
            n = info["n_rois"]
            results.append({"stem": stem, "n_rois": n, "skipped": False})
            if status_writer:
                status_writer.write(f"  [{i+1}/{len(fovs)}] {stem}: {n} tonic ROIs")
        except Exception as e:
            errors.append((stem, str(e)))
            if status_writer:
                status_writer.write(f"  {stem}: ERROR — {e}")

    return results, errors


# ── Stem discovery ───────────────────────────────────────────────────────────

def _discover_stems(cellpose_dir: Path, s2p_dir: Path, tonic_dir: Path) -> list[str]:
    """Find all FOV stems across the three branch output directories."""
    stems = set()
    if cellpose_dir.exists():
        for p in cellpose_dir.glob("*_masks.tif"):
            stems.add(p.stem.replace("_masks", ""))
    if s2p_dir.exists():
        for d in s2p_dir.iterdir():
            if d.is_dir() and (d / "suite2p" / "plane0" / "stat.npy").exists():
                stems.add(d.name)
    if tonic_dir.exists():
        for p in tonic_dir.glob("*_tonic_masks.tif"):
            stems.add(p.stem.replace("_tonic_masks", ""))
    return sorted(stems)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="ROI G. Biv", layout="wide")
    st.title("ROI G. Biv")
    st.caption("Three-branch ROI detection for two-photon calcium imaging")

    cfg = _load_config()
    cp_cfg = cfg.get("cellpose", {})
    tc_cfg = cfg.get("tonic", {})
    mg_cfg = cfg.get("merge", {})
    cl_cfg = cfg.get("classify", {})
    tr_cfg = cfg.get("traces", {})
    s2p_cfg = cfg.get("suite2p", {})

    # ── GPU status ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.success(f"GPU: {gpu_name} ({vram:.1f} GB)")
    else:
        st.warning("No GPU detected — Cellpose and Suite2p will run on CPU (slower).")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Pipeline Parameters")

        tif_dir = st.text_input(
            "TIF directory",
            placeholder="/path/to/your/tif/stacks",
            help="Directory containing pre-motion-corrected TIF stacks",
        )
        output_dir = st.text_input(
            "Output directory",
            placeholder="/path/to/output",
            help="Where all pipeline outputs will be saved",
        )

        st.divider()
        fs = st.number_input("Frame rate (Hz)", value=30.0, min_value=1.0, step=1.0)
        tau = st.number_input(
            "GCaMP tau (s)", value=float(s2p_cfg.get("tau", 1.0)),
            min_value=0.1, step=0.1,
            help="GCaMP6s=1.0, GCaMP6f=0.4, GCaMP7f=0.7")
        model_path = st.text_input("Model checkpoint", value=str(DEFAULT_MODEL))
        do_registration = st.checkbox(
            "Run motion correction", value=False,
            help="Enable if TIFs are NOT pre-corrected (_mc suffix)")

        # Branch A: Cellpose
        with st.expander("Cellpose (Branch A)"):
            diameter = st.number_input(
                "Cell diameter (px)", value=int(cp_cfg.get("diameter", 12)),
                min_value=5, step=1)
            flow_threshold = st.number_input(
                "Flow threshold", value=float(cp_cfg.get("flow_threshold", 0.6)),
                min_value=0.0, max_value=2.0, step=0.1,
                help="Higher = more permissive boundaries")
            cellprob_threshold = st.number_input(
                "Cell probability threshold",
                value=float(cp_cfg.get("cellprob_threshold", -2.0)),
                min_value=-6.0, max_value=6.0, step=0.5,
                help="Lower = accepts dimmer cells")
            tile_norm_blocksize = st.number_input(
                "Tile norm block size", value=int(cp_cfg.get("tile_norm_blocksize", 128)),
                min_value=0, step=32,
                help="Compensates GRIN vignetting (0 = standard normalization)")
            use_vcorr = st.checkbox(
                "Use Vcorr channel", value=True,
                help="Stack Vcorr as 2nd Cellpose input channel")
            denoise = st.checkbox(
                "Cellpose3 denoising", value=bool(cp_cfg.get("denoise", False)),
                help="Apply image restoration before segmentation")

        # Branch C: Tonic
        with st.expander("Tonic Detection (Branch C)"):
            tonic_band = st.selectbox(
                "Frequency band", ["neuronal", "astrocyte"],
                help="neuronal: 0.05-2.0 Hz, astrocyte: 0.01-0.3 Hz")
            corr_threshold = st.number_input(
                "Correlation threshold", value=float(tc_cfg.get("corr_threshold", 0.25)),
                min_value=0.0, max_value=1.0, step=0.05)
            tonic_min_size = st.number_input(
                "Min ROI size (px)", value=int(tc_cfg.get("min_roi_pixels", 80)),
                min_value=10, step=10)
            tonic_max_size = st.number_input(
                "Max ROI size (px)", value=int(tc_cfg.get("max_roi_pixels", 350)),
                min_value=50, step=50)
            tonic_min_solidity = st.number_input(
                "Min solidity", value=float(tc_cfg.get("min_solidity", 0.6)),
                min_value=0.0, max_value=1.0, step=0.05,
                help="Minimum area/convex-hull ratio; rejects spindly neuropil fragments")
            tonic_max_eccentricity = st.number_input(
                "Max eccentricity", value=float(tc_cfg.get("max_eccentricity", 0.85)),
                min_value=0.0, max_value=1.0, step=0.05,
                help="Maximum eccentricity of equivalent ellipse; rejects elongated fragments")
            n_components = st.number_input(
                "SVD components", value=int(tc_cfg.get("n_components", 500)),
                min_value=50, step=50)
            soma_radius = st.number_input(
                "Soma radius (px)", value=int(tc_cfg.get("soma_radius", 8)),
                min_value=2, step=1)

        # Merge & Classification
        with st.expander("Merge & Classification"):
            iou_threshold = st.slider(
                "IoU threshold", 0.1, 0.8,
                float(mg_cfg.get("iou_threshold", 0.3)), 0.05,
                help="Minimum spatial overlap to match ROIs across branches")
            snr_min = st.number_input(
                "Min SNR", value=float(cl_cfg.get("snr_min", 2.0)),
                min_value=0.0, step=0.5,
                help="Minimum signal-to-noise for cell acceptance")
            area_min = st.number_input(
                "Min area (px)", value=int(cl_cfg.get("area_min", 30)),
                min_value=5, step=5)
            area_max = st.number_input(
                "Max area (px)", value=int(cl_cfg.get("area_max", 500)),
                min_value=50, step=50)
            compact_min = st.number_input(
                "Min compactness", value=float(cl_cfg.get("compact_min", 0.15)),
                min_value=0.0, max_value=1.0, step=0.05,
                help="Circularity threshold (0-1)")
            require_spatial_support = st.checkbox(
                "Require spatial support for Branch C-only ROIs",
                value=bool(mg_cfg.get("require_spatial_support", True)),
                help="Discard C-only ROIs whose centroid falls below the FOV 25th-percentile "
                     "intensity in the mean projection (rejects dark-region neuropil artifacts)")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_instr, tab_run, tab_results = st.tabs(
        ["Instructions", "Run Pipeline", "View Results"])

    with tab_instr:
        _instructions_tab()

    with tab_run:
        _run_tab(
            tif_dir=tif_dir, output_dir=output_dir, fs=fs, tau=tau,
            do_registration=do_registration, model_path=model_path,
            cfg=cfg,
            # Cellpose params
            diameter=diameter, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            tile_norm_blocksize=tile_norm_blocksize,
            use_vcorr=use_vcorr, denoise=denoise,
            # Tonic params
            n_components=n_components, soma_radius=soma_radius,
            corr_threshold=corr_threshold,
            tonic_min_size=tonic_min_size, tonic_max_size=tonic_max_size,
            tonic_min_solidity=tonic_min_solidity,
            tonic_max_eccentricity=tonic_max_eccentricity,
            # Merge/classify params
            iou_threshold=iou_threshold, snr_min=snr_min,
            area_min=area_min, area_max=area_max, compact_min=compact_min,
            require_spatial_support=require_spatial_support,
        )

    with tab_results:
        _results_tab(output_dir)


# ── Instructions tab ─────────────────────────────────────────────────────────

def _instructions_tab():
    st.markdown("""
## What is ROI G. Biv?

ROI G. Biv is a three-branch consensus pipeline for detecting regions of interest
(ROIs) in two-photon calcium imaging data acquired through GRIN lenses. It combines
three independent detection methods, merges their results, extracts fluorescence
traces, and classifies each ROI by quality and activity type.

---

## Required Input

| What | Format | Notes |
|------|--------|-------|
| **TIF stacks** | Multi-frame TIFF (frames x height x width) | One file per field of view (FOV). Pre-motion-corrected files should have `_mc` in the filename (registration will be skipped automatically). |
| **Model checkpoint** | Cellpose model file | A fine-tuned Cellpose model for your tissue type. The default model is pre-loaded if one has been deployed. |

All TIF files must be acquired at the **same frame rate**. Place them in a single
directory (subdirectories are searched recursively).

---

## Pipeline Overview (7 Stages)

| Stage | What it does |
|-------|-------------|
| **1. Suite2p** | Runs Suite2p on each TIF stack to detect ROI candidates (Branch B) and produce a registered movie (data.bin) for downstream steps. |
| **2. Projections** | Extracts mean and Vcorr (temporal correlation) projection images from Suite2p output. These serve as input for Cellpose. |
| **3. Cellpose** | Runs the trained Cellpose model on mean+Vcorr projections to segment cell bodies (Branch A). |
| **4. Tonic Detection** | Detects tonically active neurons via bandpass filtering and local correlation in SVD space (Branch C). These are low-variance neurons missed by Suite2p and Cellpose. |
| **5. Three-Way Merge** | Matches ROIs across all three branches using pairwise IoU and assigns confidence tiers (ABC, AB, AC, BC, A, B, C) based on which branches agree. |
| **6. Trace Extraction** | Extracts raw fluorescence (F), neuropil (Fneu), corrected dF/F, and deconvolved spikes for every merged ROI from the registered movie. |
| **7. Classification** | Computes QC features (SNR, area, compactness), rejects non-cells, and labels activity types: phasic, tonic, sparse, or ambiguous. |

---

## How to Use

1. Enter the **TIF directory** and **output directory** in the sidebar
2. Set the **frame rate** to match your acquisition
3. Adjust parameters in the sidebar expanders if needed (defaults work for GRIN lens data)
4. Switch to the **Run Pipeline** tab and click **Run Pipeline**
5. After completion, switch to the **View Results** tab to explore the output

The pipeline is **resumable** — if interrupted, click Run Pipeline again and
completed stages will be skipped automatically.

---

## Key Parameters

| Parameter | Default | When to change |
|-----------|---------|----------------|
| Frame rate | 30 Hz | Must match your acquisition rate |
| GCaMP tau | 1.0 s | GCaMP6f: 0.4, GCaMP7f: 0.7, GCaMP8: 0.3 |
| Cell diameter | 12 px | Adjust for your optics/magnification |
| Motion correction | OFF | Enable if TIFs are NOT pre-corrected |
| IoU threshold | 0.3 | Higher = stricter cross-branch matching |
| Min SNR | 2.0 | Lower = keep dimmer cells |

---

## Output Files

| Directory | Contents |
|-----------|----------|
| `suite2p/` | Suite2p output per FOV (stat.npy, ops.npy, data.bin) |
| `projections/` | Mean and Vcorr projection images |
| `cellpose/` | Branch A: uint16 labeled ROI masks |
| `tonic/` | Branch C: tonic ROI masks and correlation maps |
| `merged/` | Merged masks, per-ROI records CSVs, merge_summary.csv |
| `traces/` | F, Fneu, dF/F, spikes, alpha arrays per FOV (.npy) |
| `classified/` | Per-FOV classification CSVs and classification_summary.csv |

The final output — `classified/classification_summary.csv` — contains one row per
ROI with QC features, cell/not-cell status, and activity type label.
""")


# ── Run tab ──────────────────────────────────────────────────────────────────

def _run_tab(*, tif_dir, output_dir, fs, tau, do_registration, model_path, cfg,
             diameter, flow_threshold, cellprob_threshold, tile_norm_blocksize,
             use_vcorr, denoise, n_components, soma_radius, corr_threshold,
             tonic_min_size, tonic_max_size, tonic_min_solidity, tonic_max_eccentricity,
             iou_threshold, snr_min, area_min, area_max, compact_min,
             require_spatial_support):

    if not tif_dir or not output_dir:
        st.info("Enter a TIF directory and output directory in the sidebar to begin.")
        return

    tr_cfg = cfg.get("traces", {})

    tif_path = Path(tif_dir)
    out_path = Path(output_dir)

    if not tif_path.exists():
        st.error(f"TIF directory not found: {tif_path}")
        return

    # Discover TIFs
    from roigbiv.io import discover_tifs, validate_tif

    tif_files = discover_tifs(tif_path)
    if not tif_files:
        st.error(f"No TIF files found under {tif_path}")
        return

    st.subheader(f"Found {len(tif_files)} TIF file(s)")
    with st.expander("File details", expanded=False):
        for tif in tif_files:
            try:
                stem, shape = validate_tif(tif)
                st.text(f"  {tif.name}  —  {shape[0]} frames, "
                        f"{shape[1]}x{shape[2]} px")
            except ValueError as e:
                st.warning(str(e))

    # Run button
    if not st.button("Run Pipeline", type="primary", use_container_width=True):
        return

    # Output subdirectories
    s2p_dir = out_path / "suite2p"
    proj_dir = out_path / "projections"
    cp_dir = out_path / "cellpose"
    tonic_dir = out_path / "tonic"
    merged_dir = out_path / "merged"
    traces_dir = out_path / "traces"
    classified_dir = out_path / "classified"

    progress = st.progress(0, text="Starting pipeline...")
    all_errors = []

    # ── Stage 1: Suite2p (Branch B) ──────────────────────────────────────
    with st.status("Stage 1/7: Suite2p processing...", expanded=True) as status:
        from roigbiv.suite2p import run_suite2p_batch
        t0 = time.time()
        run_suite2p_batch(
            tif_files, output_dir=str(s2p_dir),
            fs=fs, tau=tau, anatomical_only=0,
            do_registration=do_registration, cfg=cfg,
        )
        elapsed = time.time() - t0
        status.update(label=f"Stage 1: Suite2p ({elapsed:.0f}s)", state="complete")
    progress.progress(15, text="Suite2p complete")

    # ── Stage 2: Extract projections ─────────────────────────────────────
    with st.status("Stage 2/7: Extracting projections...", expanded=True) as status:
        from roigbiv.io import extract_projections
        t0 = time.time()
        n_proj = extract_projections(str(s2p_dir), str(proj_dir))
        elapsed = time.time() - t0
        status.update(
            label=f"Stage 2: Projections ({n_proj} FOVs, {elapsed:.0f}s)",
            state="complete")
    progress.progress(25, text="Projections extracted")

    # ── Stage 3: Cellpose (Branch A) ─────────────────────────────────────
    with st.status("Stage 3/7: Cellpose inference...", expanded=True) as status:
        t0 = time.time()
        cp_results, cp_errors = _run_cellpose_branch(
            projections_dir=proj_dir, out_dir=cp_dir,
            model_path=model_path, diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            tile_norm_blocksize=tile_norm_blocksize,
            use_vcorr=use_vcorr, denoise=denoise,
            status_writer=status,
        )
        all_errors.extend(("Cellpose", s, e) for s, e in cp_errors)
        n_cp = sum(1 for r in cp_results if not r.get("skipped"))
        elapsed = time.time() - t0
        status.update(
            label=f"Stage 3: Cellpose ({n_cp} processed, {elapsed:.0f}s)",
            state="complete")
    progress.progress(45, text="Cellpose complete")

    # ── Stage 4: Tonic detection (Branch C) ──────────────────────────────
    with st.status("Stage 4/7: Tonic detection...", expanded=True) as status:
        t0 = time.time()
        tc_results, tc_errors = _run_tonic_branch(
            s2p_dir=s2p_dir, out_dir=tonic_dir, fs=fs,
            n_components=n_components, soma_radius=soma_radius,
            corr_threshold=corr_threshold,
            min_size=tonic_min_size, max_size=tonic_max_size,
            min_solidity=tonic_min_solidity,
            max_eccentricity=tonic_max_eccentricity,
            status_writer=status,
        )
        all_errors.extend(("Tonic", s, e) for s, e in tc_errors)
        n_tc = sum(1 for r in tc_results if not r.get("skipped"))
        elapsed = time.time() - t0
        status.update(
            label=f"Stage 4: Tonic ({n_tc} processed, {elapsed:.0f}s)",
            state="complete")
    progress.progress(60, text="Tonic detection complete")

    # ── Stage 5: Three-way merge ─────────────────────────────────────────
    with st.status("Stage 5/7: Merging branches...", expanded=True) as status:
        from roigbiv.merge import merge_batch
        t0 = time.time()
        stems = _discover_stems(cp_dir, s2p_dir, tonic_dir)
        status.write(f"  Found {len(stems)} FOV stems across branches")
        merge_df = merge_batch(
            stems=stems, branch_a_dir=cp_dir, s2p_dir=s2p_dir,
            branch_c_dir=tonic_dir, out_dir=merged_dir,
            iou_threshold=iou_threshold,
            require_spatial_support=require_spatial_support,
        )
        elapsed = time.time() - t0
        n_merged = len(merge_df) if not merge_df.empty else 0
        status.update(
            label=f"Stage 5: Merge ({n_merged} ROIs, {elapsed:.0f}s)",
            state="complete")
    progress.progress(75, text="Merge complete")

    # ── Stage 6: Trace extraction ────────────────────────────────────────
    with st.status("Stage 6/7: Extracting traces...", expanded=True) as status:
        from roigbiv.traces import extract_traces_fov
        t0 = time.time()
        traces_dir.mkdir(parents=True, exist_ok=True)

        trace_stems = sorted(
            p.stem.replace("_merged_masks", "")
            for p in merged_dir.glob("*_merged_masks.tif"))
        n_done = 0

        for i, stem in enumerate(trace_stems):
            # Load merge records for tonic identification
            rec_path = merged_dir / f"{stem}_merge_records.csv"
            merge_records = None
            if rec_path.exists():
                merge_records = pd.read_csv(str(rec_path)).to_dict("records")

            try:
                result = extract_traces_fov(
                    stem=stem, merged_mask_dir=merged_dir, s2p_dir=s2p_dir,
                    out_dir=traces_dir, merge_records=merge_records,
                    fs=fs, tau=tau,
                    inner_radius=int(tr_cfg.get("inner_radius", 2)),
                    min_neuropil_pixels=int(tr_cfg.get("min_neuropil_pixels", 350)),
                    baseline_window=float(tr_cfg.get("baseline_window_seconds", 60.0)),
                    baseline_percentile=float(tr_cfg.get("baseline_percentile", 10.0)),
                    tonic_multiplier=float(tr_cfg.get("tonic_baseline_multiplier", 2.0)),
                    do_deconvolve=bool(tr_cfg.get("do_deconvolve", True)),
                    chunk_frames=int(tr_cfg.get("chunk_frames", 200)),
                )
                if result.get("status") == "done":
                    n_done += 1
                    status.write(
                        f"  [{i+1}/{len(trace_stems)}] {stem}: "
                        f"{result.get('n_rois', '?')} ROIs, "
                        f"{result.get('n_frames', '?')} frames")
                else:
                    status.write(f"  {stem}: {result.get('status', 'unknown')}")
            except Exception as e:
                all_errors.append(("Traces", stem, str(e)))
                status.write(f"  {stem}: ERROR — {e}")

        elapsed = time.time() - t0
        status.update(
            label=f"Stage 6: Traces ({n_done} FOVs, {elapsed:.0f}s)",
            state="complete")
    progress.progress(90, text="Traces extracted")

    # ── Stage 7: Classification ──────────────────────────────────────────
    with st.status("Stage 7/7: Classifying ROIs...", expanded=True) as status:
        from roigbiv.classify import classify_fov
        t0 = time.time()
        classified_dir.mkdir(parents=True, exist_ok=True)

        cls_stems = sorted(
            p.name.removesuffix("_F.npy")
            for p in traces_dir.glob("*_F.npy"))
        cls_dfs = []

        for i, stem in enumerate(cls_stems):
            try:
                rec_path = merged_dir / f"{stem}_merge_records.csv"
                df = classify_fov(
                    stem=stem, traces_dir=traces_dir,
                    merged_mask_dir=merged_dir,
                    merge_records_path=rec_path,
                    out_dir=classified_dir, fs=fs,
                    snr_min=snr_min, area_min=area_min,
                    area_max=area_max, compact_min=compact_min,
                )
                if not df.empty:
                    cls_dfs.append(df)
                    n_cells = int(df["is_cell"].sum())
                    status.write(
                        f"  [{i+1}/{len(cls_stems)}] {stem}: "
                        f"{n_cells}/{len(df)} cells")
            except Exception as e:
                all_errors.append(("Classify", stem, str(e)))
                status.write(f"  {stem}: ERROR — {e}")

        # Write combined summary
        if cls_dfs:
            combined = pd.concat(cls_dfs, ignore_index=True)
            combined.to_csv(
                str(classified_dir / "classification_summary.csv"), index=False)

        elapsed = time.time() - t0
        status.update(
            label=f"Stage 7: Classification ({len(cls_dfs)} FOVs, {elapsed:.0f}s)",
            state="complete")
    progress.progress(100, text="Pipeline complete!")

    # ── Summary ──────────────────────────────────────────────────────────
    st.balloons()

    if all_errors:
        with st.expander(f"{len(all_errors)} error(s) occurred", expanded=False):
            for stage, stem, err in all_errors:
                st.warning(f"**{stage}** — {stem}: {err}")

    # Quick summary
    if cls_dfs:
        combined = pd.concat(cls_dfs, ignore_index=True)
        n_total = len(combined)
        n_cells = int(combined["is_cell"].sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total ROIs", n_total)
        col2.metric("Cells Accepted", n_cells)
        col3.metric("Cells Rejected", n_total - n_cells)
        col4.metric("FOVs", combined["fov"].nunique())

        if not merge_df.empty:
            st.subheader("Merge Tier Breakdown")
            tier_counts = merge_df["tier"].value_counts().sort_index()
            st.dataframe(
                tier_counts.rename("count").to_frame(),
                use_container_width=True)

        st.subheader("Activity Types")
        type_counts = combined.loc[
            combined["is_cell"], "activity_type"].value_counts()
        st.dataframe(
            type_counts.rename("count").to_frame(),
            use_container_width=True)
    else:
        st.info("No ROIs were classified. Check for errors above.")


# ── Visualization helpers (Napari launcher) ───────────────────────────────────

def _launch_napari(
    out_path: Path, stem: str,
    load_mean: bool, load_a: bool, load_b: bool, load_c: bool,
    load_tiers: bool, load_types: bool,
) -> None:
    """Generate a self-contained Napari viewer script and launch it as a subprocess."""
    import subprocess
    import tempfile

    proj_dir = out_path / "projections"
    cp_dir = out_path / "cellpose"
    tonic_dir = out_path / "tonic"
    merged_dir = out_path / "merged"
    classified_dir = out_path / "classified"
    s2p_plane = out_path / "suite2p" / stem / "suite2p" / "plane0"

    L = [
        "import sys",
        f"sys.path.insert(0, r'{PROJECT_ROOT}')",
        "import napari",
        "import numpy as np",
        "import pandas as pd",
        "import tifffile",
        "from napari.utils.colormaps import DirectLabelColormap",
        "",
        "def _hex_rgba(h, a=1.0):",
        "    h = h.lstrip('#')",
        "    return np.array([int(h[i:i+2], 16)/255 for i in (0,2,4)]+[a], dtype=np.float32)",
        "",
        f"viewer = napari.Viewer(title='ROI G. Biv \u2014 {stem}')",
        "",
    ]

    # Mean projection
    if load_mean:
        p = proj_dir / f"{stem}_mean.tif"
        if p.exists():
            L.append(
                f"viewer.add_image(tifffile.imread(r'{p}').astype('float32'),"
                f" name='Mean Projection', colormap='gray')"
            )

    # Branch A — Cellpose
    if load_a:
        p = cp_dir / f"{stem}_masks.tif"
        if p.exists():
            L.append(
                f"viewer.add_labels(tifffile.imread(r'{p}'),"
                f" name='Branch A (Cellpose)', opacity=0.5)"
            )

    # Branch B — Suite2p (reconstruct from stat.npy)
    if load_b:
        stat_p = s2p_plane / "stat.npy"
        ops_p = s2p_plane / "ops.npy"
        if stat_p.exists() and ops_p.exists():
            L += [
                "from roigbiv.merge import stat_to_mask",
                f"_stat = np.load(r'{stat_p}', allow_pickle=True)",
                f"_ops = np.load(r'{ops_p}', allow_pickle=True).item()",
                "viewer.add_labels(stat_to_mask(_stat, _ops['Ly'], _ops['Lx']),"
                " name='Branch B (Suite2p)', opacity=0.5)",
            ]

    # Branch C — Tonic
    if load_c:
        p = tonic_dir / f"{stem}_tonic_masks.tif"
        if p.exists():
            L.append(
                f"viewer.add_labels(tifffile.imread(r'{p}'),"
                f" name='Branch C (Tonic)', opacity=0.5)"
            )

    # Merged mask (shared by tier + activity type layers)
    merged_p = merged_dir / f"{stem}_merged_masks.tif"
    rec_p = merged_dir / f"{stem}_merge_records.csv"
    cls_p = classified_dir / f"{stem}_classification.csv"

    need_merged = (
        (load_tiers and merged_p.exists() and rec_p.exists()) or
        (load_types and merged_p.exists() and cls_p.exists())
    )
    if need_merged:
        L.append(f"_merged = tifffile.imread(r'{merged_p}')")

    # Merged — Tiers
    if load_tiers and merged_p.exists() and rec_p.exists():
        L += [
            f"_rec = pd.read_csv(r'{rec_p}')",
            "_tier_id = {'ABC': 1, 'AB': 2, 'AC': 3, 'BC': 4, 'A': 5, 'B': 6, 'C': 7}",
            "_tier_mask = np.zeros_like(_merged)",
            "for _, _row in _rec.iterrows():",
            "    _tier_mask[_merged == int(_row['roi_id'])] ="
            " _tier_id.get(str(_row.get('tier', 'A')).upper(), 0)",
            "_tier_cmap = DirectLabelColormap(color_dict={",
            "    None: np.zeros(4, dtype=np.float32), 0: np.zeros(4, dtype=np.float32),",
            "    1: _hex_rgba('#FFD700'), 2: _hex_rgba('#9b59b6'), 3: _hex_rgba('#1abc9c'),",
            "    4: _hex_rgba('#e67e22'), 5: _hex_rgba('#3498db'),",
            "    6: _hex_rgba('#2ecc71'), 7: _hex_rgba('#e74c3c'),",
            "})",
            "viewer.add_labels(_tier_mask, name='Merged \u2014 Tiers',"
            " colormap=_tier_cmap, opacity=0.6)",
        ]

    # Merged — Activity Types
    if load_types and merged_p.exists() and cls_p.exists():
        L += [
            f"_cls = pd.read_csv(r'{cls_p}')",
            "if 'roi_id' not in _cls.columns: _cls = _cls.reset_index()",
            "_type_id = {'phasic': 1, 'tonic': 2, 'sparse': 3, 'ambiguous': 4, 'rejected': 5}",
            "_act_mask = np.zeros_like(_merged)",
            "for _, _row in _cls.iterrows():",
            "    _act_mask[_merged == int(_row['roi_id'])] ="
            " _type_id.get(str(_row.get('activity_type', 'rejected')), 5)",
            "_act_cmap = DirectLabelColormap(color_dict={",
            "    None: np.zeros(4, dtype=np.float32), 0: np.zeros(4, dtype=np.float32),",
            "    1: _hex_rgba('#e74c3c'), 2: _hex_rgba('#3498db'), 3: _hex_rgba('#2ecc71'),",
            "    4: _hex_rgba('#f39c12'), 5: _hex_rgba('#aaaaaa'),",
            "})",
            "viewer.add_labels(_act_mask, name='Merged \u2014 Activity Types',"
            " colormap=_act_cmap, opacity=0.6)",
        ]

    L += ["", "napari.run()"]

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("\n".join(L))
        tmp_path = f.name

    subprocess.Popen([sys.executable, tmp_path])


def _viz_section(out_path: Path) -> None:
    """Napari ROI mask visualization launcher in the Results tab."""
    merged_dir = out_path / "merged"
    mask_tifs = sorted(merged_dir.glob("*_merged_masks.tif"))
    if not mask_tifs:
        st.info("No merged mask files found. Run the pipeline first.")
        return

    stems = [p.name.removesuffix("_merged_masks.tif") for p in mask_tifs]

    st.subheader("ROI Mask Visualization")
    col_fov, _ = st.columns([2, 3])
    with col_fov:
        stem = st.selectbox("FOV", stems, key="viz_fov")

    st.caption("Select layers to load:")
    c1, c2, c3 = st.columns(3)
    with c1:
        load_mean  = st.checkbox("Mean Projection",         value=True, key="viz_mean")
        load_a     = st.checkbox("Branch A (Cellpose)",     value=True, key="viz_a")
    with c2:
        load_b     = st.checkbox("Branch B (Suite2p)",      value=True, key="viz_b")
        load_c     = st.checkbox("Branch C (Tonic)",        value=True, key="viz_c")
    with c3:
        load_tiers = st.checkbox("Merged \u2014 Tiers",          value=True, key="viz_tiers")
        load_types = st.checkbox("Merged \u2014 Activity Types",  value=True, key="viz_types")

    if st.button("Open in Napari", type="primary", key="viz_open"):
        try:
            _launch_napari(
                out_path=out_path, stem=stem,
                load_mean=load_mean, load_a=load_a, load_b=load_b, load_c=load_c,
                load_tiers=load_tiers, load_types=load_types,
            )
            st.success("Napari launched \u2014 check for a new window.")
        except Exception as e:
            st.error(f"Failed to launch Napari: {e}")


# ── Results tab ──────────────────────────────────────────────────────────────

def _results_tab(output_dir):
    if not output_dir:
        st.info("Enter an output directory in the sidebar.")
        return

    out_path = Path(output_dir)
    merged_dir = out_path / "merged"
    classified_dir = out_path / "classified"

    cls_csv = classified_dir / "classification_summary.csv"
    merge_csv = merged_dir / "merge_summary.csv"

    if not cls_csv.exists() and not merge_csv.exists():
        st.info("No results found. Run the pipeline first.")
        return

    # ── Classification summary ────────────────────────────────────────
    if cls_csv.exists():
        df = pd.read_csv(str(cls_csv))
        n_total = len(df)
        n_cells = int(df["is_cell"].sum())
        n_fovs = df["fov"].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total ROIs", n_total)
        col2.metric("Cells Accepted", n_cells)
        col3.metric("Cells Rejected", n_total - n_cells)
        col4.metric("FOVs", n_fovs)

        # Activity type breakdown
        st.subheader("Activity Types")
        type_counts = df.loc[df["is_cell"], "activity_type"].value_counts()
        st.dataframe(
            type_counts.rename("count").to_frame(), use_container_width=True)

    # ── Merge tier breakdown ──────────────────────────────────────────
    if merge_csv.exists():
        merge_df = pd.read_csv(str(merge_csv))
        st.subheader("Merge Tier Breakdown")
        tier_counts = merge_df["tier"].value_counts().sort_index()
        st.dataframe(
            tier_counts.rename("count").to_frame(), use_container_width=True)

    # ── Per-FOV detail ────────────────────────────────────────────────
    if cls_csv.exists():
        st.subheader("Per-FOV Detail")
        fovs = sorted(df["fov"].unique())
        selected_fov = st.selectbox("Select FOV", fovs)

        if selected_fov:
            fov_df = df[df["fov"] == selected_fov]
            n_cells_fov = int(fov_df["is_cell"].sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("ROIs", len(fov_df))
            c2.metric("Cells", n_cells_fov)
            if n_cells_fov > 0:
                dominant = fov_df.loc[
                    fov_df["is_cell"], "activity_type"].value_counts().index[0]
                c3.metric("Dominant Type", dominant)

            st.dataframe(fov_df, use_container_width=True, height=300)

    # ── Mask visualization ────────────────────────────────────────────
    st.divider()
    _viz_section(out_path)

    # ── Downloads ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download Results")

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    if cls_csv.exists():
        dl_col1.download_button(
            "classification_summary.csv",
            data=cls_csv.read_bytes(),
            file_name="classification_summary.csv",
            mime="text/csv",
        )
    if merge_csv.exists():
        dl_col2.download_button(
            "merge_summary.csv",
            data=merge_csv.read_bytes(),
            file_name="merge_summary.csv",
            mime="text/csv",
        )

    if dl_col3.button("Download all as ZIP"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for subdir in ["merged", "classified"]:
                d = out_path / subdir
                if d.exists():
                    for f in d.iterdir():
                        if f.is_file() and f.suffix == ".csv":
                            zf.write(f, f"{subdir}/{f.name}")
        st.download_button(
            "Save ZIP",
            data=buf.getvalue(),
            file_name="roigbiv_results.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
