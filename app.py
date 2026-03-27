"""
ROI G. Biv — Streamlit web interface.

Launch with:  streamlit run app.py
Access from LAN:  streamlit run app.py --server.address 0.0.0.0
"""
import io
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# ── Project root & defaults ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "deployed" / "current_model"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


def main():
    st.set_page_config(page_title="ROI G. Biv", layout="wide")
    st.title("ROI G. Biv")
    st.caption("Consensus cell detection for two-photon calcium imaging")

    # ── GPU status ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.success(f"GPU: {gpu_name} ({vram:.1f} GB)")
    else:
        st.warning("No GPU detected. Cellpose will run on CPU (slower).")

    # ── Sidebar: parameters ───────────────────────────────────────────────
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
            help="Where results will be saved",
        )

        st.divider()
        fs = st.number_input("Frame rate (Hz)", value=30.0, min_value=1.0, step=1.0)
        tau = st.number_input("GCaMP tau (s)", value=1.0, min_value=0.1, step=0.1,
                              help="GCaMP6s=1.0, GCaMP6f=0.4, GCaMP7f=0.7")
        diameter = st.number_input("Cell diameter (px)", value=30, min_value=5, step=5)
        iou_threshold = st.slider("IoU threshold (GOLD)", 0.1, 0.8, 0.3, 0.05)
        use_vcorr = st.checkbox("Use Vcorr channel", value=True,
                                help="Stack Vcorr as 2nd Cellpose input channel")
        do_registration = st.checkbox("Run motion correction", value=False,
                                      help="Enable if TIFs are NOT pre-corrected")

        tiers = st.multiselect("Output tiers", ["gold", "silver", "bronze"],
                               default=["gold", "silver"])

        st.divider()
        model_path = st.text_input("Model checkpoint", value=str(DEFAULT_MODEL))

    # ── Main area ─────────────────────────────────────────────────────────
    tab_run, tab_results = st.tabs(["Run Pipeline", "View Results"])

    with tab_run:
        _run_tab(tif_dir, output_dir, fs, tau, diameter, iou_threshold,
                 use_vcorr, do_registration, tiers, model_path)

    with tab_results:
        _results_tab(output_dir, tiers)


# ── Run tab ───────────────────────────────────────────────────────────────────

def _run_tab(tif_dir, output_dir, fs, tau, diameter, iou_threshold,
             use_vcorr, do_registration, tiers, model_path):

    if not tif_dir or not output_dir:
        st.info("Enter a TIF directory and output directory in the sidebar to begin.")
        return

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
    for tif in tif_files:
        try:
            stem, shape = validate_tif(tif)
            st.text(f"  {tif.name}  —  {shape[0]} frames, {shape[1]}x{shape[2]} px")
        except ValueError as e:
            st.warning(str(e))

    # Run button
    if not st.button("Run Pipeline", type="primary", use_container_width=True):
        return

    s2p_activity_dir = out_path / "s2p_activity"
    s2p_anatomy_dir = out_path / "s2p_anatomy"
    projections_dir = out_path / "projections"
    results_dir = out_path / "results"

    progress = st.progress(0, text="Starting pipeline...")

    # Stage 1: Suite2p activity pass
    with st.status("Suite2p activity pass...", expanded=True) as status:
        from roigbiv.suite2p import run_suite2p_batch
        t0 = time.time()
        run_suite2p_batch(
            tif_files, output_dir=str(s2p_activity_dir),
            fs=fs, tau=tau, anatomical_only=0, do_registration=do_registration,
        )
        status.update(label=f"Suite2p activity pass ({time.time()-t0:.0f}s)",
                      state="complete")
    progress.progress(25, text="Activity pass complete")

    # Stage 2: Suite2p anatomy pass
    with st.status("Suite2p anatomy pass...", expanded=True) as status:
        t0 = time.time()
        run_suite2p_batch(
            tif_files, output_dir=str(s2p_anatomy_dir),
            fs=fs, tau=tau, anatomical_only=1, do_registration=do_registration,
        )
        status.update(label=f"Suite2p anatomy pass ({time.time()-t0:.0f}s)",
                      state="complete")
    progress.progress(50, text="Anatomy pass complete")

    # Stage 3: Extract projections
    with st.status("Extracting projections...", expanded=True) as status:
        from roigbiv.io import extract_projections
        t0 = time.time()
        extract_projections(str(s2p_activity_dir), str(projections_dir))
        status.update(label=f"Projections extracted ({time.time()-t0:.0f}s)",
                      state="complete")
    progress.progress(65, text="Projections extracted")

    # Stage 4: Union ROI building + Cellpose scoring
    with st.status("Union ROI building + Cellpose scoring...", expanded=True) as status:
        from roigbiv.union import build_union_batch
        t0 = time.time()
        summary = build_union_batch(
            tif_files=tif_files,
            activity_dir=str(s2p_activity_dir),
            anatomy_dir=str(s2p_anatomy_dir),
            projections_dir=str(projections_dir),
            model_path=model_path,
            output_dir=str(results_dir),
            diameter=diameter,
            iou_threshold=iou_threshold,
            tiers=tiers,
            use_vcorr=use_vcorr,
        )
        status.update(label=f"Union ROIs scored ({time.time()-t0:.0f}s)",
                      state="complete")
    progress.progress(100, text="Pipeline complete!")

    st.balloons()

    if not summary.empty:
        st.subheader("Tier breakdown")
        st.dataframe(summary.groupby("tier")["roi_id"].count().rename("count"))
        st.subheader("Results preview")
        st.dataframe(summary.head(20), use_container_width=True)

        csv_path = results_dir / "scored_rois_summary.csv"
        if csv_path.exists():
            st.download_button(
                "Download scored_rois_summary.csv",
                data=csv_path.read_bytes(),
                file_name="scored_rois_summary.csv",
                mime="text/csv",
            )
    else:
        st.info("No ROIs processed (all FOVs may have been skipped).")


# ── Napari launcher ───────────────────────────────────────────────────────────

def _launch_napari(stem: str, results_dir: str, projections_dir: str,
                   outline_tiers: list[str] | None = None):
    """Launch Napari viewer for a single FOV in a subprocess."""
    try:
        import napari  # noqa: F401 — pre-flight check
    except ImportError:
        st.error(
            "Napari is not installed. Install it with:\n\n"
            "```\npip install 'napari[all]'\n```"
        )
        return

    cmd = [
        sys.executable, "-m", "roigbiv.napari_viewer",
        "--stem", stem,
        "--results-dir", results_dir,
        "--projections-dir", projections_dir,
    ]
    if outline_tiers:
        cmd.extend(["--outline-tiers"] + [t.upper() for t in outline_tiers])
    try:
        subprocess.Popen(cmd)
        st.success(f"Napari is opening for **{stem}**. Close Napari to return here.")
    except Exception as e:
        st.error(f"Failed to launch Napari: {e}")


# ── Results tab ───────────────────────────────────────────────────────────────

def _results_tab(output_dir, tiers):
    if not output_dir:
        st.info("Enter an output directory in the sidebar.")
        return

    results_dir = Path(output_dir) / "results"
    if not results_dir.exists():
        st.info(f"No results found at {results_dir}. Run the pipeline first.")
        return

    csv_path = results_dir / "scored_rois_summary.csv"
    mask_files = sorted(results_dir.glob("*_all_s2p_masks.tif"))

    if not mask_files:
        st.info("No mask files found. Run the pipeline first.")
        return

    stems = [f.name.replace("_all_s2p_masks.tif", "") for f in mask_files]

    # Summary table
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
        df["tier"] = df["tier"].str.upper()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total ROIs", len(df))
        col2.metric("FOVs", df["fov"].nunique())
        col3.metric("GOLD", (df["tier"] == "GOLD").sum())

        st.subheader("ROI Summary")
        st.dataframe(df, use_container_width=True, height=300)
    else:
        df = None

    # FOV viewer
    st.subheader("FOV Viewer")
    selected_stem = st.selectbox("Select FOV", stems)
    projections_dir = Path(output_dir) / "projections"

    if st.button(f"Open '{selected_stem}' in Napari", type="primary"):
        _launch_napari(selected_stem, str(results_dir), str(projections_dir),
                       outline_tiers=tiers)

    # Download all results as ZIP
    st.subheader("Download Results")
    if st.button("Download all results as ZIP"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in results_dir.iterdir():
                if f.is_file():
                    zf.write(f, f.name)
        st.download_button(
            "Save ZIP",
            data=buf.getvalue(),
            file_name=f"roigbiv_results.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
