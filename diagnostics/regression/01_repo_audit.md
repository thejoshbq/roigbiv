# 01 — Repository and Environment Audit

## Environment (conda env `roigbiv`)

- Python 3.10
- numpy 2.0.2
- cellpose 3.1.1.2
- suite2p 0.14.6 (from memory; __version__ attr absent)
- torch 2.10.0+cu128, CUDA available
- skimage 0.25.2, tifffile 2025.5.10
- GPU: RTX 4060 8 GB

## Git state

- Branch: `develop`
- HEAD: `70dbe1f 2026-04-14 feat: interactive ROI curator + Suite2p UI tuning, simplified napari viewer`
- Previous: `ff58732 2026-04-07 feat: expand Cellpose pipeline with new modules, data processing, and retrained model`
- Uncommitted changes (git status --short):
  - `M app.py` — Streamlit entry (suspect, contains hardcoded DEFAULT_MODEL)
  - `M pyproject.toml`
  - `M roigbiv/suite2p.py` — hardlink optimization per Agent 3's diff
  - `D notebooks/roigbiv.ipynb` — archived per project notes
  - `?? roigbiv/pipeline/` — **entire new multi-stage pipeline is untracked**
  - `?? roigbiv/cli.py`, `?? roigbiv/overlay.py`
  - `?? docs/roi-pipeline-specification.md` — design spec for the new pipeline
  - `?? docs/visualizer-plan.md`
  - `?? test_output/`, `?? test_raw/` — one test FOV + two pipeline runs

## Suspect files (prime candidates for regression cause)

| File | Why it's suspicious |
|------|---------------------|
| `roigbiv/pipeline/gate1.py` | New filter gate that rejects raw Cellpose candidates; thresholds set per spec §18 but not empirically tuned to this FOV |
| `roigbiv/pipeline/stage1.py` | Runs Cellpose with new dual-channel input (`mean_M` + `vcorr_S`) and `use_denoise=True` wrapper (Cellpose3 `denoise_cyto3`) |
| `roigbiv/pipeline/types.py` | `PipelineConfig` defaults — any threshold default change here cascades |
| `roigbiv/pipeline/foundation.py` | New L+S SVD preprocessing; introduces `mean_S ≈ 0` / `vcorr_S` domain shift |
| `models/deployed/current_model` | Swapped on 2026-04-07 from run014 → run015_epoch_0280 (AP@0.5 dropped 0.68→0.61) |

## Scripts inventory (confirmed present)

- `scripts/run_inference.py` — standalone Cellpose-only inference (baseline for "old" pipeline comparison)
- `scripts/compare_models.py` — existing AP@0.5 evaluation utility (reusable)
- `scripts/train.py`, `scripts/eval_model.py`, `scripts/match_rois.py` — training/eval
- `scripts/ingest_corrections.py`, `scripts/merge_rois.py`, `scripts/validate_dataset.py` — data ops
