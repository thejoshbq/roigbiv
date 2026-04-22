# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**ROI G. Biv** — ROI detection pipeline for two-photon calcium imaging. Feeds ROI masks downstream to [pynapse](https://github.com/Otis-Lab-MUSC/pynapse) (signal extraction) and axplorer (visualization).

Part of the Otis Lab workspace. The parent `CLAUDE.md` at `~/Otis-Lab/CLAUDE.md` covers workspace-wide conventions and the Trello board.

## Environment

Conda env `roigbiv` (Python 3.10) — defined in `environment.yml`. Key pinned pieces: `cellpose<4.0.0` (CP3 is canonical for this repo; all checkpoints are CP3 format), `suite2p`, `torch` with CUDA. `pyproject.toml` intentionally omits `suite2p` and `cellpose` because installation order matters.

```bash
conda activate roigbiv              # always first
pip install -e .                    # editable install (already in environment.yml)
```

GPU: RTX 4060 8 GB. Cellpose inference is GPU; Suite2p is CPU-only.

## Common Commands

```bash
# Dash web app (primary interface for lab members — supersedes the old Streamlit UI)
roigbiv-ui                                         # http://127.0.0.1:8050
roigbiv-ui --host 0.0.0.0                          # LAN access

# Pipeline — workspace mode (recommended): outputs + registry live inside the input dir;
# migrate + backfill run automatically.
roigbiv-pipeline --workspace --input PATH --fs 7.5 [--model PATH]

# Pipeline — classic single-FOV or batch mode
roigbiv-pipeline --input PATH --fs 7.5 [--model PATH] [--output-dir PATH] [--no-viewer]

# Terminal runner that emails a PNG overlay when done (wraps roigbiv-pipeline)
roigbiv-cli --input test_raw/ --fs 7.5 --email-to user@x --smtp-user user@gmail.com

# Cross-session FOV registry
roigbiv-registry {list|show|match|track|backfill|migrate}

# Tests
pytest                                             # whole suite
pytest roigbiv/pipeline/tests/test_stage4.py       # single file
pytest roigbiv/registry/tests/ -k orchestrator     # filter by name

# Cellpose fine-tuning
python scripts/train.py --run_id runXXX [--epochs 100] [--lr 0.1]

# Cellpose GUI (interactive correction → scripts/ingest_corrections.py)
cellpose --pretrained_model models/deployed/current_model
```

**Frame rate:** Use `--fs 7.5` for the reference FOV (4× frame averaging). The 30 Hz default is only valid for un-averaged stacks.

## Architecture

ROIGBIV is a **sequential subtractive detection pipeline** — not the older parallel three-branch consensus design. Each stage operates on the residual signal after previous stages subtract their detections. Authoritative spec: `docs/roi-pipeline-specification.md`.

```
Foundation  (motion correction + truncated SVD + L+S background + summary images)
  → Stage 1  Cellpose spatial detection      → Gate 1 (morphology)      → subtract
  → Stage 2  Suite2p temporal detection      → Gate 2 (cross-validation)→ subtract
  → Stage 3  Template sweep on residual      → Gate 3 (waveform)        → subtract
  → Stage 4  Tonic neuron search             → Gate 4 (corr contrast)
  → Unified QC + classification + HITL package
```

Provenance is tracked per ROI (`source_stage`, `gate_outcome`, `confidence`, per-stage scores). See `roigbiv/pipeline/types.py` for `ROI` / `FOVData` / `PipelineConfig`.

### Package layout

- `roigbiv/pipeline/` — the sequential pipeline. Entry point `run.py::run_pipeline`; stages are `stage1.py`…`stage4.py` with matching `gate1.py`…`gate4.py`. `foundation.py` does motion correction + SVD + L+S. `subtraction.py` removes detected sources from the residual movie on disk. `batch.py` runs ≥2 FOVs concurrently with a shared GPU lock. `napari_viewer.py` / `hitl.py` handle review.
- `roigbiv/registry/` — cross-session FOV + cell tracking. `orchestrator.py::register_or_match` produces one of `hash_match | auto_match | review | new_fov` per session. Storage: SQLAlchemy store (`store/sqlalchemy_store.py`) + filesystem blob store (`blob/`). Matching uses ROICaT embeddings (`roicat_adapter.py`) with a calibrated logistic posterior (`calibration.py`). Alembic migrations in `migrations/versions/`.
- `roigbiv/` top level — legacy single-stage modules kept for reference (`suite2p.py`, `match.py`, `union.py`, `tonic.py`, `curator.py`, `napari_viewer.py`). Prefer the `pipeline/` equivalents for new work.
- `roigbiv/ui/` — Dash + Plotly frontend (pages: Process, Registry, Viewer, Review/HITL). Entry: `roigbiv-ui` / `python -m roigbiv.ui`. Reuses `run_with_workspace` for the in-directory output + registry convention, and writes additive HITL corrections under each FOV's `corrections/` subdir without mutating pipeline outputs.
- `roigbiv/pipeline/workspace.py` — `resolve_workspace` + `run_with_workspace`: in-input `output/`, `registry.db`, auto-migrate, auto-backfill. Used by both the UI and `roigbiv-pipeline --workspace`.
- `roigbiv/pipeline/corrections.py` — HITL ops model (add / delete / merge / split / edit / relabel), JSONL append log, idempotent replay + materialization.
- `app.py` — deprecation shim pointing `streamlit run app.py` users to `roigbiv-ui`.
- `scripts/` — training, evaluation, and one-off data utilities (not shipped with the package).
- `configs/pipeline.yaml` — all tunable defaults. CLI flags override individual values.
- `models/deployed/current_model` — the deployed Cellpose checkpoint. `models/checkpoints/` holds training runs.

### GPU concurrency

`roigbiv/pipeline/batch.py` caps at 2 workers with `spawn` start method (forking after a CUDA init deadlocks). A shared `multiprocessing.Manager().Lock()` serializes Cellpose, Suite2p, Stage 3 FFT, and source subtraction; CPU phases overlap freely. When running single-FOV via `run_pipeline`, pass `gpu_lock=None` (default) — the `_gpu_section` helper is a no-op.

### Registry configuration

Env-driven (`roigbiv/registry/config.py`). Defaults: SQLite at `inference/registry.db`, blobs at `inference/fingerprints/`, calibration at `inference/registry_calibration.json`. Key vars: `ROIGBIV_REGISTRY_DSN`, `ROIGBIV_BLOB_ROOT`, `ROIGBIV_ROICAT_DEVICE`, `ROIGBIV_FOV_ACCEPT_THRESHOLD` (default 0.9), `ROIGBIV_FOV_REVIEW_THRESHOLD` (default 0.5). Run migrations with `roigbiv-registry migrate` (wraps `alembic upgrade head`).

### Output layout

Pipeline writes to `inference/pipeline/{stem}/` by default:
- `stage1/`…`stage4/` — per-stage mask TIFFs, probability/contrast maps, and `*_report.json` with full ROI metadata
- `summary/` — mean_M, mean_S, std_S, vcorr_S, dog_map
- `traces/`, `classified/`, `hitl/`, `registry_match.json`

## Conventions

- Raw stacks are pre-motion-corrected (`*_mc.tif`); Suite2p registration inside the pipeline is disabled for these. Foundation still runs its own registration when the input lacks `_mc`.
- Indicator is GCaMP6s, `tau=1.0`.
- GOLD+SILVER tiers are the default downstream set (configured under `consensus.default_tiers` in `pipeline.yaml`) — this naming is legacy from the parallel-consensus era; the sequential pipeline uses `gate_outcome ∈ {accept, flag, reject}` instead.
- "Labrynth" (sister project) is intentional branding — do not autocorrect.
- Git LFS is used for `models/checkpoints/` and `models/deployed/`.
