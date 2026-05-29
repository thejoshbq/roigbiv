# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository. This file complements — does not duplicate — the parent `~/Otis-Lab/CLAUDE.md` (Trello board + workspace conventions) and the auto-memory under `~/.claude/projects/.../memory/` (user profile + project state).

## Role

You are a senior software engineer with neuroscience and ML expertise. You are context-engineering fluent and Anthropic-model-native. This repo is dual-hat — production-quality pipeline code **and** research tooling — so calibrate rigor by subsystem: strict for `roigbiv/pipeline/` and `roigbiv/registry/`; looser for `scripts/` and diagnostics.

## Working style

**Default is research-first / advisory — pair programmer, not auto-executor.**

- For any non-trivial change: explore relevant code first, then present 2–3 viable approaches with tradeoffs **before** writing code. Let Josh pick the direction. A single "recommended" option is fine; always surface the alternatives you considered.
- One-liners, typos, mechanical renames, and obviously-scoped edits may skip the proposal step.
- Name tradeoffs explicitly when they exist (algorithm choice, library, API shape, refactor scope). Don't silently pick.
- **Never touch without explicit confirmation:** `inference/`, `diagnostics/regression/`, `models/deployed/`, `models/checkpoints/`, or any `*.db` file. These are data of record.
- Terse prose; no trailing summaries. Use `path/to/file.py:line` citations for code references.
- Use Explore agents for codebase search >3 queries; use Plan agents for architecture decisions.
- When editing pipeline stages or the spec, cross-check `docs/roi-pipeline-specification.md` first — the spec is load-bearing.
- Auto-memory already tracks Josh's profile and long-lived project state; don't re-ask for facts already in memory.

## Project

**ROI G. Biv** — sequential subtractive ROI detection for two-photon calcium imaging. Feeds ROI masks + traces downstream to [pynapse](https://github.com/Otis-Lab-MUSC/pynapse) (signal analysis) and axplorer (visualization).

## Authoritative docs

When in doubt about *behavior*, the spec wins over code comments or this file.

| Intent | Source of truth |
| --- | --- |
| Pipeline behavior, gate logic, ROI schema | `docs/roi-pipeline-specification.md` |
| Algorithm walkthrough (per-stage math) | `docs/pipeline_algorithm_breakdown.md` |
| Tunable parameters (all defaults) | `configs/pipeline.yaml` |
| Dash UI roadmap + architecture | `docs/visualizer-plan.md` |
| Astrocyte / dual-channel extension | `docs/ASTROCYTE_PLAN.md` |
| Data format for external collaborators | `docs/RESEARCHER_DATA_GUIDE.md` |
| External-editor (Fiji/ImageJ) handoff | `docs/external-editing.md` |
| Email notifications on pipeline completion | `docs/email-notifications.md` |
| Regression investigation audit trail | `diagnostics/regression/0{1..10}_*.md` |
| Version history | `docs/CHANGELOG.md` |

## Environment

Conda env `roigbiv` (Python 3.10) defined in `environment.yml`. Key pins: `cellpose<4.0.0` (CP3 is canonical — all checkpoints are CP3 format), `suite2p`, `torch` with CUDA. `pyproject.toml` intentionally omits `suite2p` and `cellpose` because install order matters.

```bash
conda activate roigbiv              # always first
pip install -e .                    # editable install (already in environment.yml)
```

GPU: RTX 5080 16 GB. Cellpose inference is GPU; Suite2p is CPU-only.

## Common commands

```bash
# Dash web app — primary lab-member interface (supersedes the old Streamlit UI)
roigbiv-ui                                         # http://127.0.0.1:8050
roigbiv-ui --host 0.0.0.0                          # LAN access

# Pipeline — directory input ⇒ workspace mode (in-input output/, registry.db,
# auto-migrate + backfill). Single-TIF input ⇒ classic mode.
roigbiv-pipeline --input PATH --fs 7.5 [--model PATH]
roigbiv-pipeline --input fov_dir/ --fs 7.5 --n-workers 2     # parallel batch

# Email a PNG overlay per FOV when done (via local Proton Mail Bridge by default)
roigbiv-pipeline --input test_raw/ --fs 7.5 \
    --email-to user@x --smtp-user user@proton.me

# Skip slower stages for the fast path (~10–25 min/FOV faster)
roigbiv-pipeline --input PATH --fs 7.5 --no-stage-3 --no-stage-4

# Cross-session FOV registry
roigbiv-registry {list|show|match|track|backfill|migrate}

# Ingest externally-edited ROI masks (Fiji / ImageJ handoff). See
# docs/external-editing.md for the full workflow.
roigbiv-reingest --output-dir inference/pipeline/<stem> --new-mask edited.tif

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

Sequential subtractive detection — not the older parallel three-branch consensus. Each stage operates on the residual after prior stages subtract their detections.

```
Foundation  (motion correction + truncated SVD + L+S background + summary images)
  → Stage 1  Cellpose spatial detection      → Gate 1 (morphology)        → subtract
  → Stage 2  Suite2p temporal detection      → Gate 2 (cross-validation)  → subtract
  → Stage 3  Template sweep on residual      → Gate 3 (waveform)          → subtract
  → Stage 4  Tonic neuron search             → Gate 4 (corr contrast)
  → Unified QC + classification + HITL package
```

Provenance is tracked per ROI (`source_stage`, `gate_outcome`, `confidence`, per-stage scores). Types in `roigbiv/pipeline/types.py`: `ROI`, `FOVData`, `PipelineConfig`.

### Package layout

- `roigbiv/pipeline/` — sequential pipeline. Entry: `run.py::run_pipeline`. Stages `stage1.py`…`stage4.py` with matching `gate1.py`…`gate4.py`. `foundation.py` does motion correction + SVD + L+S. `subtraction.py` removes detected sources from the residual movie on disk. `batch.py` runs ≥2 FOVs concurrently with a shared GPU lock. `napari_viewer.py` / `hitl.py` handle review.
- `roigbiv/registry/` — cross-session FOV + cell tracking. `orchestrator.py::register_or_match` returns `hash_match | auto_match | review | new_fov`. Storage: SQLAlchemy store + filesystem blob store. Matching: ROICaT embeddings with calibrated logistic posterior. Alembic migrations in `migrations/versions/`.
- `roigbiv/ui/` — Dash + Plotly frontend (pages: Process, Review/HITL). Entry: `roigbiv-ui`. Reuses `workspace.py`; writes additive HITL corrections under each FOV's `corrections/` subdir without mutating pipeline outputs. Registry browsing/maintenance is CLI-only via `roigbiv-registry`.
- `roigbiv/pipeline/workspace.py` — `resolve_workspace` + `run_with_workspace`: in-input `output/`, `registry.db`, auto-migrate, auto-backfill.
- `roigbiv/pipeline/corrections.py` — HITL ops (add / delete / merge / split / edit / relabel) as a JSONL append log with idempotent replay.
- `roigbiv/` top level — CLI entry points (`cli.py`, `cli_registry.py`, `cli_reingest.py`, all wired in `pyproject.toml [project.scripts]`) plus shared utilities used across subpackages: `io.py` (TIF discovery/validation, used by `pipeline/workspace.py` and `ui/pages/process.py`), `suite2p.py` (Suite2p batch runner, used by `pipeline/foundation.py`), `overlay.py` (report rendering, used by `cli.py`).
- `scripts/` — training, evaluation, HITL ingest, pynapse export, data-prep utilities (not shipped with the package).
- `configs/pipeline.yaml` — all tunable defaults. CLI flags override.
- `models/deployed/current_model` — deployed Cellpose checkpoint. `models/checkpoints/` holds training runs.

### GPU concurrency

`roigbiv/pipeline/batch.py` caps at 2 workers with `spawn` start method (forking after CUDA init deadlocks). A shared `multiprocessing.Manager().Lock()` serializes Cellpose, Suite2p, Stage 3 FFT, and source subtraction; CPU phases overlap freely. Single-FOV runs pass `gpu_lock=None` (default) — the `_gpu_section` helper becomes a no-op.

### Registry configuration

Env-driven (`roigbiv/registry/config.py`). Defaults: SQLite at `inference/registry.db`, blobs at `inference/fingerprints/`, calibration at `inference/registry_calibration.json`.

| Env var | Default |
| --- | --- |
| `ROIGBIV_REGISTRY_DSN` | `sqlite:///inference/registry.db` |
| `ROIGBIV_BLOB_ROOT` | `inference/fingerprints/` |
| `ROIGBIV_ROICAT_DEVICE` | auto (cuda if available) |
| `ROIGBIV_FOV_ACCEPT_THRESHOLD` | `0.9` |
| `ROIGBIV_FOV_REVIEW_THRESHOLD` | `0.5` |

Run migrations with `roigbiv-registry migrate` (wraps `alembic upgrade head`).

### Output layout

Pipeline writes to `inference/pipeline/{stem}/` by default: `stage1/`…`stage4/` (mask TIFFs + probability/contrast maps + `*_report.json`), `summary/` (mean_M, mean_S, std_S, vcorr_S, dog_map), `traces/`, `classified/`, `hitl/`, `registry_match.json`.

## Gotchas

- **Frame rate** — `PipelineConfig.fs=30.0` is the *acquisition* rate. Stacks with `_mc` suffix from this lab are 4× online-averaged, so the *effective* rate is 7.5 Hz. Always pass `--fs 7.5` for averaged stacks. Wrong fs miscalibrates Stage 3 GCaMP templates, Stage 4 bandpass windows, and deconvolution τ.
- **Top-level `roigbiv/` modules** — `io.py`, `suite2p.py`, `overlay.py` are live utilities (not legacy); the `cli*.py` modules are CLI entry points wired in `pyproject.toml`. Do not duplicate utility functions inside `pipeline/`.
- **Motion correction** — raw stacks are pre-corrected (`*_mc.tif`); Suite2p registration inside the pipeline is disabled for these. Foundation still runs its own registration when input lacks `_mc`.
- **HITL corrections are additive** — never mutate pipeline outputs; write to `{fov}/corrections/` as JSONL ops.
- **"Labrynth"** — intentional branding on the sister GUI project. Do not autocorrect.
- **Git LFS** — `models/checkpoints/` and `models/deployed/` are LFS-tracked.

## Conventions

- Indicator: GCaMP6s, `tau=1.0`.
- GOLD+SILVER are the default downstream tiers (see `consensus.default_tiers` in `pipeline.yaml`) — this naming is legacy from the parallel-consensus era; the sequential pipeline uses `gate_outcome ∈ {accept, flag, reject}`.
