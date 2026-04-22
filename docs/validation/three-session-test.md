# Three-session validation — v3 ROICaT-integrated registry

**Dataset**: three T1 PrL-NAc sessions (2022-12-09, 2022-12-15, 2023-01-16).
**Expected**: 1 FOV with 3 sessions.
**Status**: **PASS** as of Run 3 (`alignment_method=RoMa`).
**Script**: `scripts/validation/three_session_run.py`
**Transcript (raw)**: `docs/validation/three_session_transcript.json` (always reflects the most recent run)
**Fresh registry**: `inference/registry_roicat.db` (teardown+rebuild each run)
**Fresh blob store**: `inference/fingerprints_v3/`

This document records the remediation chronology so every failed hypothesis is legible without having to diff the git log.

---

## Headline result (Run 3): 1 FOV, 3 sessions, both steps `auto_match`

| Step | Posterior | Inlier rate | Shared clusters | Fraction clustered |
|---|---|---|---|---|
| Step 1 (S2 vs S1) | **0.993** | 0.755 | 57 | 0.760 |
| Step 2 (S3 vs S1+S2) | **0.978** | 0.634 | 46 | 0.676 |

Both above the 0.9 `accept_threshold`. Registry ends with exactly **1 FOV, 3 sessions, 123 unique cells, 27 session-missing markers** (cells present in earlier sessions but not step 3).

---

## Run log

### Run 1 — 2026-04-22 — baseline (Phase 4 original)

**Config**: `alignment_method=PhaseCorrelation`, `roi_mixing_factor=0.5`, `all_to_all=False`, `nonrigid=False`, `accept_threshold=0.9`, `review_threshold=0.5`.

| Step | Stem | Decision | `alignment_inlier_rate` | `n_shared_clusters` | `fraction_query_clustered` | `mean_cluster_cohesion` | `fov_posterior` |
|---|---|---|---|---|---|---|---|
| 0 | `T1_221209_…_HI-D1_…_PRE-002` | `new_fov` | — | — | — | — | — (empty registry) |
| 1 | `T1_221215_…_LOW-D1_…_PRE-000` | `new_fov` (reject) | 0.214 | 12 | 0.160 | 0.451 | 0.330 |
| 2 | `T1_230116_…_EXT-D9_…_PRE-000` | `review` | 0.443 | 27 | 0.397 | 0.339 | **0.791** |

**Verdict**: **FAIL** — 2 FOVs minted, session 3 parked in review against the step-1 FOV.
**Diagnosis**: alignment stage is the dominant blocker. `PhaseCorrelation` inlier rate on step 1 (6-day gap) is 0.21; logs contain repeated *"Could not find a path to alignment for image idx: [0]"* warnings. Mean projections visually differ substantially session-to-session (bleaching, activity shifts).

### Run 2 — 2026-04-22 — hypothesis: raise ROI mixing factor

**Config**: same as run 1 except `roi_mixing_factor=0.9` (up from 0.5). Based on visual inspection showing ROI footprint layout is stable across days while mean projection is session-variable.

| Step | Decision | `alignment_inlier_rate` | `n_shared_clusters` | `fraction_query_clustered` | `mean_cluster_cohesion` | `fov_posterior` |
|---|---|---|---|---|---|---|
| 0 | `new_fov` | — | — | — | — | — |
| 1 | `new_fov` (reject) | **0.090** | 12 | 0.160 | 0.451 | **0.230** |
| 2 | `new_fov` (reject) | **0.078** | 13 | 0.191 | 0.488 | **0.268** |

**Verdict**: **FAIL — and worse than run 1** — 3 FOVs minted, no match even made it to the review band.

**Why it failed**: stability and distinctiveness are not the same thing. The mean projection is session-variable but carries distinctive spatial frequency content that phase correlation can lock onto. ROI footprint density on this dataset is ~70–85 blob-like regions fairly evenly distributed — low frequency-domain distinctiveness, so driving the alignment signal toward footprints-only *removed* information rather than concentrating it.

**Action**: default reverted to `0.5`. Env var `ROIGBIV_ROICAT_ROI_MIXING` remains exposed as an opt-in knob.

### Run 3 — 2026-04-22 — hypothesis: switch alignment to RoMa

**Config**: `ROIGBIV_ROICAT_ALIGNMENT=RoMa`, `ROIGBIV_ROICAT_DEVICE=cuda`. All other knobs unchanged (mixing=0.5, all_to_all=False, nonrigid=False, accept=0.9, review=0.5).

Required one-time infra change: `roigbiv/registry/roicat_adapter.py` gained `_maybe_patch_romatch_for_native_fallback()`, which detects the missing `local_corr` compiled CUDA extension and monkey-patches `romatch.roma_{outdoor,indoor,tiny_v1_outdoor}` to pass `use_custom_corr=False`. This routes through the PyTorch-native correlation path (~2–3× slower, numerically equivalent). No change to the ROICaT or romatch packages themselves. ROICaT's `Aligner` doesn't expose `use_custom_corr` so the patch is the only clean injection point; when the extension is installed, the patch is a no-op.

| Step | Decision | `alignment_inlier_rate` | `n_shared_clusters` | `fraction_query_clustered` | `mean_cluster_cohesion` | `fov_posterior` | Elapsed |
|---|---|---|---|---|---|---|---|
| 0 | `new_fov` | — | — | — | — | — | 0.2s |
| 1 | **`auto_match`** | **0.755** | **57** | **0.760** | 0.277 | **0.993** | 27.2s |
| 2 | **`auto_match`** | **0.634** | **46** | **0.676** | 0.315 | **0.978** | 23.3s |

**Verdict**: **PASS.** Final DB: 1 FOV, 3 sessions, 123 cells, 27 `n_missing` markers. RoMa is the winning lever.

**Interpretation**: RoMa's dense feature matcher (DINOv2 ViT-L/14 backbone + learned correlation) is robust to the cross-session photometric drift that broke PhaseCorrelation. Inlier rate on the hardest pair (6-day gap, session 2 vs session 1) jumped from 0.214 → 0.755 (3.5×); `fraction_query_clustered` from 0.16 → 0.76 (4.8×); `n_shared_clusters` from 12 → 57.

One feature *did* drop: `mean_cluster_cohesion` fell from 0.451 → 0.277. This is an expected consequence, not a regression — when alignment is bad you only find the tightest same-cell pairs (artificially high cohesion over few clusters); when alignment is good you find many more pairs including the merely-similar-but-correct ones (average cohesion diluted). The logistic's other features compensate: `0.05·57 + 3.0·0.76 + 4.0·0.755 = 8.00` overwhelms the cohesion contribution.

**Cost**: 50.5 s of RoMa inference total (CUDA). One-time 1.5 GB model download (`roma_outdoor.pth` 425 MB + `dinov2_vitl14_pretrain.pth` 1.13 GB) on first run, cached to `~/.cache/torch/hub/checkpoints/`.

### Run 4 — 2026-04-22 — defaults flipped

**Config**: no env overrides. `AdapterConfig` defaults now `alignment_method="RoMa"` + `device=field(default_factory=_auto_device)` (→ `"cuda"` on this machine). `RegistryConfig` matches.

| Step | Decision | `fov_posterior` |
|---|---|---|
| 1 | `auto_match` | 0.993 |
| 2 | `auto_match` | 0.987 |

**Verdict**: **PASS** — same as Run 3. Confirms the defaults alone (no `ROIGBIV_ROICAT_*` env vars set) produce the validated behaviour. This is the new committed state.

---

## Remediation options — updated state

Of the seven options listed after run 1:

| # | Option | State |
|---|---|---|
| 1 | Different alignment method (ECC_cv2 / RoMa) | **RESOLVED — RoMa is the winning lever (Run 3).** |
| 2 | Non-rigid refinement | Unused. Not needed at RoMa posteriors of 0.978–0.993. |
| 3 | `all_to_all=True` | Unused. Not needed at posteriors that clear 0.9. |
| 4 | Lower accept threshold → 0.75 | Unused. Not needed. |
| 5 | Collect labeled pairs + refit logistic | Still valuable as a long-term hardening, not blocking. |
| 6 | Streamlit manual review of session 3 | Not needed — session 3 auto-matched. |
| 7 | Visually inspect mean projections | **Done** (`~/Pictures/mean_projections.png`); confirmed inter-session drift. |
| — | Raise `roi_mixing_factor` | Rejected in Run 2. |

---

## Follow-up validation plan

When the next remediation lever is pulled, follow this protocol to ensure a like-for-like comparison and full pipeline validation — not just a single-script verdict.

### 1. Pre-change: confirm baseline is reproducible

```bash
cd /home/thejoshbq/Otis-Lab/Projects/roigbiv
# Rebuild fresh DB from scratch each time. Do NOT touch inference/registry.db.
python scripts/validation/three_session_run.py
```

Expected: the run log row for the current default matches the table above within ±0.01 on every metric. If it does not, investigate environment drift (dependency versions, ROInet cache, ROICaT version) *before* changing any knobs.

### 2. Unit tests — adapter

```bash
pytest roigbiv/registry/tests/test_roicat_adapter.py -q
```

13 tests. Must pass before and after any adapter edit. The end-to-end two-session synthetic test is gated on ROInet weight cache presence; if it skips on CI, run locally with weights downloaded.

### 3. Unit tests — rest of the v3 registry

```bash
pytest roigbiv/registry/tests/ -q --ignore=roigbiv/registry/tests/test_roicat_adapter.py
```

Covers: `test_filename.py` (animal/region/date parsing), `test_fingerprint.py` (v3 footprint-derived hash stability), `test_calibration.py` (logistic coefficients + sigmoid), `test_orchestrator.py` (hash-shortcut path + new-FOV mint path). All must pass.

### 4. Pipeline smoke test — fresh session end-to-end

Run one of the three T1 sessions through the full pipeline from raw to registry with `--registry` enabled. Asserts the pipeline → registry handoff is intact (not just the `three_session_run.py` path that calls the orchestrator directly).

```bash
python -m roigbiv.pipeline.run \
  --input data/raw/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002_mc.tif \
  --output data/output/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002_SMOKE \
  --registry \
  --fs 7.5
```

Expect: the run ends with a registry report in the output dir and a new FOV in the v3 DB. If the output dir already exists, either delete it or point at a fresh path — the pipeline's resume logic may skip stages.

### 5. Three-session validation (the headline test)

```bash
python scripts/validation/three_session_run.py
```

Produces `docs/validation/three_session_transcript.json`. Compare against the run log in this document. Update this document with a new "Run N" section containing:
- Config diff vs the previous run.
- Per-step metrics table.
- Verdict.
- One-paragraph interpretation of *why* the change helped or didn't.

### 6. CLI surface

```bash
roigbiv-registry list
roigbiv-registry show --fov <fov_id>
roigbiv-registry match --output-dir data/output/<session_stem>
```

Exercises the `cli_registry.py` entry points against the post-change DB. Mostly regression coverage — checks that schema + report shape haven't drifted.

### 7. Streamlit maintenance panel

```bash
streamlit run app.py
```

Open the registry tab → confirm: migrate button, FOV list, session list, cell browser all render against the v3 DB. If step 5 left sessions in the review band, visually open them here and confirm the manual-accept path works.

### 8. Before declaring a remediation successful

All of the following must hold on the three-session dataset:

- [ ] Run 5 produces exactly 1 FOV with 3 sessions.
- [ ] Every step has `fov_posterior ≥ 0.9` (accept band).
- [ ] `alignment_inlier_rate ≥ 0.4` on every step.
- [ ] `fraction_query_clustered ≥ 0.4` on every step.
- [ ] Steps 2–4 (unit tests, CLI, Streamlit) all pass.
- [ ] A new "Run N" section is appended here with the numbers.
- [ ] Design doc `docs/design/roicat-integration.md` gains a short addendum recording the change.

---

## Registry state on disk (end of Run 3 — PASS)

```
inference/registry_roicat.db: 1 FOV row, 3 sessions, 123 cells
  FOV 79995aa3-443… (latest 2023-01-16, 123 cells)
    session 2022-12-09  n_new=83  n_matched=0   n_missing=0   posterior=—
    session 2022-12-15  n_new=18  n_matched=57  n_missing=0   posterior=0.993
    session 2023-01-16  n_new=22  n_matched=46  n_missing=27  posterior=0.978
```

`inference/registry.db` (the pre-v3 legacy database) was **not** touched.

---

## Outstanding / optional

- **Labeled-pair calibration refit** (Option 5 in the remediation table). Coefficients are still hand-priors; on the three-session dataset we have 2 known same-FOV positive examples and 0 known negatives. Refit becomes actionable once we have at least a few known-different FOV pairs to anchor the negative class.

---

## Managing the registry from the Streamlit app

The Streamlit app's **Registry** tab (`app.py::_registry_tab`) is the primary UI surface for inspecting + maintaining the v3 registry. It is read-first / write-via-backfill — no manual review-band approval UI, no adapter-knob UI.

### Launch

```bash
# Point at the v3 DB + blob store you want to manage.
export ROIGBIV_REGISTRY_DSN="sqlite:///inference/registry_roicat.db"
export ROIGBIV_BLOB_ROOT="inference/fingerprints_v3"

# RoMa is the default alignment; device auto-detects cuda. Override only if needed.
# export ROIGBIV_ROICAT_ALIGNMENT=PhaseCorrelation
# export ROIGBIV_ROICAT_DEVICE=cpu

streamlit run app.py
```

Open **Cross-Session Registry** (the rightmost tab).

### 1. Registry maintenance (first-time / after schema changes)

Expander: **"Registry maintenance"** → button **"Run database migrations"**.

- Runs `roigbiv.registry.migrate.ensure_alembic_head()` against the DSN in `ROIGBIV_REGISTRY_DSN`.
- Idempotent — safe to re-run. Displays the resolved revision (`0003` is the current head).
- Run this once after cloning or upgrading the repo on any non-fresh DB. Fresh DBs are auto-stamped when a session is first registered.

### 2. Backfill existing pipeline outputs into the registry

Expander: **"Backfill existing runs"**.

1. Set **"Root directory to scan"** to the directory containing your pipeline outputs (defaults to the sidebar's output dir, usually `data/output/`). Each subdirectory with `merged_masks.tif + summary/mean_M.tif` is a candidate session.
2. **Dry run** — walks the directory, prints how many candidates would be registered, does NOT write. Use this first on any unfamiliar directory.
3. **Backfill now** — registers/matches every candidate in chronological order via `roigbiv.registry.backfill.run_backfill`. Each session that matches above `fov_accept_threshold` is auto-merged into its FOV; sessions that fall into review or reject follow the normal orchestrator rules.

First-time backfills with the default RoMa alignment will trigger the ~1.5 GB model download on the first session that has a candidate FOV to match against — expect a multi-minute delay and no progress bar in the UI for the first pair.

### 3. Browse FOVs

Below the expanders you get a table of every FOV in the registry with `animal_id`, `region`, `latest_session`, session count, cell count, and fingerprint prefix.

### 4. Per-FOV session detail

Select a FOV from the **"Select a FOV"** dropdown.

- **Sessions table** — one row per session with `session_date`, `fov_posterior`, `n_matched`, `n_new`, `n_missing`, `output_dir`. `fov_posterior` is `None` for the founding session (which had no candidates to match against).
- **Expandable "Cells in this FOV"** — one row per `global_cell_id` with the session it was first seen in and the local label id it had there.

This is the main "is the match quality what I expect?" view. If a row shows `n_missing > 0`, those are cells present in an earlier session but not observed in this one; `n_new > 0` flags genuinely new cells.

### 5. Longitudinal cell browser

Text input: paste a `global_cell_id` from the cells expander.

- Produces a per-session table showing the `local_label_id` that cell had in each session and its `match_score` for that session's merge.
- Use this to confirm a cell is stably tracked across days (same `global_cell_id`, different `local_label_id`s per session) and to grab the local labels you need for downstream analyses (e.g., selecting traces from `F.npy` per session by index).

### What the UI does NOT expose (use the CLI or env)

- **Changing alignment / adapter knobs** — set `ROIGBIV_ROICAT_*` env vars *before* launching Streamlit. Runtime changes won't apply to calls already in flight.
- **Manual review-band accept** — there is no button to accept a `review`-band match. To promote one, either (a) delete the minted stub FOV from the DB and re-run backfill against a lower `ROIGBIV_FOV_ACCEPT_THRESHOLD`, or (b) manually edit `cell_observation` rows (advanced; not recommended).
- **Deleting / merging FOVs** — no UI. Use sqlite directly or write a short script against `SQLAlchemyStore`.
- **Viewing ROI overlays** — use the Results tab's napari launcher, or open `merged_masks.tif` directly in napari/Fiji. The registry tab doesn't render masks.
- **Re-clustering existing FOVs with a new alignment method** — not currently supported. Requires deleting the v3 DB and re-running backfill. Planned as an admin action but not implemented.

### Recommended first-run workflow for a fresh dataset

1. `streamlit run app.py` (with env vars set).
2. Registry tab → **Run database migrations** (once).
3. **Backfill existing runs → Dry run** pointed at `data/output/`. Confirm the session count matches what you expect.
4. **Backfill now**. Wait for the RoMa weights download on the first match.
5. Browse the FOV overview. Expect 1 row per distinct FOV across animals/regions.
6. Spot-check one FOV's **session detail** — `n_matched` should be dominant vs `n_new` after the founding session.
7. Sample one `global_cell_id` in the **longitudinal browser** to verify consistent tracking.
