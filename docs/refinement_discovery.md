# Refinement Engagement — Phase 0 Discovery

**Scope:** Improve recall of high-baseline (bright) and tonic pyramidal neurons and modernize Stage 1
segmentation, without altering the sequential-subtractive spine, per-stage gates, provenance/
confidence system, HITL export, cross-session registry, or the virtual-residual engine. Every change
is an A/B-gated, one-variable experiment validated on the held-out set; no default flips without a
benchmark and explicit sign-off.

**Phase 0 is read-only.** No functional code changed. Per the engagement's
`discovery_before_code` rule, if the codebase contradicts a directive assumption, the work STOPS and
reports rather than adapting code to the directive. **Two such contradictions were found** (Findings A
and B below) and resolved at the Phase-0 gate.

---

## ⚠️ Blocking findings

### FINDING A — Phase 1's core premise is ALREADY SATISFIED
Phase 1 proposed switching the Stage 1 morphological channel from `mean_S` (≈0 after SVD) to the raw
`mean_M`. **The code already feeds `mean_M`.**

- `roigbiv/pipeline/run.py:548-554` — Stage 1 call passes `fov.mean_M` as the morphological channel,
  with an inline comment stating the directive's exact rationale ("mean_S ≈ 0 under SVD-based L+S …
  so mean_M preserves the morphological contrast Cellpose's training expects").
- `roigbiv/pipeline/stage1.py:284` — channels assembled as `np.stack([morph_input, vcorr_S], -1)`.
- Doc tension is real but stale: spec §3.1/§3.4 say "denoised mean_S"; the **code uses mean_M**.

The only thing missing is that the choice is **hardcoded** rather than config-selectable.

**Resolution (gate decision): SKIP Phase 1.** Premise met; the bright-cell recovery is already the
default behavior. Sequence proceeds from Phase 2.

### FINDING B — Cellpose-SAM (Phases 3/4) requires a major-version bump
- Installed `cellpose==3.1.1.2`; pinned `cellpose<4.0.0` (`environment.yml:22`).
- `cpsam` is in the resolver allow-list (`roigbiv/pipeline/stage1.py:55-59`) but is **illusory under
  3.x** — CP3 silently falls back to the default model. `cpsam` requires **cellpose ≥4.x**. The repo
  documents this deliberately (`roigbiv/pipeline/profiles.py:22-24`: "cpsam … cannot load under this
  repo's cellpose<4.0.0 pin … a separate, deferred sidecar track").
- A 3.x→4.x bump risks breaking: (a) `denoise_cyto3` via `cellpose.denoise.DenoiseModel`
  (`stage1.py:145-164`); (b) the `channels=(1,2)` role API (`stage1.py:313-321`) — cpsam is
  channel-invariant; (c) the deployed fine-tuned **CP3 checkpoint** (`models/deployed/current_model`,
  loaded `pretrained_model=` at `stage1.py:248-256`) — high risk of load incompatibility / forced
  retrain.

**Resolution (gate decision): UPGRADE cellpose to ≥4.x** (full upgrade, not a sidecar), as a
prerequisite migration phase (Phase M) that must land and be validated before Phases 3/4. Carries
`migration` + `data_integrity` risk and touches the protected `models/deployed/` zone — any
checkpoint change/retrain requires explicit confirmation.

---

## Nine discovery answers (with citations)

1. **Background method / RPCA / k_background.** Default `background_method = "svd"`
   (`roigbiv/pipeline/types.py:187`). The `rpca` path **exists and is wired end-to-end**:
   implementation `roigbiv/pipeline/rpca.py:1-32`; dispatched from `roigbiv/pipeline/foundation.py:369-446`
   (`method = getattr(cfg, "background_method", "svd")` → `if method == "rpca":`, with GPU-memory
   adaptation + CPU fallback). `k_background = 30` (`types.py:166`, also `types.py:140`); applied at
   `foundation.py:476-481` (`k = min(int(cfg.k_background), n_svd)`).

2. **Stage 1 input (mean_S vs mean_M).** Morphological channel = **`mean_M`** (raw registered-movie
   mean), passed through `denoise_cyto3` when `cfg.use_denoise=True` (default; `types.py:304`). Call
   site `roigbiv/pipeline/run.py:548-554`; denoise applied `stage1.py:259-270` (`denoise_mean_S`,
   defined `stage1.py:145-164`, uses `cellpose.denoise.DenoiseModel(model_type="denoise_cyto3")`).
   Channel 2 = `vcorr_S` (`stage1.py:218-221`, stacked `stage1.py:284`). Model called
   `stage1.py:313-321` with `channels=list(cfg.channels)`. **The doc's "denoised mean_S" is stale;
   code uses mean_M.** See Finding A.

3. **Cellpose model resolver / version / cpsam.** Resolver `stage1.py:55-98`; builtins frozenset
   includes `cpsam` (`stage1.py:55-59`); model load `stage1.py:248-256`
   (`CellposeModel(model_type=...)` for builtins, `pretrained_model=...` for paths). Installed
   **cellpose 3.1.1.2**; pinned `<4.0.0` (`environment.yml:22`; `pyproject.toml` omits it by design,
   `pyproject.toml:28-30`). `cpsam` not usable on 3.x. Default model path `models/deployed/current_model`
   (`types.py:21-23`). See Finding B for the 4.x bump risk matrix.

4. **Evaluation / A-B harness.** EXISTS: entry `roigbiv/eval/harness.py` (CLI `harness.py:127-165`;
   batch mode reads a held-out manifest). Metrics `roigbiv/eval/metrics.py::stratified_metrics`
   (`metrics.py:20-117`) emit recall / precision / F1 / TP / FP / FN. **Already stratified by activity
   type**: `ACTIVITY_TYPES=(phasic, sparse, tonic, silent, ambiguous)` (`metrics.py:16-17`); strata
   read `activity_type` from pipeline `roi_metadata.json` (`harness.py:39-63`). tonic/silent flagged
   `lower_bound` (manual GT under-represents them, "Blindspot 13"). IoU matching at multiple
   thresholds `scripts/compare_models.py:37-39` (AP@0.5 / 0.75 / 0.5:0.95). Held-out set:
   `experiments/harness/heldout_fovs.txt` (13 FOVs; val split seed=42, val_frac=0.15 per
   `scripts/train.py::load_dataset`; format `stem|movie_path|gt_masks_path`). **No new harness needed.**

5. **Summary images.** Full (non-scout) Foundation produces and saves `mean_M, mean_S, max_S, std_S,
   vcorr_S, mean_L, dog_map` (`foundation.py:883-888`). `max_S` computed at `foundation.py:854-857`
   (via `generate_summary_images`) and `foundation.py:513-621` (`_accumulate_summaries`, `np.maximum`
   accumulator). **Scout mode saves only `mean_M, vcorr_S, dog_map`** (`foundation.py:759-761`) — so
   `max_S` is absent in scout / cv-only summaries (relevant to Phase 4).

6. **Subtraction profile source.** Source = **`std_S`** (`run.py:651-658`, with rationale that mean_S≈0
   under truncated-SVD L+S so it can't represent the spatial activity pattern). Independent of the
   Stage-1 morphological-channel choice: `std_S` is computed from residual S before any morph
   selection (`foundation.py:623-639`, `:850-858`) and recomputed for Stage 2
   (`subtraction.py:604-626`). ✅ verified.

7. **Tonic classification.** `roigbiv/pipeline/classify.py:63-68`: `tonic` iff
   `bp_std > cfg.tonic_bp_std_factor * max(noise_floor, 1e-12)` **and** `skew ≤ cfg.phasic_min_skew`
   **and** (`int(roi.source_stage) == 4` **or** `tonic_population_ok`). Population-median criterion
   `tonic_population_ok = (mean_F > median_F) and (std_F < median_std)`, medians computed
   `classify.py:84-87` ("high mean, low variance"). `tonic_bp_std_factor = 2.0` (`types.py:346`).
   Stage-4 ROIs bypass the population check (Stage 4 is the dedicated tonic hunter). Neuropil annulus:
   `neuropil_inner_buffer = 2`, `neuropil_outer_radius = 15` (`types.py:335-336`); geometry built in
   `roigbiv/pipeline/traces.py:38-85` (dilate by inner, then inner+outer; exclude other ROI masks).

8. **Gate 4 / Stage 4.** Gate 4 has **no accept tier** (`gate4.py:26-31` docstring). Decision
   `gate4.py:156-162`: pass all six checks → `gate_outcome="flag"`; any failure → `"reject"`; **both**
   set `confidence="requires_review"`. Every Stage-4 candidate routes to mandatory HITL review. ✅.

9. **Git.** Engagement work began on `fix/0-virtual-residual` with a large uncommitted diff (entry-
   criteria violation — see below); resolved by stashing and cutting `refine/` branches from clean
   `main`. Branches present: `main`, `develop`, `fix/0-virtual-residual`. No active git hooks
   (`.git/hooks/*` are samples), no `.pre-commit-config.yaml`, no CONTRIBUTING. Only CI is
   `.github/workflows/release.yml`, **tag-triggered (`v*`) only** — nothing runs per-branch. Module
   inventory confirmed present under `roigbiv/pipeline/`: `stage1-4.py`, `gate1-4.py`,
   `subtraction.py`, `residual.py`, `classify.py`, `qc_features.py`, `traces.py`, `foundation.py`.

### Secondary flags
- **Entry-criteria:** the engagement assumes a clean working tree on a fresh branch cut from `main`.
  Discovery began on `fix/0-virtual-residual` with a large uncommitted diff. Resolved by `git stash`
  (recoverable) + branching from clean `main`.
- **Scout/cv-only summaries** omit `max_S`, `std_S`, `mean_S`, `mean_L` — Phase 4's `max_S` channel is
  only available on full Foundation runs.

---

## Resolved engagement sequence
`0 → 2 → M → 3 → 4 → 5` (Phase 1 dropped per Finding A). Each phase: its own `refine/phaseN-slug`
branch off `main`, exactly one variable, A/B via the existing stratified harness, OFF-by-default
flags, and a hard gate awaiting human sign-off before the next phase.

- **Phase 2 — PMD spatiotemporal denoising** into Stages 3/4 (config flag, OFF default). Committed
  insertion-point doc first, then a patch-parallel/chunked/bounded-RAM implementation that leaves L+S,
  Stage-2 Suite2p reuse, and the ResidualView/SourceLayer reconstruction contract byte-for-byte intact.
- **Phase M — cellpose ≥4.x migration** (prerequisite for Phases 3/4). Verify/repair `denoise_cyto3`,
  the channels API, and deployed-checkpoint loading; keep CP3 selectable as fallback. `models/deployed/`
  changes require explicit confirmation. STOP and report if the checkpoint cannot load.
- **Phase 3 — model A/B** stock cpsam vs deployed CP3; optional cpsam fine-tune as a separate experiment.
- **Phase 4 — multi-channel input** `mean_M + vcorr_S + max_S` (valid once cpsam channel-invariant).
- **Phase 5 — tonic classification** 5a neuropil-relative baseline-elevation feature (annulus 2/15);
  5b accept tier for Stage-1/2 tonic only — Stage-4 path (`gate4.py`) untouched.

---

## Verification (Phase 0)
Read-only; nothing executed. This document is the Phase-0 deliverable. Later phases validate via the
existing harness, e.g.
`python -m roigbiv.eval.harness --batch experiments/harness/heldout_fovs.txt --pipeline-root experiments/runs/ --output <out>.json`
(metrics already stratified by activity type).
