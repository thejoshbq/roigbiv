# ROI G. Biv — Project & Pipeline Overview

**Sequential subtractive ROI detection for two-photon calcium imaging.**

> **Status:** Current as of `main`. This is the authoritative onboarding + as-built
> overview: what the project is *for* and how the pipeline and its integrated
> algorithms actually work today.
>
> **Companion docs.** `docs/publication/algorithms_v2.md` is the manuscript-grade
> methods reference (every threshold cited to `file:line`); this document is the
> readable, self-contained counterpart and shares its authority for current-code
> behavior. `docs/roi-pipeline-specification.md` is the original design spec
> (framed "pre-implementation") — where it and the code disagree, the code wins,
> and §14 below lists the specific divergences.
>
> **Source of truth for every default in this document is the runtime dataclass
> `PipelineConfig` in `roigbiv/pipeline/types.py:158-514`.** `configs/pipeline.yaml`
> is documentary only (its own header says the runtime does not load it) and
> disagrees in a few places; where it does, the value quoted here is the code value.

---

## Table of contents

1. [Purpose & scientific context](#1-purpose--scientific-context)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Foundation](#3-foundation)
4. [Stage 1 — Cellpose spatial detection](#4-stage-1--cellpose-spatial-detection)
5. [Source subtraction engine](#5-source-subtraction-engine)
6. [Stage 2 — Suite2p temporal detection](#6-stage-2--suite2p-temporal-detection)
7. [Stage 3 — template sweep](#7-stage-3--template-sweep)
8. [Stage 4 — tonic-neuron search](#8-stage-4--tonic-neuron-search)
9. [Post-detection: QC, traces, classification, HITL](#9-post-detection-qc-traces-classification-hitl)
10. [Cross-session FOV & cell registry](#10-cross-session-fov--cell-registry)
11. [Optional & aspirational subsystems](#11-optional--aspirational-subsystems)
12. [Orchestration, GPU, resume, output layout](#12-orchestration-gpu-resume-output-layout)
13. [Master parameter reference](#13-master-parameter-reference)
14. [Authoritative-docs map & doc-drift corrections](#14-authoritative-docs-map--doc-drift-corrections)

---

## 1. Purpose & scientific context

**ROI G. Biv** ("ROI, G. Biv" — a play on ROYGBIV) detects regions of interest (ROIs)
— individual neurons and their activity traces — in **two-photon calcium imaging**
movies of the mouse brain. The default indicator is **GCaMP6s** (`tau=1.0` s,
`types.py:173`). Its output — ROI masks plus per-ROI fluorescence traces — feeds
downstream to **[pynapse](https://github.com/Otis-Lab-MUSC/pynapse)** for signal
analysis / peri-event alignment and to **axplorer** for visualization.

> **axplorer status.** axplorer is named as a downstream consumer but there is **no
> axplorer export path in the codebase today** — treat it as external/planned. The
> **pynapse** export *is* wired (`scripts/roigbiv_to_pynapse.py`, `roigbiv/cli_export.py`,
> `roigbiv/pipeline/export_io.py`).

### The core idea: sequential subtractive detection

Older versions of this project used a **parallel three-branch consensus** design
(two Suite2p passes + Cellpose, merged by spatial IoU into GOLD/SILVER/BRONZE
tiers). **That architecture is retired.** The current pipeline is **sequential and
subtractive**: a shared Foundation prepares the movie, then four detection stages
run in order, each operating on the **residual** left after the previous stages
subtract the sources they found. A neuron detected and validated in Stage 1 is
*removed* from the movie before Stage 2 looks at it, so each stage sees only what
its predecessors missed.

### Design principles

Reconciled from the spec's "Design Principles" (`docs/roi-pipeline-specification.md`
§1) against current code:

- **Recall first.** Prefer surfacing every plausible source (even at the cost of
  false positives) and let validation gates + human review remove the bad ones.
  Missing a real neuron is worse than flagging a spurious one for review.
- **Sequential and subtractive.** Each stage detects on the residual after prior
  detections are subtracted, so the four detectors are *complementary* rather than
  redundant — spatial morphology (Stage 1), classifier-backed activity (Stage 2),
  matched-filter transients (Stage 3), and slow/tonic modulation (Stage 4).
- **Validate between stages.** Every stage is paired with a **gate** that accepts /
  flags / rejects each candidate on independent evidence before it is subtracted.
- **Per-ROI provenance.** Every ROI records which stage found it, its gate outcome,
  a confidence level, and the stage-specific score that justified it.
- **Progressive convergence through HITL.** Ambiguous ROIs are packaged for
  human-in-the-loop review; corrections are additive and can retrain the Cellpose model.
- **Existing tools as infrastructure.** Cellpose, Suite2p, and ROICaT are used as
  building blocks; the novel work is the sequencing, subtraction, and gating around them.
- **Default on.** All four stages run by default so the cheapest invocation gives
  full coverage (`enable_stage_2/3/4=True`, `types.py:499-501`); the fast path drops
  `--no-stage-3 --no-stage-4`.

---

## 2. Architecture at a glance

```
Foundation
  motion correction → truncated-SVD L+S background split → summary images (virtual residual S)
        │
        ▼
Stage 1  Cellpose spatial detection   → Gate 1  morphology            → subtract → residual S₁
        ▼
Stage 2  Suite2p temporal detection   → Gate 2  temporal cross-val    → subtract → residual S₂
        ▼
Stage 3  template sweep on residual    → Gate 3  waveform validation   → subtract → residual S₃
        ▼
Stage 4  tonic-neuron search           → Gate 4  correlation contrast  (terminal — no subtraction)
        │
        ▼
Unified QC → trace extraction → activity classification → HITL review package
```

Single-FOV orchestration lives in `roigbiv/pipeline/run.py::run_pipeline`. Two
invariants make the rest of the document readable:

### The residual is virtual

`S = M − L` (movie minus low-rank background) is **never materialized to disk.** A
single live `ResidualView` (`roigbiv/pipeline/residual.py`) accumulates one
`SourceLayer` per subtraction stage and reconstructs any chunk on demand from
`data.bin` + `svd_factors.npz`, i.e. `S_new = S_in − Σ wᵢ·cᵢ`. Each stage detects on
the **live residual view** carrying all prior subtractions: Stage 3 reads S₂ if it
exists, else S₁, else S. The former on-disk `residual_S{,1,2,3}.dat` memmap chain is
gone — `FOVData.residual_S_path` is deprecated and stays `None` (`types.py:119-124`).

### `mean_S ≈ 0`, so the code uses `mean_M` and `std_S`

Under a **truncated-SVD** low-rank/sparse split, the top-k SVD components absorb each
pixel's mean brightness into `L`, so the residual mean image `mean_S` is ≈ 0 and
carries no morphology. The pipeline therefore substitutes, **everywhere the spec said
`mean_S`**:

- **`mean_M`** — the raw registered-movie mean (from Suite2p's `ops["meanImg"]`) — as
  the morphological channel: Cellpose channel 1, Gate-1 soma-surround contrast, the DoG
  nuclear-shadow map, and Gate-4 intensity floor.
- **`std_S`** — the per-pixel RMS of the residual — as the subtraction spatial-profile
  source (a source's footprint is where its temporal variance lives, not its mean).

This is load-bearing; keep it in mind through §§3–8. (`foundation.py`, `gate4.py`,
`subtraction.py`.)

---

## 3. Foundation

`roigbiv/pipeline/foundation.py::run_foundation` prepares everything the stages
consume: a registered movie (`data.bin`), the low-rank/sparse split, summary images,
and the DoG map.

### 3.1 Motion correction — three backends

Dispatched on `cfg.motion_correction_backend` (`types.py:210`):

- **`phasecorr` (default).** Suite2p performs rigid + non-rigid registration **and**
  detection in one pass (`roigbiv/suite2p.py::run_suite2p_fov`), producing the int16
  `data.bin` plus `stat.npy` / `iscell.npy` / `ops.npy` that L+S, Stage 2, and Stage 4
  all reuse. **Tuned defaults** (`types.py:246-260`): non-rigid `block_size=[64,64]`
  and one-photon high-pass registration `1Preg=True`. Full-session validation on the
  Prism reference FOV showed the old `[128,128]`/no-1Preg default reached only ~58% of
  legacy SIMA cell-sharpness while `[64,64]+1Preg` reaches ~103%. For bright high-SNR
  2P data, disable the 1P high-pass with `--no-mc-1preg`.
- **`rowwise-pcc` (opt-in).** GPU row-wise non-rigid phase correlation
  (`registration.py::run_rowwise_pcc_register`) with strip regularization (taller
  strips, median/confidence outlier rejection, spatiotemporal smoothing) to suppress
  noise-driven per-row warps on dim FOVs. Suite2p then runs detection-only.
- **`legacy` (opt-in).** Genuine SIMA `HiddenMarkov2D(granularity='row')` run in the
  `sima-legacy` Python-3.8 conda sidecar via subprocess (`legacy_mc.py::run_sima_legacy_register`,
  SIMA 1.3.2). CPU-only and slow (tens of minutes to hours per FOV); a faithful
  reproduction of the legacy notebook for exact repro.

Inputs already motion-corrected (embedded TIFF `Software` tag, or `_mc` filename
fallback — `roigbiv.io.detect_motion_corrected`) set `do_registration=False`, so
Suite2p skips its own registration; Foundation still registers when the input lacks
`_mc`. Motion traces (`xoff`/`yoff`) are persisted to `motion_trace.npz` and reused by
Gate 4. Every backend exports `{stem}_mc.tif`.

### 3.2 Truncated-SVD low-rank / sparse (L+S) background split

`compute_background_separation` (`foundation.py`):

1. Open `data.bin` as an int16 memmap; temporally **bin** the movie to a target
   `svd_bin_frames=5000` frames (`types.py:174`) to bound the SVD cost.
2. Compute the top-`n_svd=200` (`types.py:167`) singular vectors with
   **`torch.svd_lowrank(A, q=n_svd, niter=2)`** on GPU (CPU fallback), factoring the
   transposed binned movie so `U` indexes pixels. **Seeded `torch.manual_seed(0)`** for
   run-to-run determinism.
3. **L** = the top-`k_background=30` (`types.py:166`) rank reconstruction — the slow,
   structured background (neuropil, hemodynamics, bleaching). **S = M − L** is the
   sparse residual carrying transient cellular activity.
4. `mean_L` is computed in closed form; `S` is left **virtual** (see §2). SVD factors
   persist to `svd_factors.npz`; a `residual_S.meta.json` sidecar marks `kind: virtual`.

> This default background is a **truncated-SVD** split, **not** a robust PCA. A genuine
> RPCA implementation exists in `rpca.py` but is not wired — see §11.

### 3.3 Summary images

`generate_summary_images` makes a single streaming pass over the reconstructed residual
(chunks capped at 128 frames to bound RAM), accumulating:

- **`mean_M`** — raw registered-movie mean from Suite2p `ops["meanImg"]` (the
  morphological channel; **not** `mean_S`).
- **`max_S`, `std_S`** — residual peak and per-pixel RMS.
- **`vcorr_S`** — 8-neighbour local correlation map (each undirected pixel-pair
  correlation computed once via fixed edge families).

### 3.4 DoG nuclear-shadow map

`compute_nuclear_shadow_map` computes a Difference-of-Gaussians
`G(σ_outer=6) − G(σ_inner=2)` (`scipy.ndimage.gaussian_filter`) **on `mean_M`**,
positive over dark nuclei (soma interiors appear as intensity dips ringed by bright
cytoplasm). Gate 1 uses it to rescue low-contrast somata.

### 3.5 Scout path

`scout_mode=True` (`types.py:181`) skips SVD/L+S entirely, computes Vcorr directly on
`data.bin`, and stops after Stage 1 + Gate 1 — fast FOV-clarity / model-A/B triage.
Not analysis-grade (no traces/QC/registry) and not resumable into a full run.

### Foundation parameters

| Param | Default | Meaning |
|---|---|---|
| `k_background` | 30 | rank of L (background) reconstruction |
| `n_svd` | 200 | singular vectors computed (superset of k) |
| `fs` | 30.0 | effective frame rate (Hz) after frame averaging — **pass `--fs 7.5` for 4×-averaged `_mc` stacks** |
| `frame_averaging` | 1 | temporal binning factor that produced `fs` |
| `tau` | 1.0 | indicator decay τ (GCaMP6s) |
| `svd_bin_frames` | 5000 | target binned frame count for the SVD |
| `reconstruct_chunk` | 500 | temporal chunk for L+S streaming |
| `motion_correction_backend` | `"phasecorr"` | `phasecorr` \| `rowwise-pcc` \| `legacy` |
| `mc_s2p_block_size` | `[64, 64]` | non-rigid block size (tuned) |
| `mc_s2p_one_photon_reg` | `True` | 1P high-pass registration (tuned; `--no-mc-1preg` for bright 2P) |
| `mc_max_displacement` | 50 | px clamp (shared by rowwise-pcc + legacy) |

> **Frame-rate gotcha.** `fs=30.0` is the *acquisition* rate. This lab's `_mc` stacks
> are 4×-online-averaged → effective 7.5 Hz. Always pass `--fs 7.5` for averaged stacks;
> a wrong `fs` miscalibrates Stage-3 templates, Stage-4 bandpass windows, and the
> deconvolution τ.

---

## 4. Stage 1 — Cellpose spatial detection

`roigbiv/pipeline/stage1.py::run_cellpose_detection`. Finds soma-shaped objects on the
morphological image.

- **Detector:** **Cellpose 3.x** (`cellpose<4.0.0`; CP3 is canonical — all deployed
  checkpoints are CP3 format). Default backend `stage1_backend="cellpose3"` (`types.py:311`).
- **Model:** the deployed fine-tuned checkpoint `models/deployed/current_model`
  (`_DEFAULT_CELLPOSE_MODEL`, `types.py:21-23`). The default is anchored to the package
  root, not cwd, so runs from any directory load the fine-tuned model rather than
  silently falling back to stock cyto3. It is loaded via
  `CellposeModel(gpu=..., pretrained_model=model_path)` for checkpoint paths (the
  default), and `model_type=...` only for Cellpose's built-in model names
  (`stage1.py:380-383`).
- **Dual-channel input:**
  - **ch1 = `mean_M`**, optionally passed through Cellpose3's `denoise_cyto3` image
    restoration first (`use_denoise=True`, `types.py:298`).
  - **ch2 = `stage1_ch2_source`** — default **`vcorr_max_fused`** (`types.py:330`): the
    per-image min-max-normalized `max(vcorr_S, max_S)`, i.e. "is correlated **or** has a
    bright peak." Alternatives: `vcorr_S` (legacy) or `max_S`. The default was flipped
    from `vcorr_S` after a Phase-4 A/B (+0.017 recall, 0/13 FOV regressions).
- **Diameter:** fixed `diameter=12` px by default. `diameter_auto=True` instead
  estimates a per-FOV soma scale with **roigbiv's own** DoG-peak-detection + Otsu sizing
  on the morphological channel (`_effective_diameter` → `optics.measure_soma_scale`),
  *not* Cellpose's `SizeModel` — the custom-trained `CellposeModel` carries no SizeModel
  (`stage1.py:405-411`).
- **Thresholds:** `cellprob_threshold=-2.0`, `flow_threshold=0.4` (`types.py:294-295`).
  Returns per-ROI boolean masks plus per-ROI mean `cellprob`.

### Gate 1 — morphological validation

`roigbiv/pipeline/gate1.py::evaluate_gate1`. Per candidate, on `mean_M`:

| Check | Threshold | Param |
|---|---|---|
| Area | ∈ [80, 600] px² | `min_area`, `max_area` |
| Solidity | ≥ 0.55 | `min_solidity` |
| Eccentricity | ≤ 0.90 | `max_eccentricity` |
| Soma-surround contrast | > 0.10 | `min_contrast` |
| DoG (nuclear shadow) | conjunctive with contrast | `dog_strong_negative_percentile=10.0` |

- **Contrast** is measured against an annular surround (dilated ring excluding other
  ROIs; `annulus_inner_buffer=2`, `annulus_outer_radius=15`).
- **DoG rule is conjunctive** (spec §6): a candidate is DoG-rejected only if its
  `nuclear_shadow_score` is below the 10th-percentile DoG value **and** its contrast
  check also fails — a soma sitting on a genuine nuclear shadow is not punished for low
  surround contrast.
- **Decision:** reject if DoG-reject **or** ≥ 2 non-contrast failures; accept if 0
  failures; **flag** if exactly 1 failure within its per-criterion margin
  (`flag_area_margin=20`, `flag_solidity_margin=0.05`, `flag_eccentricity_margin=0.03`,
  `flag_contrast_margin=0.03`), else reject.
- **Merge demotion:** masks larger than `gate1_merge_peak_min_area=4000` px² with ≥ 2
  intensity peaks (`peak_local_max`, `min_distance=gate1_merge_peak_min_separation=28`)
  are demoted accept→flag (splitting is a downstream/HITL concern). Inert on GRIN, where
  `max_area=600` caps masks well below 4000.

Gate 1 sets `source_stage=1`, `gate_outcome`, and `confidence`. Accepted + flagged
Stage-1 ROIs are then subtracted.

---

## 5. Source subtraction engine

`roigbiv/pipeline/subtraction.py::run_source_subtraction`. Removes detected sources
from the residual so the next stage sees only what remains. **Only `accept` + `flag`
ROIs are subtracted; rejects never are.** Five steps:

1. **Spatial profiles** (`estimate_spatial_profiles`). For each ROI,
   `wᵢ = profile_source / max_over_mask(profile_source)`, peaking at 1.0 inside the mask
   and 0 outside. **`profile_source = std_S`** (per-pixel RMS), *not* the spec's
   `mean_t[S]` (≈ 0 under SVD L+S) — see §2.
2. **Simultaneous trace estimation** (`estimate_traces_simultaneous` →
   `solve_traces_from_chunks`). GPU-chunked ridge normal equations over the union of ROI
   pixels: `c = (WᵀW + λI)⁻¹ Wᵀb`, with `λ` scaled by
   `subtract_ridge_lambda_scale=1e-6`. An optional one-sided-Huber **IRLS robust
   solver** (`subtract_solver="robust"`) down-weights positive residuals (real
   unmodeled transients) so they aren't absorbed into a neighbour's trace — off by
   default (see §11).
3. **Lazy rank-1 subtraction** (`subtract_sources`). Appends a
   `SourceLayer(flat_idx, W_design, traces)` to the `ResidualView` via `with_source` —
   nothing dense is written. Reading any chunk reconstructs `S_new = S_in − Σ wᵢ·cᵢ` on
   demand. Only a small `{output_name}.sources.npz` + `.meta.json` sidecar persists for
   `--resume`.
4. **Streaming validation** (`validate_subtraction`). One streaming pass computes, per
   ROI: `mean_ratio` (< 3.0), `std_ratio` (∈ (0.3, 3.0)), and `anticorr_max`
   (> `subtract_anticorr_threshold=-0.3`) against an annular surround. A clean
   subtraction leaves flat, uncorrelated residual over the footprint; strong negative
   anticorrelation means over-subtraction.
5. **NNLS fallback** (`_nnls_refine_flagged`). Triggered only if the anticorr failure
   fraction exceeds `subtract_anticorr_failure_fraction=0.10`: re-estimates the flagged
   ROIs' traces over local mask pixels with a single-variable closed-form NNLS
   (`max(0, ·)` — no negative activity), capped at `subtract_nnls_fallback_max_rois=30`,
   then re-subtracts and re-validates just those.

---

## 6. Stage 2 — Suite2p temporal detection

`roigbiv/pipeline/stage2.py::run_stage2`. Recovers active neurons Stage 1 missed
morphologically.

- **Does NOT re-run Suite2p.** It **reuses** the `stat.npy` / `iscell.npy` already
  produced by Foundation's Suite2p pass — Suite2p's detection is a byproduct of
  registration, so there is nothing to recompute.
- **Classifier filter:** keeps candidates with `iscell[:,1] ≥ iscell_threshold=0.3`
  (`types.py:395`), converting sparse `stat` entries to dense masks.
- **Novelty filter:** an **IoU filter against Stage-1 `accept|flag` masks** keeps only
  candidates with max IoU ≤ `gate2_iou_threshold=0.3` — genuinely new detections, not
  rediscoveries of Stage-1 cells.
- **Traces** are extracted from the **live residual view (S₁)**, not the raw movie.

### Gate 2 — temporal cross-validation

`roigbiv/pipeline/gate2.py::evaluate_gate2`. Fills solidity/eccentricity via
`regionprops`, then, against nearby Stage-1 ROIs (centroid within
`gate2_spatial_radius=20` px), **rejects** if:

- `|r| ≥ gate2_max_correlation=0.7` with any nearby Stage-1 ROI (spillover/redundant), or
- `r ≤ gate2_anticorr_threshold=-0.5` (subtraction artifact / cascade defense), or
- centroid within `gate2_near_distance=5` px **and** `|r| > gate2_near_corr_threshold=0.5`
  (near-duplicate), or
- relaxed morphology fails: area ∈ [60, 400], solidity ≥ 0.4, eccentricity ≤ 0.85
  (looser than Gate 1 — Suite2p footprints are noisier than Cellpose).

**Flags** rather than accepts if max `|r| > gate2_flag_corr_threshold=0.5`. Accepted +
flagged Stage-2 ROIs are subtracted (producing S₂) if any downstream stage is enabled.

---

## 7. Stage 3 — template sweep

`roigbiv/pipeline/stage3.py::run_stage3` with `stage3_templates.py`. A **matched
filter** on the residual (typically S₂) for isolated calcium transients too small or
too infrequent for Suite2p.

- **FFT matched filter.** Per spatial chunk (auto-sized by
  `stage3_chunk_budget_bytes=1 GB`): estimate per-pixel σ from the global MAD, then
  cross-correlate every pixel trace against each template via
  `torch.fft.rfft`/`irfft` (`ifft(fft(trace) · conj(fft(template)))`), take the running
  max across templates, and threshold at `template_threshold=6.0σ` (`types.py:416`).
- **GCaMP kernel bank** (`build_template_bank`). Each template is
  `(1 − e^{−t/τ_rise})·e^{−t/τ_decay}`, truncated at `5·τ_decay`, sampled at `fs`, and
  **L2-normalized to unit energy**. Three kinetics per indicator family; the family is
  auto-selected by τ (`τ < 0.75 s → jGCaMP8f`, else GCaMP6s).
- **6σ is deliberately high.** In real residual data the per-pixel noise has a heavier
  right tail than Gaussian (structured neuropil leakage); 4σ produced 150M+ false
  crossings on a single FOV, so 6σ brings counts into a clusterable 1e3–1e5 range. An
  adaptive per-chunk threshold bump fires above ~200k events; a hard cap
  `stage3_max_events=2,000,000` keeps the top events by score.
- **Spatial clustering** (`_cluster_events_spatial`). SciPy single-linkage `fcluster` at
  `cluster_distance=12` px (an O(n) grid-snap fallback above 20k events).
- **Per cluster:** a **temporally-independent event count** (greedy pick of events
  ≥ `min_event_separation=2.0` s · fs apart), a disk mask of radius
  `spatial_pool_radius=8`, and a trace from the residual.

> **Documented spec deviation.** σ is a **per-pixel *global* MAD**, not the spec's
> sliding-window `σ_local(p,t)` (`stage3.py`). The `stage3_sigma_window_frames=500`
> param is retained for the sliding variant but the shipped path uses the global MAD.

### Gate 3 — waveform validation

`roigbiv/pipeline/gate3.py::evaluate_gate3`. Per candidate:

- **Solidity** ≥ `gate3_min_solidity=0.5`.
- **Waveform R²** vs the best-matching template per event (peak-aligned, least-squares
  amplitude) ≥ `gate3_min_waveform_r2=0.6` (≥ `gate3_min_waveform_r2_single_event=0.5`
  for single-event candidates).
- **Rise/decay ratio** < `gate3_max_rise_decay_ratio=0.5` (10→90% rise vs peak→37%
  decay) — enforces the fast-rise/slow-decay calcium shape.
- **Anti-correlation** ≤ `gate3_anticorr_threshold=-0.5` vs prior Stage 1–2 ROIs within
  `gate2_spatial_radius=20` px (Gate 3 reuses the Gate-2 radius). Marginal R² (within
  +0.1 of threshold) → flag.

Confidence is graded by event count (≥ 6 high, ≥ 2 moderate, 1 low). Accepted + flagged
Stage-3 ROIs are subtracted (producing S₃) if Stage 4 is enabled.

---

## 8. Stage 4 — tonic-neuron search

`roigbiv/pipeline/stage4.py::run_stage4`. Finds **tonically active** neurons — steady
firers with no discrete transients, invisible to transient-based detectors. Runs on the
residual (S₃), all memmap-backed:

1. **Per-pixel linear detrend** (`detrend_to_memmap`) — vectorized OLS, once.
2. For each of three **bandpass windows** — fast (0.5–2.0 Hz), medium (0.1–1.0 Hz),
   slow (0.05–0.5 Hz) — apply a **zero-phase Butterworth** filter
   (`scipy.signal.sosfiltfilt`, `bandpass_order=4`), chunked in space. Windows optionally
   run in a `ThreadPoolExecutor` (`stage4_n_workers=3`). A window is skipped if the
   recording is shorter than ~5 cycles of its low-frequency edge.
3. **Temporal compression** to `n_svd_components_stage4=300` dims via binned averaging,
   preserving pairwise correlations.
4. **Correlation-contrast map** (`compute_correlation_contrast`) — z-score each pixel
   trace, then `inner_corr − outer_corr` where inner = a self-excluded disk of radius
   `corr_neighbor_radius_inner=6` and outer = an annulus out to
   `corr_neighbor_radius_outer=15`, computed by spatial convolution (disk kernels) to
   avoid O(N²). A tonic cell is a patch of pixels correlated with each other but not with
   the surround.
5. **Cluster** pixels with contrast > `corr_contrast_threshold=0.10`, connected
   components + morphology filter (area ∈ [80, 350], solidity ≥ 0.6, ecc ≤ 0.85).
6. **Cross-window IoU merge** (greedy, `stage4_iou_merge_threshold=0.3`), recording
   `n_windows_detected`.

### Gate 4 — correlation-contrast validation

`roigbiv/pipeline/gate4.py::evaluate_gate4`. Six checks:

| Check | Threshold | Param |
|---|---|---|
| Correlation contrast | > 0.10 | `gate4_min_corr_contrast` |
| Eccentricity | ≤ 0.85 | (stage4) |
| Solidity | ≥ 0.6 | (stage4) |
| **Motion correlation** | \|Pearson(trace, `xoff`/`yoff`)\| < 0.3 | `gate4_max_motion_corr` |
| Anti-correlation vs prior ROIs | > −0.5 within 20 px | `gate4_anticorr_threshold`, `gate4_spatial_radius` |
| Intensity floor | `mean_M` ≥ 25th-percentile | `gate4_min_mean_intensity_pct` |

The **motion-correlation check is unique to Gate 4** — it catches ring artifacts that
track the motion-correction shifts rather than real signal. Gate 4 uses **`mean_M`** for
the intensity floor (not `mean_S`).

> **Stage 4 has no accept tier.** Passing candidates get `gate_outcome="flag"` /
> `confidence="requires_review"`; failures get `reject`. Tonic ROIs *always* go to human
> review by default — the accepted count from Stage 4 is 0 unless the optional tonic
> accept tier is enabled (§11).

---

## 9. Post-detection: QC, traces, classification, HITL

After Stage 4, `run_pipeline` finalizes every non-rejected ROI:

- **Trace extraction** (`traces.py::extract_all_traces`): `F_raw`, `F_neu` (neuropil),
  `F_corrected = F_raw − neuropil_coeff·F_neu` with `neuropil_coeff=0.7`. Overlapping ROI
  groups (`overlap_correction.py`) get their traces re-estimated jointly to avoid
  double-counting shared pixels.
- **QC feature battery** (`qc_features.py::compute_all_features`): spatial
  (boundary gradient, spatial blur/FWHM, FOV distance), temporal (std, skew, SNR,
  transient count, mean fluorescence, bandpass std/power ratio, autocorr τ), and
  provenance (`n_stages_detected`). A 0.05–2.0 Hz `trace_bandpass` array is stored on
  each ROI's `features` — the **primary HITL evidence for Stage-4 tonic candidates**.
- **dF/F + deconvolution** (`dff.py`, `deconvolution.py`): sliding-baseline dF/F and
  spike deconvolution (τ from `cfg.tau`).
- **Activity classification** (`classify.py::classify_all_rois`): a **rule-based decision
  tree** (not a model), five labels evaluated top-to-bottom — **phasic → sparse → tonic
  → silent → ambiguous** (`phasic_min_transients=5`, `sparse_min_transients=1`,
  `tonic_bp_std_factor=2.0`, etc.). Tonic requires elevated bandpass std, low skew, and
  either `source_stage==4` or a population "high-mean/low-variance" signature.
  **Silent-cell retention** (spec "Blindspot 8"): a silent ROI is kept only if
  `nuclear_shadow_score > 0` **or** `solidity > 0.7`.

### ROI schema & provenance

`roigbiv/pipeline/types.py::ROI` (`types.py:26-104`). Every ROI carries:

| Field | Values | Set by |
|---|---|---|
| `label_id` | unique across all stages on the FOV | detection |
| `source_stage` | 1 \| 2 \| 3 \| 4 | each stage |
| `gate_outcome` | `accept` \| `flag` \| `reject` | the matching gate |
| `confidence` | `high` \| `moderate` \| `requires_review` | the matching gate |
| `cellpose_prob` | Stage-1 mean cellprob | Stage 1 |
| `iscell_prob` | Stage-2 classifier prob | Stage 2 |
| `event_count` | Stage-3 independent events | Stage 3 |
| `corr_contrast` | Stage-4 contrast score | Stage 4 |

Plus spatial features (`area`, `solidity`, `eccentricity`, `nuclear_shadow_score`,
`soma_surround_contrast`), traces, `activity_type`, a free-form `features` dict, and
human-readable `gate_reasons`. `to_serializable()` defines the JSON schema written to
`roi_metadata.json` (drops masks/arrays).

> **Stale comment note.** `iscell_prob`/`event_count`/`corr_contrast` are annotated
> "future" in `types.py:48-50`, but the current Stages 2–4 **do** populate them.

### Human-in-the-loop package

`roigbiv/pipeline/hitl.py`. `build_review_queue` builds a **4-tier priority queue**
(P1 = Stage-4 `requires_review`, ascending corr-contrast; P2 = flagged/moderate by
descending stage; P3 = Stage-3 single-event/low; P4 = informational). `export_hitl_package`
writes `review_queue.json`, `merged_masks.tif` (uint16, label IDs preserved for every
non-rejected ROI), per-ROI evidence (`hitl/stage4/{id}/bandpass_trace.npy` +
`corr_contrast_crop`, `hitl/stage3/{id}/event_frame_indices.json`), and a Cellpose-GUI
training staging dir. **HITL corrections are additive** — they append JSONL ops under
`{fov}/corrections/` (`corrections.py`, idempotent replay) and never mutate pipeline
outputs.

---

## 10. Cross-session FOV & cell registry

`roigbiv/registry/` tracks the same FOV and the same cells across imaging sessions.

- **Matching pipeline** (`registry/roicat_adapter.py`) uses **ROICaT**: an **Aligner**
  (geometric → optional non-rigid; default method **RoMa**, a deep-learning image
  matcher — `PhaseCorrelation` selectable via `ROIGBIV_ROICAT_ALIGNMENT`; RoMa stays the
  aligner regardless of device — the pipeline's `--cpu` flag does *not* switch it, only
  runs it slower — and the compute device is set separately by `ROIGBIV_ROICAT_DEVICE`)
  → **ROI_Blurrer** → **ROInet** SimCLR/ConvNeXt embedding → **Scattering
  Wavelet Transform** → ROI graph → **Clusterer**. ROICaT is an optional dependency
  (`pip install 'roigbiv[embeddings]'`); absent it falls back to legacy phase correlation.
- **Orchestration** (`registry/orchestrator.py::register_or_match`) returns one of
  `hash_match | auto_match | review | new_fov`, using a **calibrated logistic posterior**
  over the embedding similarity with thresholds `ROIGBIV_FOV_ACCEPT_THRESHOLD=0.9` /
  `ROIGBIV_FOV_REVIEW_THRESHOLD=0.5`.
- **Storage:** SQLAlchemy store (SQLite at `inference/registry.db` by default) +
  filesystem blob store (`inference/fingerprints/`); Alembic migrations in
  `registry/migrations/versions/`. Configured entirely via env vars
  (`registry/config.py`).
- **CLI:** `roigbiv-registry {list|show|match|track|backfill|migrate}` (browsing/maintenance
  is CLI-only; the Dash UI does not expose the registry). Design doc:
  `docs/design/roicat-integration.md`.

---

## 11. Optional & aspirational subsystems

Everything below is **off by default**. Each is documented at the same depth as the
default path per the "full coverage" intent, with its toggle and current state called
out explicitly.

### PMD spatiotemporal denoiser — `use_pmd_denoise=False`
A patch-wise penalized-matrix-decomposition denoiser (Buchanan et al. lineage) applied
to the residual feeding Stages 3–4. When on, it materializes a denoised `(T,H,W)` float32
memmap and swaps `fov.residual_view` for a dense-backed `ResidualView` at a single
insertion point; the L+S split, Stage-2 Suite2p reuse, and the `ResidualView` contract
are untouched. Params: `pmd_patch_size=32`, `pmd_patch_overlap=8`, `pmd_max_rank=30`,
`pmd_rank_margin=0.0`, `pmd_band_budget_bytes≈1 GB`. **Phase 2; no default flip.**

### Cellpose-SAM / CP4 sidecar — `stage1_backend="cpsam_sidecar"`
Runs **Cellpose-SAM (cellpose 4.x)** out-of-process in a separate `cp-sam` conda env
(4.x needs numpy 2.x and cannot share this interpreter; the deployed CP3 checkpoint
cannot load under 4.x). Stage-1 inputs/outputs are identical either way, so gates /
subtraction / provenance / the residual engine are untouched. CP-SAM is channel-invariant
and noise-robust, so the sidecar drops denoise and ignores the `channels=(1,2)`
convention. Env python resolved via `cpsam_sidecar_python` / `$ROIGBIV_CPSAM_PYTHON`.
**Phase M; OFF by default.**

### Tonic accept tier — `tonic_accept_tier=False`
When enabled, anatomically-detected (`source_stage ∈ {1,2}`) ROIs that classify as tonic
**and** whose `neuropil_baseline_elevation` ≥ `tonic_accept_min_elevation=0.5` are promoted
`gate_outcome→accept` so they skip human review (the original outcome is recorded in
`gate_reasons`). **Stage-4 tonics are never touched.** Threshold is provisional pending a
held-out elevation sweep. **Phase 5b; OFF pending A/B + explicit approval.**

### Robust IRLS subtraction solver — `subtract_solver="robust"`
A one-sided-Huber IRLS variant of the trace solver (§5 step 2) that down-weights positive
residuals so unmodeled transients aren't absorbed into a neighbour's estimated trace
(`subtract_robust_kappa=0.5`, `subtract_robust_max_iter=5`). **Fully wired**, but the
default is `"ridge"`. (Distinct from RPCA below.)

### `rowwise-pcc` and `legacy`/SIMA motion-correction backends
See §3.1. Opt-in via `--motion-correction {rowwise-pcc|legacy}`; `phasecorr` is default.

### RPCA robust background — **implemented but NOT wired**
`roigbiv/pipeline/rpca.py` implements a genuine robust PCA L+S split
(`robust_lowrank_sparse`: inexact-ALM Principal Component Pursuit with an L1 soft-threshold
prox, plus a GoDec solver). It is gated on `cfg.background_method == "rpca"` — **but
`background_method` does not exist in `PipelineConfig`**, and no runtime module imports the
function outside tests. **The default and only wired background is the truncated-SVD L+S of
§3.2.** Treat RPCA as aspirational/disabled until a config field and call site are added.
(Not to be confused with the robust *trace* solver above, which is wired.)

### Smaller optional paths
- **Auto-diameter** (`diameter_auto=False`) — per-FOV soma-scale estimate via roigbiv's
  own DoG-peak/Otsu sizing (`optics.measure_soma_scale`), not Cellpose's SizeModel.
- **Scout mode** (`scout_mode=False`) — Cellpose-only triage (§3.5).
- **Foundation-only** (`foundation_only=False`) — stop after Foundation for MC inspection;
  resumable into Stage 1.
- **Optics auto-scale** (`auto_scale=True`, `assume_optics=False`) — after Foundation,
  measure the FOV's soma scale and derive the numeric gates (areas, separations, pool
  radii; the full `SCALE_DERIVED_FIELDS` set), overriding profile numbers but never a
  field the user pinned (`explicit_fields`). It runs whenever `auto_scale` is set and the
  profile is one of `grin` / `prism` / `generic` (`optics.py:55`,
  `auto_scale_active`) — i.e. it can adjust the default GRIN gates too; a run-time comment
  claiming GRIN is "gated away / byte-identical" is stale (the guarantee holds only when
  `auto_scale` is off). Re-derives identically on resume (reads the on-disk `mean_M`), so
  these fields are excluded from the resume fingerprint. May pause for
  `needs_optics_confirmation.json` unless `assume_optics=True`.
- **Astrocyte / dual-channel extension** — planning only; see `docs/ASTROCYTE_PLAN.md`.

---

## 12. Orchestration, GPU, resume, output layout

### Single-FOV flow
`run_pipeline(tif_path, cfg, gpu_lock, abort_event)` runs: pre-flight (resolve output
dir, detect motion-correction, resume planning, disk-budget check) → Foundation →
optional optics auto-scale → Stage 1 → Gate 1 → subtract S₁ → Stage 2 → Gate 2 →
subtract S₂ → optional PMD → Stage 3 → Gate 3 → subtract S₃ → Stage 4 → Gate 4 →
post-detection (§9). A per-stage manifest update after every stage backs `--resume`.

> **Conditional subtraction.** Inter-stage subtraction is *skipped* when no downstream
> stage is enabled (`_any_downstream_enabled` in `run.py`). Consequence: `--no-stage-3
> --no-stage-4` means Stage-2 sources are never subtracted. The Stage-1 subtraction always
> runs (only gated by resume).

### GPU concurrency & batch
`roigbiv/pipeline/batch.py` runs ≥ 2 FOVs concurrently, capped at 2 workers with the
`spawn` start method (forking after CUDA init deadlocks). A shared
`multiprocessing.Manager().Lock()` serializes the GPU-bound sections (Cellpose, Suite2p,
Stage-3 FFT, subtraction) while CPU phases overlap. Single-FOV runs pass `gpu_lock=None`
(the `_gpu_section` helper becomes a no-op). GPU: RTX 5080 16 GB; Cellpose inference is
GPU, Suite2p is CPU-only.

### Output layout
Default `inference/pipeline/{stem}/` (`outputs.py::save_pipeline_outputs`). Top-level
trace arrays (rows sorted by `label_id`) are the canonical downstream products:

```
F.npy, Fneu.npy, F_corrected.npy    raw / neuropil / neuropil-corrected traces
dFF.npy, spks.npy                   dF/F and deconvolved spikes
F_bandpass.npy, F_bandpass_index.npy  tonic-ROI bandpass traces + their label IDs
roi_metadata.json                   per-ROI schema: activity_type, gate_outcome,
                                     confidence, provenance scores, review_priority
pipeline_log.json                   full cfg.summary_for_log() + stage/activity/review counts
stage1/ … stage4/                   mask TIFFs + probability/contrast maps + stageN_report.json
summary/                            mean_M, mean_S, max_S, std_S, vcorr_S, dog_map
traces/                             traces.npy (PRIMARY, neuropil-corrected), traces_raw.npy,
                                    traces_neuropil.npy, traces_meta.json   ← pynapse bundle
hitl/                               review_queue.json, merged_masks.tif, per-ROI evidence, GUI staging
registry_match.json                 ROICaT match outcome
svd_factors.npz, residual_S.meta.json   Foundation sidecars
```

Activity-type labels live in each ROI's `roi_metadata.json` entry (`activity_type`) and
are aggregated in `pipeline_log.json` — there is no separate `classified/` directory.

**pynapse export.** `scripts/roigbiv_to_pynapse.py` / `roigbiv/cli_export.py` build the
`traces/` bundle; `traces_meta.json` records row→ID map, `fs`, `frame_averaging`, and
session/FOV IDs, and pynapse's `Sample` builder reconstructs the raw fps as
`fs × frame_averaging`.

---

## 13. Master parameter reference

Every `PipelineConfig` field, grouped by subsystem, sourced from `types.py:158-514`.
For the exhaustive per-threshold rationale with method-level citations, see
`docs/publication/algorithms_v2.md` §19.

### Foundation
| Param | Default | Meaning |
|---|---|---|
| `k_background` | 30 | rank of L background |
| `n_svd` | 200 | singular vectors computed |
| `batch_size` | 500 | **unwired** — not forwarded to Suite2p, which uses its own default of 250 (`suite2p.py:61`) |
| `nonrigid` | True | non-rigid registration |
| `do_registration` | False | overridden True for non-`_mc` inputs |
| `fs` | 30.0 | effective Hz (pass `--fs 7.5` for averaged stacks) |
| `frame_averaging` | 1 | binning factor behind `fs` |
| `tau` | 1.0 | GCaMP6s decay |
| `svd_bin_frames` | 5000 | binned frame target |
| `reconstruct_chunk` | 500 | L+S streaming chunk |

### Motion correction
| Param | Default | Meaning |
|---|---|---|
| `motion_correction_backend` | `"phasecorr"` | `phasecorr`\|`rowwise-pcc`\|`legacy` |
| `mc_max_displacement` | 50 | px clamp (rowwise-pcc + legacy) |
| `mc_strip_height` | 32 | rowwise-pcc strip rows |
| `mc_smooth_sigma_rows` / `_time` | 6.0 / 1.0 | rowwise-pcc field smoothing |
| `mc_prefilter` | False | DoG band-pass on shift inputs |
| `mc_sima_env` / `mc_granularity` | `"sima-legacy"` / `"row"` | legacy SIMA |
| `mc_s2p_block_size` | `[64, 64]` | phasecorr non-rigid block (tuned) |
| `mc_s2p_smooth_sigma` | 1.15 | reference blur |
| `mc_s2p_maxregshift` | 0.1 | rigid shift clamp (frame fraction) |
| `mc_s2p_one_photon_reg` | True | 1P high-pass reg (tuned) |
| `mc_s2p_spatial_hp_reg` / `_taper` | 42 / 40.0 | HP window / edge taper |

### Optics / profile
| Param | Default | Meaning |
|---|---|---|
| `profile` | `"grin"` | lens profile bundle |
| `auto_scale` | True | derive numeric gates from measured soma scale when profile ∈ {grin, prism, generic} |
| `assume_optics` | False | suppress optics-confirmation pause (headless) |

### Stage 1 / Gate 1
| Param | Default | Meaning |
|---|---|---|
| `cellpose_model` | deployed `current_model` | fine-tuned CP3 checkpoint |
| `diameter` / `diameter_auto` | 12 / False | soma diameter (px) |
| `cellprob_threshold` | −2.0 | Cellpose cellprob cutoff |
| `flow_threshold` | 0.4 | Cellpose flow error cutoff |
| `channels` / `tile_norm_blocksize` | (1,2) / 128 | dual-channel input |
| `use_denoise` | True | `denoise_cyto3` on ch1 |
| `stage1_backend` | `"cellpose3"` | vs `cpsam_sidecar` |
| `stage1_ch2_source` | `"vcorr_max_fused"` | ch2 content |
| `min_area` / `max_area` | 80 / 600 | Gate-1 area band (px²) |
| `min_solidity` / `max_eccentricity` | 0.55 / 0.90 | Gate-1 shape |
| `min_contrast` | 0.10 | soma-surround contrast |
| `gate1_merge_peak_min_area` / `_separation` | 4000 / 28 | merge demotion |
| `dog_strong_negative_percentile` | 10.0 | conjunctive DoG rule |
| `annulus_inner_buffer` / `annulus_outer_radius` | 2 / 15 | contrast annulus |

### Subtraction engine
| Param | Default | Meaning |
|---|---|---|
| `subtract_chunk_frames` | 2000 | trace-solve temporal chunk |
| `subtract_ridge_lambda_scale` | 1e-6 | ridge λ scale |
| `subtract_anticorr_threshold` | −0.3 | validation anticorr floor |
| `subtract_anticorr_failure_fraction` | 0.10 | NNLS-fallback trigger |
| `subtract_nnls_fallback_max_rois` | 30 | NNLS-fallback cap |
| `subtract_solver` | `"ridge"` | vs `robust` (IRLS) |
| `subtract_robust_kappa` / `_max_iter` | 0.5 / 5 | robust solver |

### Stage 2 / Gate 2
| Param | Default | Meaning |
|---|---|---|
| `iscell_threshold` | 0.3 | Suite2p classifier cutoff |
| `gate2_iou_threshold` | 0.3 | rediscovery IoU |
| `gate2_max_correlation` | 0.7 | redundant/spillover \|r\| |
| `gate2_anticorr_threshold` | −0.5 | artifact r |
| `gate2_spatial_radius` | 20 | correlation neighborhood (px) |
| `gate2_min_area`/`max_area`/`min_solidity`/`max_eccentricity` | 60/400/0.4/0.85 | relaxed morphology |
| `gate2_near_distance` / `gate2_near_corr_threshold` | 5 / 0.5 | near-duplicate |
| `gate2_flag_corr_threshold` | 0.5 | flag vs accept |

### Stage 3 / Gate 3
| Param | Default | Meaning |
|---|---|---|
| `template_threshold` | 6.0 | event σ (deliberately high) |
| `spatial_pool_radius` | 8 | soma-radius disk (px) |
| `cluster_distance` | 12 | fcluster px |
| `min_event_separation` | 2.0 | temporal-independence (s) |
| `stage3_max_events` | 2,000,000 | hard event cap |
| `stage3_chunk_budget_bytes` | 1 GB | per-chunk working-set cap |
| `stage3_sigma_window_frames` | 500 | sliding-MAD window (unused; global MAD ships) |
| `gate3_min_waveform_r2` / `_single_event` | 0.6 / 0.5 | waveform fit |
| `gate3_max_rise_decay_ratio` | 0.5 | calcium-shape check |
| `gate3_anticorr_threshold` / `gate3_min_solidity` | −0.5 / 0.5 | anticorr / shape |
| `gate3_waveform_window_tau_multiple` | 5.0 | window = 5·τ·fs |

### Stage 4 / Gate 4
| Param | Default | Meaning |
|---|---|---|
| `bandpass_windows` | fast/medium/slow | (0.5–2.0)/(0.1–1.0)/(0.05–0.5) Hz |
| `bandpass_order` | 4 | Butterworth order |
| `n_svd_components_stage4` | 300 | temporal compression dims |
| `corr_neighbor_radius_inner` / `_outer` | 6 / 15 | contrast disk / annulus (px) |
| `corr_contrast_threshold` | 0.10 | clustering cutoff |
| `stage4_min_area`/`max_area`/`min_solidity`/`max_eccentricity` | 80/350/0.6/0.85 | morphology |
| `stage4_iou_merge_threshold` | 0.3 | cross-window merge |
| `stage4_n_workers` | 3 | parallel bandpass windows |
| `gate4_min_corr_contrast` | 0.10 | contrast floor |
| `gate4_max_motion_corr` | 0.3 | motion-artifact reject |
| `gate4_anticorr_threshold` / `gate4_spatial_radius` | −0.5 / 20 | anticorr |
| `gate4_min_mean_intensity_pct` | 25 | `mean_M` intensity percentile floor |

### Traces / classification
| Param | Default | Meaning |
|---|---|---|
| `neuropil_coeff` | 0.7 | neuropil subtraction coefficient |
| `neuropil_inner_buffer` / `_outer_radius` | 2 / 15 | neuropil annulus |
| `baseline_window_s` / `baseline_percentile` | 60.0 / 10 | F0 sliding window |
| `tonic_baseline_window_s` | 120.0 | wider F0 for tonic |
| `phasic_min_transients` / `phasic_min_skew` | 5 / 0.5 | phasic rule |
| `sparse_min_transients` / `sparse_min_skew` | 1 / 0.3 | sparse rule |
| `tonic_bp_std_factor` | 2.0 | tonic bandpass-std multiple |
| `tonic_accept_tier` / `tonic_accept_min_elevation` | False / 0.5 | optional accept tier |

### Optional / batch / output
| Param | Default | Meaning |
|---|---|---|
| `use_pmd_denoise` (+ `pmd_*`) | False | PMD denoiser |
| `batch_n_workers` | 1 | parallel FOV pool (cap 2) |
| `enable_stage_2/3/4` | True | per-stage toggles |
| `force_cpu` | False | disable GPU |
| `output_dir` | None → `inference/pipeline/{stem}/` | output root |
| `no_viewer` | False | skip napari viewer |
| `resume` | False | resume from prior artifacts |

---

## 14. Authoritative-docs map & doc-drift corrections

### Where each source of truth lives
| Intent | Source of truth |
|---|---|
| Architecture decisions (direction, deprecations) | `docs/adr/` — [ADR-0001: non-destructive candidate union](../adr/0001-non-destructive-candidate-union.md), [ADR-0002: cascade deprecation criteria](../adr/0002-cascade-default-deprecation-criteria.md) |
| Pipeline behavior, gate logic, ROI schema | `docs/roi-pipeline-specification.md` (design intent) |
| Algorithm methods, per-threshold citations | `docs/publication/algorithms_v2.md` (as-built, authoritative) |
| Runtime defaults | `roigbiv/pipeline/types.py::PipelineConfig` (**canonical**) |
| Tunable-parameter documentation | `configs/pipeline.yaml` (**documentary only** — not loaded at runtime) |
| Version history | `docs/CHANGELOG.md` |
| Astrocyte extension | `docs/ASTROCYTE_PLAN.md` |
| Registry design | `docs/design/roicat-integration.md` |

### Stale statements this document supersedes
- **README "Overview" (GOLD/SILVER/BRONZE consensus)** — the parallel three-branch
  IoU-consensus architecture is retired; the current pipeline is sequential subtractive
  (§2). The README Overview now points here.
- **`flow_threshold`** — some module docstrings say `0.6`; the real default is **0.4**
  (`types.py:295`).
- **`mean_S` vs `mean_M`** — the spec's `mean_S` inputs are served by **`mean_M`** in code
  (Cellpose ch1, Gate-1 contrast, DoG, Gate-4 intensity), because `mean_S ≈ 0` under
  SVD L+S; the subtraction profile source is **`std_S`** (§2).
- **On-disk residual** — the `residual_S*.dat` memmap chain no longer exists; the residual
  is **virtual** (`ResidualView`), and `FOVData.residual_S_path` stays `None`.
- **`iscell_prob` / `event_count` / `corr_contrast` "future" comments** (`types.py:48-50`)
  — these fields **are** populated by Stages 2–4.
- **`configs/pipeline.yaml`** — where it disagrees with `types.py` (e.g. `flow_threshold`,
  `use_denoise`, `consensus:` tiers), the YAML is stale/documentary and the code wins.
