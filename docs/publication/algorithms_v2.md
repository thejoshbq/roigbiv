# ROIGBIV — Algorithmic methods (v2, current implementation)

> **Scope.** This is a publication-grade reference for every algorithm in the
> ROIGBIV two-photon calcium-imaging pipeline as **currently implemented on the
> `main` branch**. It is intended to be cited in manuscripts and to accompany the
> code as a methods-section companion. Every numeric threshold is sourced from the
> implementation; any code reference is given as `file:line`.
>
> **Relationship to v1.** This document supersedes
> [`algorithms.md`](algorithms.md) (referred to here as *v1*). v1 remains the
> historical record of an earlier revision; it predates several architectural
> changes that landed afterwards — the **virtual residual** (no dense
> `residual_S*.dat`), the **multi-backend motion correction** with tuned
> phase-correlation defaults, deterministic SVD seeding, and a family of
> **optional / experimental** subsystems (robust background, PMD denoise, robust
> subtraction solver, tonic-accept tier). Wherever v2 and v1 disagree on a default
> or an output artefact, **v2 is authoritative for current code.** Each change is
> explained inline at the point it arises.
>
> **Canonical source of defaults.** `roigbiv/pipeline/types.py` — the dataclass
> `PipelineConfig` (`types.py:158-482`). The legacy YAML at `configs/pipeline.yaml`
> describes the superseded parallel-consensus architecture and is *not*
> authoritative for the sequential pipeline documented here.
>
> **Optional-feature convention.** Subsystems that are OFF in the default
> configuration are tagged **`Optional — OFF by default`** with the controlling
> flag. The default path (the spine of this document) is everything *not* so
> tagged.

---

## Table of contents

1. [Overview and notation](#1-overview-and-notation)
2. [Foundation: motion correction, background decomposition, summary images](#2-foundation)
3. [Stage 1 — Cellpose spatial detection](#3-stage-1--cellpose-spatial-detection)
4. [Gate 1 — morphological validation](#4-gate-1--morphological-validation)
5. [Source subtraction engine](#5-source-subtraction-engine)
6. [Stage 2 — Suite2p temporal detection](#6-stage-2--suite2p-temporal-detection)
7. [Gate 2 — temporal cross-validation](#7-gate-2--temporal-cross-validation)
8. [Optional — PMD spatiotemporal residual denoiser](#8-optional--pmd-spatiotemporal-residual-denoiser)
9. [Stage 3 — template sweep on residual](#9-stage-3--template-sweep-on-residual)
10. [Gate 3 — waveform validation](#10-gate-3--waveform-validation)
11. [Stage 4 — tonic-neuron search](#11-stage-4--tonic-neuron-search)
12. [Gate 4 — correlation-contrast validation](#12-gate-4--correlation-contrast-validation)
13. [Quality-control features, trace extraction, classification](#13-quality-control-features-trace-extraction-classification)
14. [Human-in-the-loop review package](#14-human-in-the-loop-review-package)
15. [Pipeline orchestration, resume, GPU management, output layout](#15-pipeline-orchestration-resume-gpu-management-output-layout)
16. [Acquisition / optics profiles](#16-acquisition--optics-profiles)
17. [Evaluation: residual-retention diagnostic](#17-evaluation-residual-retention-diagnostic)
18. [Cross-session FOV and cell registry](#18-cross-session-fov-and-cell-registry)
19. [Parameter reference (master table)](#19-parameter-reference-master-table)
20. [Bibliography](#20-bibliography)

---

## 1. Overview and notation

### 1.1 Architecture

ROIGBIV is a sequential subtractive pipeline for detecting regions of interest
(ROIs) in two-photon calcium-imaging movies. Detection proceeds through four
stages. After each detection stage, a gate accepts, flags, or rejects each
candidate; accepted and flagged candidates are then subtracted from the residual
before the next stage operates. This removes the need for any single detector to
discriminate among all neuron types and provides per-ROI provenance (which stage
discovered the ROI, under which gate outcome, at which confidence).

```
input TIF  ─►  Foundation  (motion correction → SVD L+S → summary images → DoG)
                   │           residual S is VIRTUAL: reconstructed on demand
                   ▼                       (ResidualView, no dense .dat)
               Stage 1  (Cellpose on {mean_M, vcorr⊕max_S})  ─►  Gate 1 (morphology)
                   │                                               │
                   ▼                                               ▼
               +SourceLayer₁ (lazy)                        accept | flag ⟶ subtract
                   │
                   ▼
               Stage 2  (Suite2p reuse, IoU vs Stage 1)    ─►  Gate 2 (temporal)
                   │                                               │
                   ▼                                               ▼
               +SourceLayer₂ (lazy)                        accept | flag ⟶ subtract
                   │
                   ├───[Optional: PMD denoise of the residual feeding S3/S4]
                   ▼
               Stage 3  (FFT matched filter, event cluster) ─►  Gate 3 (waveform)
                   │                                               │
                   ▼                                               ▼
               +SourceLayer₃ (lazy)                        accept | flag ⟶ subtract
                   │
                   ▼
               Stage 4  (bandpass correlation contrast)     ─►  Gate 4 (contrast)
                   │
                   ▼
               QC features → dF/F → OASIS → classification → [tonic-accept] → HITL export
```

The residual chain $S_0 \to S_1 \to S_2 \to S_3$ is **not materialised on disk**.
Each subtraction appends a small in-RAM *source layer* to a lazily-reconstructing
`ResidualView` (§5.3, §2.3). This is the single largest architectural change since
v1, which described dense `residual_S*.dat` memory-maps throughout.

### 1.2 Design principles

- **Recall-first, precision through review.** Every gate is tuned so that false
  negatives are rarer than false positives; precision is obtained by the
  human-in-the-loop review package rather than by aggressive automated rejection.
- **Sequential and subtractive.** Each stage operates on the residual of previous
  stages, so no stage wastes effort rediscovering prior detections.
- **Provenance-tracked.** Every ROI carries `source_stage`, `gate_outcome ∈ {accept,
  flag, reject}`, `confidence ∈ {high, moderate, low, requires_review}`, per-stage
  scores, and a list of human-readable gate-failure reasons
  (`roigbiv/pipeline/types.py:26-82`).
- **HITL-closed.** Stage 4 has no automated accept tier; its survivors enter a
  prioritised review queue (§14).
- **Reconstruct, don't store.** Anything derivable from `data.bin` + the SVD
  factors + the per-stage source layers is reconstructed on demand rather than
  written out, bounding peak disk to a single registered movie (§2.3, §15).
- **Opt-in for the uncertain.** Experimental detectors and substitutions
  (robust background, robust subtraction, PMD, tonic-accept) default OFF and flip
  only after a gate-aware A/B plus explicit approval.

### 1.3 Notation

| symbol | meaning |
|:---|:---|
| $T$ | number of frames |
| $H \times W$ | spatial dimensions (also $L_y \times L_x$) |
| $N_\text{pix} = H \cdot W$ | total pixel count |
| $M \in \mathbb{R}^{T \times H \times W}$ | registered movie (`data.bin`, int16) |
| $L$, $S$ | low-rank (background) and sparse (foreground) components, $M = L + S$ |
| $S_k$ | residual after $k$ stages of subtraction ($S_0 = S$), reconstructed lazily |
| $f_s$ | acquisition frame rate (Hz) |
| $\tau$ | indicator decay constant (s); 1.0 for GCaMP6s |
| $\sigma_p$ | per-pixel noise scale (MAD-based) |
| $K$ | number of templates in the Stage 3 matched-filter bank |
| $W$ (in §5) | ROI spatial-profile design matrix, not to be confused with image width |

All per-pixel accumulators run in `float64` to avoid catastrophic cancellation;
final outputs are `float32`.

---

## 2. Foundation

Implementation: `roigbiv/pipeline/foundation.py`.

The foundation stage produces (a) a rigidly- and non-rigidly-registered movie
$M \in \mathbb{R}^{T \times H \times W}$, stored as an `int16` `data.bin`
memory-map; (b) a low-rank-plus-sparse decomposition $M = L + S$ where $S$ is a
**virtual residual** reconstructed on demand (no dense file); (c) summary images
(mean_M, mean_S, max_S, std_S, vcorr_S, mean_L); and (d) a
difference-of-Gaussians (DoG) nuclear-shadow map. Entry point: `run_foundation`
(`foundation.py:708`).

### 2.1 Motion correction — three backends

Motion correction is dispatched by `run_motion_correction` (`foundation.py:52`)
on `cfg.motion_correction_backend` (`types.py:210`), with three options. The
default reflects a tuning campaign that v1 predates.

**(a) `phasecorr` (default).** Suite2p [Pachitariu et al. 2017] rigid registration
by FFT subpixel phase correlation to a reference image, followed by piecewise
non-rigid block registration. The registration ops dict is assembled at
`foundation.py:160-205`. The default knobs were **retuned** relative to Suite2p
stock and relative to v1:

| parameter | v2 default | `types.py` | v1 value | purpose |
|:---|:---|:---|:---|:---|
| `mc_s2p_block_size` | **`[64, 64]`** | 246 | `128` | non-rigid block size (px) |
| `mc_s2p_one_photon_reg` | **`True`** → ops `"1Preg"` | 257 | (absent) | 1-photon-style spatial high-pass before registration |
| `mc_s2p_smooth_sigma` | 1.15 px | 247 | 1.15 | reference-image Gaussian smoothing |
| `mc_s2p_maxregshift` | 0.1 | 249 | 0.1 | rigid shift clamp (fraction of frame) |
| `mc_s2p_maxregshift_nr` | 5 px | 251 | — | max non-rigid block shift |
| `mc_s2p_nimg_init` | 300 | 252 | 300 | frames used to build the reference image |
| `mc_s2p_spatial_hp_reg` | 42 px | 258 | — | spatial high-pass window for 1Preg |

**Why these defaults (`types.py:238-245`).** Full-session validation on the Logan
Prism FOV (a 2271-frame registered mean vs a grid-aligned legacy SIMA mean)
showed the old `[128,128]` / no-1Preg default reached only **58 %** of legacy
cell-sharpness, whereas `[64,64]` + 1Preg reach **~103 %** (at or above legacy)
with no over-fit banding. `[64,64]` alone gets 91 %; the 1-photon high-pass
(`1Preg`) supplies the remainder and is **load-bearing** on dim, low-contrast
GRIN/Prism frames. Because `1Preg` is a 1-photon high-pass, bright high-SNR 2P
data should pass `--no-mc-1preg` (and/or `--mc-block-size 128 128`).

When the input filename ends in `_mc.tif` the movie is assumed pre-registered and
Suite2p registration is disabled (`do_registration=False`, `types.py:170`);
Suite2p still runs its detection pass so `stat.npy`/`iscell.npy` are available for
Stage 2 reuse (§6).

**(b) `rowwise-pcc`** — **`Optional — OFF by default`** (`motion_correction_backend="rowwise-pcc"`).
GPU row-wise non-rigid phase correlation (`foundation.py:101`;
`roigbiv/pipeline/registration.py`). The movie is corrected in horizontal strips
(`mc_strip_height=32` rows, `types.py:212`); strip-displacement *regularization*
— taller strips for per-strip SNR, median + confidence outlier rejection
(`mc_strip_confidence_weight=True`, `types.py:220`), and spatial/temporal
smoothing of the displacement field (`mc_smooth_sigma_rows=6.0`,
`mc_smooth_sigma_time=1.0`, `types.py:218-219`) — suppresses the noise-driven
per-row warps (~30× less spurious warp on a still frame) that otherwise regress
dim/low-SNR FOVs. An optional DoG band-pass on the shift-estimation inputs
(`mc_prefilter=False`, `types.py:224-226`) helps only when structured background
dominates and degrades white-noise-limited frames, hence OFF. This backend
remains opt-in pending parity validation against `phasecorr` on the real stack;
Suite2p then runs detection-only.

**(c) `legacy`** — **`Optional — OFF by default`** (`motion_correction_backend="legacy"`).
A genuine SIMA `HiddenMarkov2D(granularity='row')` correction
[Kaifosh et al. 2014] run in the `sima-legacy` Python-3.8 sidecar conda env via
subprocess (`foundation.py:126`; `roigbiv/pipeline/legacy_mc.py`). CPU-only and
slow (tens of minutes to hours per FOV); a faithful reproduction of the legacy
notebook's correction, for exact legacy repro. Suite2p runs detection-only
afterwards.

### 2.2 Temporally binned truncated SVD (with deterministic seeding)

The decomposition operates on a temporally binned copy of the movie to bound
compute. Given target $T_\text{bin} \approx 5000$ frames (`svd_bin_frames=5000`,
`types.py:174`), the bin width is $b = \lceil T / T_\text{bin} \rceil$ and the
binned movie averages each block (`_compute_binned_movie`, `foundation.py:228`).

A rank-$n_\text{svd}$ truncated SVD (`n_svd=200`, `types.py:167`) is computed on
$\tilde M^\top$ via `torch.svd_lowrank(A, q, niter=2)` — two power iterations of
randomised subspace SVD [Halko et al. 2011] — at `_binned_svd_gpu`
(`foundation.py:256`). The transpose orientation is deliberate: factoring
$\tilde M^\top \in \mathbb{R}^{N_\text{pix} \times T_\text{bin}}$ makes the
returned $U$ index pixels directly, convenient for spatial reconstruction. On a
`torch.cuda.OutOfMemoryError` the computation falls back transparently to CPU.

**Deterministic seeding (new in v2; `foundation.py:276-293`).** `torch.svd_lowrank`
is a *randomised* algorithm. Without a fixed seed the top-$k$ subspace drifts
run-to-run (observed mean principal-angle cosine ≈ 0.65 on real movies), and
because that subspace defines $L$, the drift propagates into $S \to$ `vcorr_S`
$\to$ the Cellpose channel-2 input $\to$ borderline detections. The SVD therefore
seeds `torch.manual_seed(0)` (and `cuda.manual_seed_all(0)`) before factoring,
and re-seeds on the CPU-fallback path, making detection reproducible.

The temporal components are nearest-neighbour upsampled from $T_\text{bin}$ to $T$
(`_upsample_V`, `foundation.py:306`) — acceptable for the background subspace
because binning already preserves its dominant low-frequency structure.

### 2.3 Low-rank / sparse decomposition — virtual residual

Denote the top $k = k_\text{background}$ SVD factors (default $k=30$,
`types.py:166`) as $U_k$, $\Sigma_k$, $V_k$ (upsampled). The background and
residual at frame $t$ are

$$
L_t = U_k \Sigma_k V_k^{(t)\top}, \qquad S_t = M_t - L_t.
$$

This is a *truncated-SVD L+S decomposition*: not the iterative principal-component
pursuit of Candès et al. [2011] (no nuclear-norm or $\ell_1$ objective), but a
direct rank-$k$ projection that separates slowly-varying photobleach / neuropil /
illumination drift ($L$) from the sparse cellular signal ($S$). The choice $k=30$
is validated empirically against summary-image contrast and the residual-retention
diagnostic of §17.

**The residual is virtual (new in v2).** Implementation:
`roigbiv/pipeline/residual.py`. Materialising every link of the chain
$S \to S_1 \to S_2 \to S_3$ as a dense `(T,H,W)` float32 memmap costs ~10–19 GB
*each* and peaks at 40–60 GB across the chain — the source of a silent `SIGBUS`
crash when the disk filled mid-write (`residual.py:1-25`). Nothing actually needs
the dense array: every link is reconstructible on demand from artefacts that
already exist. `compute_background_separation` (`foundation.py:327`) therefore
builds a `ResidualView.from_factors` (`residual.py:117`, called at
`foundation.py:400`) that holds only the SVD factors (a few MB) plus a zero-cost
int16 memmap of `data.bin`, and reconstructs any temporal chunk, spatial band, or
pixel set on demand via three read primitives:

- `read_chunk(t0,t1)` → `(cs, Ly, Lx)` — temporal slab (`residual.py:216`),
- `read_rows(y0,y1)` → `(T, h, Lx)` — full-$T$ spatial band (`residual.py:231`),
- `read_pixels(ys,xs)` → `(T, P)` — arbitrary pixel timecourses (`residual.py:256`).

Each primitive computes $M - L$ (the int16 movie minus the rank-$k$ background)
and then applies each accumulated `SourceLayer` in order, so the reconstructed
value matches the previously-written `.dat` within float32 tolerance
(`residual.py:22-24`). No `residual_S.dat` is written; a JSON sentinel
`residual_S.meta.json` with `kind:"virtual"` records the contract. **`mean_L` is
computed in closed form** as $U_k \Sigma_k \cdot \overline{V_k}$
(`foundation.py:398`), avoiding a second full-movie pass. SVD factors
($U$, $\Sigma$, $V_\text{bin}$, `bin_size`) persist to `svd_factors.npz` so the
view can be rebuilt on resume (`ResidualView.from_foundation`, `residual.py:146`).

#### 2.3.1 Optional — robust low-rank/sparse background (RPCA)

**`Optional — present but not wired into the default pipeline.`** Implementation:
`roigbiv/pipeline/rpca.py`. Plain top-$k$ SVD aligns its leading components with
per-pixel mean brightness, so the static structural image *and* the brightest /
tonic somata get absorbed into $L$, leaving `mean_S ≈ 0` and erasing the easiest
cells from the detection substrate (`rpca.py:1-14`). The RPCA module replaces the
SVD step with a *robust* decomposition $M_\text{bin} \approx L_\text{bin} +
S_\text{bin}$ in which $L_\text{bin}$ is genuinely low-rank background and
$S_\text{bin}$ is the sparse foreground a plain SVD would pull into $L$. Two
solvers are available: inexact-ALM Principal Component Pursuit (IALM-PCP)
[Lin, Chen & Ma 2010] as primary, and GoDec [Zhou & Tao 2011] as a lighter
fallback (`robust_lowrank_sparse`, `rpca.py:109`). To preserve the virtual-residual
contract it takes the **exact** SVD of the robust $L_\text{bin}$ and emits
$(U, \Sigma, V_\text{bin})$ in the *identical* pixel-indexed convention as
`_binned_svd_gpu`, so `residual.py` and the `svd_factors.npz` format are untouched
— only *how* $L$ is estimated changes (`rpca.py:16-23`). Because IALM holds ~6
live full-size copies on the GPU, `estimate_rpca_bin_frames` (`rpca.py:80`) caps
the temporal-bin target against currently-free VRAM and the foundation retries
coarser, then on CPU.

> **Status.** The module is fully implemented and self-consistent but is **not
> referenced by `foundation.py` or `run.py`**, and the current `PipelineConfig`
> exposes **no** `background_method` flag (the docstring's `cfg.background_method
> == "rpca"` selector is aspirational). RPCA is therefore available for
> programmatic/experimental use but is not reachable from the default or
> CLI-driven pipeline as of this revision.

### 2.4 Streaming summary images

Summary images of $S$ are computed in a single temporal pass over the virtual
residual (`_accumulate_summaries`, `foundation.py:430`; driven by
`generate_summary_images`, `foundation.py:540`) with `float64` accumulators and a
chunked iterator (`_iter_S_chunks`, `foundation.py:417`).

**Mean, maximum, standard deviation.** Ordinary running moments:

$$
\mu(p) = \tfrac{1}{T}\sum_t S_t(p), \quad
\max(p) = \max_t S_t(p), \quad
\sigma(p) = \sqrt{\max\!\bigl(0, \tfrac{1}{T}\sum_t S_t(p)^2 - \mu(p)^2\bigr)}.
$$

**8-neighbour local correlation (vcorr).** For each offset
$(\delta_y,\delta_x) \in \{-1,0,1\}^2 \setminus \{(0,0)\}$ the pass maintains
$\sum x,\sum y,\sum x^2,\sum y^2,\sum xy$ and a count per offset and forms the
per-offset Pearson correlation from second moments:

$$
r_{\delta_y,\delta_x}(p) = \frac{n\sum xy - \sum x\sum y}
{\sqrt{(n\sum x^2 - (\sum x)^2)(n\sum y^2 - (\sum y)^2) + \varepsilon}},
$$

averaging over the offsets whose neighbour lies inside the FOV
(`foundation.py:430-537`). The second-moment formulation is algebraically
equivalent to the mean-subtracted Pearson estimator but needs one pass. A
`scout_vcorr_neighbors` knob (`types.py:183`) selects the 8-stencil (full) or
4-stencil (von Neumann) variant for scout mode.

**Raw morphological mean (mean_M).** The mean of the *registered* movie (not the
residual) is read from Suite2p's `meanImg` or reconstructed from `data.bin`. Under
a top-$k$ SVD L+S the first components absorb per-pixel brightness, so
`mean_S ≈ 0` and is unsuitable as a morphological channel; `mean_M` preserves the
raw anatomical contrast that Cellpose's training regime expects. This substitution
recurs at Stage 1 (§3), subtraction profiles (§5.1), and Gate 4 (§12).

**Scout / foundation-only modes (new in v2).** Two fast paths short-circuit the
foundation:

- **Scout mode** (`scout_mode=True`, `types.py:177-183`) skips SVD/L+S/residual
  entirely and computes the Cellpose channel-2 correlation map directly on the
  registered movie (`vcorr_on_movie`, `foundation.py:558`; `_run_foundation_scout`,
  `foundation.py:632`), stopping after Stage 1 + Gate 1. It is a fast FOV-clarity /
  model-A-B triage — *not* analysis-grade (no traces, QC, registry; not resumable).
- **Foundation-only** (`foundation_only=True`, `types.py:185-190`) runs the full
  foundation (motion correction + SVD/L+S + summaries) then stops before Stage 1,
  writing a `foundation_only.json` sentinel so the motion-corrected FOV can be
  inspected before committing to detection. A later `--resume` run continues from
  Stage 1.

### 2.5 Difference-of-Gaussians nuclear-shadow map

GCaMP is excluded from the neuronal nucleus, so healthy somata typically show a
darker central region against a brighter cytoplasmic annulus. A DoG applied to
`mean_M` quantifies this (`compute_nuclear_shadow_map`, `foundation.py:605`):

$$
\mathrm{DoG}(x,y) = G_{\sigma_\text{outer}} * M_\mu(x,y) - G_{\sigma_\text{inner}} * M_\mu(x,y),
\qquad \sigma_\text{inner}=2,\ \sigma_\text{outer}=6.
$$

The polarity is chosen so a pixel at the dark nuclear centre gives a *positive*
response: the narrow Gaussian picks up the dark nucleus (low value), the wide
Gaussian averages over soma+surround (higher value), so the difference is positive
at the nucleus. Gate 1 uses the 10th percentile of this map as the
"strongly-negative" threshold for rejection. All summary images are written to
`summary/*.tif` as `float32`.

---

## 3. Stage 1 — Cellpose spatial detection

Implementation: `roigbiv/pipeline/stage1.py`.

Stage 1 segments morphologically clear somata via Cellpose 3 [Stringer et al.
2021; Pachitariu & Stringer 2022] on a dual-channel stack. Channel 1 is always the
raw morphological mean `mean_M`; channel 2 carries a temporal-activity cue. The
pair is complementary: the mean projection misses dim/tonic cells; the activity
channel misses bright-but-silent cells.

### 3.1 Channel-2 content (changed default in v2)

`stage1_ch2_source` (`types.py:291-305`) selects the second channel's content
(`_resolve_stage1_ch2`, `stage1.py`):

| value | channel-2 content |
|:---|:---|
| `vcorr_S` | pixel-correlation map (legacy / v1 behavior) |
| `max_S` | residual peak-intensity (single-firer / sparse cue) |
| **`vcorr_max_fused`** (default) | per-image min-max-normalised $\max(\text{vcorr\_S}, \text{max\_S})$ |

**Why fused (`types.py:303-304`).** The default flipped `vcorr_S → vcorr_max_fused`
after a Phase-4 A/B: the union of "is temporally correlated" OR "has a bright peak"
recovered cells that vcorr alone missed, for **recall +0.017, 0/13 FOV
regressions, FP +2.4 %** (see `docs/phase4_channel_ab_report.md`). CP3's deployed
checkpoint is architecturally 2-channel (`conv1 in_channels=2`), so enrichment
happens by swapping channel-2 *content*, not by adding a third channel. Gate 1
always uses `vcorr_S` regardless, so this changes the Stage-1 *detector input*
only. When `max_S` is unavailable (e.g. scout-mode foundation) it falls back to
`vcorr_S` with a warning.

### 3.2 Preprocessing, diameter, and backend

Channel 1 (`mean_M`) is optionally passed through Cellpose 3's `denoise_cyto3`
image-restoration model [Pachitariu & Stringer 2022] (`use_denoise=True`,
`types.py:273`). Inference uses `CellposeModel` loaded from the deployed checkpoint
(`models/deployed/current_model`, `types.py:263`) — the path is anchored to the
package root, not cwd, so runs from any directory load the fine-tuned model rather
than silently falling back to stock cyto3 (`types.py:17-23`) — or the built-in
`cyto3` if the checkpoint fails to load.

**Table 3.1 — Cellpose parameters (`types.py:262-273`).**

| parameter | v2 default | `types.py` | note |
|:---|:---|:---|:---|
| `diameter` | 12 px | 264 | expected soma diameter under GRIN optics |
| `diameter_auto` | `False` | 268 | when `True`, a calibration pass with `diameter=None` on downsampled `mean_M` uses Cellpose's `SizeModel` estimate, overriding `diameter` on success |
| `cellprob_threshold` | $-2.0$ | 269 | permissive cell-probability cut for recall |
| `flow_threshold` | **0.4** | 270 | flow-field error threshold |
| `channels` | $(1,2)$ | 271 | 1-indexed Cellpose channel roles |
| `tile_norm_blocksize` | 128 px | 272 | tile-normalisation block (counters GRIN vignetting) |

> **`flow_threshold` nuance.** The `PipelineConfig` dataclass default is **0.4**
> (`types.py:270`), but the **CLI** backfills **0.6** when `--flow-threshold` is
> omitted (`profiles.py:49-54`, `STAGE1_CLI_DEFAULTS`) to preserve byte-identical
> historical GRIN CLI behavior. v1 documented 0.6; both values are "correct"
> depending on entry point. Programmatic `PipelineConfig()` callers get 0.4; CLI
> users who omit the flag get 0.6.

**Auto-diameter** (`diameter_auto`, new in v2) guards against the one-size-fits-all
failure mode when soma scale is unknown — it is the default in the `generic` optics
profile (§16). **CP-SAM sidecar backend** (`stage1_backend`, `types.py:275-289`)
is **`Optional — OFF by default`**: setting `stage1_backend="cpsam_sidecar"` runs
Cellpose-SAM (Cellpose 4.x) *out-of-process* in the `cp-sam` conda env, because
4.x needs numpy 2.x and cannot share this interpreter, and the deployed CP3
checkpoint cannot load under 4.x. Stage-1 inputs/outputs are identical either way,
so gates, subtraction, provenance, and the residual engine are untouched; the
sidecar is channel-invariant and noise-robust, so it drops denoise and ignores the
`channels` role convention.

### 3.3 Output

Cellpose returns a uint16 label image, XY flow maps, and the cell-probability map
$\Pi \in \mathbb{R}^{H \times W}$. For each non-zero label $\ell$ the binary mask
is $M_\ell = \{p: L(p)=\ell\}$ and the per-ROI probability is
$\Pi_\ell = |M_\ell|^{-1}\sum_{p\in M_\ell}\Pi(p)$.

---

## 4. Gate 1 — morphological validation

Implementation: `roigbiv/pipeline/gate1.py`. (Logic unchanged from v1; thresholds
re-verified against `types.py`.)

Gate 1 converts raw Cellpose candidates into `ROI` objects with an
`accept | flag | reject` outcome based on five features.

### 4.1 Features

**Area, solidity, eccentricity** from `skimage.measure.regionprops`. Area is the
pixel count, solidity is $A/A_\text{convex hull}$, eccentricity is that of the
equivalent ellipse.

**Soma–surround contrast.** Construct an annulus around the mask excluding other
ROI pixels, with `annulus_inner_buffer=2 px`, `annulus_outer_radius=15 px`
(`types.py:322-323`). Exclusion prevents neighbour-soma contamination of the
annular background. Contrast is

$$
c_i = \frac{\mu_S(\text{mask}_i) - \mu_S(\text{ring}_i)}{\max(|\mu_S(\text{ring}_i)|, 10^{-6})},
$$

with the sign of the denominator preserved when the ring mean is near zero.

**Nuclear shadow score.** $n_i = \text{mean}_{p\in\text{mask}_i}\mathrm{DoG}(p)$;
the mean over the full mask (not the centroid) is robust to labelling jitter.

### 4.2 Decision logic

Define a "strongly negative DoG" threshold as the `dog_strong_negative_percentile`
(default 10th, `types.py:319`) of the DoG map. The decision:

- **Reject** if the *DoG conjunction rule* triggers — strongly-negative DoG AND
  contrast $\le$ `min_contrast` — OR two or more criteria other than contrast fail.
- **Accept** (`confidence=high`) if no criterion fails.
- **Flag** (`confidence=moderate`) if exactly one criterion fails within its
  per-criterion absolute margin (Table 4.2).
- Otherwise **reject** (`confidence=requires_review`).

The DoG conjunction rule captures "likely astrocyte or out-of-focus ghost" while
treating DoG as advisory: a dim cell with negative DoG but healthy contrast is not
penalised. Marginal flagging preserves borderline cells for review, consistent
with the recall-first principle.

**Table 4.1 — Gate 1 thresholds (`types.py:308-319`).**

| threshold | default | action if breached |
|:---|:---|:---|
| `min_area` | 80 px | reject (unless marginal single failure) |
| `max_area` | 600 px | reject (unless marginal single failure) |
| `min_solidity` | 0.55 | reject (unless marginal single failure) |
| `max_eccentricity` | 0.90 | reject (unless marginal single failure) |
| `min_contrast` | 0.10 | reject; also triggers DoG conjunction check |
| `dog_strong_negative_percentile` | 10.0 | DoG rejection if contrast also fails |

**Table 4.2 — Per-criterion flag margins (`types.py:314-317`).**

| criterion | flag margin |
|:---|:---|
| area | ±20 px |
| solidity | ±0.05 |
| eccentricity | ±0.03 |
| contrast | ±0.03 |

---

## 5. Source subtraction engine

Implementation: `roigbiv/pipeline/subtraction.py`.

Between detection stages, source subtraction removes the fluorescence contribution
of accepted+flagged ROIs from the residual so the next stage operates on a cleaner
substrate. In v2 the subtraction is **lazy**: it appends a `SourceLayer` to the
`ResidualView` (§2.3) rather than writing a new dense memmap.

### 5.1 Spatial profile estimation

Each ROI is assigned a normalised spatial profile $w_i(p)$ supported on
$\text{mask}_i$, peaking at 1.0 inside the mask, where the *profile source* field
$\psi$ is **`std_S`** (per-pixel temporal standard deviation), not the spec's
$\mu_t[S]$ (`estimate_spatial_profiles`, `subtraction.py:43-94`). Under
truncated-SVD L+S the top-$k$ components absorb per-pixel mean brightness so
$\mu_t[S] \approx 0$ with no spatial structure; `std_S` faithfully preserves the
spatial pattern of residual activity (active pixels have higher variance than
neuropil).

### 5.2 Simultaneous trace estimation (ridge solve — default)

At each frame the per-ROI activity $c(t)$ is estimated by ridge-regularised least
squares over the union $P = \bigcup_i \text{mask}_i$ of supports:

$$
\hat c(t) = \arg\min_c \lVert S(P,t) - Wc\rVert_2^2 + \lambda\lVert c\rVert_2^2
= (W^\top W + \lambda I)^{-1} W^\top S(P,t),
$$

with $W \in \mathbb{R}^{|P|\times N}$ stacking the union-restricted profiles.
$W^\top W + \lambda I$ is precomputed once and one linear system is solved per
temporal chunk on the GPU (`subtraction.py`, ridge path), with

$$
\lambda = \rho\cdot\frac{\operatorname{tr}(W^\top W)}{N},
\qquad \rho = \text{`subtract\_ridge\_lambda\_scale`} = 10^{-6}\ (\texttt{types.py:353}).
$$

Scaling $\lambda$ by $\operatorname{tr}(W^\top W)$ keeps regularisation
proportional to the data scale. The GPU path uses `torch.linalg.solve`; CUDA OOM
falls back to CPU. Temporal chunking (`subtract_chunk_frames=2000`,
`types.py:352`) streams the residual through RAM. The solver is selected by
`subtract_solver` (`types.py:357`); the default is `"ridge"`.

#### 5.2.1 Optional — robust Huber-IRLS solver

**`Optional — OFF by default`** (`subtract_solver="robust"`, `types.py:357`).
A one-sided Huber M-estimator solved by iteratively-reweighted least squares
down-weights *positive* residuals (pixels where the linear model under-predicts —
i.e. a brighter source is bleeding through) while leaving negative residuals at
full weight:

$$
p(r) = \begin{cases} 1 & r \le 0 \\ \kappa_\text{abs}/\max(\kappa, r) & r > 0 \end{cases}
$$

The noise scale $\sigma$ that sets $\kappa_\text{abs} = \kappa\sigma$ is estimated
**from negative residuals only**, because the contaminating artefact (a bad ROI's
ridge trace, or ghost leakage) always pushes residuals *positive*; the negative-eps
pool is uncontaminated and reflects the true noise floor. Controlled by
`subtract_robust_kappa=0.5` (sigma units, `types.py:358`) and
`subtract_robust_max_iter=5` (`types.py:359`). Experimental, pending an A/B against
the ridge default.

### 5.3 Rank-1 lazy subtraction

For every pixel $p \in P$ and frame $t$, $S_\text{out}(p,t) = S_\text{in}(p,t) -
\sum_i w_i(p)\hat c_i(t)$. Instead of writing this out, the engine calls
`view.with_source(flat_idx, W_design, traces, stage_idx)` (`residual.py:176`),
which returns a **new `ResidualView` sharing the movie/SVD arrays by reference**
and appending one small `SourceLayer` (union pixel indices, per-ROI weight design,
per-ROI traces; `residual.py:40-54`). The subtraction is then applied on every
subsequent read by the view's read primitives (`residual.py:225-228, 248-253,
267-272`). The arithmetic is identical to the former materialising
`subtract_sources`, so results match the old `.dat` within float32 tolerance — but
peak disk stays at one registered movie regardless of stage count. Each layer is
persisted (`SourceLayer.save`, `residual.py:56`) for resume.

### 5.4 Post-subtraction validation

Three per-ROI ratios are tested on the residual *after* subtraction (one streaming
pass with `float64` moments; Pearson from second moments):

| check | definition | pass range |
|:---|:---|:---|
| mean ratio | $|\mu(S_\text{out}[\text{mask}])| / (|\mu(S_\text{out}[\text{ring}])| + 10^{-6})$ | $< 3$ |
| std ratio | $\sigma(S_\text{out}[\text{mask}]) / \sigma(S_\text{out}[\text{ring}])$ | $(0.3, 3)$ |
| anti-correlation | Pearson$\bigl(\mu(S_\text{out}[\text{mask}])_t, \hat c_i(t)\bigr)$ | $> \rho_\text{anti}$ |

with $\rho_\text{anti} = \text{`subtract\_anticorr\_threshold`} = -0.3$
(`types.py:354`). All three must hold for `pass=True`. A strong mean ratio $\gg 1$
indicates a bright residual spot left behind; std ratio outside $(0.3,3)$ indicates
over/under-subtraction; strong anti-correlation indicates the estimated trace
cancelled into the noise rather than extracting a true source.

### 5.5 Single-variable NNLS fallback

If the anti-correlation failure fraction exceeds
`subtract_anticorr_failure_fraction = 0.10` (`types.py:355`), up to
`subtract_nnls_fallback_max_rois = 30` (`types.py:356`) flagged ROIs are
re-estimated with a non-negativity constraint. Because each profile is localised,
this reduces to single-variable non-negative least squares on the local support,

$$
\hat c_i(t) = \max\!\Bigl(0,\ \frac{w_i^\top S_\text{in}(\text{mask}_i, t)}{w_i^\top w_i}\Bigr),
$$

closed-form. Refined traces are substituted, subtraction re-run, and only the
refined ROIs re-validated; unflagged entries from the first pass are retained.

---

## 6. Stage 2 — Suite2p temporal detection

Implementation: `roigbiv/pipeline/stage2.py`. (Behavior as v1; reads the virtual
$S_1$.)

Stage 2 recovers neurons whose morphology is insufficient for Cellpose but whose
temporal activity drives Suite2p's SVD-based detector [Pachitariu et al. 2017] —
burst-firers, task-locked neurons, and cells occluded in the mean projection by a
brighter neighbour.

### 6.1 Reuse of foundation Suite2p outputs

Rather than re-run Suite2p, Stage 2 reads the `stat.npy` / `iscell.npy` already
produced by the foundation step (`_load_suite2p_outputs`, `stage2.py:88-111`),
converts each `stat` entry to a dense binary mask, and drops entries with
`iscell[i,1] < iscell_threshold = 0.3` (`types.py:363`).

### 6.2 IoU novelty filter

Against the union of Stage 1 accept+flag masks, $\mathrm{IoU}(A,B) = |A\cap
B|/|A\cup B|$ is computed and only candidates whose maximum IoU does not exceed
`gate2_iou_threshold = 0.3` (`types.py:366`) are retained. 0.3 is within the
0.3–0.5 literature range for consensus ROI matching [Giovannucci et al. 2019] and
is conservative for Suite2p's irregular footprints against Cellpose contours.

### 6.3 Trace extraction on the residual

Each retained candidate's trace is extracted from the virtual $S_1$ (not the raw
movie) via `view.read_pixels` over the mask support
(`extract_traces_from_residual`, `stage2.py:38-81`):
$\text{trace}_i(t) = |\text{mask}_i|^{-1}\sum_{p\in\text{mask}_i}S_1(p,t)$.
Resulting ROIs carry `source_stage=2`, `iscell_prob`, `trace`, and provisional
`gate_outcome="accept"` pending Gate 2.

---

## 7. Gate 2 — temporal cross-validation

Implementation: `roigbiv/pipeline/gate2.py`. (Logic as v1; thresholds re-verified.)

Gate 2 verifies that Stage 2 candidates are genuinely independent sources rather
than rediscoveries the IoU filter missed, spillover from an imperfectly subtracted
Stage 1 neighbour, or subtraction artefacts.

### 7.1 Features

Area and morphology via `regionprops`; centroid as the mean of mask coordinates;
Pearson correlations against each Stage 1 ROI trace whose centroid lies within
`gate2_spatial_radius = 20 px` (`types.py:369`), computed in a single vectorised
pass over the row-wise mean-centred Stage 1 trace matrix (`_pearson_row`).

### 7.2 Decision logic

Let $r_i$ be candidate-to-nearby-Stage-1 correlations. Failures:

| failure | condition |
|:---|:---|
| morphology | area $\notin [60, 400]$ or solidity $< 0.4$ or eccentricity $> 0.85$ |
| redundancy / spillover | $\max|r_i| \ge$ `gate2_max_correlation = 0.7` |
| subtraction artefact | $\min r_i \le$ `gate2_anticorr_threshold = -0.5` |
| near-duplicate | any Stage 1 within `gate2_near_distance = 5 px` with $|r| >$ `gate2_near_corr_threshold = 0.5` |

Decision: **reject** if any failure; **flag** (`confidence=moderate`) if all pass
but $\max|r_i| >$ `gate2_flag_corr_threshold = 0.5`; **accept** (`confidence=high`)
otherwise. Relaxed thresholds vs Gate 1 (`gate2_min_area=60`, `gate2_max_area=400`,
`gate2_min_solidity=0.4`; `types.py:370-373`) acknowledge that Suite2p footprints
are noisier than Cellpose contours and lack a supplementary morphological signal.

---

## 8. Optional — PMD spatiotemporal residual denoiser

**`Optional — OFF by default`** (`use_pmd_denoise`, `types.py:431`). Implementation:
`roigbiv/pipeline/pmd.py`. This is the one optional substitution that is *wired
into the run path*: `run.py:783` applies it after Stage 2 and before Stage 3 when
the flag is set (and the resume planner says Stage 3 should run).

### 8.1 Motivation

SVD truncation in the foundation denoises the *background* but leaves shot noise in
the residual; Stage 3 (per-pixel MAD) and Stage 4 (z-scored bandpassed residual)
must work against that noise. PMD [Buchanan et al. 2018 lineage] lifts residual SNR
spatiotemporally — the gain faint sparse/tonic detections need most
(`pmd.py:11-15`).

### 8.2 Algorithm

Per overlapping spatial patch, the `(T, P)` pixel-time matrix is mean-centred and
decomposed by a truncated SVD; only components above the Marchenko–Pastur noise
edge are kept (`_pmd_denoise_patch`, `pmd.py:45-80`):

- Robust per-patch noise std from lag-1 temporal differences,
  $\sigma = \sqrt{\operatorname{median}((X_c[1:]-X_c[:-1])^2)/2}$, valid when the
  signal is temporally smooth.
- Economy SVD $X_c \approx U\,\Sigma\,V^\top$ via `torch.svd_lowrank(q, niter=2)`.
- MP edge $= \sigma(\sqrt T + \sqrt P)(1 + \text{margin})$ with
  `pmd_rank_margin=0.0` (`types.py:435`); components with $\Sigma > $ edge are
  retained (capped at `pmd_max_rank=30`, `types.py:434`), the rest discarded.

Horizontal bands of `pmd_patch_size=32` rows (`types.py:432`) advance by
`patch_size − pmd_patch_overlap` (`pmd_patch_overlap=8`, `types.py:433`) and are
blended by overlap-add to suppress block seams (`pmd_denoise_to_memmap`,
`pmd.py:87`), with a `pmd_band_budget_bytes ≈ 1 GB` soft RAM cap (`types.py:436`).
torch on GPU with CPU fallback.

### 8.3 Insertion contract

PMD reads the input through `ResidualView` primitives only and never mutates the
L+S factors or the reconstruction math. Because it is a *global* patch
decomposition it cannot be applied coherently per on-demand read, so it
materialises the denoised `(T,H,W)` float32 memmap **once** and wraps it as the
*dense base* of a fresh `ResidualView` (`pmd_denoise_view`, imported at
`run.py:329`, called at `run.py:788`). The Stage-3 subtraction carries that dense
base forward via `with_source(dense=self._dense)`, so Stage 4 inherits the denoised
residual automatically. Stage 1/2 are never affected.

### 8.4 Current status

An A/B on the fused default found PMD **recall-neutral** with a false-positive cost,
so it ships OFF (commits `956512f`, `39afbfa`; `docs/phase2_pmd_insertion_point.md`).
It is documented here because it is fully implemented and one flag away.

---

## 9. Stage 3 — template sweep on residual

Implementation: `roigbiv/pipeline/stage3.py`, `stage3_templates.py`. (As v1;
re-verified.)

Stage 3 targets sparsely-firing neurons whose transient count is too low for
Suite2p's SVD but which produce identifiable calcium-transient waveforms in the
post-Stage-2 residual $S_2$ (or its PMD-denoised replacement, §8).

### 9.1 Dual-exponential template bank

Each template is a double-exponential transient
$w(t;\tau_r,\tau_d) = (1 - e^{-t/\tau_r})e^{-t/\tau_d}$, $t\in[0,5\tau_d]$, sampled
at $f_s$ and $L_2$-normalised to unit energy so scores compare across templates
(`stage3_templates.py:39-50`). Three templates per indicator family parameterise
single, doublet, and burst kinetics, selected by a decay-constant threshold of
0.75 s: GCaMP6s [Chen et al. 2013] if $\tau\ge 0.75$ s, else jGCaMP8f [Zhang et al.
2023].

**Table 9.1 — Template bank (`stage3_templates.py:18-30`).**

| indicator | shape | $\tau_\text{rise}$ (s) | $\tau_\text{decay}$ (s) |
|:---|:---|:---|:---|
| GCaMP6s | single | 0.05 | 1.0 |
| GCaMP6s | doublet | 0.075 | 1.2 |
| GCaMP6s | burst | 0.10 | 1.5 |
| jGCaMP8f | single | 0.04 | 0.5 |
| jGCaMP8f | doublet | 0.06 | 0.6 |
| jGCaMP8f | burst | 0.08 | 0.75 |

### 9.2 FFT-based cross-correlation

$S_2$ is scanned in spatial row-chunks of `stage3_pixel_chunk_rows = 8`
(`types.py:389`; auto-scaled to `stage3_chunk_budget_bytes ≈ 1 GB`,
`types.py:390`). For each chunk the per-pixel noise scale is
$\sigma_p = \max(\text{MAD}(x_p)/0.6745, 10^{-6})$, each pixel trace is FFT'd, and
for each template $k$ the normalised cross-correlation
$\xi_k(p,t) = \mathcal F^{-1}[\mathcal F\{x_p\}\cdot\overline{\mathcal F\{w_k\}}](t)/\sigma_p$
is accumulated into a running `score_max` / `template_idx_max`, so no
$(N_\text{pix},K,T)$ array is materialised.

**Spec deviation (global MAD).** The implementation uses a *global* per-pixel MAD
rather than the spec's sliding-window local MAD (`stage3_sigma_window_frames=500`,
`types.py:391`, retained as config but not the active path) to save ~300 MB of
intermediate storage per chunk. Per-pixel global MAD already normalises away the
dominant scale variation (brightness heterogeneity); residual temporal
non-stationarity is mitigated by the subsequent template match, which penalises
non-transient waveforms.

### 9.3 Thresholding and event extraction

A pixel-time pair is an *event* when $\text{score\_max}(p,t) > \theta$,
$\theta = $ `template_threshold = 6.0` (`types.py:384`). **Why 6σ
(`types.py:379-383`):** in real residual data the per-pixel noise has a heavier
right tail than pure Gaussian (structured neuropil/background leakage); at 4σ,
150M+ false crossings were observed on a single FOV, while 6σ brings counts into
the $10^3$–$10^5$ range where clustering is tractable. To bound memory, $\theta$ is
adaptively raised by 1.0σ per iteration (up to 8) if a chunk emits more than
$2\times10^5$ events, and a global cap `stage3_max_events = 2\times10^6`
(`types.py:392`) retains the top-$K$ by score.

### 9.4 Spatial clustering

Events are clustered in 2-D by single-linkage hierarchical clustering with distance
threshold `cluster_distance = 12 px` (`types.py:387`). Above $2\times10^4$ events
the $O(n^2)$-memory `pdist` becomes prohibitive, so it switches to a grid-snap
approximation (cell size = the distance threshold). This under-merges chained cells
at adjacent grid boundaries but is acceptable because real somata produce many
events per cell and the switch-over is reached only in pathological cases.

### 9.5 Temporal-independence filter

For each cluster, events are greedily selected in descending score order, retaining
one only if no previously-selected event is within `min_event_separation = 2.0 s`
(`types.py:388`), i.e. $\lceil 2 f_s\rceil$ frames. The retained count is
`event_count`; clusters with zero independent events are discarded.

### 9.6 Candidate packaging

Each surviving cluster becomes a candidate ROI: a filled disk of radius
`spatial_pool_radius = 8 px` (`types.py:385`) centred on the cluster mean; the
trace is extracted from $S_2$ via the same residual extractor as Stage 2;
provisional `gate_outcome="accept"` pending Gate 3.

---

## 10. Gate 3 — waveform validation

Implementation: `roigbiv/pipeline/gate3.py`. (As v1; thresholds re-verified.)

### 10.1 Waveform extraction

For each event frame $t_e$ an asymmetric window
$\mathcal W(t_e) = \text{trace}[t_e - L/4,\ t_e + 3L/4)$ with
$L = \lceil 5\tau f_s\rceil$ (`gate3_waveform_window_tau_multiple = 5.0`,
`types.py:400`) is extracted and a baseline (mean of the first 10 %) subtracted to
remove slow drift.

### 10.2 Template $R^2$ fit

For each template $w_k$ the peak is aligned by index shift and an amplitude fit by
least squares ($\hat A_k = w_k^\top\mathcal W / w_k^\top w_k$, positive only), with

$$
R^2_k = 1 - \frac{\lVert\mathcal W - \hat A_k w_k\rVert_2^2}{\lVert\mathcal W - \bar{\mathcal W}\rVert_2^2}.
$$

The best $R^2$ across all events drives the decision:

| criterion | threshold |
|:---|:---|
| single-event candidate | $R^2_{k^\star} \ge$ `gate3_min_waveform_r2_single_event = 0.5` (`types.py:396`) |
| multi-event candidate | $R^2_{k^\star} \ge$ `gate3_min_waveform_r2 = 0.6` (`types.py:395`) |
| marginal flag band | $[\min_{r2}, \min_{r2}+0.1)$ → flag rather than reject |

### 10.3 Rise/decay asymmetry

$\rho = (t_{90}-t_{10})/(t_{37}-t_\text{peak})$; reject if $\rho \ge$
`gate3_max_rise_decay_ratio = 0.5` (`types.py:397`). Slow-rise/fast-decay patterns
indicate noise, astrocyte-slow events, or motion artefacts.

### 10.4 Anti-correlation cascade defence

For prior-stage ROIs with centroid within `gate2_spatial_radius = 20 px`, Pearson
correlations are computed against their traces and the candidate is rejected if the
minimum is $\le$ `gate3_anticorr_threshold = -0.5` (`types.py:398`) — a defence
against traces left anti-correlated by imperfect upstream subtraction.

### 10.5 Morphology and confidence

Disk-mask solidity must meet `gate3_min_solidity = 0.5` (`types.py:399`).
Confidence is graded by event count: 1 → low, 2–5 → moderate, ≥6 → high. Any
failure rejects; passing within the 0.1-marginal $R^2$ band flags; otherwise
accept.

---

## 11. Stage 4 — tonic-neuron search

Implementation: `roigbiv/pipeline/stage4.py`. (As v1; re-verified.)

Tonic neurons fire quasi-continuously and pile up into a nearly-constant
fluorescence level under $\tau\approx 1$ s kinetics: low temporal variance
(Suite2p misses them), no discrete transients (Stage 3 misses them), partially
absorbed into $L$. Stage 4 detects them via local spatial-temporal correlation
contrast on $S_3$.

### 11.1 Per-pixel linear detrend

A vectorised OLS detrend removes residual drift / photobleaching per pixel
(`detrend_to_memmap`, `stage4.py:56-91`), $\tilde S_3(p,t) = S_3(p,t) - (\alpha_p +
\beta_p t)$, written once and reused across bandpass windows. Spatial chunking
(`stage4_pixel_chunk_rows = 16`, `types.py:418`) bounds RAM independently of $T$.

### 11.2 Zero-phase Butterworth bandpass at three windows

Three windows isolate different tonic-firing rates (`bandpass_windows`,
`types.py:403-407`):

| window | passband (Hz) | targets |
|:---|:---|:---|
| fast | 0.5–2.0 | 3–5 Hz firing |
| medium | 0.1–1.0 | 1–3 Hz firing |
| slow | 0.05–0.5 | < 1 Hz firing / slow modulation |

Each is an order-4 zero-phase Butterworth (`bandpass_order = 4`, `types.py:408`) as
an SOS cascade via `scipy.signal.sosfiltfilt` (`bandpass_to_memmap`,
`stage4.py:98-132`). Zero-phase filtering preserves temporal alignment for the
correlation step, so chunking is spatial, not temporal. A stability check skips any
window whose lower cutoff requires a recording length $> T/f_s$ ($5/f_\text{low}$).

### 11.3 Temporal compression

The filtered movie is compressed to $(H\cdot W, D)$ via binned temporal averaging
with $D = \min(\text{`n\_svd\_components\_stage4` = 300}, T)$ (`types.py:409`;
`compress_temporal`, `stage4.py:139-170`). Shared bin edges make this an
orthogonal projection that preserves pairwise correlations exactly, reducing the
correlation step from $O(T)$ to $O(D)$ per pair.

### 11.4 Correlation-contrast map via spatial convolution

For per-pixel $z$-scored vectors $z_p \in \mathbb{R}^D$ and neighbourhood
$\mathcal N(p)$, the mean Pearson correlation over $\mathcal N$ is
$\overline r(p) = \tfrac1D z_p^\top(\tfrac1{|\mathcal N|}\sum_q z_q)$, where the
inner sum is a spatial convolution with a uniform disk kernel. With $K_\text{in}$ a
self-excluded disk of `corr_neighbor_radius_inner = 6` and $K_\text{ann}$ the
annulus to `corr_neighbor_radius_outer = 15` (`types.py:410-411`),

$$
C(p) = \overline r_\text{in}(p) - \overline r_\text{ann}(p)
$$

(`compute_correlation_contrast`, `stage4.py:197-265`). Somata exhibit high inner
and low annular correlation; neuropil correlates broadly at both radii. The
convolution form reduces an $O(N_\text{pix}^2)$ all-pairs correlation to
$O(D\cdot N_\text{pix}\cdot|\text{kernel}|)$ per radius.

### 11.5 Thresholding, morphology, cross-window merge

The contrast map is thresholded at `corr_contrast_threshold = 0.10`
(`types.py:412`) and labelled by connected components. Components are kept if
$a\in[80,350]$, $s\ge 0.6$, $e\le 0.85$ (`types.py:413-416`). Candidates from the
three windows are pooled and merged greedily in descending contrast by IoU with
`stage4_iou_merge_threshold = 0.3` (`types.py:417`), each winner recording the set
of windows it was detected in. Windows run serially or on a thread pool of up to
`stage4_n_workers = 3` (`types.py:419`); `sosfiltfilt` and the convolution release
the GIL so parallelism is real, with BLAS threads capped to avoid oversubscription.

Candidate ROIs receive `source_stage=4`, `confidence="requires_review"` (locked by
design), and provisional `gate_outcome="flag"` pending Gate 4.

---

## 12. Gate 4 — correlation-contrast validation

Implementation: `roigbiv/pipeline/gate4.py`. (As v1; re-verified.)

Gate 4 has **no accept tier**: every candidate passing all six checks receives
`gate_outcome="flag"`, `confidence="requires_review"`; any failure rejects. This is
a deliberate epistemic-humility stance — the automated pipeline cannot confirm
tonic candidates with the confidence of Stages 1–3, so human review of the bandpass
trace plus correlation map is mandatory.

The six checks:

1. **Correlation contrast.** $C \ge$ `gate4_min_corr_contrast = 0.10` (`types.py:442`).
2. **Eccentricity.** $e \le$ `stage4_max_eccentricity = 0.85` (`types.py:416`).
3. **Solidity.** $s \ge$ `stage4_min_solidity = 0.60` (`types.py:415`).
4. **Motion correlation.** $\max(|r_x|,|r_y|) <$ `gate4_max_motion_corr = 0.3`
   (`types.py:443`), Pearson of the **raw** trace against the rigid displacement
   fields. Sub-pixel motion leaves fluctuating ring artefacts at soma boundaries
   that mimic tonic signals; the raw trace is used because motion power spreads
   across frequencies.
5. **Cascade anti-correlation.** For prior-stage ROIs within `gate4_spatial_radius
   = 20 px` (`types.py:446`), the minimum Pearson correlation must exceed
   `gate4_anticorr_threshold = -0.5` (`types.py:444`).
6. **Intensity floor on mean_M.** $\mu_M(\text{mask}) \ge$
   percentile(mean_M, `gate4_min_mean_intensity_pct = 25`) (`types.py:445`). The
   floor uses **mean_M**, not `mean_S` (≈0 under SVD L+S), so a percentile filter is
   meaningful.

---

## 13. Quality-control features, trace extraction, classification

Implementation: `roigbiv/pipeline/qc_features.py`, `classify.py`,
`gate_tonic_elevation.py`, plus the neuropil and dF/F utilities invoked by `run.py`.

### 13.1 Spatial QC features

For every non-rejected ROI (`compute_spatial_features`): **boundary gradient**
(mean $\lVert\nabla\text{mean\_S}\rVert$ over the 1-pixel mask boundary — sharp
somata score high); **spatial blur (radial FWHM)** (out-of-focus ghosts are
broader); **FOV distance** (contextualises GRIN-edge aberration failures).

### 13.2 Temporal QC features

Each ROI gets a raw trace $F_i$ and neuropil-corrected $F^c_i = F_i - \alpha
F^{np}_i$ with `neuropil_coeff = 0.7` (`types.py:326`) and a neuropil annulus of
inner buffer 2 px / outer radius 15 px (`types.py:327-328`). Features
(`compute_temporal_features`): `std`, `skew`, `mean_fluorescence`, `noise_floor`
(MAD/0.6745), `snr`, `n_transients` (FFT matched-filter peak count, 3σ height, min
distance $\lceil 2\tau f_s\rceil$), `trace_bandpass` (zero-phase order-4
Butterworth 0.05–2 Hz), `bp_std`, `bp_power_ratio` (Welch PSD in band / total), and
`autocorr_tau`. The bandpass trace is the **primary evidence** for tonic review:
tonic transients pile up as slow fluctuations that the 0.05–2 Hz band isolates,
whereas the raw trace looks nearly flat.

### 13.3 dF/F and OASIS deconvolution

$\Delta F/F_0$ uses a sliding-window baseline of `baseline_window_s = 60 s`
(`tonic_baseline_window_s = 120 s` for tonic-classified ROIs; `types.py:329-331`)
and `baseline_percentile = 10`. Spike deconvolution uses OASIS
[Friedrich et al. 2017] configured for GCaMP6s kinetics ($\tau=1.0$ s).

### 13.4 Provenance feature

`n_stages_detected` is computed post hoc as the count of distinct `source_stage`
values held by ROIs overlapping the target with IoU > 0.3.

### 13.5 Activity-type classification

A rule-based decision tree evaluated top to bottom (`classify_activity_type`),
using per-FOV medians $\tilde F$, $\tilde\sigma$ of `mean_fluorescence` and `std`
for the tonic population criterion:

| class | condition (first match wins) |
|:---|:---|
| phasic | `n_transients` ≥ 5 AND `skew` > 0.5 |
| sparse | 1 ≤ `n_transients` < 5 AND `skew` > 0.3 |
| tonic | `bp_std` > 2.0 × max(`noise_floor`, $10^{-12}$) AND `skew` ≤ 0.5 AND (`source_stage`=4 OR (`mean_F` > $\tilde F$ AND `std` < $\tilde\sigma$)) |
| silent | `n_transients` = 0 AND `bp_std` < `noise_floor` AND (nuclear_shadow_score > 0 OR solidity > 0.7) |
| ambiguous | fallback |

Thresholds: `phasic_min_transients=5`, `phasic_min_skew=0.5`,
`sparse_min_transients=1`, `sparse_min_skew=0.3`, `tonic_bp_std_factor=2.0`
(`types.py:334-338`). The silent tier is retained only when morphology is
convincing (positive nuclear shadow or solid mask), keeping cells that may fire in
another session while rejecting flat traces at fragmented low-contrast locations.

### 13.6 Optional — tonic accept tier

**`Optional — OFF by default`** (`tonic_accept_tier`, `types.py:340-349`).
Implementation: `gate_tonic_elevation.py`. A narrow, auditable promotion that runs
*after* classification: an **anatomically-detected** tonic soma skips human review
when its baseline sits convincingly above the surrounding neuropil
(`apply_tonic_accept_tier`, `gate_tonic_elevation.py:45`):

$$
\text{promote to accept iff}\quad
\begin{cases}
\texttt{tonic\_accept\_tier} = \text{True} \\
\texttt{activity\_type} = \text{tonic} \\
\texttt{source\_stage} \in \{1, 2\} & \text{(anatomical detectors only)} \\
\texttt{gate\_outcome} \neq \text{reject} \\
\texttt{neuropil\_baseline\_elevation} \ge \texttt{tonic\_accept\_min\_elevation}\ (0.5)
\end{cases}
$$

**Invariants (`gate_tonic_elevation.py:14-23`).** Stage-4 tonics
(`source_stage==4`) are **never** touched — Gate 4's `requires_review` contract is
load-bearing. No mask in `merged_masks.tif` changes (those carry every non-rejected
ROI regardless of outcome); the only effect is review-queue membership. The
promotion is **strictly additive provenance**: the original outcome/confidence is
recorded in `gate_reasons`, so it is reversible/auditable. OFF by default; flipping
requires a gate-aware A/B plus explicit approval.

---

## 14. Human-in-the-loop review package

Implementation: `roigbiv/pipeline/hitl.py`. (As v1.)

ROIGBIV exports a prioritised queue and per-ROI evidence files that drop into the
Cellpose GUI for manual correction.

### 14.1 Four-tier priority queue

Assembled from all non-rejected ROIs (`build_review_queue`):

| priority | criterion | sort key |
|:---|:---|:---|
| 1 | `source_stage=4` AND `confidence=requires_review` | `corr_contrast` ascending |
| 2 | any stage, `confidence=moderate` | `source_stage` descending |
| 3 | `source_stage=3` AND (`event_count=1` OR `confidence=low`) | `label_id` ascending |
| 4 | remaining non-rejected | `label_id` ascending |

When the tonic-accept tier (§13.6) is enabled, promoted anatomical tonics drop out
of priority 1/2 by virtue of becoming `accept`/`high`.

### 14.2 Exported artefacts

Per-FOV (`export_hitl_package`): `review_queue.json` (priority list with reasoning
strings); `merged_masks.tif` (uint16 label image of every non-rejected ROI, label
IDs aligned 1:1 with downstream trace rows); `hitl/stage4/{label_id}/`
(`bandpass_trace.npy` — primary tonic evidence; `corr_contrast_crop.npy` — 61×61
crop; `info.json`); `hitl/stage3/{label_id}/event_frame_indices.json` (frames ±10
around each event); and `hitl_staging/{images,masks}/` in Cellpose-GUI-ready layout
for training-data correction.

---

## 15. Pipeline orchestration, resume, GPU management, output layout

Implementation: `roigbiv/pipeline/run.py`, `outputs.py`, `batch.py`,
`resume.py`, `gpuguard.py`, `workspace.py`.

### 15.1 Control flow

`run_pipeline` (`run.py`) threads one `FOVData` container through the stages.
GPU-heavy sections (Cellpose inference, Suite2p detection, Stage 3 FFT, the
subtraction ridge/NNLS solve) are wrapped in a `_gpu_section` context that acquires
a shared `multiprocessing.Manager().Lock()` under batch execution and is a
zero-cost no-op for single-FOV runs. With the virtual residual there is no
inter-stage `.dat` to delete; each subtraction only grows the in-RAM source-layer
list, and the optional PMD step (§8) is the sole place a dense residual is
materialised (`run.py:783-789`).

### 15.2 Resume

`resume=True` (`types.py:452-457`; `resume.py`) consults `output_dir` for prior
artefacts and skips completed stages, refusing to resume if the config or input
fingerprint differs from what wrote them. The per-stage opt-in flags
(`enable_stage_2/3/4`, `types.py:467-469`) are **excluded** from the resume
fingerprint, so flipping a stage flag on a prior workspace runs only the
now-enabled stage(s). The virtual residual is rebuilt from `svd_factors.npz` +
saved `SourceLayer` blobs (`ResidualView.from_foundation`, `residual.py:146`).

### 15.3 Cascade-monotonicity check

After all stages complete, the summary verifies that detected counts decrease
across stages — a soft sanity check that no stage is re-discovering prior
detections (artifact propagation). Violations are recorded as warnings in
`pipeline_log.json` rather than raised.

### 15.4 GPU management

**`gpuguard`** (`roigbiv/pipeline/gpuguard.py`, new in v2). On this lab's
single-GPU box (**RTX 5080, 16 GB**) the local-Qwen MCP server shares the card and
keeps a large model (≈18 GB) resident for minutes after each call; while loaded the
pipeline gets almost no free VRAM and Cellpose / Foundation SVD silently fall back
to CPU. `free_gpu_for_run` (`gpuguard.py`) is a best-effort preflight: if free VRAM
is below a threshold and `ollama` holds it, it asks `ollama` to unload
(`keep_alive=0`) and waits briefly for reclaim. It **never raises** — any failure
(no CUDA, `ollama` unreachable, memory held elsewhere) degrades to "leave things as
they are," and the existing per-stage CPU fallback still applies. The unload is
non-destructive (`ollama` reloads on its next request).

### 15.5 Batch runner and GPU lock

`batch.py` runs ≥2 FOVs concurrently via `ProcessPoolExecutor` with the `spawn`
start method (`batch_n_workers`, `types.py:439`; default 1 = sequential, hard-capped
at 2). `spawn` is mandatory because forking after CUDA-context init deadlocks. A
`Manager().Lock()` passed to every worker serialises GPU phases; CPU-only phases
(foundation summaries, Stage 4 bandpass+convolution, trace extraction, QC, dF/F,
OASIS) overlap freely. The cap of 2 saturates the 16 GB card — more workers cannot
reduce wall-time because the GPU lock serialises GPU phases.

### 15.6 Output layout

Per-FOV output directory (default `inference/pipeline/{stem}/`):

```
suite2p/plane0/{ops.npy, data.bin, stat.npy, iscell.npy, ...}
svd_factors.npz                        (U, S, V_bin, bin_size)
residual_S.meta.json                   (sentinel: kind="virtual" — NO dense .dat)
source_layers/stage{1,2,3}.npz         (per-stage SourceLayer: flat_idx, W_design, traces)
subtraction_report_stage{1,2,3}.json   (post-subtraction validation report)
motion_trace.npz                       (xoff, yoff, fs)
summary/{mean_M, mean_S, max_S, std_S, vcorr_S, mean_L, dog_map}.tif

stage1/{stage1_masks.tif, stage1_probs.tif, stage1_report.json}
stage2/{stage2_masks.tif, stage2_report.json}
stage3/{stage3_masks.tif, stage3_events.npy, stage3_report.json}
stage4/{stage4_masks.tif, corr_contrast_{fast,medium,slow}.tif,
        stage4_corr_contrast.npy, stage4_report.json}

hitl/ , hitl_staging/                  (see §14)

F.npy / Fneu.npy / F_corrected.npy     (N × T)
dFF.npy / spks.npy                     (N × T)
F_bandpass.npy / F_bandpass_index.npy  (tonic ROIs only + label→row map)
merged_masks.tif                       (uint16 final label image)
roi_metadata.json                      (per-ROI full metadata)
pipeline_log.json                      (execution summary)
review_queue.json
```

> **Changed vs v1.** v1 listed dense `residual_S.dat` and
> `residual_S1/2/3.dat`. Those no longer exist: the residual is virtual
> (`residual_S.meta.json` sentinel + `source_layers/*.npz`). Exact subdirectory
> names for the source-layer blobs are an implementation detail of
> `outputs.py`/`resume.py`; the contract is "SVD factors + per-stage source layers
> are sufficient to reconstruct any $S_k$."

Each `stage{1..4}_report.json` has `{detected, accepted, flagged, rejected, rois:
[ROI.to_serializable(), ...]}` where `to_serializable` (`types.py:63-82`) emits
`label_id`, `source_stage`, `confidence`, `gate_outcome`, spatial features,
per-stage scores, `activity_type`, `gate_reasons`, and a stage-specific `features`
dict. `pipeline_log.json` (written by `save_pipeline_outputs`, `outputs.py`)
captures input/output paths, shape, `k_background`, a `summary_for_log()` config
snapshot (`types.py:472-482`), per-stage counts, subtraction pass/fail tallies,
activity-type counts, overlap groups, review-queue summary, per-stage timings
(including `pmd_denoise_s` when PMD runs), and warnings.

---

## 16. Acquisition / optics profiles

Implementation: `roigbiv/pipeline/profiles.py` (new in v2).

The `PipelineConfig` defaults are tuned for 512² **GRIN** imaging (bright, round,
~12 px somata). Dim, diffuse **PRISM** FOVs (~56 px) need a different Stage-1
configuration. Rather than have a user hand-set ~8 flags, a *profile* bundles the
corrections behind one selector. It is a **Python dict registry** (not a YAML
loader): flat dicts keyed by `PipelineConfig` field names that splat directly and
cannot drift from the dataclass (`profiles.py:8-10`). Merge precedence enforced by
the callers (`merged_overrides`, `profiles.py:122`):

```
PipelineConfig defaults  <  profile bundle  <  explicit user flags
```

| profile | overrides | rationale |
|:---|:---|:---|
| `grin` (default) | none (= dataclass defaults) | the working 512² baseline; selecting it is a no-op |
| `prism` | `channels=(0,0)`, `cellpose_model="cyto3"`, `use_denoise=False`, `diameter=56`, `min_area=1500`, `max_area=5000`, looser solidity/eccentricity, `tile_norm_blocksize=256`, `flow_threshold=0.4`, `cellprob_threshold=0.0`, `mc_strip_height=48` | **single-channel** Stage-1 input is the dominant PRISM fix: feeding `vcorr_S` as Cellpose's nucleus channel suppresses segmentation on PRISM's diffuse correlation map; `channels=(0,0)` on `mean_M` takes cyto3 from 0→13 detections, and dropping denoise + the generalist model add the rest (→16). The GRIN-fine-tuned deployed model is the *worst* on PRISM. Area bounds from `scripts/measure_prism_scale.py` (median diameter≈56). |
| `generic` | `channels=(0,0)`, `cellpose_model="cyto3"`, `use_denoise=False`, `diameter_auto=True` | conservative "unknown optics" fallback: single-channel generalist with per-FOV diameter estimate; deliberately no experimental gates — the least-certain path must not also be the most experimental. |

`auto` must be resolved to a concrete profile upstream (in the CLI/UI, where
explicit-vs-default flags are still distinguishable) before `get_profile`
(`profiles.py:96-114`). Several PRISM constants (`min_solidity`,
`max_eccentricity`, `tile_norm_blocksize`, `flow_threshold`, `cellprob_threshold`)
are marked PENDING A/B against PRISM ground truth and should be confirmed before
being treated as load-bearing (`profiles.py:28-32`). CP-SAM is deliberately **not**
referenced here (CP4, incompatible with this repo's `cellpose<4.0.0` pin).

---

## 17. Evaluation: residual-retention diagnostic

Implementation: `roigbiv/eval/retention.py` (new in v2).

Stage 1 detects on `mean_M`, but subtraction and Stages 2–4 consume the residual
$S = M - L$. If the top-$k$ background $L$ absorbs a soma's brightness (the
*k-too-high* regime), that cell vanishes from $S$ and downstream stages can never
recover it — even when Stage 1 detected it. This module quantifies that absorption
per ground-truth soma, replacing the spec's eyeball "visible in mean(S), absent
from mean(L)" check with a number.

Using $M = L + S$ (hence $\text{mean\_M} = \text{mean\_L} + \text{mean\_S}$), the
retained-brightness fraction is

$$
r_S(\text{mask}) = \frac{\sum_\text{mask}\text{mean\_S}}{\sum_\text{mask}\text{mean\_M}}
$$

(`mask_retention`, `retention.py:30`). $r_S\to 1$ ⇒ the soma survives into the
residual (good); $r_S\to 0$ ⇒ absorbed into background (bad). Values can fall
slightly outside $[0,1]$ because `mean_S` is signed. `retention_summary`
(`retention.py:55`) aggregates median/mean/min $r_S$ and the fraction of somata
clearing $\tau_\text{retain}=0.5$ — the quantity that gates the background-calibration
phase and that the $k$-sweep maximises. `count_vcorr_maxima` (`retention.py:81`)
counts localized hotspots in the `vcorr_S` map (via `skimage.peak_local_max`, with
a threshold-count fallback) to provide an automated $k$-plateau selector: count
peaks vs $k$ and take the plateau where extra rank stops surfacing new hotspots.

---

## 18. Cross-session FOV and cell registry

Implementation: `roigbiv/registry/`. (Unchanged from v1; reproduced for
completeness. Relocatable to supplementary methods if the manuscript focuses
narrowly on detection.)

The registry identifies whether a newly-pipelined FOV is a re-recording of a
previously-seen FOV and, if so, which cells correspond. Decisions are one of
`hash_match`, `auto_match`, `review`, or `new_fov`, persisted to
`registry_match.json`.

### 18.1 Fingerprinting

Each session's fingerprint (`FINGERPRINT_VERSION = 3`, `fingerprint.py:34`) is a
deterministic SHA-256 over a canonical tuple of the merged mask — `[H,W]` plus the
label-sorted `[label_id, y, x, area]` integer rows (`compute_fingerprint`,
`fingerprint.py:56-115`). The mean projection is stored as context but is not
hashed. Identical fingerprints give an $O(1)$ shortcut to a full re-run match.

### 18.2 ROICaT alignment, embedding, clustering

Matching between fingerprint-distinct sessions uses ROICaT [Landry et al. 2024]
(`roicat_adapter.py`): build a `Data_roicat` from padded mean projections + sparse
CSR footprints, then (1) **geometric alignment** — default RoMa [Edstedt et al.
2024], with `PhaseCorrelation`/`ECC_cv2` alternatives; when the RoMa CUDA
`local_corr` extension is unavailable a PyTorch-native correlation is patched in
(2–3× slower, numerically equivalent); blends mean projection and footprint density
via `roi_FOV_mixing_factor=0.5`; (2) an **alignment-quality proxy** (RANSAC inlier
rate, or post-warp Pearson clipped to $[0,1]$); (3) a **ROI blurrer**
(`kernel_halfWidth=2`); (4) **ROInet embedding** (weights cached under
`~/.cache/roigbiv/roinet`); (5) a **scattering wavelet** feature; (6) a
**similarity graph** combining spatial footprints, ROInet latents, and SWT latents
with power weights (SF 1.0, NN 0.5, SWT 0.5); (7) **sequential-Hungarian
clustering** with cost threshold 0.6 (singletons labelled $-1$). Output is a
`ClusterResult` (per-ROI cluster labels, session-membership matrix, inlier rate,
quality metrics).

### 18.3 Calibrated logistic posterior

A four-feature vector (`compute_fov_features`, `match.py:105-169`) —
`n_shared_clusters`, `fraction_query_clustered`, `alignment_quality`,
`mean_cluster_cohesion` — drives a single-layer logistic $p_\text{same FOV} =
\sigma(z)$ with hand priors `DEFAULT_FOV_COEFS = (-4.0, 0.05, 3.0, 4.0, 3.0)`
(`calibration.py:37`), replaceable by `fit_from_labels` (scikit-learn
`LogisticRegression`) once a labelled pair set exists. Coefficients persist to
`inference/registry_calibration.json`.

### 18.4 Decision thresholds and storage

`auto_match` if $p\ge 0.9$; `review` if $0.5\le p<0.9$ (no DB write); `new_fov` if
$p<0.5$ (`match.py:31-32`). `register_or_match` (`orchestrator.py`) short-circuits
to `hash_match` on fingerprint collision, then iterates candidate FOVs scoped by
`(animal_id, region)` parsed from the filename. Storage is a four-table SQLAlchemy
schema (FOV, Session, Cell, CellObservation) plus a local blob store under
`inference/fingerprints/{fov_id}/`, with Alembic migrations for the v2 embedding
URIs and v3 cluster-label columns. Configuration is env-driven
(`ROIGBIV_REGISTRY_DSN`, `ROIGBIV_BLOB_ROOT`, `ROIGBIV_ROICAT_DEVICE`,
`ROIGBIV_FOV_ACCEPT_THRESHOLD=0.9`, `ROIGBIV_FOV_REVIEW_THRESHOLD=0.5`). The
`roigbiv-registry` CLI exposes `list/show/match/track/backfill/migrate`; `backfill`
is idempotent via the fingerprint pre-filter.

---

## 19. Parameter reference (master table)

Every parameter below is a field of `PipelineConfig` (`roigbiv/pipeline/types.py`).
CLI flags override these defaults (note the `flow_threshold` CLI-vs-dataclass nuance
in §3.2).

### Foundation

| parameter | default | `types.py` |
|:---|:---|:---|
| `k_background` | 30 | 166 |
| `n_svd` | 200 | 167 |
| `batch_size` | 500 | 168 |
| `nonrigid` | True | 169 |
| `do_registration` | False (True when input lacks `_mc`) | 170 |
| `fs` | 30.0 Hz (CLI-required) | 171 |
| `frame_averaging` | 1 | 172 |
| `tau` | 1.0 s (GCaMP6s) | 173 |
| `svd_bin_frames` | 5 000 | 174 |
| `reconstruct_chunk` | 500 | 175 |
| `scout_mode` / `scout_vcorr_stride` / `scout_vcorr_neighbors` | False / 1 / 8 | 181–183 |
| `foundation_only` | False | 190 |

### Motion correction

| parameter | default | `types.py` |
|:---|:---|:---|
| `motion_correction_backend` | `"phasecorr"` | 210 |
| `mc_max_displacement` | 50 px | 211 |
| `mc_strip_height` | 32 rows | 212 |
| `mc_smooth_sigma_rows` / `_time` | 6.0 / 1.0 | 218–219 |
| `mc_strip_confidence_weight` | True | 220 |
| `mc_prefilter` / `_sigma_low` / `_sigma_high` | False / 1.0 / 8.0 | 224–226 |
| `mc_sima_env` / `mc_granularity` | `"sima-legacy"` / `"row"` | 228–229 |
| `mc_s2p_block_size` | **`[64, 64]`** | 246 |
| `mc_s2p_smooth_sigma` | 1.15 | 247 |
| `mc_s2p_maxregshift` | 0.1 | 249 |
| `mc_s2p_nonrigid` | True | 250 |
| `mc_s2p_maxregshift_nr` | 5 px | 251 |
| `mc_s2p_nimg_init` | 300 | 252 |
| `mc_s2p_one_photon_reg` | **True** (→ `1Preg`) | 257 |
| `mc_s2p_spatial_hp_reg` | 42 px | 258 |
| `mc_s2p_spatial_taper` | 40.0 px | 260 |

### Stage 1 (Cellpose)

| parameter | default | `types.py` |
|:---|:---|:---|
| `cellpose_model` | `models/deployed/current_model` | 263 |
| `diameter` | 12 px | 264 |
| `diameter_auto` | False | 268 |
| `cellprob_threshold` | $-2.0$ | 269 |
| `flow_threshold` | 0.4 (dataclass) / 0.6 (CLI backfill) | 270 |
| `channels` | (1, 2) | 271 |
| `tile_norm_blocksize` | 128 px | 272 |
| `use_denoise` | True | 273 |
| `stage1_backend` | `"cellpose3"` (opt: `cpsam_sidecar`) | 286 |
| `stage1_ch2_source` | **`"vcorr_max_fused"`** | 305 |

### Gate 1

| parameter | default | `types.py` |
|:---|:---|:---|
| `min_area` / `max_area` | 80 / 600 px | 308–309 |
| `min_solidity` | 0.55 | 310 |
| `max_eccentricity` | 0.90 | 311 |
| `min_contrast` | 0.10 | 312 |
| `flag_{area,solidity,eccentricity,contrast}_margin` | 20 / 0.05 / 0.03 / 0.03 | 314–317 |
| `dog_strong_negative_percentile` | 10.0 | 319 |
| `annulus_inner_buffer` / `_outer_radius` | 2 / 15 px | 322–323 |

### Subtraction

| parameter | default | `types.py` |
|:---|:---|:---|
| `subtract_chunk_frames` | 2 000 | 352 |
| `subtract_ridge_lambda_scale` | $10^{-6}$ | 353 |
| `subtract_anticorr_threshold` | $-0.3$ | 354 |
| `subtract_anticorr_failure_fraction` | 0.10 | 355 |
| `subtract_nnls_fallback_max_rois` | 30 | 356 |
| `subtract_solver` | `"ridge"` (opt: `"robust"`) | 357 |
| `subtract_robust_kappa` / `_max_iter` | 0.5 / 5 | 358–359 |

### Stage 2 / Gate 2

| parameter | default | `types.py` |
|:---|:---|:---|
| `threshold_scaling` | 1.0 | 362 |
| `iscell_threshold` | 0.3 | 363 |
| `gate2_iou_threshold` | 0.3 | 366 |
| `gate2_max_correlation` | 0.7 | 367 |
| `gate2_anticorr_threshold` | $-0.5$ | 368 |
| `gate2_spatial_radius` | 20 px | 369 |
| `gate2_min_area` / `_max_area` | 60 / 400 px | 370–371 |
| `gate2_min_solidity` | 0.4 | 372 |
| `gate2_max_eccentricity` | 0.85 | 373 |
| `gate2_near_distance` | 5 px | 374 |
| `gate2_near_corr_threshold` / `gate2_flag_corr_threshold` | 0.5 / 0.5 | 375–376 |

### Stage 3 / Gate 3

| parameter | default | `types.py` |
|:---|:---|:---|
| `template_threshold` | 6.0 σ | 384 |
| `spatial_pool_radius` | 8 px | 385 |
| `spatial_pool_threshold` | 3.0 σ | 386 |
| `cluster_distance` | 12 px | 387 |
| `min_event_separation` | 2.0 s | 388 |
| `stage3_pixel_chunk_rows` | 8 | 389 |
| `stage3_chunk_budget_bytes` | 1 GB | 390 |
| `stage3_sigma_window_frames` | 500 (config-only; global MAD used) | 391 |
| `stage3_max_events` | 2 000 000 | 392 |
| `gate3_min_waveform_r2` | 0.6 | 395 |
| `gate3_min_waveform_r2_single_event` | 0.5 | 396 |
| `gate3_max_rise_decay_ratio` | 0.5 | 397 |
| `gate3_anticorr_threshold` | $-0.5$ | 398 |
| `gate3_min_solidity` | 0.5 | 399 |
| `gate3_waveform_window_tau_multiple` | 5.0 | 400 |

### Stage 4 / Gate 4

| parameter | default | `types.py` |
|:---|:---|:---|
| `bandpass_windows` | {fast 0.5–2.0, medium 0.1–1.0, slow 0.05–0.5} Hz | 403–407 |
| `bandpass_order` | 4 | 408 |
| `n_svd_components_stage4` | 300 | 409 |
| `corr_neighbor_radius_inner` / `_outer` | 6 / 15 px | 410–411 |
| `corr_contrast_threshold` | 0.10 | 412 |
| `stage4_min_area` / `_max_area` | 80 / 350 px | 413–414 |
| `stage4_min_solidity` | 0.60 | 415 |
| `stage4_max_eccentricity` | 0.85 | 416 |
| `stage4_iou_merge_threshold` | 0.3 | 417 |
| `stage4_pixel_chunk_rows` | 16 | 418 |
| `stage4_n_workers` | 3 | 419 |
| `gate4_min_corr_contrast` | 0.10 | 442 |
| `gate4_max_motion_corr` | 0.3 | 443 |
| `gate4_anticorr_threshold` | $-0.5$ | 444 |
| `gate4_min_mean_intensity_pct` | 25 | 445 |
| `gate4_spatial_radius` | 20 px | 446 |

### Classification / neuropil / tonic-accept

| parameter | default | `types.py` |
|:---|:---|:---|
| `neuropil_coeff` | 0.7 | 326 |
| `neuropil_inner_buffer` / `_outer_radius` | 2 / 15 px | 327–328 |
| `baseline_window_s` | 60.0 s | 329 |
| `baseline_percentile` | 10 | 330 |
| `tonic_baseline_window_s` | 120.0 s | 331 |
| `phasic_min_transients` / `phasic_min_skew` | 5 / 0.5 | 334–335 |
| `sparse_min_transients` / `sparse_min_skew` | 1 / 0.3 | 336–337 |
| `tonic_bp_std_factor` | 2.0 | 338 |
| `tonic_accept_tier` | **False** | 348 |
| `tonic_accept_min_elevation` | 0.5 | 349 |

### Optional subsystems / execution

| parameter | default | `types.py` |
|:---|:---|:---|
| `use_pmd_denoise` | **False** | 431 |
| `pmd_patch_size` / `pmd_patch_overlap` | 32 / 8 px | 432–433 |
| `pmd_max_rank` / `pmd_rank_margin` | 30 / 0.0 | 434–435 |
| `pmd_band_budget_bytes` | 1 GB | 436 |
| `batch_n_workers` | 1 (cap 2) | 439 |
| `resume` | False | 457 |
| `enable_stage_2` / `_3` / `_4` | True / True / True | 467–469 |
| `force_cpu` | False | 470 |

### Registry

| parameter | default | location |
|:---|:---|:---|
| `FINGERPRINT_VERSION` | 3 | `fingerprint.py:34` |
| `DEFAULT_FOV_COEFS` | $(-4.0, 0.05, 3.0, 4.0, 3.0)$ | `calibration.py:37` |
| `AUTO_ACCEPT_THRESHOLD` / `REVIEW_THRESHOLD` | 0.9 / 0.5 | `match.py:31-32` |
| ROICaT alignment / cost / mixing | RoMa / 0.6 / 0.5 | `roicat_adapter.py` |

### Robust background (RPCA — present, not wired)

| parameter | value | location |
|:---|:---|:---|
| `MIN_RPCA_FRAMES` | 150 | `rpca.py:55` |
| `_IALM_LIVE_COPIES` | 6 | `rpca.py:56` |
| `_RPCA_MEM_FRACTION` | 0.6 | `rpca.py:57` |
| solver `method` | `"ialm"` (opt: `"godec"`) | `rpca.py:120` |

---

## 20. Bibliography

**Motion correction, Suite2p, OASIS, SIMA.**

- Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L.F., Dalgleish,
  H., Carandini, M. & Harris, K.D. (2017). Suite2p: beyond 10,000 neurons with
  standard two-photon microscopy. *bioRxiv* 061507. (Registration + SVD-based
  detection used in foundation, Stage 2, and the `phasecorr` motion backend.)
- Kaifosh, P., Zaremba, J.D., Danielson, N.B. & Losonczy, A. (2014). SIMA: Python
  software for analysis of dynamic fluorescence imaging data. *Frontiers in
  Neuroinformatics* 8:80. (HiddenMarkov2D row-granularity correction used by the
  opt-in `legacy` motion backend.)
- Friedrich, J., Zhou, P. & Paninski, L. (2017). Fast online deconvolution of
  calcium imaging data. *PLoS Comp. Biol.* 13(3): e1005423. (OASIS, §13.3.)

**Cellpose and image restoration.**

- Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. (2021). Cellpose: a
  generalist algorithm for cellular segmentation. *Nature Methods* 18, 100–106.
- Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model.
  *Nature Methods* 19, 1634–1641. (`denoise_cyto3` image restoration, Stage 1.)

**SVD, robust PCA, and matrix-decomposition denoising.**

- Halko, N., Martinsson, P.-G. & Tropp, J.A. (2011). Finding structure with
  randomness. *SIAM Review* 53, 217–288. (Randomised-subspace SVD;
  `torch.svd_lowrank` in §2.2, §8.2.)
- Candès, E.J., Li, X., Ma, Y. & Wright, J. (2011). Robust principal component
  analysis? *J. ACM* 58, 11:1–11:37. (Conceptual precedent for L+S; the default
  foundation uses direct rank-$k$ truncation, not nuclear-norm PCP.)
- Lin, Z., Chen, M. & Ma, Y. (2010). The augmented Lagrange multiplier method for
  exact recovery of corrupted low-rank matrices. *arXiv:1009.5055*. (IALM-PCP, the
  primary RPCA solver, §2.3.1.)
- Zhou, T. & Tao, D. (2011). GoDec: randomized low-rank & sparse matrix
  decomposition in noisy case. *ICML 2011*. (GoDec fallback, §2.3.1.)
- Marchenko, V.A. & Pastur, L.A. (1967). Distribution of eigenvalues for some sets
  of random matrices. *Mat. Sb.* 72(114):4, 507–536. (Noise-edge rank selection,
  PMD, §8.2.)
- Buchanan, E.K., Kinsella, I., Zhou, D., et al. (2018). Penalized matrix
  decomposition for denoising, compression, and improved demixing of functional
  imaging data. *arXiv:1807.06203*. (PMD lineage, §8.)

**Calcium indicators.**

- Chen, T.-W., et al. (2013). Ultrasensitive fluorescent proteins for imaging
  neuronal activity. *Nature* 499, 295–300. (GCaMP6s, $\tau\approx 1$ s.)
- Zhang, Y., et al. (2023). Fast and sensitive GCaMP calcium indicators for imaging
  neural populations. *Nature* 615, 884–891. (jGCaMP8f, Stage 3 template bank.)

**Consensus ROI matching and registry.**

- Giovannucci, A., et al. (2019). CaImAn: an open source tool for scalable calcium
  imaging data analysis. *eLife* 8, e38173. (IoU 0.3 in Stage 2 and §13.4.)
- Landry, J.R., Nagy, D.G., Pachitariu, M. & Harris, K.D. (2024). ROICaT: Region of
  Interest Classification and Tracking. *bioRxiv*. (FOV alignment, ROInet
  embedding, scattering-wavelet, sequential-Hungarian clustering, §18.)
- Edstedt, J., Bökman, G., Wadenbäck, M. & Felsberg, M. (2024). RoMa: Robust Dense
  Feature Matching. *CVPR 2024*. (Default registry alignment, §18.2.)

**Filter design.**

- Butterworth, S. (1930). On the theory of filter amplifiers. *Exp. Wireless &
  Wireless Engineer* 7, 536–541. (Stage 4 + QC bandpass.)
