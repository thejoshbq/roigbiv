# ROIGBIV — Algorithmic methods

> **Scope.** This document is a publication-grade reference for every algorithm in the
> ROIGBIV two-photon calcium-imaging pipeline and its cross-session registry. It is
> intended to be cited in manuscripts and to accompany the code as a methods-section
> companion. Every numeric threshold is sourced from the implementation
> (`roigbiv/pipeline/types.py::PipelineConfig`); any code reference is given as
> `file:line`.
>
> This supersedes `docs/pipeline_algorithm_breakdown.md`, which describes an older
> parallel three-branch Suite2p+Cellpose consensus architecture and does not reflect
> the current sequential subtractive pipeline.
>
> **Canonical source of defaults.** `roigbiv/pipeline/types.py` (the dataclass
> `PipelineConfig`). The legacy YAML at `configs/pipeline.yaml` describes the
> superseded consensus architecture and is *not* authoritative for the sequential
> pipeline documented here.

---

## Table of contents

1. [Overview and notation](#1-overview-and-notation)
2. [Foundation: motion correction, background decomposition, summary images](#2-foundation)
3. [Stage 1 — Cellpose spatial detection](#3-stage-1--cellpose-spatial-detection)
4. [Gate 1 — morphological validation](#4-gate-1--morphological-validation)
5. [Source subtraction engine](#5-source-subtraction-engine)
6. [Stage 2 — Suite2p temporal detection](#6-stage-2--suite2p-temporal-detection)
7. [Gate 2 — temporal cross-validation](#7-gate-2--temporal-cross-validation)
8. [Stage 3 — template sweep on residual](#8-stage-3--template-sweep-on-residual)
9. [Gate 3 — waveform validation](#9-gate-3--waveform-validation)
10. [Stage 4 — tonic-neuron search](#10-stage-4--tonic-neuron-search)
11. [Gate 4 — correlation-contrast validation](#11-gate-4--correlation-contrast-validation)
12. [Quality-control features, trace extraction, classification](#12-quality-control-features-trace-extraction-classification)
13. [Human-in-the-loop review package](#13-human-in-the-loop-review-package)
14. [Pipeline orchestration and output layout](#14-pipeline-orchestration-and-output-layout)
15. [Cross-session FOV and cell registry](#15-cross-session-fov-and-cell-registry)
16. [Parameter reference (master table)](#16-parameter-reference-master-table)
17. [Bibliography](#17-bibliography)

---

## 1. Overview and notation

### 1.1 Architecture

ROIGBIV is a sequential subtractive pipeline for detecting regions of interest (ROIs)
in two-photon calcium-imaging movies. Detection proceeds through four stages. After
each detection stage, a gate accepts, flags, or rejects each candidate; accepted and
flagged candidates are then subtracted from the residual movie before the next stage
operates. This removes the need for any single detector to discriminate among all
neuron types and provides per-ROI provenance (which stage discovered the ROI, under
which gate outcome, at which confidence).

```
input TIF  ─►  Foundation  (motion correction → SVD L+S → summary images → DoG)
                   │
                   ▼
               Stage 1  (Cellpose on {mean_M, vcorr_S})   ─►  Gate 1  (morphology)
                   │                                              │
                   ▼                                              ▼
               subtract_1                                 accept | flag ⟶ subtract
                   │
                   ▼
               Stage 2  (Suite2p reuse, IoU vs Stage 1)   ─►  Gate 2  (temporal)
                   │                                              │
                   ▼                                              ▼
               subtract_2                                 accept | flag ⟶ subtract
                   │
                   ▼
               Stage 3  (FFT matched filter, event cluster) ─►  Gate 3  (waveform)
                   │                                              │
                   ▼                                              ▼
               subtract_3                                 accept | flag ⟶ subtract
                   │
                   ▼
               Stage 4  (bandpass correlation contrast)    ─►  Gate 4  (contrast)
                   │
                   ▼
               QC features → dF/F → OASIS → classification → HITL export
```

### 1.2 Design principles

- **Recall-first, precision through review.** Every gate is tuned so that false
  negatives are rarer than false positives; precision is obtained by the
  human-in-the-loop review package rather than by aggressive automated rejection.
- **Sequential and subtractive.** Each stage operates on the residual of previous
  stages, so no stage wastes effort rediscovering prior detections.
- **Provenance-tracked.** Every ROI carries `source_stage`, `gate_outcome ∈ {accept,
  flag, reject}`, `confidence ∈ {high, moderate, low, requires_review}`, per-stage
  scores, and a list of human-readable gate-failure reasons
  (`roigbiv/pipeline/types.py:17-73`).
- **HITL-closed.** Stage 4 has no automated accept tier; its survivors enter a
  prioritised review queue.

### 1.3 Notation

| symbol | meaning |
|:---|:---|
| $T$ | number of frames |
| $H \times W$ | spatial dimensions (also $L_y \times L_x$) |
| $N_\text{pix} = H \cdot W$ | total pixel count |
| $M \in \mathbb{R}^{T \times H \times W}$ | registered movie |
| $L$, $S$ | low-rank (background) and sparse (foreground) components, $M = L + S$ |
| $S_k$ | residual after $k$ stages of subtraction ($S_0 = S$) |
| $f_s$ | acquisition frame rate (Hz) |
| $\tau$ | indicator decay constant (s); 1.0 for GCaMP6s |
| $\sigma_p$ | per-pixel noise scale (MAD-based) |
| $K$ | number of templates in the Stage 3 matched-filter bank |
| $W$ (in §5) | ROI spatial-profile design matrix, not to be confused with image width |

All per-pixel accumulators run in `float64` to avoid catastrophic cancellation; final
outputs are `float32`.

---

## 2. Foundation

Implementation: `roigbiv/pipeline/foundation.py`.

The foundation stage produces (a) a rigidly- and non-rigidly-registered movie
$M \in \mathbb{R}^{T \times H \times W}$, stored as an `int16` memory-map; (b) a
low-rank-plus-sparse decomposition $M = L + S$ with $S$ persisted as a `float32`
memory-map; (c) five summary images (mean_M, mean_S, max_S, std_S, vcorr_S); and
(d) a difference-of-Gaussians (DoG) nuclear-shadow map. All arrays larger than
$H \cdot W$ are streamed through temporal chunks so peak RAM is bounded by a single
chunk.

### 2.1 Motion correction (Suite2p phase-correlation)

Motion correction is delegated to Suite2p [Pachitariu et al. 2017], invoked through
`run_motion_correction` at `foundation.py:38-95`. Suite2p performs rigid
registration by FFT subpixel phase correlation to a reference image computed from
`nimg_init = 300` frames, followed by optional non-rigid registration over blocks of
$128 \times 128$ pixels. When the input filename ends in `_mc.tif` the pipeline
assumes the movie is pre-registered and disables this step (`do_registration=False`,
`types.py:153`); in that case only the existing displacement fields (if any) are
loaded. Suite2p writes a contiguous `int16` `data.bin` which is subsequently opened
as an `np.memmap` (`foundation.py:102-113`).

Relevant defaults (`types.py:151-157`, `pipeline.yaml:51-71`):

| parameter | default | purpose |
|:---|:---|:---|
| `batch_size` | 500 | GPU frames per registration batch |
| `nonrigid` | True | enable piecewise non-rigid block registration |
| `nimg_init` | 300 | frames used to compute reference image |
| `smooth_sigma` | 1.15 px | reference-image Gaussian smoothing |
| `maxregshift` | 0.1 | maximum shift as fraction of frame size |
| `block_size` | 128 × 128 px | non-rigid block size |

Suite2p's internal ROI-detection pass is also executed at this point and its
outputs (`stat.npy`, `iscell.npy`) are retained for reuse by Stage 2.

### 2.2 Temporally binned truncated SVD

The decomposition operates on a temporally binned copy of the movie to bound
compute. Given target $T_\text{bin} \approx 5000$ frames
(`svd_bin_frames=5000`, `types.py:156`), the bin width is
$b = \lceil T / T_\text{bin} \rceil$ and the binned movie is

$$
\tilde M_{b,p} \;=\; \frac{1}{|B_b|} \sum_{t \in B_b} M_{t,p}, \qquad
B_b = [b \cdot |B_b|,\; (b+1) \cdot |B_b|) \cap [0, T),
$$

computed at `foundation.py:116-141`.

A rank-$n_\text{svd}$ truncated SVD (`n_svd=200`, `types.py:150`) is computed on
$\tilde M^\top$ via `torch.svd_lowrank(A, q, niter=2)` — two power iterations of
randomised subspace SVD [Halko et al. 2011] — at `foundation.py:144-181`. The
transpose orientation is deliberate: we factor $\tilde M^\top \in
\mathbb{R}^{N_\text{pix} \times T_\text{bin}}$ so that the returned $U$ indexes pixels
directly (convenient for spatial reconstruction). On a `torch.cuda.OutOfMemoryError`
the computation falls back transparently to CPU.

The temporal components are then nearest-neighbour upsampled from $T_\text{bin}$ to
$T$ (`_upsample_V` at `foundation.py:184-202`). This is an acceptable approximation
for the background subspace because the binning already preserves its dominant
low-frequency structure.

### 2.3 Low-rank / sparse decomposition

Denote the top $k = k_\text{background}$ SVD factors (default $k=30$, `types.py:149`)
as $U_k \in \mathbb{R}^{N_\text{pix} \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$
diagonal, $V_k \in \mathbb{R}^{T \times k}$ (upsampled). The background and
residual at each frame $t$ are

$$
L_t \;=\; U_k \, \Sigma_k \, V_k^{(t)\top}, \qquad
S_t \;=\; M_t - L_t,
$$

implemented as a streamed chunked reconstruction with chunk width
`reconstruct_chunk=500` (`types.py:157`, `foundation.py:274-291`). Throughout the
codebase we refer to this as a *truncated-SVD L+S decomposition*: it is not the
iterative principal component pursuit of Candès et al. [2011] (no nuclear-norm or
$\ell_1$ objective), but rather a direct rank-$k$ projection that achieves the same
separation of slowly-varying photobleach / neuropil / illumination drift ($L$) from
the sparse cellular signal ($S$). The choice $k=30$ is validated empirically
against summary-image contrast (internal design doc, spec §3.3).

The SVD factors ($U$, $\Sigma$, $V_\text{bin}$) are persisted to `svd_factors.npz`
for reuse by later stages; $S$ is persisted to `residual_S.dat` as a `float32`
memory-map alongside a JSON metadata sidecar.

### 2.4 Streaming summary images

The summary images of $S$ are computed in a single temporal pass
(`generate_summary_images`, `foundation.py:315-423`) with `float64` accumulators.
Per-chunk peak RAM is capped at ~500 MB by hard-limiting the chunk width to 128
frames regardless of `reconstruct_chunk` (`foundation.py:497-505`).

**Mean, maximum, standard deviation.** Ordinary running moments:

$$
\mu(p) = \frac{1}{T} \sum_t S_t(p), \quad
\max(p) = \max_t S_t(p), \quad
\sigma(p) = \sqrt{\,\max\bigl(0, \tfrac{1}{T}\sum_t S_t(p)^2 - \mu(p)^2\bigr)\,}.
$$

**8-neighbour local correlation (vcorr).** For each of the eight offsets
$(\delta_y, \delta_x) \in \{-1, 0, 1\}^2 \setminus \{(0,0)\}$ we maintain
accumulators $\sum x, \sum y, \sum x^2, \sum y^2, \sum xy$ and a count $n$ per
offset, where $x$ is the shifted neighbour and $y$ is the centre pixel. The
per-offset Pearson correlation is computed via second moments:

$$
r_{\delta_y, \delta_x}(p)
\;=\; \frac{n \sum x y - \sum x \sum y}
          {\sqrt{\bigl(n\sum x^2 - (\sum x)^2\bigr)\bigl(n\sum y^2 - (\sum y)^2\bigr) + \varepsilon}},
$$

and vcorr at each pixel is the average of $r_{\delta_y, \delta_x}(p)$ over the
offsets whose shifted neighbour lies inside the FOV
(`foundation.py:333-420`). Boundary pixels average over fewer neighbours. The
second-moment formulation is algebraically equivalent to the mean-subtracted Pearson
estimator but requires only one pass over the data.

**Raw morphological mean (mean_M).** The mean of the *registered* movie (not the
residual) is read from Suite2p's `meanImg` field or reconstructed from `data.bin`
if absent (`foundation.py:516-525`). Under a top-$k$ SVD L+S decomposition the
first few components absorb per-pixel brightness, so `mean_S ≈ 0` and is unsuitable
as a morphological channel. `mean_M` preserves the raw anatomical contrast that
Cellpose's training regime expects.

### 2.5 Difference-of-Gaussians nuclear-shadow map

GCaMP is excluded from the neuronal nucleus, so healthy somata typically show a
darker central region against a brighter cytoplasmic annulus. A
difference-of-Gaussians (DoG) applied to `mean_M` quantifies this:

$$
\mathrm{DoG}(x, y)
\;=\; G_{\sigma_\text{outer}} * M_\mu(x, y)
\;-\; G_{\sigma_\text{inner}} * M_\mu(x, y),
\qquad \sigma_\text{inner} = 2, \sigma_\text{outer} = 6.
$$

Implementation: `compute_nuclear_shadow_map`, `foundation.py:430-450`. The
polarity is chosen so that a pixel at the dark nuclear centre gives a positive
response: the narrow Gaussian picks up the dark nucleus (low value), the wide
Gaussian averages over soma+surround (higher value), so $G_{\sigma_\text{outer}} -
G_{\sigma_\text{inner}}$ is positive at the nucleus. Gate 1 uses the 10th
percentile of this map as the "strongly negative" threshold for rejection.

All summary images are written to `summary/*.tif` as `float32`.

---

## 3. Stage 1 — Cellpose spatial detection

Implementation: `roigbiv/pipeline/stage1.py`.

Stage 1 segments morphologically clear somata via Cellpose 3 [Stringer et al. 2021;
Pachitariu & Stringer 2022] operating on a dual-channel image stack. The mean
projection alone misses dim and tonic cells; the local-correlation projection vcorr
misses bright but silent cells; the pair is complementary.

### 3.1 Preprocessing and model

The first channel is `mean_S` optionally passed through Cellpose 3's
`denoise_cyto3` image-restoration model [Pachitariu & Stringer 2022]
(`denoise_mean_S`, `stage1.py:45-77`; enabled via `use_denoise=True`,
`types.py:166`). The restoration output is reshaped to `(H, W)` and cast to
`float32`. The second channel is `vcorr_S` unchanged.

Inference is carried out by `CellposeModel` loaded either from a deployed checkpoint
path (`models/deployed/current_model`, `types.py:160`) or — if the path fails to
load — from the built-in `cyto3` model. The input is stacked as
$x \in \mathbb{R}^{H \times W \times 2}$ with `channel_axis=-1`, and inference is
called with the parameters in Table 3.1.

**Table 3.1 — Cellpose parameters (`types.py:160-166`, `pipeline.yaml:5-25`).**

| parameter | default | meaning |
|:---|:---|:---|
| `diameter` | 12 px | expected soma diameter under GRIN-lens optics |
| `cellprob_threshold` | $-2.0$ | permissive cell-probability cut to maximise recall |
| `flow_threshold` | 0.6 | flow-field error threshold |
| `channels` | $(1, 2)$ | 1-indexed Cellpose channel roles: cyto=mean, nucleus=vcorr |
| `tile_norm_blocksize` | 128 px | tile-normalisation block size (counters GRIN vignetting) |

Cellpose returns a uint16 label image, a list of XY flow maps, and the cell-probability
map $\Pi \in \mathbb{R}^{H \times W}$ (`flows[2]` in Cellpose 3.x). For each
non-zero label $\ell$, the binary mask is $M_\ell = \{p : L(p) = \ell\}$ and the
per-ROI probability is $\Pi_\ell = (|M_\ell|)^{-1} \sum_{p \in M_\ell} \Pi(p)$
(`stage1.py:154-166`).

---

## 4. Gate 1 — morphological validation

Implementation: `roigbiv/pipeline/gate1.py`.

Gate 1 converts raw Cellpose candidates into `ROI` objects with an
`accept | flag | reject` outcome based on five features.

### 4.1 Features

**Area, solidity, eccentricity.** Computed from
`skimage.measure.regionprops`. Area is the pixel count, solidity is
$A / A_\text{convex hull}$, and eccentricity is that of the equivalent ellipse
(`gate1.py:105-116`).

**Soma–surround contrast.** Construct an annulus around the mask,

$$
\text{ring}
\;=\; \operatorname{dilate}(\text{mask}, r_\text{out})
\;\wedge\; \neg \operatorname{dilate}(\text{mask}, r_\text{in})
\;\wedge\; \neg \bigl(\bigcup_{j \neq i} \text{mask}_j\bigr),
$$

with `annulus_inner_buffer=2 px`, `annulus_outer_radius=15 px`
(`types.py:183-184`; `gate1.py:29-43`). Exclusion of other ROI pixels prevents
neighbour-soma contamination of the annular background. Contrast is

$$
c_i \;=\; \frac{\mu_{S}(\text{mask}_i) - \mu_{S}(\text{ring}_i)}
               {\max\bigl(|\mu_{S}(\text{ring}_i)|,\; 10^{-6}\bigr)},
$$

with the sign of the denominator preserved when $\mu_S(\text{ring}_i)$ is near zero
(`gate1.py:46-58`).

**Nuclear shadow score.** $n_i = \text{mean}_{p \in \text{mask}_i}\text{DoG}(p)$.
Taking the mean over the full mask rather than sampling the centroid is more
robust to labelling jitter (`gate1.py:128`).

### 4.2 Decision logic

Define a "strongly negative DoG" threshold as the `dog_strong_negative_percentile`
(default 10th, `types.py:180`) of the DoG map over the FOV. A candidate's failure
set collects criteria that breach their thresholds (`gate1.py:131-141`).

The decision (`gate1.py:143-178`) is:

- **Reject** if the *DoG conjunction rule* triggers — strongly-negative DoG AND
  contrast $\le$ `min_contrast` — OR two or more criteria other than contrast fail.
- **Accept** with `confidence=high` if no criterion fails.
- **Flag** with `confidence=moderate` if exactly one criterion fails within its
  per-criterion absolute margin (Table 4.2).
- Otherwise **reject** with `confidence=requires_review`.

The DoG conjunction rule captures the intended semantics of "likely astrocyte or
out-of-focus ghost" while treating DoG as advisory: a dim cell with negative DoG but
healthy contrast is not penalised. Marginal flagging preserves borderline cells for
review rather than discarding them, consistent with the recall-first principle.

**Table 4.1 — Gate 1 thresholds (`types.py:169-180`).**

| threshold | default | action if breached |
|:---|:---|:---|
| `min_area` | 80 px | reject (unless marginal and single failure) |
| `max_area` | 600 px | reject (unless marginal and single failure) |
| `min_solidity` | 0.55 | reject (unless marginal and single failure) |
| `max_eccentricity` | 0.90 | reject (unless marginal and single failure) |
| `min_contrast` | 0.10 | reject; also triggers DoG conjunction check |
| `dog_strong_negative_percentile` | 10.0 | DoG rejection if contrast also fails |

**Table 4.2 — Per-criterion flag margins (`types.py:175-178`).**

| criterion | flag margin |
|:---|:---|
| area | ±20 px |
| solidity | ±0.05 |
| eccentricity | ±0.03 |
| contrast | ±0.03 |

---

## 5. Source subtraction engine

Implementation: `roigbiv/pipeline/subtraction.py`.

Between every pair of detection stages, source subtraction removes the fluorescence
contribution of accepted+flagged ROIs from the residual movie so the next stage
operates on a cleaner substrate. Let $\mathcal{I}$ be the index set of ROIs to
subtract and $\text{mask}_i$ each ROI's binary support.

### 5.1 Spatial profile estimation

Each ROI is assigned a normalised spatial profile $w_i(p)$ supported on
$\text{mask}_i$,

$$
w_i(p) \;=\;
\begin{cases}
\dfrac{\psi(p)}{\max_{p' \in \text{mask}_i} \psi(p')} & p \in \text{mask}_i \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

where $\psi$ is a *profile source* field (`estimate_spatial_profiles`,
`subtraction.py:43-92`). The spec [§5.1] calls for $\psi = \mu_t[S]$ (temporal
mean of the residual), but under truncated-SVD L+S the top-$k$ components absorb
per-pixel mean brightness so $\mu_t[S] \approx 0$ with no spatial structure. We
instead pass $\psi = \sigma_t[S]$ (per-pixel temporal standard deviation), which
faithfully preserves the spatial pattern of residual activity: active pixels have
higher variance than neuropil. The `compute_std_map` routine at
`subtraction.py:565-589` computes this via a two-pass running-moment stream with
`float64` accumulators. Profiles peak at 1.0 inside the mask.

### 5.2 Simultaneous trace estimation (ridge-regularised GPU solve)

At each frame $t$ we estimate the per-ROI activity vector $c(t) \in \mathbb{R}^N$ by
ridge-regularised least squares over the union $P = \bigcup_i \text{mask}_i$ of ROI
supports:

$$
\hat c(t) \;=\; \arg\min_{c}\,
\bigl\lVert S(P, t) - W c \bigr\rVert_2^2 + \lambda \lVert c \rVert_2^2,
$$

where $W \in \mathbb{R}^{|P| \times N}$ stacks the union-restricted profiles as
columns. The normal equations $\hat c(t) = (W^\top W + \lambda I)^{-1} W^\top S(P,
t)$ admit a closed-form solution; we precompute $W^\top W + \lambda I$ once and
solve one linear system per temporal chunk on the GPU
(`solve_traces_from_chunks`, `subtraction.py:127-175`):

$$
\lambda \;=\; \rho \cdot \frac{\operatorname{tr}(W^\top W)}{N},
\qquad \rho = \text{`subtract\_ridge\_lambda\_scale`} = 10^{-6}
\;(\text{\texttt{types.py:203}}).
$$

Scaling $\lambda$ by the trace of $W^\top W$ keeps regularisation strength
proportional to the data scale, so the same $\rho$ works across ROI counts and
profile norms. The GPU path uses `torch.linalg.solve`; CUDA OOM falls back to CPU
(`subtraction.py:153-174`).

Temporal chunking (`subtract_chunk_frames=2000`, `types.py:202`) streams the
residual memory-map through RAM to avoid materialising the full $(|P|, T)$ design
response.

### 5.3 Rank-1 streaming subtraction

For every pixel $p \in P$ and every frame $t$,

$$
S_\text{out}(p, t) \;=\; S_\text{in}(p, t) - \sum_{i} w_i(p)\, \hat c_i(t).
$$

Pixels outside $P$ are copied unchanged (`subtract_sources`,
`subtraction.py:224-283`). The operation is streamed: one sequential read of the
input memory-map, an in-RAM rank-1 update, one sequential write to the output
memory-map. The new residual is written to a fresh memory-map
(`residual_S1.dat`, `residual_S2.dat`, `residual_S3.dat`); the predecessor is
retained for validation and is optionally unlinked after NNLS fallback completes.

### 5.4 Post-subtraction validation

Three per-ROI ratios are tested on the residual *after* subtraction
(`_validate_streaming`, `subtraction.py:303-480`):

| check | definition | pass range |
|:---|:---|:---|
| mean ratio | $\bigl|\mu(S_\text{out}[\text{mask}])\bigr| / \bigl(|\mu(S_\text{out}[\text{ring}])| + 10^{-6}\bigr)$ | $< 3$ |
| std ratio | $\sigma(S_\text{out}[\text{mask}]) / \sigma(S_\text{out}[\text{ring}])$ | $(0.3,\,3)$ |
| anti-correlation | Pearson $r\bigl(\mu(S_\text{out}[\text{mask}])_t,\; \hat c_i(t)\bigr)$ | $> \rho_\text{anti}$ |

with $\rho_\text{anti} = \text{`subtract\_anticorr\_threshold`} = -0.3$
(`types.py:204`). All three ratios must hold for `pass=True`. Values are
accumulated in `float64` moments during one streaming pass through the post-subtraction
residual; Pearson correlations are computed from second moments
(`subtraction.py:458-466`).

Rationale: a strong mean ratio ≫ 1 indicates a bright residual spot left at the
subtracted location; std ratio outside $(0.3, 3)$ indicates over- or
under-subtraction; strong anti-correlation indicates that the estimated trace
cancelled into the noise rather than extracting a true source.

### 5.5 Single-variable NNLS fallback

If the anti-correlation failure fraction exceeds
`subtract_anticorr_failure_fraction = 0.10` (`types.py:205`), up to
`subtract_nnls_fallback_max_rois = 30` (`types.py:206`) flagged ROIs are
re-estimated with a non-negativity constraint. Because each ROI's profile is
localised, the problem reduces to single-variable non-negative least squares on
the local support:

$$
\hat c_i(t) \;=\; \max\!\Bigl(0,\;
\frac{w_i^\top\, S_\text{in}(\text{mask}_i, t)}{w_i^\top w_i}\Bigr),
$$

closed-form (`_nnls_refine_flagged`, `subtraction.py:514-558`). The re-estimated
traces are substituted back, subtraction is re-run, and only the refined ROIs are
re-validated; unflagged entries from the first pass are retained
(`run_source_subtraction`, `subtraction.py:670-702`).

---

## 6. Stage 2 — Suite2p temporal detection

Implementation: `roigbiv/pipeline/stage2.py`.

Stage 2 recovers neurons whose morphology is insufficient for Cellpose but whose
temporal activity drives Suite2p's SVD-based detector [Pachitariu et al. 2017] —
canonically burst-firers, task-locked neurons, and cells occluded in the mean
projection by a brighter neighbour.

### 6.1 Reuse of foundation Suite2p outputs

Re-running Suite2p would duplicate foundation cost; instead, Stage 2 reads the
`stat.npy` and `iscell.npy` files already produced by the foundation step
(`_load_suite2p_outputs`, `stage2.py:93-116`). Each entry in `stat` is converted to
a dense binary mask by indexing `ypix`/`xpix` and clipping to the FOV
(`_stat_entries_to_masks`, `stage2.py:119-145`). Entries with
`iscell[i, 1] < iscell_threshold = 0.3` (`types.py:210`) are dropped.

### 6.2 IoU novelty filter

Against the union of Stage 1 accept+flag masks, we compute

$$
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|},
$$

and retain only those Stage 2 candidates whose maximum IoU against any Stage 1 mask
does not exceed `gate2_iou_threshold = 0.3` (`types.py:213`, `stage2.py:161-183`).
The 0.3 cut-off is within the 0.3–0.5 literature range for consensus ROI matching
[Giovannucci et al. 2019] and is conservative for Suite2p's irregular footprints
against Cellpose's smooth contours.

### 6.3 Trace extraction on the residual

Each retained candidate's trace is extracted from $S_1$ (the Stage-1-subtracted
residual) rather than from the raw movie, via a matrix–vector product
(`extract_traces_from_residual`, `stage2.py:38-86`):

$$
\text{trace}_i(t) \;=\; \frac{1}{|\text{mask}_i|}\,\sum_{p \in \text{mask}_i} S_1(p, t).
$$

Extraction uses a dense $(N, H \cdot W)$ mask matrix multiplied by temporally
chunked slabs of the residual; at typical $N \sim 30$–$200$ and $H \cdot W = 2.6 \times 10^5$
the mask matrix is 60–400 MB — small enough to avoid the overhead of sparse
algebra. The residual fallback to $S_0$ (if Stage 1 produced no subtractions) is
handled at `stage2.py:240`.

Resulting ROI objects carry `source_stage=2`, `iscell_prob`, `trace`, and
provisional `gate_outcome="accept"` pending Gate 2.

---

## 7. Gate 2 — temporal cross-validation

Implementation: `roigbiv/pipeline/gate2.py`.

Gate 2 verifies that Stage 2 candidates are genuinely independent sources rather
than rediscoveries the IoU filter missed, spatial spillover from an imperfectly
subtracted Stage 1 neighbour, or subtraction artefacts.

### 7.1 Features

For each candidate we compute:

- Area (pre-computed) and morphology via `regionprops` (`gate2.py:85-89`).
- Centroid $\bar p_i$ as the mean of mask coordinates.
- Correlations against each Stage 1 ROI trace whose centroid lies within
  `gate2_spatial_radius = 20 px` (`types.py:216`). Pearson correlations are
  computed in a single vectorised pass over the row-wise mean-centred
  Stage 1 trace matrix (`_pearson_row`, `gate2.py:42-55`):
  $r = \tfrac{B_c a_c}{\lVert a_c\rVert\, \lVert B_{c,\cdot}\rVert}$.

### 7.2 Decision logic

Let $r_i$ be the vector of candidate-to-nearby-Stage-1 Pearson correlations and
let $r^\text{near}_i$ be that subset restricted to Stage 1 ROIs within
`gate2_near_distance = 5 px` (`types.py:220`).

Failures are (`gate2.py:82-120`):

| failure | condition |
|:---|:---|
| morphology | area $\notin$ [60, 400] or solidity < 0.4 |
| redundancy / spillover | $\max \lvert r_i\rvert \ge$ `gate2_max_correlation = 0.7` |
| subtraction artefact | $\min r_i \le$ `gate2_anticorr_threshold = -0.5` |
| near-duplicate | any Stage 1 within `gate2_near_distance` with $\lvert r\rvert >$ `gate2_near_corr_threshold = 0.5` |

Decision:

- **Reject** if any failure triggers.
- **Flag** with `confidence=moderate` if all checks pass but $\max \lvert r_i\rvert
  >$ `gate2_flag_corr_threshold = 0.5` (borderline redundant).
- **Accept** with `confidence=high` otherwise.

Relaxed Gate 2 thresholds vs Gate 1 (`gate2_min_area=60`, `gate2_max_area=400`,
`gate2_min_solidity=0.4`; `types.py:217-219`) acknowledge that Suite2p footprints
are inherently noisier than Cellpose contours and do not have a supplementary
morphological signal to cross-check.

---

## 8. Stage 3 — template sweep on residual

Implementation: `roigbiv/pipeline/stage3.py` and `roigbiv/pipeline/stage3_templates.py`.

Stage 3 targets sparsely-firing neurons whose transient count is too low to
appear in Suite2p's SVD but which produce identifiable calcium-transient waveforms
in the post-Stage-2 residual $S_2$.

### 8.1 Dual-exponential template bank

Each template is a double-exponential calcium transient,

$$
w(t;\,\tau_r,\tau_d) \;=\; \bigl(1 - e^{-t / \tau_r}\bigr)\, e^{-t / \tau_d},
\qquad t \in [0,\, 5\tau_d],
$$

sampled at the acquisition rate $f_s$ and $L_2$-normalised to unit energy so that
cross-correlation scores are directly comparable across templates
(`_build_waveform`, `stage3_templates.py:39-50`). Three templates per indicator
family parameterise single, doublet, and burst kinetics (Table 8.1), selected by a
decay-constant threshold of 0.75 s (`stage3_templates.py:33-36`): GCaMP6s
[Chen et al. 2013] if $\tau \ge 0.75$ s, else jGCaMP8f [Zhang et al. 2023].

**Table 8.1 — Template bank (`stage3_templates.py:18-30`).**

| indicator | shape | $\tau_\text{rise}$ (s) | $\tau_\text{decay}$ (s) |
|:---|:---|:---|:---|
| GCaMP6s | single | 0.05 | 1.0 |
| GCaMP6s | doublet | 0.075 | 1.2 |
| GCaMP6s | burst | 0.10 | 1.5 |
| jGCaMP8f | single | 0.04 | 0.5 |
| jGCaMP8f | doublet | 0.06 | 0.6 |
| jGCaMP8f | burst | 0.08 | 0.75 |

### 8.2 FFT-based cross-correlation

The residual $S_2$ is scanned in spatial row-chunks of
`stage3_pixel_chunk_rows = 8` rows (`types.py:235`), which amounts to ~4 000
pixels per chunk on a 512-wide FOV. For each chunk we move the pixel-traces tensor
to the GPU, compute per-pixel noise scale

$$
\sigma_p \;=\; \max\bigl(\text{MAD}(x_p)\,/\,0.6745,\;\, 10^{-6}\bigr),
$$

with MAD = median absolute deviation (`stage3.py:93-96`), and FFT-transform each
pixel trace to length $n_\text{FFT} = 2^{\lceil \log_2(T + L_\text{max} - 1)\rceil}$
where $L_\text{max}$ is the longest template length. Templates are padded to
$L_\text{max}$ and pre-FFT'd once per stage invocation. For each template $k$ the
normalised cross-correlation

$$
\xi_k(p, t) \;=\; \frac{\bigl(\mathcal{F}^{-1}[\,\mathcal{F}\{x_p\} \cdot \overline{\mathcal{F}\{w_k\}}\,]\bigr)(t)}{\sigma_p}
$$

is accumulated into a running `score_max` and running `template_idx_max` tensor,
so no $(N_\text{pix}, K, T)$ array is ever materialised (`stage3.py:101-116`).

**Spec deviation.** Spec §9.2 specifies a sliding-window local MAD
$\sigma_\text{local}(p, t)$ with a `stage3_sigma_window_frames=500`-frame window
(`types.py:236`); the current implementation uses a *global* per-pixel MAD to save
~300 MB of intermediate storage per chunk (`stage3.py:27-32`). The approximation
is defensible: per-pixel global MAD already normalises away the dominant source of
scale variation (pixel-wise brightness heterogeneity); temporal non-stationarity is
mitigated by Stage 3's subsequent template matching, which penalises non-transient
waveforms.

### 8.3 Thresholding and event extraction

A pixel-time pair $(p, t)$ is declared an *event* when
$\text{score\_max}(p, t) > \theta$ with $\theta = $ `template_threshold = 6.0`
(`types.py:230`). The 6σ threshold is at the top of spec §18.6's 3–6σ range
because real residual distributions have heavier right tails than pure Gaussian
noise (structured neuropil leakage).

To bound per-chunk memory we adaptively raise $\theta$ by 1.0σ per iteration (up
to 8 iterations) if more than $2 \times 10^5$ events cross in a single chunk,
effectively converting the test into a top-$K$ selector when the chunk is
pathological (`stage3.py:118-131`). A global cap of `stage3_max_events =
2 \times 10^6` (`types.py:237`) retains the top-$K$ events by score if exceeded
(`stage3.py:326-332`).

### 8.4 Spatial clustering

Events are clustered in 2-D by single-linkage hierarchical clustering
(`scipy.cluster.hierarchy.linkage` + `fcluster`) with distance threshold
`cluster_distance = 12 px` (`types.py:233`). Above $2 \times 10^4$ events the
$O(n^2)$-memory `pdist` becomes prohibitive, so we switch to a grid-snap
approximation: each event is assigned to a $d \times d$ grid cell where $d$ equals
the distance threshold, and events in the same cell become the same cluster
(`_cluster_events_spatial`, `stage3.py:150-196`). This under-merges chained
cells at adjacent grid boundaries but is acceptable because real somata produce
many events per grid cell and the switch-over is only reached in pathological
cases.

### 8.5 Temporal-independence filter

For each spatial cluster we greedily select events in descending score order,
retaining an event only if no previously selected event is within
`min_event_separation = 2.0 s` (`types.py:234`) — i.e.,
$\lceil 2 f_s \rceil$ frames
(`_count_temporally_independent`, `stage3.py:199-223`). The retained count is
`event_count`; clusters with zero independent events are discarded.

### 8.6 Candidate packaging

Each surviving cluster becomes a candidate ROI: mask is a filled disk of radius
`spatial_pool_radius = 8 px` (`types.py:231`) centred on the cluster's mean
$(y, x)$; the trace is extracted from $S_2$ via the same matrix-vector extractor
as Stage 2 (`stage3.py:382-394`); provisional `gate_outcome="accept"` pending
Gate 3 (`stage3.py:395-426`).

---

## 9. Gate 3 — waveform validation

Implementation: `roigbiv/pipeline/gate3.py`.

Gate 3 tests whether each candidate's per-event waveforms are consistent with true
calcium transients rather than noise crossings or subtraction artefacts.

### 9.1 Waveform extraction

For each event frame $t_e$ we extract an asymmetric window around the event,

$$
\mathcal{W}(t_e) \;=\; \text{trace}\bigl[t_e - L/4,\; t_e + 3L/4\bigr),
\qquad L \;=\; \lceil 5 \tau f_s \rceil,
$$

where $L$ is the `gate3_waveform_window_tau_multiple = 5.0`-multiple of $\tau f_s$
(`types.py:245`). For $\tau = 1$ s and $f_s = 30$ Hz, $L = 150$ frames. A baseline
computed as the mean of the first 10 % of the window is subtracted to remove
slow drift (`gate3.py:36-60`).

### 9.2 Template $R^2$ fit

For each template $w_k$ in the bank we align its peak to the waveform's peak by
index shift and fit an amplitude by least squares,

$$
\hat A_k \;=\; \frac{w_k^\top \mathcal{W}}{w_k^\top w_k},
$$

keeping only positive amplitudes. The coefficient of determination is

$$
R^2_k \;=\; 1 - \frac{\lVert \mathcal{W} - \hat A_k\, w_k\rVert_2^2}
                      {\lVert \mathcal{W} - \bar{\mathcal{W}}\rVert_2^2},
$$

computed on the overlapping portion of the peak-aligned waveform and template
(`_waveform_r2`, `gate3.py:63-113`). The best template is
$k^\star = \arg\max_k R^2_k$, and the best $R^2$ across all events of a cluster is
used for the gate decision. Thresholds:

| criterion | threshold |
|:---|:---|
| single-event candidate | $R^2_{k^\star} \ge$ `gate3_min_waveform_r2_single_event = 0.5` (`types.py:241`) |
| multi-event candidate | $R^2_{k^\star} \ge$ `gate3_min_waveform_r2 = 0.6` (`types.py:240`) |
| marginal flag band | $[\min_r2,\; \min_r2 + 0.1)$ → flag rather than reject |

### 9.3 Rise/decay asymmetry

Calcium transients rise fast and decay slow. We compute

$$
\rho \;=\; \frac{t_{90} - t_{10}}{t_{37} - t_\text{peak}},
$$

where $t_{10}, t_{90}, t_{37}$ are the first frame indices at which the waveform
crosses 10 %, 90 %, and 37 % of peak amplitude (rise measured before the peak,
decay after). Reject if $\rho \ge$ `gate3_max_rise_decay_ratio = 0.5`
(`types.py:242`; `_rise_decay_ratio`, `gate3.py:116-154`). Slow-rise / fast-decay
patterns (large $\rho$) indicate noise, astrocyte-slow events, or back-propagated
motion artefacts.

### 9.4 Anti-correlation cascade defence

Imperfect Stage 1 or 2 subtraction can leave traces that anti-correlate with the
surviving pixels of a partially-subtracted neighbour. For each candidate we find
all prior-stage ROIs with centroid within `gate2_spatial_radius = 20 px` (the
same radius used by Gate 2), compute Pearson correlations against their traces,
and reject if the minimum is $\le$ `gate3_anticorr_threshold = -0.5`
(`types.py:243`; `gate3.py:259-272`).

### 9.5 Morphology and confidence grading

Solidity of the disk mask, computed via `regionprops`, must meet
`gate3_min_solidity = 0.5` (`types.py:244`). Confidence is graded by event count
(`gate3.py:274-280`):

| `event_count` | confidence |
|:---|:---|
| 1 | low |
| 2–5 | moderate |
| ≥ 6 | high |

### 9.6 Decision

Any failure triggers reject. If all checks pass but $R^2_{k^\star}$ is within the
0.1-marginal band, the outcome is flag (with the event-count confidence);
otherwise accept (`gate3.py:282-294`).

---

## 10. Stage 4 — tonic-neuron search

Implementation: `roigbiv/pipeline/stage4.py`.

Tonic neurons fire quasi-continuously (2–5+ Hz) and their individual transients
pile up into a nearly-constant fluorescence level under $\tau \approx 1$ s GCaMP6s
kinetics. They have low temporal variance (Suite2p misses them), no discrete
transients (Stage 3 misses them), and are partially absorbed into the low-rank $L$
component during the foundation step. Stage 4 detects them via local
spatial-temporal correlation contrast on the residual $S_3$.

### 10.1 Per-pixel linear detrend

A vectorised ordinary-least-squares detrend removes residual drift and
photobleaching artefacts per pixel (`detrend_to_memmap`, `stage4.py:56-87`):

$$
\tilde S_3(p, t) \;=\; S_3(p, t) \;-\; \bigl(\alpha_p + \beta_p t\bigr),
\qquad \beta_p = \frac{\sum_t (t - \bar t) S_3(p, t)}{\sum_t (t - \bar t)^2},
\quad \alpha_p = \overline{S_3(p, \cdot)}.
$$

The intercept and slope are computed once and the detrended movie is written to a
memory-map for reuse across bandpass windows. Chunking in space
(`stage4_pixel_chunk_rows = 16`, `types.py:263`) bounds RAM independently of $T$.

### 10.2 Zero-phase Butterworth bandpass at three windows

Three bandpass windows isolate different tonic-firing rates
(`bandpass_windows`, `types.py:248-252`):

| window | passband (Hz) | targets |
|:---|:---|:---|
| fast | 0.5–2.0 | 3–5 Hz firing |
| medium | 0.1–1.0 | 1–3 Hz firing |
| slow | 0.05–0.5 | < 1 Hz firing and slow modulation |

Each window's filter is an order-4 zero-phase Butterworth
(`bandpass_order = 4`, `types.py:253`) realised as a second-order-sections cascade
applied with `scipy.signal.sosfiltfilt` (forward-then-reverse)
(`bandpass_to_memmap`, `stage4.py:94-126`). Zero-phase filtering preserves the
temporal alignment needed by the downstream correlation step; consequently the
chunking is spatial, not temporal. A stability check skips any window whose
lower cutoff yields a required minimum recording length $> T / f_s$
($5 / f_\text{low}$; `stage4.py:388-394`).

### 10.3 Temporal compression

After filtering we compress the $(T, H, W)$ movie to a $(H \cdot W, D)$ matrix via
binned temporal averaging with $D = \min($`n_svd_components_stage4 = 300`, $T)$
bins (`types.py:254`; `compress_temporal`, `stage4.py:133-164`). Because every
pixel shares the same bin edges, pairwise correlations are preserved exactly — it
is an orthogonal projection onto the constant-per-bin basis. This reduces the
subsequent correlation computation from $O(T)$ to $O(D)$ per pixel pair.

### 10.4 Correlation-contrast map via spatial convolution

For per-pixel $z$-scored vectors $z_p \in \mathbb{R}^D$ (`stage4.py:225-230`) and
neighbourhood $\mathcal{N}(p)$, the mean Pearson correlation over the neighbourhood
is

$$
\overline r(p; \mathcal{N}) \;=\; \frac{1}{|\mathcal{N}(p)|}
\sum_{q \in \mathcal{N}(p)} \frac{1}{D}\, z_p^\top z_q
\;=\; \frac{1}{D}\,z_p^\top \Bigl(\tfrac{1}{|\mathcal{N}(p)|}\! \sum_q z_q\Bigr).
$$

The inner sum is a spatial convolution of $z$ with a uniform disk kernel. Let

- $K_\text{in}$ = disk of radius `corr_neighbor_radius_inner = 6` with the central
  pixel excluded (`types.py:255`), normalised to unit $\ell_1$ norm,
- $K_\text{out}^\text{full}$ = disk of radius
  `corr_neighbor_radius_outer = 15` (`types.py:256`), and
- $K_\text{in}^\text{full}$ = disk of radius 6 with centre included.

The annulus mean is

$$
K_\text{ann} \;=\;
\frac{|K_\text{out}^\text{full}|\, K_\text{out}^\text{full}
      \;-\; |K_\text{in}^\text{full}|\, K_\text{in}^\text{full}}
     {|K_\text{out}^\text{full}| - |K_\text{in}^\text{full}|}.
$$

Accumulating over the compressed dimension $D$,

$$
C(p) \;=\; \underbrace{\frac{1}{D}\sum_d z_{p,d}\,(z_{\cdot,d} * K_\text{in})(p)}_{\text{local}} \;-\;
\underbrace{\frac{1}{D}\sum_d z_{p,d}\,(z_{\cdot,d} * K_\text{ann})(p)}_{\text{annular}}
\;=\; \overline r_\text{in}(p) - \overline r_\text{ann}(p)
$$

(`compute_correlation_contrast`, `stage4.py:191-259`). The interpretation: somata
exhibit high inner correlation and low outer correlation (a local-coherent
micro-region); neuropil exhibits broad spatial correlation at both radii. The
convolution-based formulation reduces an otherwise $O(N_\text{pix}^2)$ all-pairs
correlation to $O(D \cdot N_\text{pix} \cdot |\text{kernel}|)$ per radius.

### 10.5 Thresholding, morphological filter, cross-window merge

The contrast map is thresholded at
`corr_contrast_threshold = 0.10` (`types.py:257`) and labelled via connected
components (`scipy.ndimage.label`; `cluster_contrast_map`, `stage4.py:266-306`).
Each component is kept if all of

$$
a \in [80, 350], \quad s \ge 0.6, \quad e \le 0.85
$$

hold (`types.py:258-261`), where $a$ is the pixel area, $s$ is regionprops
solidity, $e$ is eccentricity.

Candidates from the three windows are pooled and merged greedily in descending
correlation-contrast order by IoU with the threshold
`stage4_iou_merge_threshold = 0.3` (`types.py:262`;
`merge_across_windows`, `stage4.py:321-358`). Each winner records the set of
windows in which it was detected (`bandpass_windows_detected` feature), which
contributes evidence at review.

### 10.6 Trace extraction and parallel execution

For every surviving candidate the trace is extracted from $S_3$ using the
matrix-vector extractor (§6.3). The three bandpass windows are processed either
serially or with a thread pool of up to `stage4_n_workers = 3` threads
(`types.py:264`); `sosfiltfilt` and `ndi_convolve` release the GIL so parallelism
is real. BLAS threads per worker are capped to `cpu_count // n_workers` via
`threadpool_limits` to prevent oversubscription (`stage4.py:482-497`).

Candidate ROIs receive `source_stage=4`, `confidence="requires_review"` (locked by
design), and provisional `gate_outcome="flag"` pending Gate 4.

---

## 11. Gate 4 — correlation-contrast validation

Implementation: `roigbiv/pipeline/gate4.py`.

Gate 4 has no accept tier. Every candidate that passes all six checks below
receives `gate_outcome="flag"` and `confidence="requires_review"`; any failure
triggers reject. This reflects a deliberate epistemic-humility stance on tonic
detection: the automated pipeline cannot confirm tonic candidates with the same
confidence as Stages 1–3, and human review of the bandpass trace plus correlation
map is mandatory.

The six checks (`gate4.py:107-172`):

1. **Correlation contrast.** $C \ge$ `gate4_min_corr_contrast = 0.10`
   (`types.py:270`).
2. **Eccentricity.** $e \le$ `stage4_max_eccentricity = 0.85`.
3. **Solidity.** $s \ge$ `stage4_min_solidity = 0.60`.
4. **Motion correlation.** Pearson correlations of the raw ROI trace against the
   per-frame rigid displacement fields $(x_\text{off}, y_\text{off})$ from the
   Suite2p `ops`:
   $\max\bigl(\lvert r_x\rvert, \lvert r_y\rvert\bigr) <$
   `gate4_max_motion_corr = 0.3` (`types.py:271`). Sub-pixel motion leaves
   fluctuating ring artefacts at soma boundaries that mimic tonic signals; the
   raw trace is used (not bandpass) because motion power spreads across
   frequencies.
5. **Cascade anti-correlation.** For prior-stage ROIs with centroid within
   `gate4_spatial_radius = 20 px` (`types.py:274`), the minimum Pearson
   correlation against the candidate trace must exceed
   `gate4_anticorr_threshold = -0.5` (`types.py:272`).
6. **Intensity floor on mean_M.** $\mu_{M}(\text{mask}) \ge$
   percentile($\mathrm{mean\_M}$, `gate4_min_mean_intensity_pct = 25`)
   (`types.py:273`). The spec originally prescribed `mean_S`; we substitute
   `mean_M` because `mean_S ≈ 0` under SVD L+S (`foundation.py:513-515`;
   `gate4.py:14-24`) making a percentile filter on it meaningless.

---

## 12. Quality-control features, trace extraction, classification

Implementation: `roigbiv/pipeline/qc_features.py`, `roigbiv/pipeline/classify.py`,
plus the neuropil and dF/F utilities invoked by `run.py`.

### 12.1 Spatial QC features

For every non-rejected ROI (`compute_spatial_features`, `qc_features.py:98-128`)
we compute:

- **Boundary gradient.** $\mu_{p \in \partial \text{mask}} \lVert \nabla \text{mean\_S}(p) \rVert_2$
  with $\partial \text{mask}$ the set difference of the mask and its 1-pixel
  erosion (`_boundary_gradient`, `qc_features.py:41-54`). Sharp somata have high
  boundary gradient.
- **Spatial blur (radial FWHM).** The intensity profile is radially binned
  around the centroid out to twice the effective radius; the FWHM is twice the
  smallest radius at which the profile drops below half its peak value
  (`_spatial_blur_fwhm`, `qc_features.py:57-95`). Ghost cells from out-of-focus
  volumes have broader FWHM than in-focus somata.
- **FOV distance.** $\lVert (\bar y, \bar x) - (H/2, W/2)\rVert_2$. Used to
  contextualise optical-aberration-induced failures near GRIN-lens edges.

### 12.2 Temporal QC features

Each ROI receives a raw trace $F_i(t)$ and neuropil-corrected trace
$F^c_i(t) = F_i(t) - \alpha F^{np}_i(t)$, with neuropil coefficient
`neuropil_coeff = 0.7` (`types.py:187`) and the neuropil trace extracted from an
annulus of inner buffer 2 px and outer radius 15 px
(`neuropil_inner_buffer=2`, `neuropil_outer_radius=15`; `types.py:188-189`).

Temporal features (`compute_temporal_features`, `qc_features.py:206-267`):

| feature | definition |
|:---|:---|
| `std` | $\sigma(F^c)$ |
| `skew` | `scipy.stats.skew(F^c, bias=False)` |
| `mean_fluorescence` | $\mu(F)$ |
| `noise_floor` | $\text{MAD}(F^c) / 0.6745$ |
| `snr` | $(\max F^c - \mu F^c) / \text{noise\_floor}$ |
| `n_transients` | FFT matched-filter peak count using the first template, peak-height 3σ, min distance $\lceil 2 \tau f_s\rceil$ |
| `trace_bandpass` | zero-phase order-4 Butterworth bandpass 0.05–2.0 Hz applied to $F^c$ |
| `bp_std` | $\sigma$ of `trace_bandpass` |
| `bp_power_ratio` | Welch-PSD power in [0.05, 2] Hz divided by total power |
| `autocorr_tau` | first lag at which the FFT autocorrelation of `trace_bandpass` falls below $e^{-1}$, expressed in seconds |

The bandpass trace is the *primary evidence* for tonic-ROI review: tonic transients
pile up in the raw trace as slow fluctuations that the 0.05–2 Hz band cleanly
isolates, whereas the raw trace looks nearly flat.

### 12.3 dF/F and OASIS deconvolution

$\Delta F / F_0$ is computed with a sliding-window baseline of length
`baseline_window_s = 60 s` (`types.py:190`; `tonic_baseline_window_s = 120 s` for
tonic-classified ROIs) and `baseline_percentile = 10`. Spike deconvolution uses
the OASIS algorithm [Friedrich et al. 2017] configured for GCaMP6s kinetics
($\tau = 1.0$ s).

### 12.4 Provenance feature

Cross-stage matches are not tracked during detection; `n_stages_detected` is
computed post hoc as the count of distinct `source_stage` values held by ROIs
overlapping the target with IoU > 0.3 (`compute_provenance_features`,
`qc_features.py:282-298`).

### 12.5 Activity-type classification

Classification is a rule-based decision tree evaluated top to bottom
(`classify_activity_type`, `classify.py:32-76`). Population medians $\tilde F$ and
$\tilde \sigma$ of `mean_fluorescence` and `std` are computed once per FOV and
drive the tonic population criterion.

| class | condition (first match wins) |
|:---|:---|
| phasic | `n_transients` ≥ 5 AND `skew` > 0.5 |
| sparse | 1 ≤ `n_transients` < 5 AND `skew` > 0.3 |
| tonic | `bp_std` > 2.0 × max(`noise_floor`, $10^{-12}$) AND `skew` ≤ 0.5 AND (`source_stage` = 4 OR (`mean_F` > $\tilde F$ AND `std` < $\tilde \sigma$)) |
| silent | `n_transients` = 0 AND `bp_std` < `noise_floor` AND (nuclear_shadow_score > 0 OR solidity > 0.7) |
| ambiguous | fallback |

Thresholds live in `types.py:195-199`: `phasic_min_transients=5`,
`phasic_min_skew=0.5`, `sparse_min_transients=1`, `sparse_min_skew=0.3`,
`tonic_bp_std_factor=2.0`. The silent tier is retained only when morphology is
convincing (positive nuclear shadow or solid mask), which keeps cells that may
fire in a different session while rejecting flat traces at fragmented
low-contrast locations.

---

## 13. Human-in-the-loop review package

Implementation: `roigbiv/pipeline/hitl.py`.

ROIGBIV does not include a GUI; instead it exports a prioritised queue and
per-ROI evidence files that drop into the Cellpose GUI for manual correction.

### 13.1 Four-tier priority queue

The queue is assembled from all non-rejected ROIs (`build_review_queue`,
`hitl.py:58-106`):

| priority | criterion | sort key |
|:---|:---|:---|
| 1 | `source_stage = 4` AND `confidence = requires_review` | `corr_contrast` ascending (most uncertain first) |
| 2 | any stage, `confidence = moderate` | `source_stage` descending |
| 3 | `source_stage = 3` AND (`event_count = 1` OR `confidence = low`) | `label_id` ascending |
| 4 | remaining non-rejected | `label_id` ascending |

### 13.2 Exported artefacts

Per-FOV (`export_hitl_package`, `hitl.py:141-253`):

- `review_queue.json` — the priority list with per-entry reasoning strings.
- `merged_masks.tif` — `uint16` label image of every non-rejected ROI, with label
  IDs preserved 1:1 with the row index in downstream trace arrays.
- `hitl/stage4/{label_id}/`
  - `bandpass_trace.npy` — **primary evidence** for tonic review
    (`trace_bandpass` from §12.2).
  - `corr_contrast_crop.npy` — a $61 \times 61$ crop of the best-scoring
    correlation-contrast map centred on the ROI.
  - `info.json` — metadata, best-scoring bandpass window, activity type.
- `hitl/stage3/{label_id}/event_frame_indices.json` — frames ±10 around each
  detected event, for targeted video review of single-event candidates.
- `hitl_staging/images/{stem}.tif`, `hitl_staging/masks/{stem}_seg.tif` —
  Cellpose-GUI-ready layout for training-data correction.

---

## 14. Pipeline orchestration and output layout

Implementation: `roigbiv/pipeline/run.py`, `roigbiv/pipeline/outputs.py`,
`roigbiv/pipeline/batch.py`.

### 14.1 Control flow

`run_pipeline` at `run.py` executes stages sequentially, threading one `FOVData`
container through them. GPU-heavy sections (Cellpose inference, Suite2p detection,
Stage 3 FFT, subtraction ridge solve and NNLS) are wrapped in a `_gpu_section`
context that acquires a shared `multiprocessing.Manager().Lock()` when the
pipeline is invoked from `batch.py`, and is a zero-cost no-op otherwise
(`run.py:32-40`). Between stages, the subtraction engine of §5 deletes the
predecessor residual memory-map once validation and NNLS complete, keeping peak
disk usage to one residual at a time.

### 14.2 Monotonicity check

After all stages complete, `print_detection_summary` (`run.py:61`–…) verifies that
detected counts decrease across stages — a soft sanity check that no stage is
re-discovering prior detections. Violations are recorded as warnings in
`pipeline_log.json` rather than raising errors.

### 14.3 Output layout

Per-FOV output directory (default
`inference/pipeline/{stem}/`, `run.py:43-58`) contains:

```
suite2p/plane0/{ops.npy, data.bin, stat.npy, iscell.npy, ...}
svd_factors.npz
residual_S.dat                         (foundation output, float32 memmap)
residual_S1.dat  /  residual_S2.dat  /  residual_S3.dat   (per-stage residuals)
residual_S*.meta.json                  (shape + dtype)
subtraction_report_residual_S*.json    (post-subtraction validation report)
motion_trace.npz                       (xoff, yoff, fs)
summary/{mean_M, mean_S, max_S, std_S, vcorr_S, mean_L, dog_map}.tif

stage1/
  stage1_masks.tif                     (uint16 label image, accept+flag)
  stage1_probs.tif                     (cellprob map)
  stage1_report.json
stage2/{stage2_masks.tif, stage2_report.json}
stage3/
  stage3_masks.tif
  stage3_events.npy                    (per-cluster pickled event list)
  stage3_report.json
stage4/
  stage4_masks.tif
  corr_contrast_fast.tif / medium.tif / slow.tif
  stage4_corr_contrast.npy
  stage4_report.json

hitl/                                  (see §13)
hitl_staging/

F.npy               (N × T raw traces)
Fneu.npy            (N × T neuropil)
F_corrected.npy     (N × T neuropil-subtracted)
dFF.npy             (N × T ΔF/F)
spks.npy            (N × T OASIS deconvolved)
F_bandpass.npy      (N_tonic × T, tonic ROIs only)
F_bandpass_index.npy (label_id → row mapping for F_bandpass)
merged_masks.tif    (uint16 final label image)
roi_metadata.json   (per-ROI full metadata)
pipeline_log.json   (execution summary; see §14.4)
review_queue.json
```

Each `stage{1..4}_report.json` has the schema

```json
{
  "detected": <int>,
  "accepted": <int>,
  "flagged": <int>,
  "rejected": <int>,
  "rois": [ <ROI.to_serializable()>, ... ]
}
```

where `ROI.to_serializable` returns `label_id`, `source_stage`, `confidence`,
`gate_outcome`, spatial features (area, solidity, eccentricity,
nuclear_shadow_score, soma_surround_contrast), per-stage scores
(cellpose_prob, iscell_prob, event_count, corr_contrast), `activity_type`,
`gate_reasons`, and a stage-specific `features` dict (`types.py:54-73`).

### 14.4 `pipeline_log.json` schema

Written by `save_pipeline_outputs` (`outputs.py:42-147`):

```json
{
  "input": "<absolute path to input TIFF>",
  "output_dir": "<absolute output path>",
  "fov_name": "<input stem with _mc stripped>",
  "timestamp": "<ISO-8601 UTC>",
  "shape": [T, H, W],
  "k_background": 30,
  "config": { <PipelineConfig.summary_for_log()> },
  "stage_counts": {
    "stage1": { "detected": ..., "accepted": ..., "flagged": ..., "rejected": ... },
    "stage2": { ... },
    "stage3": { ... },
    "stage4": { ... }
  },
  "subtraction": {
    "stage1": { "n_rois": ..., "n_passed": ..., "n_failed": ... },
    "stage2": { ... },
    "stage3": { ... }
  },
  "activity_type_counts": {
    "phasic": ..., "sparse": ..., "tonic": ..., "silent": ..., "ambiguous": ...
  },
  "overlap_groups": { "n_groups": ..., "group_sizes": [...], "groups": [[...]] },
  "total_rois": ...,
  "review_queue_summary": {
    "total": ..., "priority_1": ..., "priority_2": ..., "priority_3": ..., "priority_4": ...
  },
  "timings_s": { "foundation_s": ..., "stage1_detect_s": ..., ... },
  "warnings": [ ... ]
}
```

### 14.5 Batch runner and GPU lock

`batch.py` runs ≥ 2 FOVs concurrently via `ProcessPoolExecutor` with the `spawn`
start method (`batch.py:29-61`). `spawn` is mandatory because forking a process
that has already initialised a CUDA context deadlocks on the first CUDA call in
the child. A `multiprocessing.Manager().Lock()` is passed to every worker via the
pool initialiser and acquired around all GPU-heavy phases; CPU-only phases
(foundation summary images, Stage 4 bandpass+convolution, trace extraction,
QC features, dF/F, OASIS) overlap freely. The hard cap is
`MAX_BATCH_WORKERS = 2` (`batch.py:43`), which saturates the RTX 4060 8 GB
GPU — adding more workers cannot reduce wall-time because the GPU lock
serialises the GPU phases.

Worker stdout is redirected through a `_QueuedStdout` shim that pushes completed
lines, tagged with the FOV index, onto a shared `multiprocessing.Queue`
(`batch.py:64-88`). A main-thread pump drains the queue into a log callback for
the caller.

---

## 15. Cross-session FOV and cell registry

Implementation: `roigbiv/registry/`.

The registry identifies whether a newly-pipelined FOV is a re-recording of a
previously-seen FOV and, if so, which cells in the two sessions correspond.
Decisions are one of `hash_match`, `auto_match`, `review`, or `new_fov`,
persisted to `registry_match.json` alongside the pipeline outputs.

> This section can be relocated to supplementary methods if the manuscript's
> focus is narrowly on detection.

### 15.1 Fingerprinting

Implementation: `roigbiv/registry/fingerprint.py`.

Each session's fingerprint (`FINGERPRINT_VERSION = 3`, `fingerprint.py:34`) is a
deterministic SHA-256 over a canonical tuple representation of the merged mask:

$$
\text{hash\_input} \;=\;
\mathtt{b"roigbiv\text{-}v3;shape="} \;\|\; [H, W]_{\mathtt{int64}} \;\|\;
\mathtt{b";rois="} \;\|\;
\operatorname{sort}_{\text{label\_id}}
\begin{bmatrix} \text{label\_id} & y & x & a \end{bmatrix}_{\mathtt{int64}}
$$

where $(y, x)$ is the integer-pixel centroid and $a$ is the pixel area per ROI
(`compute_fingerprint`, `fingerprint.py:56-115`). The mean projection is stored as
context but is *not* part of the hash. Identical fingerprints indicate identical
mask geometry and enable an $O(1)$ shortcut to a full re-run match.

### 15.2 ROICaT alignment, embedding, and clustering

Implementation: `roigbiv/registry/roicat_adapter.py`.

Matching between fingerprint-distinct sessions relies on ROICaT
[ROICaT, bioRxiv 2023/2024]. The pipeline builds a `Data_roicat` container from
the padded `(H, W)` mean projections and sparse CSR footprint matrices of the
candidate and query sessions, then applies:

1. **Geometric alignment** — default RoMa [Edstedt et al. 2024], an affine
   deep-learning alignment method; alternatives `PhaseCorrelation` and
   `ECC_cv2` are available via environment variable. When the RoMa CUDA
   `local_corr` extension is unavailable the adapter patches RoMa to use a
   PyTorch-native correlation (2–3× slower; numerically equivalent)
   (`roicat_adapter.py:61-113`). The alignment blends mean projection and
   footprint-density via `roi_FOV_mixing_factor = 0.5`; the template is the
   middle session.
2. **Alignment-quality proxy.** For deep-learning methods the reported score is
   the RANSAC inlier rate; for geometric methods it is the post-warp
   Pearson correlation against the template, clipped to $[0, 1]$.
3. **ROI blurrer** — `kernel_halfWidth = 2`, a mild spatial smoothing that
   regularises subpixel misalignment before embedding.
4. **ROInet embedding.** Each ROI crop is passed through the public ROInet
   model (latent pass, weights fetched on-demand and cached under
   `~/.cache/roigbiv/roinet`), producing a per-ROI latent vector.
5. **Scattering wavelet transform.** An additional ROICaT-internal wavelet
   feature, batch size 100.
6. **Similarity graph.** Three feature types — spatial footprints, ROInet
   latents, scattering-wavelet latents — are combined with fixed power weights
   (SF: 1.0, NN: 0.5, SWT: 0.5). The centroid-kNN graph uses
   $k_\text{max} = \min(n_\text{sessions} \cdot 100,\, n_\text{rois} - 1)$.
7. **Sequential-Hungarian clustering** [ROICaT] with cost threshold
   `sequential_hungarian_thresh_cost = 0.6`. Singletons are labelled $-1$.

Output is a `ClusterResult` dataclass carrying per-ROI cluster labels, a boolean
session-membership matrix, the alignment inlier rate, and ROICaT quality metrics
(`cluster_labels_unique`, `cluster_intra_means`).

### 15.3 Calibrated logistic posterior

Implementation: `roigbiv/registry/calibration.py`, `roigbiv/registry/match.py`.

From a `ClusterResult` a four-feature vector is derived
(`compute_fov_features`, `match.py:105-169`):

- `n_shared_clusters`: number of clusters containing at least one query ROI
  AND at least one candidate ROI.
- `fraction_query_clustered`: the share of query ROIs that landed in a shared
  cluster.
- `alignment_quality`: the ROICaT alignment proxy $\in [0, 1]$.
- `mean_cluster_cohesion`: $1 - \mu(\text{cluster\_intra\_means})$ over shared
  clusters, clipped to $[0, 1]$; defaults to 0.5 in degenerate cases.

A single-layer logistic produces the same-FOV posterior:

$$
z = \beta_0
+ \beta_1 n_\text{shared clusters}
+ \beta_2 f_\text{query clustered}
+ \beta_3 q_\text{alignment}
+ \beta_4 c_\text{cohesion},
\qquad
p_\text{same FOV} = \sigma(z).
$$

Implementation uses the numerically-stable piecewise sigmoid at
`calibration.py:171-176`. Prior to a labelled pair set being collected, the
coefficients are hand priors (`DEFAULT_FOV_COEFS`, `calibration.py:37`):

| coefficient | value |
|:---|---:|
| $\beta_0$ (intercept) | $-4.0$ |
| $\beta_1$ (shared clusters) | $0.05$ |
| $\beta_2$ (fraction query clustered) | $3.0$ |
| $\beta_3$ (alignment quality) | $4.0$ |
| $\beta_4$ (cluster cohesion) | $3.0$ |

These priors yield $p \approx 0.9$ when ~60 % of query ROIs join a shared cluster
AND alignment quality ≥ 0.5 AND cohesion ≥ 0.5. The priors can be replaced at any
time by calling `fit_from_labels` with a labelled $(f, y)$ set; the function fits
a scikit-learn `LogisticRegression(max_iter=1000)` (`calibration.py:131-168`).
Coefficients and a `trained` flag are persisted as JSON at
`inference/registry_calibration.json`.

### 15.4 Decision thresholds

`match.py:31-32`:

| decision | condition |
|:---|:---|
| `auto_match` | $p \ge 0.9$ |
| `review` | $0.5 \le p < 0.9$ (no DB write; flagged for manual review) |
| `reject` / `new_fov` | $p < 0.5$ (new FOV minted) |

The orchestrator (`roigbiv/registry/orchestrator.py::register_or_match`)
short-circuits to `hash_match` on fingerprint collision, then iterates over
candidate FOVs scoped by `(animal_id, region)` parsed from the filename
(`filename.py::parse_filename_metadata`), keeping the highest-posterior
candidate, and branches into the four outcomes above.

### 15.5 Storage

Implementation: `roigbiv/registry/models.py`,
`roigbiv/registry/store/sqlalchemy_store.py`, `roigbiv/registry/blob/local.py`.

The relational schema (SQLAlchemy ORM, four tables):

- **FOV**
  `(fov_id PK, fingerprint_hash UNIQUE, animal_id, region, mean_m_uri,
  centroid_table_uri, created_at, latest_session_date, fingerprint_version,
  fov_embedding_uri, roi_embeddings_uri)`. Indexed on `(animal_id, region)` and
  on `fingerprint_hash`.
- **Session**
  `(session_id PK, fov_id FK, session_date, output_dir, fov_sim, fov_posterior,
  n_matched, n_new, n_missing, created_at, cluster_labels_uri)`. Indexed on
  `(fov_id, session_date)`.
- **Cell**
  `(global_cell_id PK, fov_id FK, first_seen_session_id, morphology_summary JSON)`.
- **CellObservation**
  `(observation_id PK, global_cell_id FK, session_id FK, local_label_id,
  match_score, cluster_label)` with `UNIQUE(session_id, local_label_id)`.

Blobs are stored under `inference/fingerprints/{fov_id}/` in a local backend
(`LocalBlobStore`), with `merged_masks.npy`, `mean_M.npy`, `centroids.npy`, and
per-session `sessions/{session_id}/cluster_labels.npy`. Alembic migrations at
`roigbiv/registry/migrations/versions/` add the v2 embedding URIs
(`0002_embeddings.py`) and the v3 cluster-label URIs and cluster-label column
(`0003_roicat.py`) in a backward-compatible manner.

Configuration is env-driven; key variables and defaults
(`roigbiv/registry/config.py`):

| variable | default | purpose |
|:---|:---|:---|
| `ROIGBIV_REGISTRY_DSN` | `sqlite:///inference/registry.db` | database URL |
| `ROIGBIV_BLOB_ROOT` | `inference/fingerprints` | blob store root |
| `ROIGBIV_ROICAT_DEVICE` | auto (cuda else cpu) | PyTorch device |
| `ROIGBIV_FOV_ACCEPT_THRESHOLD` | `0.9` | auto-match cutoff |
| `ROIGBIV_FOV_REVIEW_THRESHOLD` | `0.5` | review-band floor |
| `ROIGBIV_CALIBRATION_PATH` | `inference/registry_calibration.json` | logistic coefficients |

### 15.6 Backfill and CLI

`roigbiv-registry` exposes `list`, `show`, `match`, `track`, `backfill`, and
`migrate` (`roigbiv/cli_registry.py`). The `backfill` command walks a directory
tree of pipeline outputs in chronological order (using filename-parsed dates
where available, falling back to `pipeline_log.json` timestamps) and registers
every FOV. Idempotency is guaranteed by the fingerprint pre-filter: re-runs of
identical outputs produce identical fingerprints and take the `hash_match`
branch without writing new rows.

---

## 16. Parameter reference (master table)

Every parameter below is a field of `PipelineConfig`
(`roigbiv/pipeline/types.py:142-290`). All CLI flags override these defaults.

### Foundation

| parameter | default | `types.py` line |
|:---|:---|:---|
| `k_background` | 30 | 149 |
| `n_svd` | 200 | 150 |
| `batch_size` | 500 | 151 |
| `nonrigid` | True | 152 |
| `do_registration` | False (True when input lacks `_mc`) | 153 |
| `fs` | 30.0 Hz (CLI-required) | 154 |
| `tau` | 1.0 s (GCaMP6s) | 155 |
| `svd_bin_frames` | 5 000 | 156 |
| `reconstruct_chunk` | 500 frames | 157 |

### Stage 1 (Cellpose)

| parameter | default | line |
|:---|:---|:---|
| `cellpose_model` | `models/deployed/current_model` | 160 |
| `diameter` | 12 px | 161 |
| `cellprob_threshold` | $-2.0$ | 162 |
| `flow_threshold` | 0.6 | 163 |
| `channels` | (1, 2) | 164 |
| `tile_norm_blocksize` | 128 px | 165 |
| `use_denoise` | True | 166 |

### Gate 1

| parameter | default | line |
|:---|:---|:---|
| `min_area` / `max_area` | 80 / 600 px | 169–170 |
| `min_solidity` | 0.55 | 171 |
| `max_eccentricity` | 0.90 | 172 |
| `min_contrast` | 0.10 | 173 |
| `flag_area_margin` | 20 px | 175 |
| `flag_solidity_margin` | 0.05 | 176 |
| `flag_eccentricity_margin` | 0.03 | 177 |
| `flag_contrast_margin` | 0.03 | 178 |
| `dog_strong_negative_percentile` | 10.0 | 180 |
| `annulus_inner_buffer` / `annulus_outer_radius` | 2 / 15 px | 183–184 |

### Subtraction

| parameter | default | line |
|:---|:---|:---|
| `subtract_chunk_frames` | 2 000 | 202 |
| `subtract_ridge_lambda_scale` | $10^{-6}$ | 203 |
| `subtract_anticorr_threshold` | $-0.3$ | 204 |
| `subtract_anticorr_failure_fraction` | 0.10 | 205 |
| `subtract_nnls_fallback_max_rois` | 30 | 206 |

### Stage 2 / Gate 2

| parameter | default | line |
|:---|:---|:---|
| `iscell_threshold` | 0.3 | 210 |
| `gate2_iou_threshold` | 0.3 | 213 |
| `gate2_max_correlation` | 0.7 | 214 |
| `gate2_anticorr_threshold` | $-0.5$ | 215 |
| `gate2_spatial_radius` | 20 px | 216 |
| `gate2_min_area` / `gate2_max_area` | 60 / 400 px | 217–218 |
| `gate2_min_solidity` | 0.4 | 219 |
| `gate2_near_distance` | 5 px | 220 |
| `gate2_near_corr_threshold` | 0.5 | 221 |
| `gate2_flag_corr_threshold` | 0.5 | 222 |

### Stage 3 / Gate 3

| parameter | default | line |
|:---|:---|:---|
| `template_threshold` | 6.0 σ | 230 |
| `spatial_pool_radius` | 8 px | 231 |
| `cluster_distance` | 12 px | 233 |
| `min_event_separation` | 2.0 s | 234 |
| `stage3_pixel_chunk_rows` | 8 | 235 |
| `stage3_max_events` | 2 000 000 | 237 |
| `gate3_min_waveform_r2` | 0.6 | 240 |
| `gate3_min_waveform_r2_single_event` | 0.5 | 241 |
| `gate3_max_rise_decay_ratio` | 0.5 | 242 |
| `gate3_anticorr_threshold` | $-0.5$ | 243 |
| `gate3_min_solidity` | 0.5 | 244 |
| `gate3_waveform_window_tau_multiple` | 5.0 | 245 |

### Stage 4 / Gate 4

| parameter | default | line |
|:---|:---|:---|
| `bandpass_windows` | {fast (0.5–2.0), medium (0.1–1.0), slow (0.05–0.5)} Hz | 248–252 |
| `bandpass_order` | 4 | 253 |
| `n_svd_components_stage4` | 300 | 254 |
| `corr_neighbor_radius_inner` / `_outer` | 6 / 15 px | 255–256 |
| `corr_contrast_threshold` | 0.10 | 257 |
| `stage4_min_area` / `_max_area` | 80 / 350 px | 258–259 |
| `stage4_min_solidity` | 0.60 | 260 |
| `stage4_max_eccentricity` | 0.85 | 261 |
| `stage4_iou_merge_threshold` | 0.3 | 262 |
| `stage4_pixel_chunk_rows` | 16 | 263 |
| `stage4_n_workers` | 3 | 264 |
| `gate4_min_corr_contrast` | 0.10 | 270 |
| `gate4_max_motion_corr` | 0.3 | 271 |
| `gate4_anticorr_threshold` | $-0.5$ | 272 |
| `gate4_min_mean_intensity_pct` | 25 | 273 |
| `gate4_spatial_radius` | 20 px | 274 |

### Classification / neuropil

| parameter | default | line |
|:---|:---|:---|
| `neuropil_coeff` | 0.7 | 187 |
| `neuropil_inner_buffer` / `_outer_radius` | 2 / 15 px | 188–189 |
| `baseline_window_s` | 60.0 s | 190 |
| `baseline_percentile` | 10 | 191 |
| `tonic_baseline_window_s` | 120.0 s | 192 |
| `phasic_min_transients` | 5 | 195 |
| `phasic_min_skew` | 0.5 | 196 |
| `sparse_min_transients` | 1 | 197 |
| `sparse_min_skew` | 0.3 | 198 |
| `tonic_bp_std_factor` | 2.0 | 199 |

### Registry

| parameter | default | location |
|:---|:---|:---|
| `FINGERPRINT_VERSION` | 3 | `fingerprint.py:34` |
| `DEFAULT_FOV_COEFS` | $(-4.0, 0.05, 3.0, 4.0, 3.0)$ | `calibration.py:37` |
| `AUTO_ACCEPT_THRESHOLD` | 0.9 | `match.py:31` |
| `REVIEW_THRESHOLD` | 0.5 | `match.py:32` |
| ROICaT alignment method | `RoMa` | `roicat_adapter.py` |
| ROICaT sequential-Hungarian cost | 0.6 | `roicat_adapter.py` |
| ROICaT ROI-FOV mixing factor | 0.5 | `roicat_adapter.py` |

---

## 17. Bibliography

**Motion correction, Suite2p, OASIS.**

- Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L.F., Dalgleish,
  H., Carandini, M. & Harris, K.D. (2017). Suite2p: beyond 10,000 neurons with
  standard two-photon microscopy. *bioRxiv* 061507. (Registration and SVD-based
  detection used in foundation and Stage 2.)
- Friedrich, J., Zhou, P. & Paninski, L. (2017). Fast online deconvolution of
  calcium imaging data. *PLoS Computational Biology* 13(3): e1005423.
  (OASIS spike deconvolution used in §12.3.)

**Cellpose and image restoration.**

- Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. (2021). Cellpose: a
  generalist algorithm for cellular segmentation. *Nature Methods* 18, 100–106.
- Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own
  model. *Nature Methods* 19, 1634–1641.
  (Image restoration model `denoise_cyto3` used in Stage 1 preprocessing.)

**SVD and background decomposition.**

- Halko, N., Martinsson, P.-G. & Tropp, J.A. (2011). Finding structure with
  randomness: probabilistic algorithms for constructing approximate matrix
  decompositions. *SIAM Review* 53, 217–288. (Randomised-subspace SVD, the
  algorithmic basis of `torch.svd_lowrank`.)
- Candès, E.J., Li, X., Ma, Y. & Wright, J. (2011). Robust principal component
  analysis? *Journal of the ACM* 58, 11:1–11:37. (Conceptual precedent for L+S
  decompositions; the ROIGBIV implementation uses a direct rank-$k$ truncation
  rather than nuclear-norm principal-component pursuit.)

**Calcium indicators.**

- Chen, T.-W., Wardill, T.J., Sun, Y., Pulver, S.R., Renninger, S.L., Baohan, A.,
  Schreiter, E.R., Kerr, R.A., Orger, M.B., Jayaraman, V., Looger, L.L., Svoboda,
  K. & Kim, D.S. (2013). Ultrasensitive fluorescent proteins for imaging
  neuronal activity. *Nature* 499, 295–300. (GCaMP6s kinetics, $\tau \approx 1$ s.)
- Zhang, Y., et al. (2023). Fast and sensitive GCaMP calcium indicators for
  imaging neural populations. *Nature* 615, 884–891. (jGCaMP8f kinetics used in
  the Stage 3 template bank.)

**Consensus ROI matching and registry.**

- Giovannucci, A., Friedrich, J., Gunn, P., Kalfon, J., Brown, B.L.,
  Koay, S.A., Taxidis, J., Najafi, F., Gauthier, J.L., Zhou, P., Khakh, B.S.,
  Tank, D.W., Chklovskii, D.B. & Pnevmatikakis, E.A. (2019). CaImAn: an open
  source tool for scalable calcium imaging data analysis. *eLife* 8, e38173.
  (IoU threshold 0.3 used in Stage 2 and §12.4.)
- Landry, J.R., Nagy, D.G., Pachitariu, M. & Harris, K.D. (2024). ROICaT:
  Region of Interest Classification and Tracking. *bioRxiv* (see the ROICaT
  project on GitHub for the current version). (FOV alignment, ROInet
  embedding, scattering-wavelet transform, and sequential-Hungarian clustering
  used in the registry.)
- Edstedt, J., Bökman, G., Wadenbäck, M. & Felsberg, M. (2024). RoMa: Robust
  Dense Feature Matching. *CVPR 2024*. (Default alignment method in the
  registry.)

**Filter design.**

- Butterworth, S. (1930). On the theory of filter amplifiers. *Experimental
  Wireless and the Wireless Engineer* 7, 536–541. (Butterworth bandpass used in
  Stage 4 and QC features.)
