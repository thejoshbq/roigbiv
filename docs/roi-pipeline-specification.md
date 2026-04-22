# ROI G. Biv — Pipeline Specification

> **Purpose:** Formal specification for the sequential subtractive ROI
> detection pipeline. Defines each stage's inputs, outputs, methods,
> parameters, and validation gates. Supersedes the parallel three-branch
> architecture described in `roi-pipeline-algorithm.md`.
>
> **Last updated:** 2026-04-14
> **Status:** Pre-implementation specification
> **Companion documents:**
> - `roi-pipeline-reference.md` — Lab methods, tool documentation, literature
> - `roi-pipeline-algorithm.md` — Detection challenges (Part 1 remains valid;
>   Part 2 pipeline architecture is superseded by this document)
> - `blindspots-and-mitigations.md` — Known failure modes and mitigation plans

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Foundation: Motion Correction, SVD, Background Separation](#3-foundation)
4. [Stage 1: Spatial Detection](#4-stage-1-spatial-detection)
5. [Source Subtraction Engine](#5-source-subtraction-engine)
6. [Gate 1: Morphological Validation](#6-gate-1-morphological-validation)
7. [Stage 2: Temporal Detection](#7-stage-2-temporal-detection)
8. [Gate 2: Temporal Cross-Validation](#8-gate-2-temporal-cross-validation)
9. [Stage 3: Template Sweep](#9-stage-3-template-sweep)
10. [Gate 3: Waveform Validation](#10-gate-3-waveform-validation)
11. [Stage 4: Tonic Neuron Search](#11-stage-4-tonic-neuron-search)
12. [Gate 4: Correlation Contrast Validation](#12-gate-4-correlation-contrast-validation)
13. [Unified Quality Control & Classification](#13-unified-quality-control--classification)
14. [Human-in-the-Loop Correction & Retraining](#14-human-in-the-loop-correction--retraining)
15. [Phase 2 Extension: Astrocyte Detection](#15-phase-2-extension-astrocyte-detection)
16. [In-House Components Specification](#16-in-house-components-specification)
17. [Data Flow & File Outputs](#17-data-flow--file-outputs)
18. [Parameter Reference](#18-parameter-reference)
19. [Challenge-to-Stage Mapping](#19-challenge-to-stage-mapping)
20. [Implementation Roadmap](#20-implementation-roadmap)

---

## 1. Design Principles

These principles govern every architectural and implementation decision in
the pipeline. When tradeoffs arise, resolve them in this priority order.

### 1.1 Recall First, Precision Through Review

It is better to retain a questionable ROI than to discard a real neuron.
False positives are correctable in HITL review; false negatives are
unrecoverable. Every threshold, gate, and classifier should be tuned to
minimize false negatives at the cost of accepting more false positives.
The HITL step is the precision filter, not the automated pipeline.

### 1.2 Sequential and Subtractive

Each detection stage operates on the residual signal left after previous
stages have found and removed their detections. This ensures that every
stage's computation is additive — no stage wastes effort rediscovering
what a previous stage already found. Each stage receives progressively
cleaner data, optimized for its specific detection target.

### 1.3 Validate Between Stages

Source subtraction is imperfect. Between every detection stage, a
validation gate checks new candidates for artifacts, redundancy, and
biological plausibility before allowing subtraction to proceed. The gates
prevent error propagation through the pipeline.

### 1.4 Provenance Tracking

Every ROI in the final output carries metadata recording which stage
detected it, which gates it passed, and its confidence features at each
step. This provenance enables stage-specific HITL review protocols,
targeted retraining, and diagnostic analysis of pipeline behavior.

### 1.5 Progressive Convergence Through HITL

The pipeline improves across iterations. Neurons initially found only by
later stages become training data for earlier stages. Over HITL rounds,
Stage 1 (Cellpose) learns to detect progressively harder neurons
independently, and later stages find diminishing numbers of new cells.
Convergence criterion: consecutive HITL rounds change fewer than 5% of
ROIs across representative FOVs.

### 1.6 Existing Tools as Infrastructure

Use established tools (Suite2p, Cellpose) for what they were designed to
do. Build custom components only where existing tools have documented
blind spots or domain mismatches. Suite2p provides motion correction, SVD
decomposition, and GPU-accelerated linear algebra. Cellpose provides
spatial segmentation with a learned morphological prior. Custom code
fills the gaps between them.

---

## 2. Architecture Overview

```
                    ┌──────────────────────────────────┐
                    │       RAW .TIF STACK (T,H,W)     │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │          FOUNDATION               │
                    │  Motion correction → SVD → M=L+S  │
                    └───────────────┬──────────────────┘
                                    │
                              Residual movie S
                              Summary images
                                    │
                    ┌───────────────▼──────────────────┐
                    │     STAGE 1: SPATIAL DETECTION    │
                    │  Cellpose on residual projections  │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │   GATE 1: MORPHOLOGICAL CHECK     │
                    │   + SOURCE SUBTRACTION             │
                    └───────────────┬──────────────────┘
                                    │
                              Residual S₁
                              (Stage 1 ROIs removed)
                                    │
                    ┌───────────────▼──────────────────┐
                    │    STAGE 2: TEMPORAL DETECTION    │
                    │  Suite2p on raw movie, filtered    │
                    │  against Stage 1 results           │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │   GATE 2: TEMPORAL CROSS-CHECK    │
                    │   + SOURCE SUBTRACTION             │
                    └───────────────┬──────────────────┘
                                    │
                              Residual S₂
                              (Stages 1+2 ROIs removed)
                                    │
                    ┌───────────────▼──────────────────┐
                    │      STAGE 3: TEMPLATE SWEEP     │
                    │  Matched filter for sparse events  │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │   GATE 3: WAVEFORM VALIDATION    │
                    │   + SOURCE SUBTRACTION             │
                    └───────────────┬──────────────────┘
                                    │
                              Residual S₃
                              (Stages 1+2+3 ROIs removed)
                                    │
                    ┌───────────────▼──────────────────┐
                    │    STAGE 4: TONIC NEURON SEARCH   │
                    │  Bandpass + correlation clustering  │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │  GATE 4: CORRELATION CONTRAST     │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │    UNIFIED QC & CLASSIFICATION    │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │     HITL CORRECTION & RETRAIN     │
                    └──────────────────────────────────┘
```

**Why this ordering:**

The pipeline progresses from highest-confidence, most-detectable signals
to lowest-confidence, hardest-to-detect signals:

- **Stage 1** catches cells with clear spatial morphology — the "obvious"
  cells visible in a projection image. These are the highest-confidence
  detections and the most numerous.
- **Stage 2** catches cells that are spatially ambiguous but temporally
  active — burst-firing neurons, task-locked neurons. Removing Stage 1's
  bright cells first allows Suite2p's SVD to allocate variance budget to
  dimmer, remaining signals.
- **Stage 3** catches remaining sparse-firing neurons via matched
  filtering. This runs before the tonic search because sparse-firing
  transients contaminate the correlation landscape that Stage 4 depends
  on, but tonic neurons do not contaminate the template sweep.
- **Stage 4** searches for tonic neurons last, when all transient-
  producing cells have been removed and the correlation landscape is
  maximally clean.

Each stage targets a specific population that previous stages are blind
to, and each subtraction step cleans the data for the next stage's
specific detection method.

---

## 3. Foundation

**Purpose:** Produce the shared intermediate products that all subsequent
stages consume: the registered movie, the SVD decomposition, the
background-separated residual, and summary images.

### 3.1 Motion Correction

**Method:** Suite2p rigid + non-rigid registration (phase correlation with
subpixel FFT translation). GPU-accelerated.

**Why it is first:** Every downstream operation — projections, SVD,
correlation maps, bandpass filtering, source subtraction — assumes
spatially stable pixels across time. Uncorrected motion introduces
spurious correlations at soma boundaries (see Blindspot 9) and
invalidates pixel-level temporal analysis.

**Outputs:**
- Registered movie M_reg of shape (T, H, W)
- Motion trace: `ops['xoff']`, `ops['yoff']` (XY displacement per frame,
  retained for use as a motion regressor in Gate 4)

**Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `do_registration` | 1 | Always enable for in vivo data |
| `nonrigid` | True | Handles local tissue deformation |
| `batch_size` | 500 | Safe for RTX 4060 8GB at 512×512 |
| `nplanes` | 1 | Single imaging plane |

### 3.2 SVD Decomposition

**Method:** Suite2p's GPU-accelerated SVD factorization of the registered
movie. Reshape M_reg from (T, H, W) to (T, N_pixels) and compute the
top-N singular value decomposition.

**Why here:** The SVD is the single most expensive computation in the
pipeline and produces intermediate products used by multiple downstream
stages: temporal components for Stage 2, the low-rank background estimate
for the L+S separation, and the compressed temporal representation for
Stage 4's correlation computation.

**Output:** Temporal components U, singular values Σ, spatial components
V^T. Suite2p stores these internally; they are also available as
intermediate products.

**Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `nSVD` | 1000 | Number of SVD components to compute |
| `fs` | Acquisition frame rate | Must match microscope exactly |
| `tau` | 1.0 (GCaMP6s) | Calcium indicator decay constant |

### 3.3 Background Separation (L+S)

**Method:** Reconstruct the low-rank background L from the top-k SVD
components: L = U_k Σ_k V_k^T. Compute the residual foreground:
S = M_reg − L. This separates spatially broad, temporally smooth
background (neuropil, photobleaching, uneven illumination) from spatially
localized, temporally structured cellular signals.

**Critical tuning point:** The choice of k determines the quality of every
downstream stage. See Blindspots 3 and 4 for detailed failure modes and
the required calibration procedure.

**Calibration procedure:**
1. Compute S at k = 10, 20, 30, 50, 100
2. For each k, generate the mean projection of S and the mean projection
   of L
3. Also compute the Vcorr map from S at each k
4. Select k where soma shapes are clearly visible in mean(S), absent from
   mean(L), and Vcorr(S) shows localized hotspots at expected cell
   locations
5. Document chosen k with rationale in the pipeline run log
6. Calibrate once per dataset type (experimental cohort), not per FOV

**Outputs:**
- Residual movie S of shape (T, H, W) — primary input for Stages 1, 3, 4
- Background movie L of shape (T, H, W) — inspected for absorbed tonic
  neurons (Blindspot 4)
- Chosen k value, logged

**Parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `k_background` | 20-50 | Determined by calibration procedure |

### 3.4 Summary Image Generation

**Method:** Compute multiple complementary 2D summary images from the
residual movie S. Each captures different information about neuron
locations.

**Images generated:**

| Image | Computation | What it captures |
|-------|-------------|------------------|
| Mean of S | Average of all T frames of S | All neurons proportional to mean fluorescence minus background; best contrast for most cells |
| Max of S | Per-pixel maximum across T frames of S | Peak fluorescence; captures sparsely firing neurons invisible in the mean |
| Std of S | Per-pixel standard deviation across T frames of S | Variance-weighted; bright for bursty, high-variance cells |
| Vcorr of S | Per-pixel Pearson correlation with immediate spatial neighbors, computed on S | Spatially localized temporal coherence; captures any neuron with temporally coherent fluctuations, including tonic neurons with micro-fluctuations |

**Why from S, not from M_reg:** Computing projections from the
background-subtracted residual produces cleaner images with less neuropil
haze, more uniform contrast across the FOV, and better soma-surround
differentiation. This is the primary benefit of the L+S separation for
Stage 1 — Cellpose receives fundamentally better input than if run on raw
projections.

**Denoising:** Apply Cellpose3 image restoration to the mean projection
of S before passing to Stage 1. This further improves effective SNR,
particularly for dim neurons near the detection threshold and for neurons
at FOV edges where GRIN lens PSF broadening reduces contrast.

**Outputs:**
- `mean_S.tif` — denoised mean projection of S
- `max_S.tif` — max projection of S
- `std_S.tif` — standard deviation projection of S
- `vcorr_S.tif` — local correlation map of S

---

## 4. Stage 1: Spatial Detection

**Purpose:** Detect neurons with clear spatial morphology in the summary
images. This is the highest-confidence detection stage, analogous to the
first pass in the manual workflow where "obvious" cells are identified
from the projection image.

**Method:** Cellpose (fine-tuned from `cyto3` base or Cellpose-SAM) on
dual-channel input: denoised mean of S as channel 1, Vcorr of S as
channel 2. The mean channel provides morphological contrast (soma shape,
nuclear shadow, brightness); the Vcorr channel provides correlation-based
contrast (pixels belonging to an active cell are correlated with their
neighbors, providing boundary information independent of absolute
brightness).

**Why dual-channel:** The mean projection alone misses neurons that are
dim or tonically active (low contrast against neuropil). The Vcorr map
highlights these neurons through their temporal coherence, giving Cellpose
a second contrast channel that is complementary to brightness. Max
projection may be substituted for Vcorr as channel 2 when targeting
sparse firers specifically.

### Fine-Tuning Protocol

1. Start from `cyto3` or `cellpose-sam` pretrained weights. **Never from
   a previous fine-tuned checkpoint** — chaining from checkpoints biases
   the network toward earlier images and degrades generalization.
2. Annotate 20-50 neurons across representative FOVs. Include dim neurons,
   edge neurons, neurons with visible nuclear shadows, and neurons with
   pyramidal (non-circular) morphology. Avoid over-representing bright,
   round interneurons (see Blindspot 16).
3. Train with learning rate 0.05, 200 epochs, min_train_masks=1.
4. Evaluate on held-out FOVs qualitatively (inspect missed cells, false
   positives, boundary quality) in addition to loss metrics.
5. Iterate through HITL: run inference → correct in Cellpose GUI →
   retrain from `cyto3` base with expanded training set.
6. Grow training set monotonically across HITL rounds.

### Inference Parameters

| Parameter | Default | Recommended | Effect |
|-----------|---------|-------------|--------|
| `cellprob_threshold` | 0.0 | -2.0 | Accepts dimmer, less certain cells (recall-first) |
| `flow_threshold` | 0.4 | 0.6 | More permissive boundary reconstruction |
| `diameter` | 17 | 12 | Match to neuron size in pixels at your magnification |
| `channels` | [0,0] | [1,2] | Dual-channel: mean as chan1, Vcorr as chan2 |
| `normalize` | default | tile_norm, block=128 | Compensates for GRIN vignetting |

### Nuclear Shadow Detector (Supplementary)

An in-house difference-of-Gaussians (DoG) filter tuned to the
soma/nucleus size ratio provides an additional high-specificity feature
for soma identification. The nuclear exclusion pattern (dark center,
bright surround from cytoplasmic GCaMP excluded from the nucleus) is
pathognomonic for a real soma — neuropil, dendrites, and astrocyte
processes never exhibit it.

**Implementation:** Convolve the denoised mean of S with a DoG kernel:
inner Gaussian sigma ≈ 2px (nucleus), outer Gaussian sigma ≈ 6px (soma).
The output is a "nuclear shadow score" map where strong positive responses
indicate likely somata. This map serves two purposes:
- As a feature input to Gate 1 (high nuclear shadow score increases
  confidence in an ROI's classification as a real soma)
- As seed points for Cellpose-SAM prompting (locations where the DoG
  response exceeds a threshold can be passed as point prompts)

**Output:** Set of labeled ROI masks (uint16), one contiguous region per
detected neuron. Each ROI carries a Cellpose probability score.

---

## 5. Source Subtraction Engine

**Purpose:** Remove the fluorescence contribution of detected ROIs from
the movie, producing a cleaner residual for the next detection stage. This
is the core custom component of the pipeline and is invoked between every
pair of detection stages.

**This component is entirely in-house.** No existing tool provides
inter-stage source subtraction with the required spatial weighting and
simultaneous estimation. See Blindspots 1, 2, and 5 for the detailed
failure analysis that motivates this design.

### 5.1 Algorithm

For a set of N ROIs detected in the current stage:

**Step 1 — Estimate spatial profiles:**

For each ROI i, compute its spatial weight map w_i(x,y):
```
w_i(x,y) = mean_t[S(x,y,t)] / max_{x,y in ROI_i}[mean_t[S(x,y,t)]]
```
This normalizes each ROI's spatial profile to peak at 1.0, tapering toward
the edges where the soma signal mixes with neuropil. Pixels outside the
ROI mask have w_i = 0.

**Step 2 — Simultaneous trace estimation:**

For all N ROIs, estimate their temporal traces simultaneously via
constrained least-squares. At each timepoint t, model the observed
fluorescence at each pixel p in the union of all ROI masks as:

```
S(p, t) ≈ Σ_i w_i(p) × c_i(t) + ε(p, t)
```

where c_i(t) is ROI i's temporal trace and ε is residual. Solve for all
c_i(t) simultaneously via non-negative least squares (NNLS) at each
timepoint. This disentangles overlapping ROIs' contributions at shared
pixels rather than subtracting them sequentially.

**Computational note:** For typical FOV (100-200 ROIs, each ~100-300
pixels), this is a small linear system at each of T timepoints. At
512×512 with T=18000 frames, total cost is dominated by the matrix
operations, which are vectorizable and GPU-acceleratable. Expected
runtime: seconds to low minutes per FOV.

**Step 3 — Subtract rank-1 estimates:**

For each ROI i, subtract its estimated contribution from every frame:
```
S_residual(p, t) = S(p, t) − w_i(p) × c_i(t)   for all p in ROI_i
```

Because traces were estimated simultaneously, subtracting all ROIs
produces a residual that is minimally distorted, even at overlap regions.

### 5.2 Post-Subtraction Validation

After each subtraction, automatically compute:
1. Mean projection of the residual at subtracted cell locations — no dark
   halos or ring artifacts should be visible
2. Standard deviation at subtracted locations — should match surrounding
   neuropil levels, not be anomalously high or low
3. Spot-check temporal traces at subtracted locations — should show no
   anti-correlation with the removed cell's trace

Flag any FOV where diagnostics indicate subtraction artifacts. Do not
proceed to the next stage until artifacts are resolved (see Blindspot 1
recovery procedure).

### 5.3 Notation Convention

Throughout this specification:
- **S** = the initial residual movie from the Foundation (M − L)
- **S₁** = residual after subtracting Stage 1 ROIs from S
- **S₂** = residual after subtracting Stage 2 ROIs from S₁
- **S₃** = residual after subtracting Stage 3 ROIs from S₂

Each Sₙ is the input to Stage n+1.

---

## 6. Gate 1: Morphological Validation

**Purpose:** Validate Stage 1 ROIs before source subtraction. Reject
obvious artifacts; flag uncertain candidates for HITL priority review.

**Input:** Stage 1 ROI masks + summary images (mean, Vcorr, DoG map)

### Acceptance Criteria

Each ROI is evaluated against these features:

| Feature | Computation | Accept if | Reject if |
|---------|-------------|-----------|-----------|
| Area (pixels) | Count of mask pixels | 80–350 px | Outside range |
| Solidity | Area / convex hull area | ≥ 0.55 | < 0.55 |
| Eccentricity | From fitted ellipse | ≤ 0.90 | > 0.90 |
| Nuclear shadow score | DoG response at ROI centroid | ≥ 0 (any positive) | Strong negative (inverted pattern) |
| Soma-surround contrast | (mean ROI brightness − mean annulus brightness) / mean annulus brightness | > 0.1 | ≤ 0.1 |
| Cellpose probability | From Cellpose output | > −2.0 (already filtered by inference) | Below threshold |

**Gate outcome categories:**
- **Accept:** Passes all criteria → subtracted in source subtraction,
  added to ROI pool with `source_stage = 1`, `confidence = high`
- **Flag:** Fails 1 criterion marginally → subtracted (recall-first
  principle) but tagged `confidence = moderate`, prioritized in HITL
- **Reject:** Fails 2+ criteria decisively → not subtracted, discarded
  (these are clear neuropil fragments, noise, or sub-cellular artifacts)

---

## 7. Stage 2: Temporal Detection

**Purpose:** Detect neurons that Cellpose missed due to weak spatial
morphology but that have distinctive temporal dynamics — burst-firing
neurons, task-locked neurons, neurons obscured by brighter neighbors in
the projection images.

**Method:** Hybrid approach (see Blindspot 6 for rationale).

### 7.1 Run Suite2p on Raw Registered Movie

Run Suite2p's full detection pipeline (SVD → clustering → cell
classification) on the original registered movie M_reg, not on the
source-subtracted residual S₁. This avoids the domain mismatch problem:
Suite2p's internal heuristics (naive Bayes classifier, enhanced mean
image, neuropil model) were designed for raw fluorescence movies and may
produce unreliable results on residual data.

**Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `fs` | Acquisition frame rate | Must match microscope exactly |
| `tau` | 1.0 (GCaMP6s), 0.5 (jGCaMP8f) | Indicator decay constant |
| `threshold_scaling` | 0.5–1.0 | Lower = more ROIs detected; start at 1.0, reduce if recall is low |
| `spatial_scale` | Adjust | Match GRIN-aberrated neuron size |
| `batch_size` | 500 | Safe for RTX 4060 8GB at 512×512 |
| `nplanes` | 1 | Single imaging plane |

### 7.2 Filter Against Stage 1 Results

For each Suite2p ROI, compute IoU overlap with every Stage 1 ROI.

- **IoU > 0.3 with any Stage 1 ROI:** Rediscovery — discard. This cell
  was already found by Cellpose.
- **IoU ≤ 0.3 with all Stage 1 ROIs:** Genuinely new detection — retain
  as a Stage 2 candidate.

### 7.3 Cell Classifier Filtering

Apply Suite2p's cell probability threshold:
- `iscell_prob > 0.3` → retain as candidate
- `iscell_prob ≤ 0.3` → discard unless the ROI shows clear morphological
  structure in the mean projection (edge case handled in Gate 2)

**Output:** Set of Suite2p-detected ROIs not previously found by Stage 1,
with Suite2p's spatial footprints (`stat.npy` entries) and temporal traces.

### 7.4 Future Refinement: Custom SVD Clustering on S₁

For later iterations, replace the hybrid approach with direct SVD-based
clustering on the source-subtracted residual S₁. Use Suite2p's
GPU-accelerated SVD computation on S₁, but replace Suite2p's cell
detection clustering with custom logic designed for residual movie
statistics. This captures the benefit of searching a cleaned SVD space
without relying on heuristics calibrated for raw movies.

---

## 8. Gate 2: Temporal Cross-Validation

**Purpose:** Verify that Stage 2 candidates represent genuinely
independent sources, not artifacts of imperfect Stage 1 subtraction or
spatial spillover from already-found cells.

**Input:** Stage 2 candidate ROIs + Stage 1 confirmed ROIs + their traces

### Acceptance Criteria

| Check | Method | Accept if | Reject if |
|-------|--------|-----------|-----------|
| Temporal independence | Pearson correlation between candidate trace and all Stage 1 ROI traces within 20px radius | abs(r) < 0.7 for all nearby Stage 1 ROIs | abs(r) ≥ 0.7 (redundant or spillover) |
| Anti-correlation check | Same correlation, checking sign | r > −0.5 (not anti-correlated) | r ≤ −0.5 with nearby Stage 1 ROI (subtraction artifact; see Blindspot 2) |
| Spatial distance | Centroid distance to nearest Stage 1 ROI | > 5 px | ≤ 5px AND high abs(correlation) |
| Morphological quality | Area, solidity (from Suite2p stat) | Area 60–400px, solidity ≥ 0.4 | Below thresholds |

**Note on relaxed morphological thresholds:** Suite2p's spatial footprints
are noisier than Cellpose masks (they represent weighted pixel
contributions, not clean binary masks). The solidity and area thresholds
are accordingly more permissive than Gate 1's.

**Gate outcome:** Same three-category system as Gate 1 (accept, flag,
reject). Accepted ROIs proceed to source subtraction; their traces are
estimated from S₁ (not from the raw movie) for subtraction purposes.

**Source subtraction:** Apply the source subtraction engine (§5) to
subtract Stage 2 confirmed ROIs from S₁, producing S₂.

---

## 9. Stage 3: Template Sweep

**Purpose:** Detect neurons that fired too few transients to rank in
Suite2p's SVD decomposition but whose individual events are identifiable
via matched filtering. This stage targets the sparse-firing population:
neurons with 1-15 transients across the recording.

**This stage is entirely in-house.** No existing tool implements a
transient-matched filter pipeline on source-subtracted calcium imaging
residuals.

### 9.1 Template Bank Definition

Rather than a single template, use a bank of templates spanning the
expected calcium transient waveform space. For each template, define a
normalized waveform sampled at the acquisition frame rate:

| Template ID | Rise time | Decay tau | FWHM | Target event |
|-------------|----------|-----------|------|--------------|
| `single_gcamp6s` | ~100 ms | 1.0 s | ~0.7 s | Single AP, GCaMP6s |
| `doublet_gcamp6s` | ~150 ms | 1.2 s | ~1.0 s | 2 APs ~100ms apart |
| `burst_gcamp6s` | ~200 ms | 1.5 s | ~1.3 s | 3-5 APs over ~300ms |
| `single_jgcamp8f` | ~80 ms | 0.5 s | ~0.35 s | Single AP, jGCaMP8f |

Template waveforms are parameterized by indicator kinetics. The bank
should be regenerated when the indicator changes.

**Waveform generation:** Each template is a convolution of a step
function (representing the [Ca²⁺] rise from an AP) with the indicator's
binding kinetics. For GCaMP6s:
```
template(t) = A × (1 − exp(−t/τ_rise)) × exp(−t/τ_decay)
```
where τ_rise ≈ 0.05s and τ_decay ≈ 1.0s. Normalize each template to
unit L2 norm for comparable match scores across templates.

### 9.2 Per-Pixel Template Convolution

For each pixel in S₂ and each template in the bank:

1. Compute the normalized cross-correlation between the pixel's
   timecourse and the template via FFT-based convolution:
   ```
   score(p, t, k) = (S₂(p, :) ⊛ template_k)[t] / (σ_local(p, t) × ||template_k||)
   ```
   where σ_local is a running estimate of the local noise standard
   deviation (e.g., MAD of the pixel's trace in a sliding window),
   and k indexes the template bank.

2. At each pixel and timepoint, take the maximum score across templates:
   ```
   score_max(p, t) = max_k[score(p, t, k)]
   ```

3. Threshold: retain (p, t) pairs where score_max > τ_template (default:
   4.0 standard deviations above noise floor).

**Computational cost:** FFT-based convolution is O(T log T) per pixel per
template. For 512×512 pixels × 4 templates × T=18000: ~4 × 262144 ×
T log T ≈ 10^10 FLOPS. On GPU, this is feasible in minutes.

### 9.3 Spatial Coherence Evaluation

At each timepoint where threshold crossings occur:

1. Compute a 2D "event map": for each pixel, its score_max at that
   timepoint
2. For each local maximum in the event map, compute the mean score within
   a soma-radius disk (r = 8 pixels, ~200 pixels)
3. Retain events where the spatial mean score exceeds a threshold (default:
   3.0 σ) — this pooling improves SNR by √N_pixels

### 9.4 Event Accumulation Across Time

**Critical for intermediate-activity neurons (see Blindspot 11).**

After spatial coherence evaluation produces a set of detected events, each
tagged with (centroid_x, centroid_y, time, score, matched_template):

1. Cluster events that occur within a soma radius (12 px) of each other
   across different timepoints. Use hierarchical clustering with a spatial
   distance threshold of 12 px.
2. For each spatial cluster, count the number of temporally independent
   events (events separated by at least 2× the indicator decay constant,
   i.e., >2s for GCaMP6s).
3. Score each spatial cluster by its event count × mean event score.
   High scores indicate a cell that fired multiple times at the same
   location — strong evidence for a real neuron, even if individual
   events are marginal.

**Output:** Set of candidate ROIs, each defined by:
- Centroid location (mean of event centroids in the spatial cluster)
- Spatial footprint (union of event footprints, or the soma-radius disk
  around the centroid)
- Event count and mean event score
- Matched template(s) and their scores

---

## 10. Gate 3: Waveform Validation

**Purpose:** Verify that Stage 3 candidates have transient waveforms
consistent with real calcium events, not artifacts from imperfect
subtraction or noise.

### Acceptance Criteria

| Check | Method | Accept if | Reject if |
|-------|--------|-----------|-----------|
| Waveform shape | Compare detected event's temporal profile to best-matching template; compute R² | R² > 0.6 for at least 1 event | R² < 0.6 for all events |
| Rise/decay asymmetry | Compute ratio of rise time to decay time | Ratio < 0.5 (fast rise, slow decay — characteristic of calcium transients) | Ratio ≥ 0.5 (symmetric or slow-rising — likely artifact) |
| Anti-correlation check | Pearson r with all previously-found ROI traces within 20px | r > −0.5 for all nearby ROIs | r ≤ −0.5 (subtraction artifact) |
| Event count | Number of temporally independent events | ≥ 1 | 0 (should not reach this gate with 0 events) |
| Spatial compactness | Solidity of the event footprint | ≥ 0.5 | < 0.5 (fragmented, non-somatic) |

**Confidence grading by event count:**
- 1 event: `confidence = low` — retain but prioritize in HITL
- 2-5 events: `confidence = moderate`
- 6+ events: `confidence = high`

**Source subtraction:** Apply the source subtraction engine to subtract
Stage 3 confirmed ROIs from S₂, producing S₃.

---

## 11. Stage 4: Tonic Neuron Search

**Purpose:** Detect neurons with sustained baseline firing rates whose
temporal signature is not transient-based but correlation-based — spatially
confined micro-fluctuations in the calcium-relevant frequency band.

**This stage is entirely in-house.** The specific combination of bandpass
filtering on a source-subtracted residual followed by local correlation
clustering with morphological constraints is novel. No published tool
implements this as a detection pathway.

### 11.1 Pre-Processing

**Per-pixel linear detrend:** Before bandpass filtering, subtract a
per-pixel best-fit linear trend from each pixel's timecourse in S₃. This
removes residual slow drift that the Foundation L+S separation missed,
including spatially non-uniform photobleaching from GRIN vignetting.
(See Blindspot 10.)

### 11.2 Multi-Scale Bandpass Filtering

Apply temporal bandpass filtering independently to each pixel's
timecourse across all T frames. Use a zero-phase Butterworth filter
(scipy `sosfiltfilt`, order 4) to avoid temporal distortion.

**Multi-scale windows (see Blindspot 12):**

| Window | Low cutoff | High cutoff | Target |
|--------|-----------|------------|--------|
| Fast | 0.5 Hz | 2.0 Hz | High-rate tonic (3-5 Hz firing) |
| Medium | 0.1 Hz | 1.0 Hz | Moderate-rate tonic (1-3 Hz) |
| Slow | 0.05 Hz | 0.5 Hz | Low-rate tonic / slow modulation |

For each bandpass window, produce a filtered version of S₃. All three
filtered movies proceed to the correlation clustering step independently.

### 11.3 Temporal Compression

For each filtered movie, reduce temporal dimensionality to make
correlation computation tractable:
- **Option A (preferred if GPU available):** Project into the top 200-500
  SVD components of the filtered movie
- **Option B (CPU fallback):** Temporally downsample by averaging 5-10
  contiguous frames

Both options preserve the correlation structure between pixels while
reducing the number of operations from O(T) to O(n_components) per pixel
pair.

### 11.4 Local Correlation Computation

For each pixel p in the filtered, compressed movie:

1. Define a local neighborhood: all pixels within a soma radius (~8 px,
   ~200 neighbors)
2. Compute Pearson correlation between p and each neighbor
3. Store the mean local correlation as a feature: corr_local(p)

Additionally, compute a **correlation contrast map**:
```
contrast(p) = mean_corr(neighbors within 6px) − mean_corr(neighbors 6-15px)
```

This contrast map is more discriminative than raw correlation: soma pixels
show high contrast (correlated locally, decorrelated at distance), while
neuropil pixels show low contrast (correlated broadly at similar levels).

### 11.5 Correlation Clustering

1. Threshold the correlation contrast map: retain pixels with
   contrast > τ_contrast (default: 0.10)
2. Connected-component analysis on the thresholded map
3. Size filter: reject clusters with area < 80 pixels or > 350 pixels
4. Morphological filter: reject clusters with solidity < 0.6 or
   eccentricity > 0.85
5. Merge candidates across the three bandpass windows using IoU-based
   matching (IoU > 0.3 = same cell; keep the version with the highest
   correlation contrast)

**Output per candidate:**
- Binary mask (cluster pixels)
- Correlation contrast score
- Mean intra-cluster correlation
- Which bandpass window(s) detected it
- Centroid location

---

## 12. Gate 4: Correlation Contrast Validation

**Purpose:** Verify that Stage 4 candidates represent real tonic somata,
not residual neuropil correlations, motion artifacts at subtracted cell
boundaries, or photobleaching residuals.

### Acceptance Criteria

| Check | Method | Accept if | Reject if |
|-------|--------|-----------|-----------|
| Correlation contrast | Intra-cluster vs. surround correlation ratio | Contrast > 0.10 | ≤ 0.10 (neuropil-like) |
| Shape: eccentricity | From regionprops | ≤ 0.85 | > 0.85 (elongated — likely motion boundary artifact) |
| Shape: solidity | From regionprops | ≥ 0.6 | < 0.6 (fragmented) |
| Motion correlation | Pearson r between candidate trace and Suite2p motion trace (`ops['xoff']`, `ops['yoff']`) | abs(r) < 0.3 | abs(r) ≥ 0.3 (motion artifact; see Blindspot 9) |
| Anti-correlation check | Pearson r with all previously-found ROI traces within 20px | r > −0.5 for all nearby ROIs | r ≤ −0.5 (subtraction artifact) |
| Mean projection intensity | Mean brightness at cluster location in denoised mean of S | Above 25th percentile of FOV mean | Below 25th percentile (sub-threshold region) |

**All Stage 4 candidates carry `source_stage = 4` and
`confidence = requires_review`.** These are the lowest-confidence
detections and receive the most HITL scrutiny using the modified review
protocol (bandpass-filtered traces, correlation maps — not raw traces).

---

## 13. Unified Quality Control & Classification

**Purpose:** Evaluate all confirmed ROIs from all stages with a unified
feature set. Assign activity-type classifications and composite confidence
scores. This step does not add or remove ROIs — it characterizes them.

### 13.1 Feature Extraction

For each ROI in the unified pool, extract from the **original registered
movie M_reg** (not from any residual):

**Spatial features:**

| Feature | Computation |
|---------|-------------|
| `area` | Pixel count of mask |
| `solidity` | Area / convex hull area |
| `eccentricity` | From fitted ellipse |
| `nuclear_shadow_score` | DoG response at centroid |
| `soma_surround_contrast` | (mean ROI − mean annulus) / mean annulus |
| `boundary_gradient` | Mean gradient magnitude at ROI boundary |
| `spatial_blur` | FWHM of ROI intensity distribution (ghost detection) |
| `fov_distance` | Distance from FOV center (edge-of-FOV flag) |

**Temporal features (from raw trace):**

| Feature | Computation |
|---------|-------------|
| `std` | Standard deviation of neuropil-subtracted trace |
| `skew` | Skewness of neuropil-subtracted trace |
| `snr` | Peak transient height / noise floor (MAD) |
| `n_transients` | Count of detected calcium events (template match) |
| `mean_fluorescence` | Mean of raw trace |

**Temporal features (from bandpass-filtered trace):**

| Feature | Computation |
|---------|-------------|
| `bp_std` | Std of bandpass-filtered trace (0.05-2.0 Hz) |
| `bp_power_ratio` | Power in calcium band / total power |
| `autocorr_tau` | Decay constant of temporal autocorrelation |

**Provenance features:**

| Feature | Source |
|---------|--------|
| `source_stage` | 1, 2, 3, or 4 |
| `n_stages_detected` | How many stages independently found this ROI |
| `gate_confidence` | Accept / flag category from originating gate |
| `cellpose_prob` | Cellpose probability score (Stage 1 only) |
| `iscell_prob` | Suite2p cell probability (Stage 2 only) |
| `event_count` | Number of template-matched events (Stage 3) |
| `corr_contrast` | Correlation contrast score (Stage 4) |

### 13.2 Neuropil Trace Extraction

For each ROI, compute the neuropil signal from a surrounding annulus
(inner radius = ROI boundary + 2px buffer, outer radius = inner + 15px),
excluding pixels belonging to any other ROI. Estimate the neuropil
contamination coefficient iteratively (Suite2p's method) or use a
fixed coefficient (default: 0.7). Apply correction:
```
F_corrected(t) = F_raw(t) − α × F_neuropil(t)
```

### 13.3 Activity-Type Classification

Assign each ROI to one of the following categories based on its
temporal features:

| Type | Criteria | Handling |
|------|----------|---------|
| **Phasic** | n_transients ≥ 5, skew > 0.5, clear discrete events | Standard dF/F analysis; event-locked responses |
| **Sparse** | n_transients 1-4, skew > 0.3 | Flag for trial-level analysis; insufficient for session averages |
| **Tonic** | bp_std > 2× noise floor, low skew, source_stage = 4 or high mean fluorescence with low variance | Rolling baseline dF/F; report mean rate, not event-locked |
| **Silent** | n_transients = 0, bp_std < noise floor, strong spatial morphology | Retain; flag as no detected activity this session |
| **Ambiguous** | Does not fit cleanly into above categories | Flag for HITL review |

### 13.4 Overlapping Cell Trace Correction

After classification, identify all ROI pairs with spatial overlap
(IoU > 0.1). For these pairs, re-estimate their temporal traces
simultaneously using the multi-component linear model described in
the source subtraction engine (§5.1, Step 2), operating on the original
movie M_reg. This corrects for cross-contamination introduced by
sequential subtraction. See Blindspot 5.

### 13.5 dF/F Computation

| Activity type | Baseline (F0) method |
|---------------|---------------------|
| Phasic, sparse | 10th percentile in a 60-second sliding window |
| Tonic | 10th percentile in a 120-second sliding window (wider window to capture the elevated baseline) |
| Silent | Not computed (no meaningful dF/F) |

```
dF/F(t) = (F_corrected(t) − F0(t)) / F0(t)
```

### 13.6 Spike Deconvolution (Optional)

Apply OASIS or Suite2p's built-in deconvolution to estimate spike times
and amplitudes from the dF/F traces. Use indicator-appropriate tau
(1.0s for GCaMP6s, 0.5s for jGCaMP8f).

**Note:** Spike deconvolution is unreliable for tonic neurons (the
underlying assumption of discrete events on a stable baseline is
violated). For tonic ROIs, report the bandpass-filtered trace and mean
activity level rather than deconvolved spikes.

---

## 14. Human-in-the-Loop Correction & Retraining

### 14.1 Review Protocol

Review is structured by source stage and confidence level:

**Priority 1 — Stage 4 candidates (`confidence = requires_review`):**

Present each candidate with:
- The bandpass-filtered trace (not the raw trace) — shows calcium-band
  micro-fluctuations invisible in the raw signal
- A local correlation map centered on the ROI — shows spatial coherence
  structure
- The correlation contrast ratio (number, not just a map)
- The ROI mask overlaid on the denoised mean projection

**Do NOT use the raw fluorescence trace as the primary confirmation
signal for Stage 4 candidates.** The raw trace of a tonic neuron shows
no recognizable transients and will be erroneously rejected by reviewers
trained on phasic-neuron traces. (See Blindspot 13.)

**Priority 2 — Flagged ROIs from any stage (`confidence = moderate`):**

Present with standard visualization: ROI mask on mean projection +
raw trace + basic spatial features. These are cases where automated
criteria were borderline.

**Priority 3 — Stage 3 candidates with event_count = 1:**

Present with the event's spatiotemporal context: the 5 frames before and
after the event, centered on the ROI location. The reviewer confirms
whether the event is a real transient or an artifact.

**Priority 4 — False negative search:**

After reviewing automated detections, scan the Vcorr map and mean
projection for missed cells. Draw new ROIs for any neurons visible in the
images that were not detected by any stage. These become high-priority
training data for Cellpose, as they represent the pipeline's current
blind spots.

### 14.2 Retraining Cycle

1. Save corrected masks and classifications from the Cellpose GUI
2. Ingest corrections into the training set (convert GUI outputs to
   Cellpose-format masks in the `annotated/` + `masks/` directories)
3. **Retrain Cellpose from `cyto3` base** — never from a previous
   checkpoint
4. Re-run the full pipeline on representative FOVs
5. Compare ROI counts and classifications to the previous round
6. Repeat until convergence (< 5% ROI change across rounds)

### 14.3 Training Set Composition Guidelines

To counteract the biases documented in Blindspots 13 and 16:

- **Include tonic neurons:** Stage 4 candidates confirmed during HITL
  review should be added to the Cellpose training set. Over iterations,
  Cellpose learns to detect these independently from the Vcorr channel.
- **Include edge neurons:** Annotations from FOV periphery where detection
  is degraded (Blindspot 15)
- **Include pyramidal morphology:** Explicitly annotate triangular somata
  with visible apical dendrites and nuclear shadows, even when they are
  dimmer than nearby interneurons (Blindspot 16)
- **Track training set composition:** Maintain counts of round vs.
  irregular cells, bright vs. dim cells, center vs. edge cells. Ensure
  proportions approximate the expected biological distribution.

---

## 15. Phase 2 Extension: Astrocyte Detection

**When to activate:** After Phase 1 pipeline is stable for pyramidal
neurons. Runs as a parallel branch, not a replacement.

### 15.1 Key Differences From Neuron Detection

| Property | Neurons (Phase 1) | Astrocytes (Phase 2) |
|----------|------------------|---------------------|
| Spatial model | ~12px diameter soma, fixed boundary | Variable: soma (15-25px), processes (thin), territory (~50px radius) |
| Temporal dynamics | Discrete transients, 0.1-2 Hz | Slow waves, 0.01-0.3 Hz, propagative |
| ROI paradigm | Fixed spatial mask per neuron | Event-based: each event has its own spatial footprint |
| Detection tool | Cellpose + Suite2p + custom stages | AQuA2 with STARDUST protocol for events; Cellpose for soma detection |
| Correlation method | Zero-lag Pearson | Cross-correlation with lag tolerance (±500ms) for propagation |

### 15.2 Two-Tier Architecture

**Tier 1 — Astrocyte soma detection:**
Fine-tune a separate Cellpose model specifically for astrocyte somata
(larger, rounder, different texture than neurons). Run on the slow-band
(0.01-0.3 Hz) bandpass-filtered mean projection, where astrocyte somata
appear as bright objects and neuronal transients are filtered out.

**Tier 2 — Process-level event detection:**
Apply AQuA2 following the STARDUST protocol to the slow-band filtered
movie. AQuA2's event-based paradigm handles the variable spatial
footprint, propagation, and temporal heterogeneity of astrocyte process
signals.

### 15.3 Integration With Phase 1

In mixed FOVs where both neurons and astrocytes are labeled:
- The neuronal bandpass (0.05-2.0 Hz) and astrocyte bandpass (0.01-0.3 Hz)
  serve as a first-pass cell-type discriminator
- Spatial regions active only in the slow band → astrocyte pipeline
- Spatial regions active only in the fast band → neuron pipeline
- Spatial regions active in both bands → require further classification
  (likely neuronal soma, as most astrocyte process signals are
  below neuronal frequency content)

### 15.4 Indicator Considerations

| Indicator | Targeting | Pipeline implications |
|-----------|-----------|---------------------|
| GCaMP6s cytoplasmic | Fills cell body + major processes | Standard detection; may miss fine process events |
| lck-GCaMP6f membrane-targeted | Localizes to plasma membrane | More spatially distributed signals; requires adjusted Cellpose diameter and AQuA2 sensitivity |
| jGCaMP8 variants | Various | Faster kinetics; adjust template bank tau values |

---

## 16. In-House Components Specification

Summary of all components that must be custom-built, ordered by
implementation priority.

### 16.1 Source Subtraction Engine

**Priority:** 1 (critical path — pipeline viability depends on this)
**Specification:** §5 of this document
**Dependencies:** Stage 1 output (ROI masks), registered movie
**Key algorithms:** Weighted spatial profile estimation, simultaneous
trace estimation via NNLS, rank-1 outer product subtraction
**Validation:** Post-subtraction diagnostic (§5.2)
**Estimated complexity:** ~500-800 lines; core is vectorized linear
algebra

### 16.2 Template Sweep (Stage 3)

**Priority:** 2 (no existing tool provides this)
**Specification:** §9 of this document
**Dependencies:** S₂ (source-subtracted residual after Stages 1-2)
**Key algorithms:** FFT-based template convolution, spatial coherence
pooling, event accumulation clustering
**Estimated complexity:** ~400-600 lines; FFT and connected-component
analysis

### 16.3 Nuclear Shadow Detector

**Priority:** 3 (high value, low difficulty)
**Specification:** §4 (supplementary section)
**Dependencies:** Denoised mean projection of S
**Key algorithms:** Difference-of-Gaussians convolution
**Estimated complexity:** ~50-100 lines; single convolution + thresholding

### 16.4 Tonic Search Pipeline (Stage 4 core)

**Priority:** 4 (builds on Foundation SVD infrastructure)
**Specification:** §11 of this document
**Dependencies:** S₃ (source-subtracted residual after Stages 1-3),
Suite2p motion trace
**Key algorithms:** Multi-scale bandpass filtering, local correlation
computation, correlation contrast mapping, connected-component clustering
**Note:** Branch C in the current codebase (`roigbiv`) implements a
partial version. This specification extends it with: multi-scale bandpass
sweep, correlation contrast (not raw correlation), per-pixel detrending,
and morphological filtering tuned for the cleaned residual.
**Estimated complexity:** ~600-900 lines

### 16.5 Gate Classifiers (Gates 1-4)

**Priority:** 5 (iteratively refined through HITL)
**Specification:** §6, §8, §10, §12 of this document
**Dependencies:** ROI features from each stage
**Key algorithms:** Threshold-based decision rules initially; upgrade to
naive Bayes or lightweight classifier after sufficient HITL training data
**Estimated complexity:** ~200-400 lines total across all gates

### 16.6 L+S Calibration Diagnostic Tool

**Priority:** 6 (required for Foundation tuning)
**Specification:** §3.3 (calibration procedure)
**Dependencies:** SVD decomposition output
**Key algorithms:** SVD reconstruction at multiple k values, projection
computation, Vcorr computation, side-by-side visualization
**Estimated complexity:** ~200-300 lines + visualization output

### 16.7 Correlation Contrast Computation

**Priority:** Included in 16.4 (Stage 4 core)
**Specification:** §11.4
**Key algorithm:** Dual-radius correlation averaging, contrast ratio
computation

---

## 17. Data Flow & File Outputs

### 17.1 Per-FOV Output Files

| File | Contents | Shape / Format |
|------|----------|---------------|
| `registered.bin` or `.tif` | Motion-corrected movie M_reg | (T, H, W) int16/float32 |
| `background_L.npy` | Low-rank background reconstruction | (T, H, W) float32 |
| `residual_S.npy` | Background-subtracted residual | (T, H, W) float32 |
| `mean_S.tif` | Denoised mean projection of S | (H, W) float32 |
| `max_S.tif` | Max projection of S | (H, W) float32 |
| `vcorr_S.tif` | Local correlation map of S | (H, W) float32 |
| `dog_map.tif` | Nuclear shadow (DoG) response map | (H, W) float32 |
| `stage1_masks.tif` | Stage 1 Cellpose ROI masks | (H, W) uint16 |
| `stage2_masks.tif` | Stage 2 Suite2p-derived ROI masks | (H, W) uint16 |
| `stage3_masks.tif` | Stage 3 template sweep ROI masks | (H, W) uint16 |
| `stage3_events.npy` | Per-ROI event list (time, score, template) | list of dicts |
| `stage4_masks.tif` | Stage 4 tonic search ROI masks | (H, W) uint16 |
| `stage4_corr.npy` | Per-ROI correlation contrast scores | (N_stage4,) float32 |
| `merged_masks.tif` | Unified ROI masks (all stages) | (H, W) uint16 |
| `roi_metadata.json` | Per-ROI: source_stage, confidence, activity_type, all QC features | list of dicts |
| `F.npy` | Raw fluorescence traces | (N_rois, T) float32 |
| `Fneu.npy` | Neuropil traces | (N_rois, T) float32 |
| `F_corrected.npy` | Neuropil-corrected traces | (N_rois, T) float32 |
| `dFF.npy` | dF/F traces | (N_rois, T) float32 |
| `spks.npy` | Deconvolved spike estimates | (N_rois, T) float32 |
| `F_bandpass.npy` | Bandpass-filtered traces (for tonic ROIs) | (N_tonic, T) float32 |
| `pipeline_log.json` | Run parameters, k value, ROI counts per stage, diagnostics | dict |
| `traces/traces.npy` | Neuropil-corrected traces, pynapse-facing primary | (N_rois, T) float32 |
| `traces/traces_raw.npy` | Raw fluorescence, same row order as `traces.npy` | (N_rois, T) float32 |
| `traces/traces_neuropil.npy` | Neuropil estimate, same row order | (N_rois, T) float32 |
| `traces/traces_meta.json` | Row→(session_id, fov_id, local_label_id, global_cell_id?) map, fs, frame_averaging, provenance | dict |
| `traces/corrections-{hash12}/…` | Revision-scoped bundle produced by HITL re-extract (same four files) | — |

### 17.1.1 Trace Persistence & Pynapse Handoff

The `traces/` bundle is the pynapse-facing contract. Pynapse's
`SignalRecording` loads the `.npy` directly and identifies neurons by row
index only, so the row→ID mapping lives in the sidecar.

**Row-ordering contract.** Row `K` of every `.npy` in `traces/` (and of
`roi_metadata.json`) corresponds to the ROI at position `K` in
`rois_sorted` (sorted by `label_id`). The same ROI's label appears at the
matching integer value in `merged_masks.tif`.

**Frame-rate convention.** `PipelineConfig.fs` (and the sidecar's `fs`
field) is the **effective** rate — i.e. the rate of the frames actually
stored in `traces.npy` (7.5 Hz for the reference 4×-averaged 30 Hz stack).
`frame_averaging` is the temporal binning factor that produced `fs`. To
hand this off to pynapse, whose `Sample.fps` expects the raw acquisition
rate, pass `fps = meta["fs"] * meta["frame_averaging"]` (pynapse computes
`effective_fps = fps / frame_averaging` — matching `meta["fs"]` exactly).
`fs` and `frame_averaging` come from `PipelineConfig` and are never
inferred from file timestamps.

**`rois[]` schema.** Each entry is
`{row_index, local_label_id, source_stage, gate_outcome, confidence,
  activity_type?, global_cell_id?}`. `global_cell_id` is **omitted** —
not `null` — when the FOV is unregistered or the row is a fresh HITL
label. Top-level `session_id` / `fov_id` / `registry_decision` are `null`
when the pipeline ran without `--registry` or the registry returned the
`review` decision.

**Determinism.** `traces_meta.json` is written with `sort_keys=True`, no
wall-clock fields, and ROI order locked by `label_id`. Rerunning the
pipeline on the same inputs with the same registry state produces a
byte-identical sidecar.

**HITL re-extract.** `roigbiv.pipeline.reextract.reextract_from_corrections`
reads `corrections/corrected_masks.tif` + `corrections/corrections.jsonl`,
computes a 12-char `corrections_rev` from the replayed ROI set (not the
JSONL bytes — that makes the hash stable under undo/redo), and writes a
full bundle to `traces/corrections-{rev}/`. The primary `traces.npy` is
never mutated. Identifiers (`session_id`, `fov_id`, per-row
`global_cell_id`) are inherited from the primary sidecar for ROIs that
survived corrections with their original `label_id`; fresh labels minted
by `add` / `merge` / `split` get no `global_cell_id` (re-extract never
writes to the registry DB).

**Example (pynapse).**

```python
from pynapse.core.io.microscopy import SignalRecording
from pynapse.core.sample import Sample
import json

meta = json.load(open("traces/traces_meta.json"))
sig = SignalRecording(source="traces/traces.npy")
sample = Sample(
    event_data=<REACHER event log>,
    signal_data=sig,
    fps=meta["fs"] * meta["frame_averaging"],
    frame_averaging=meta["frame_averaging"],
)
# sample.effective_fps == meta["effective_fps"] == meta["fs"]
# rows: row K ↔ meta["rois"][K]["local_label_id"] ↔ (optionally) global_cell_id
```

See `scripts/roigbiv_to_pynapse.py` for a runnable version.

### 17.2 Stage-Wise ROI Count Log

The pipeline log records the number of ROIs found, accepted, flagged, and
rejected at each stage. The expected pattern is monotonically decreasing
new detections:

```
Stage 1: N1 detected → N1_accepted + N1_flagged + N1_rejected
Stage 2: N2 detected → N2_accepted + N2_flagged + N2_rejected
Stage 3: N3 detected → N3_accepted + N3_flagged + N3_rejected
Stage 4: N4 detected → N4_accepted + N4_flagged + N4_rejected

Expected: N1 > N2 > N3 > N4

Total ROIs: Σ(accepted + flagged) across all stages
```

If any Nₙ > Nₙ₋₁, investigate for subtraction artifact propagation
(see Blindspot 2).

---

## 18. Parameter Reference

### 18.1 Foundation Parameters

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `k_background` | 30 | 10-100 | L+S separation; tune per dataset type |
| `batch_size` | 500 | 200-1000 | GPU memory; reduce if CUDA OOM |
| `nonrigid` | True | True/False | Registration quality |

### 18.2 Stage 1 Parameters (Cellpose)

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `base_model` | cyto3 | cyto3, cellpose-sam | Always retrain from base |
| `diameter` | 12 | 8-20 | Neuron size in pixels |
| `cellprob_threshold` | -2.0 | -6.0 to 0.0 | Recall vs. precision |
| `flow_threshold` | 0.6 | 0.4-1.0 | Boundary permissiveness |
| `channels` | [1,2] | [0,0] or [1,2] | Single vs. dual channel |
| `tile_norm_blocksize` | 128 | 64-256 | Vignetting compensation |
| `learning_rate` | 0.05 | 0.01-0.1 | Fine-tuning aggressiveness |
| `n_epochs` | 200 | 100-500 | Training duration |

### 18.3 Gate 1 Parameters

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `min_area` | 80 | 50-120 | Minimum soma size |
| `max_area` | 350 | 250-500 | Maximum soma size |
| `min_solidity` | 0.55 | 0.4-0.7 | Shape compactness |
| `max_eccentricity` | 0.90 | 0.8-0.95 | Shape elongation tolerance |
| `min_contrast` | 0.1 | 0.05-0.2 | Soma-surround contrast |

### 18.4 Stage 2 Parameters (Suite2p)

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `fs` | — | — | Must match microscope acquisition rate |
| `tau` | 1.0 | 0.5-1.5 | Indicator decay constant |
| `threshold_scaling` | 1.0 | 0.5-2.0 | Detection sensitivity |
| `spatial_scale` | — | — | Match GRIN-aberrated neuron size |
| `iscell_threshold` | 0.3 | 0.1-0.5 | Cell classifier cutoff |

### 18.5 Gate 2 Parameters

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `iou_overlap_threshold` | 0.3 | 0.2-0.5 | Rediscovery detection |
| `max_correlation` | 0.7 | 0.5-0.8 | Independence check |
| `anticorr_threshold` | -0.5 | -0.7 to -0.3 | Artifact detection |
| `spatial_radius` | 20 | 15-30 | Neighborhood for correlation check |

### 18.6 Stage 3 Parameters (Template Sweep)

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `template_threshold` | 4.0 σ | 3.0-6.0 | Event detection sensitivity |
| `spatial_pool_radius` | 8 | 6-12 | Soma-radius for spatial pooling |
| `spatial_pool_threshold` | 3.0 σ | 2.0-5.0 | Spatial coherence criterion |
| `cluster_distance` | 12 px | 8-16 | Max distance for event accumulation |
| `min_event_separation` | 2.0 s | 1.0-3.0 | Temporal independence threshold |

### 18.7 Gate 3 Parameters

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `min_waveform_r2` | 0.6 | 0.4-0.8 | Waveform shape match |
| `max_rise_decay_ratio` | 0.5 | 0.3-0.7 | Transient asymmetry |
| `anticorr_threshold` | -0.5 | -0.7 to -0.3 | Artifact detection |
| `min_solidity` | 0.5 | 0.4-0.6 | Spatial compactness |

### 18.8 Stage 4 Parameters (Tonic Search)

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `bandpass_windows` | [(0.5,2.0),(0.1,1.0),(0.05,0.5)] | — | Multi-scale frequency sweep |
| `filter_order` | 4 | 2-6 | Butterworth filter steepness |
| `n_svd_components` | 300 | 200-500 | Temporal compression |
| `neighbor_radius_inner` | 6 px | 4-8 | Correlation contrast inner radius |
| `neighbor_radius_outer` | 15 px | 10-20 | Correlation contrast outer radius |
| `min_contrast` | 0.10 | 0.05-0.20 | Correlation contrast threshold |
| `min_area` | 80 px | 50-120 | Minimum cluster size |
| `max_area` | 350 px | 250-500 | Maximum cluster size |
| `min_solidity` | 0.6 | 0.5-0.7 | Shape compactness |
| `max_eccentricity` | 0.85 | 0.8-0.9 | Shape elongation tolerance |
| `iou_merge_threshold` | 0.3 | 0.2-0.5 | Cross-window candidate merge |

### 18.9 Gate 4 Parameters

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `min_corr_contrast` | 0.10 | 0.05-0.15 | Correlation contrast cutoff |
| `max_motion_corr` | 0.3 | 0.2-0.4 | Motion artifact rejection |
| `anticorr_threshold` | -0.5 | -0.7 to -0.3 | Artifact detection |
| `min_mean_intensity_pct` | 25 | 10-40 | Minimum brightness percentile |

### 18.10 Trace Extraction Parameters

| Parameter | Default | Range | Affects |
|-----------|---------|-------|---------|
| `neuropil_coeff` | 0.7 | 0.5-1.0 | Neuropil subtraction strength |
| `neuropil_inner_buffer` | 2 px | 1-4 | Gap between ROI and neuropil annulus |
| `neuropil_outer_radius` | 15 px | 10-25 | Neuropil annulus extent |
| `baseline_window` | 60 s | 30-120 | Sliding window for F0 estimation |
| `baseline_percentile` | 10 | 5-20 | Percentile for F0 |
| `tonic_baseline_window` | 120 s | 60-300 | Wider window for tonic neurons |

---

## 19. Challenge-to-Stage Mapping

Reference to the detection challenges documented in
`roi-pipeline-algorithm.md` Part 1.

| Challenge | Primary stages that address it |
|-----------|-------------------------------|
| 1. GRIN optical aberrations | Foundation (L+S removes uneven illumination), Stage 1 (Cellpose tolerates non-circular shapes; tile normalization) |
| 2. Low SNR at depth | Foundation (denoising), Stage 1 (permissive thresholds), Stage 4 (correlation averages over time) |
| 3. Z-plane contamination | Unified QC (spatial blur + amplitude attenuation features reject ghosts) |
| 4. Sparsely firing neurons | Stage 3 (template sweep with event accumulation), Stage 2 (Suite2p SVD catches large transients) |
| 5. Tonic/baseline-firing neurons | Foundation (Vcorr summary image), Stage 1 (Vcorr as chan2), Stage 4 (multi-scale bandpass + correlation clustering on cleaned residual) |
| 6. Neuropil contamination | Foundation (L+S separation), Stage 4 bandpass (high-pass removes drift), Unified QC (per-ROI neuropil subtraction) |
| 7. Astrocyte ROI violations | Phase 2 (AQuA2 event-based detection) |
| 8. Neuron vs. astrocyte discrimination | Stage 4 bandpass windows (frequency content separates populations) |
| 9. Benchmark mismatch | Stage 1 (fine-tuning on GRIN data), Unified QC (classifier retraining), HITL (iterative adaptation) |

---

## 20. Implementation Roadmap

### Phase 0: Infrastructure (Weeks 1-2)

- [ ] Verify conda environment (`roigbiv`) has all dependencies
- [ ] Implement L+S calibration diagnostic tool (§3.3, §16.6)
- [ ] Run Foundation on 3-5 representative FOVs; select k_background
- [ ] Generate summary images from S; compare to raw projections
- [ ] Validate that Suite2p motion correction and SVD run correctly

### Phase 1A: Source Subtraction Engine (Weeks 2-4)

- [ ] Implement weighted spatial profile estimation (§5.1 Step 1)
- [ ] Implement simultaneous trace estimation via NNLS (§5.1 Step 2)
- [ ] Implement rank-1 subtraction (§5.1 Step 3)
- [ ] Implement post-subtraction diagnostics (§5.2)
- [ ] Validate on 3-5 FOVs: inspect residuals for ring artifacts
- [ ] **Milestone: source subtraction engine passes validation**

### Phase 1B: Stage 1 + Gate 1 (Weeks 3-5)

- [ ] Run Cellpose (current fine-tuned model) on S projections
- [ ] Evaluate: does the model need re-fine-tuning for S projections?
- [ ] Implement nuclear shadow detector (DoG filter)
- [ ] Implement Gate 1 feature extraction and decision logic
- [ ] Run source subtraction on Gate 1-accepted ROIs; inspect S₁
- [ ] Compare Stage 1 ROI counts to current pipeline baseline
- [ ] **Milestone: Stage 1 → Gate 1 → subtraction chain validated**

### Phase 1C: Stage 2 + Gate 2 (Weeks 5-7)

- [ ] Run Suite2p on raw registered movie
- [ ] Implement IoU-based filtering against Stage 1 results
- [ ] Implement Gate 2 temporal cross-validation checks
- [ ] Run source subtraction on Gate 2-accepted ROIs; inspect S₂
- [ ] Compare combined Stage 1+2 ROI counts to current pipeline
- [ ] **Milestone: Stages 1-2 chain validated**

### Phase 1D: Stage 3 + Gate 3 (Weeks 7-10)

- [ ] Implement template bank generation (§9.1)
- [ ] Implement FFT-based per-pixel convolution (§9.2)
- [ ] Implement spatial coherence evaluation (§9.3)
- [ ] Implement event accumulation clustering (§9.4)
- [ ] Implement Gate 3 waveform validation
- [ ] Run source subtraction on Gate 3-accepted ROIs; inspect S₃
- [ ] **Milestone: Stages 1-3 chain validated**

### Phase 1E: Stage 4 + Gate 4 (Weeks 10-13)

- [ ] Implement per-pixel detrending
- [ ] Implement multi-scale bandpass filtering (§11.2)
- [ ] Implement correlation contrast computation (§11.4)
- [ ] Implement correlation clustering with morphological filters (§11.5)
- [ ] Implement Gate 4 validation (including motion regressor)
- [ ] **Milestone: Full 4-stage pipeline operational**

### Phase 1F: Unified QC + HITL (Weeks 13-16)

- [ ] Implement unified feature extraction (§13.1)
- [ ] Implement activity-type classification (§13.3)
- [ ] Implement overlapping cell trace correction (§13.4)
- [ ] Implement full trace extraction pipeline (§13.2, §13.5, §13.6)
- [ ] Implement HITL review protocol with stage-specific visualizations
- [ ] Run 2-3 HITL rounds on representative FOVs
- [ ] **Milestone: Phase 1 pipeline complete and converging**

### Phase 1G: Validation & Optimization (Weeks 16-20)

- [ ] Process full dataset through the pipeline
- [ ] Compare to manual annotations (stratified by activity type)
- [ ] Identify remaining failure modes; update blindspots document
- [ ] Optimize performance bottlenecks
- [ ] Document final parameters in pipeline log
- [ ] **Milestone: Phase 1 pipeline validated for production use**

### Phase 2: Astrocyte Extension (Timeline TBD)

- [ ] Fine-tune separate Cellpose model for astrocyte somata
- [ ] Integrate AQuA2 with STARDUST protocol
- [ ] Implement two-tier detection (soma + process events)
- [ ] Validate on astrocyte imaging data from Paniccia et al. 2025
