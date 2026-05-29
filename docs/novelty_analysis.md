# ROI G. Biv: Architecture, Novel Methods, and Comparison with Existing Tools

A comprehensive analysis of the roigbiv consensus cell-detection pipeline for two-photon calcium imaging, its integration of third-party tools, its novel algorithmic contributions, and its position in the landscape of existing neuroscience imaging software.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: Cell Detection in Calcium Imaging](#2-background-cell-detection-in-calcium-imaging)
3. [Existing Tools: Landscape Analysis](#3-existing-tools-landscape-analysis)
4. [How roigbiv Integrates Third-Party Tools](#4-how-roigbiv-integrates-third-party-tools)
5. [Novel Methods in roigbiv](#5-novel-methods-in-roigbiv)
6. [Comparative Analysis](#6-comparative-analysis)
7. [Novelty Assessment](#7-novelty-assessment)
8. [Limitations and Future Directions](#8-limitations-and-future-directions)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

Identifying individual neurons in two-photon calcium imaging recordings is a foundational step in systems neuroscience. Every downstream analysis --- population dynamics, stimulus encoding, behavioral correlates --- depends on the accuracy and completeness of the initial ROI (region of interest) detection. Yet this step remains surprisingly fragile. The field has produced a rich ecosystem of detection tools, each built on different assumptions about what a neuron looks like and how it behaves. Suite2p finds neurons by their temporal correlations. CaImAn models the data as a nonnegative matrix factorization problem. Cellpose segments cell shapes using learned gradient flows. Each tool excels under the conditions it was designed for, and each has blind spots.

The problem is especially acute for GRIN (gradient-index) lens imaging, a widely used technique for recording from deep brain structures in freely moving animals. GRIN lenses introduce optical aberrations --- vignetting that dims the FOV periphery, reduced numerical aperture that lowers signal-to-noise ratio (SNR), and chromatic distortion that warps cell boundaries. These artifacts degrade the performance of algorithms tuned for cranial-window imaging, where optics are nearly diffraction-limited.

Beyond optical challenges, neurons themselves present a fundamental detection problem through their activity heterogeneity. Not all neurons are created equal in the eyes of a detection algorithm:

- **Phasic neurons** fire in discrete bursts, producing high-contrast calcium transients. These are the easiest to detect --- temporal correlation methods like Suite2p's activity mode find them reliably.
- **Tonic neurons** fire near-continuously, producing elevated but relatively constant fluorescence. Their low temporal variance makes them nearly invisible to correlation-based detectors.
- **Sparse neurons** fire only a handful of times during a recording session. Their rare events may not produce enough statistical power for temporal detection.
- **Silent neurons** do not fire at all during the recording window. They are anatomically present but functionally invisible.

No single existing tool reliably captures all four types. This is the gap that roigbiv fills.

**ROI G. Biv** is a consensus cell-detection pipeline that runs three independent detection branches --- Cellpose (spatial segmentation), Suite2p (temporal and anatomical detection in two passes), and a custom tonic neuron detector --- then merges their outputs through IoU-based matching with transitive grouping. The result is a confidence-tiered set of ROIs where each cell carries provenance information about which detectors found it and how strongly they agreed. Rather than replacing any single tool, roigbiv orchestrates existing tools and introduces new algorithms to cover the detection landscape more completely than any tool operating alone.

---

## 2. Background: Cell Detection in Calcium Imaging

### 2.1 Two-Photon Calcium Imaging Fundamentals

Two-photon laser scanning microscopy enables optical recording of neural activity in living tissue at cellular resolution. Genetically encoded calcium indicators (GECIs), most commonly from the GCaMP family, produce fluorescence changes proportional to intracellular calcium concentration, which rises sharply during action potentials. A typical experiment records a time-lapse movie at 15--30 Hz, producing a 3D array of shape (T, H, W) where T ranges from thousands to hundreds of thousands of frames.

The raw signal is contaminated by multiple noise sources: photon shot noise (Poisson), detector noise, neuropil fluorescence from out-of-focus processes, and hemodynamic artifacts from blood vessel pulsation. Extracting clean single-cell calcium traces requires three interleaved problems to be solved:

1. **Spatial segmentation**: Identifying which pixels belong to each cell body.
2. **Temporal demixing**: Separating overlapping fluorescence contributions from neighboring cells and background neuropil.
3. **Signal extraction**: Computing a corrected fluorescence trace (typically dF/F) for each identified cell, and optionally deconvolving to estimate spike trains.

### 2.2 GRIN Lens Challenges

GRIN (gradient-index) lenses are cylindrical relay optics implanted into the brain to image deep structures (e.g., ventral striatum, hippocampus, prefrontal cortex) that are inaccessible to conventional cranial-window microscopy. While they enable groundbreaking experiments in freely moving animals, they introduce several optical artifacts:

- **Vignetting**: Light transmission falls off toward the lens periphery, creating a brightness gradient across the FOV. Cells near the edge appear dimmer regardless of their actual activity.
- **Reduced SNR**: The numerical aperture of GRIN lenses is lower than that of conventional objectives, resulting in fewer collected photons per cell and higher relative noise.
- **Optical distortion**: Aberrations in the gradient-index medium produce non-uniform point-spread functions across the FOV, blurring cell boundaries unevenly.
- **Small FOV**: Typical GRIN lens FOVs are 200--500 micrometers in diameter, limiting the number of cells per recording and making every missed cell a larger proportional loss.

These challenges mean that detection parameters tuned for cranial-window data (where SNR is high, vignetting is negligible, and FOVs are large) systematically underperform on GRIN lens recordings. Algorithms need more permissive detection thresholds, vignetting compensation, and tolerance for irregular cell boundaries.

### 2.3 Cell Activity Heterogeneity

The four activity types introduced above --- phasic, tonic, sparse, and silent --- have distinct statistical signatures that interact differently with each detection paradigm:

| Activity Type | Temporal Variance | Mean Fluorescence | Skewness | Best Detector |
|---|---|---|---|---|
| Phasic | High | Moderate | High (positive) | Temporal correlation (Suite2p activity) |
| Tonic | Low | High | Low | Custom (roigbiv Branch C) |
| Sparse | Variable | Low/Moderate | Variable | Temporal correlation (with permissive thresholds) |
| Silent | Near-zero | Variable | ~0 | Anatomy-only (Suite2p anatomy, Cellpose) |

This heterogeneity is not an edge case --- in many brain regions, tonically active neurons constitute a significant fraction of the recorded population. A pipeline that only captures phasic neurons may be systematically biased in its population-level conclusions.

---

## 3. Existing Tools: Landscape Analysis

### 3.1 Suite2p

**Reference**: Pachitariu et al. (2017). *Suite2p: beyond 10,000 neurons with standard two-photon microscopy.* bioRxiv.

Suite2p is arguably the most widely used open-source tool for two-photon ROI detection. It provides a complete pipeline from motion correction through spike deconvolution, with GPU acceleration for large-scale datasets.

**Algorithm.** Suite2p's detection proceeds in several stages:

1. **Registration**: Rigid and nonrigid motion correction using phase correlation. Nonrigid correction divides the frame into blocks (default 128x128 pixels) and estimates independent shifts per block.
2. **SVD compression**: The registered movie is projected into a low-dimensional SVD basis (typically 500--1000 components), reducing the temporal dimension while preserving signal.
3. **Candidate detection**: In activity mode (`anatomical_only=0`), Suite2p computes a pixel-level temporal correlation map (Vcorr) --- each pixel's correlation with its spatial neighbors over time. Neurons appear as islands of high correlation. In anatomy mode (`anatomical_only=1`), detection operates on the time-averaged mean image using spatial gradient and morphology cues, independent of temporal dynamics.
4. **Source extraction**: Detected candidates are refined using iterative sparse nonnegative matrix factorization. Each ROI is represented by a spatial footprint (which pixels belong to it) and a temporal component (its fluorescence trace over time).
5. **Neuropil model**: For each ROI, an annular surround mask is constructed (default: starting 2 pixels from the ROI boundary, minimum 350 pixels). The mean fluorescence of this annulus estimates the neuropil contamination, which is subtracted with a scaling coefficient (alpha).
6. **Classification**: A built-in random forest classifier assigns each ROI an `iscell` probability based on spatial and temporal features. Users can set a threshold to accept or reject candidates.
7. **Spike deconvolution**: The OASIS (Online Active Set method for Inference of Spikes) algorithm estimates spike times from the neuropil-corrected dF/F traces, modeling the calcium indicator decay as a single-exponential kernel with time constant tau.

**Strengths.** Suite2p is fast (GPU-accelerated), well-validated, and has a large user community. Its activity-based detection mode is excellent for neurons with strong temporal signatures. The integrated pipeline (registration through deconvolution) minimizes the need for external tools.

**Limitations.** The activity-based mode fundamentally relies on temporal variance --- neurons with low or constant firing rates produce weak temporal correlations and are systematically missed. The anatomy mode addresses this partially but assumes compact, roughly circular cell shapes (the spatial scale parameter controls expected neuron size). For GRIN lens data, where cell boundaries are blurred and SNR is lower, both modes require careful parameter tuning. Suite2p does not provide confidence tiers or multi-method validation --- each ROI is either accepted or rejected by the single classifier.

### 3.2 Cellpose

**Reference**: Stringer et al. (2021). *Cellpose: a generalist algorithm for cellular segmentation.* Nature Methods; Pachitariu & Stringer (2022). *Cellpose 2.0: how to train your own model.* Nature Methods.

Cellpose is a deep-learning-based cell segmentation algorithm that has achieved state-of-the-art generalization across diverse cell types and imaging modalities.

**Algorithm.** Rather than directly predicting which pixels belong to cells, Cellpose learns to predict two vector fields:

1. **Gradient flows**: For each pixel inside a cell, the model predicts horizontal and vertical gradient components pointing toward the cell's center of mass. At inference time, these flow fields are integrated forward to find where each pixel "converges," and pixels converging to the same point are grouped into one ROI.
2. **Cell probability map**: A per-pixel probability that the pixel belongs to any cell (versus background). This map is thresholded to determine which pixels are candidates for segmentation.

This gradient-flow representation is what gives Cellpose its key advantage: it can segment arbitrarily shaped cells without assuming circular or elliptical morphology. The flow fields naturally handle touching cells because neighboring cells' flows point to different centers.

**Architecture.** Cellpose uses a U-Net backbone with residual connections and a style vector mechanism that enables size invariance. The style vector encodes global image properties (brightness, contrast, cell density) and modulates the network's behavior across different scales. The model supports multi-channel input (e.g., cytoplasm + nucleus, or mean + Vcorr).

**Pre-trained models.** Cellpose ships with progressively improved general-purpose models: `cyto` (original), `cyto2` (expanded training set), and `cyto3` (largest and most diverse training set). These models generalize well to cell types not seen during training, but performance on specialized imaging modalities (like GRIN lens two-photon) can be significantly improved with fine-tuning.

**Fine-tuning.** Cellpose 2.0 introduced a human-in-the-loop (HITL) workflow: run inference, correct segmentation errors manually, and retrain the model on the corrected labels. This iterative process typically converges in 3--5 rounds, with diminishing returns below ~5% ROI changes per round.

**Cellpose 3.0.** The latest major version added image restoration (denoising) as a preprocessing step. A separate denoising network can be applied before segmentation to improve performance on noisy images. This is optionally available in roigbiv via the `denoise` configuration flag.

**Strengths.** Cellpose produces smooth, anatomically accurate cell boundaries (compared to the pixelated footprints of NMF-based methods). It generalizes across cell types and imaging conditions. Fine-tuning from a strong pre-trained base requires relatively few annotated examples (10--50 FOVs is often sufficient). Multi-channel input allows incorporating functional information (Vcorr) alongside anatomy (mean image).

**Limitations.** Cellpose is purely spatial --- it processes single images, not movies. It has no notion of temporal dynamics, calcium transients, or fluorescence traces. This means it cannot distinguish active neurons from bright but inactive structures (blood vessels, glia, artifacts). Without fine-tuning, its pre-trained models may underperform on GRIN lens data where cell boundaries are optically distorted. Inference is relatively slow without GPU acceleration.

### 3.3 CaImAn (CNMF / CNMF-E)

**Reference**: Giovannucci et al. (2019). *CaImAn: An open source tool for scalable calcium imaging data analysis.* eLife.

CaImAn is a comprehensive calcium imaging analysis framework built around Constrained Nonnegative Matrix Factorization (CNMF).

**Algorithm.** CNMF models the recorded fluorescence movie Y as:

```
Y ≈ A · C + B + E
```

where A is a matrix of spatial footprints (one per neuron), C is a matrix of temporal components (calcium traces), B is a background model, and E is noise. The algorithm jointly estimates A, C, and B by solving a constrained optimization problem that enforces:

- **Non-negativity**: Spatial footprints and calcium traces are non-negative.
- **Sparsity**: Spatial footprints are spatially localized.
- **Temporal smoothness**: Calcium traces follow autoregressive dynamics consistent with indicator kinetics.

**CNMF-E** (Extended) modifies the background model for microendoscopic (one-photon) data, where background fluorescence is large, spatially varying, and highly correlated. CNMF-E models the background as a low-rank matrix plus a locally correlated component, which is essential for one-photon imaging but may be unnecessary (or even counterproductive) for two-photon data.

**OnACID** (Online Analysis of Calcium Imaging Data) extends CNMF to process data in a streaming fashion, enabling real-time processing of incoming frames. This is valuable for closed-loop experiments but is not necessary for offline batch analysis.

**NoRMCorre** (Non-Rigid Motion Correction) provides piecewise rigid motion correction similar to Suite2p's approach, dividing the frame into overlapping patches and estimating independent shifts per patch.

**Component evaluation** uses a CNN-based classifier to assess the quality of detected components, rejecting likely artifacts.

**Strengths.** CNMF provides a principled generative model of the imaging data. Unlike pixel-level segmentation methods, it explicitly models the spatial footprint as a potentially diffuse and overlapping distribution, enabling demixing of spatially overlapping sources. The mathematical framework is well-understood, and the constraints (non-negativity, autoregressive dynamics) are biophysically motivated.

**Limitations.** CaImAn has many hyperparameters that require tuning per dataset (number of components, spatial and temporal regularization, background rank). CNMF-E's elaborate background model adds computational cost and may not be necessary for two-photon data. Initialization is important and can affect results --- seeding with too few or too many candidates leads to missed cells or split ROIs. The spatial footprints from NMF tend to be diffuse and lack the sharp boundaries that cell morphology would suggest.

### 3.4 MIN1PIPE

**Reference**: Lu et al. (2018). *MIN1PIPE: A Miniscope 1-Photon-Based Calcium Imaging Signal Extraction Pipeline.* Cell Reports.

MIN1PIPE is an automated pipeline specifically designed for microendoscopic (one-photon) calcium imaging through GRIN lenses.

**Algorithm.** MIN1PIPE proceeds in three stages:

1. **Preprocessing**: Background removal using morphological operations designed to handle the large, spatially varying background fluorescence characteristic of one-photon imaging.
2. **Seed detection**: ROI candidates are seeded from a local correlation image (similar to Suite2p's Vcorr), identifying locations where neighboring pixels are temporally correlated.
3. **CNMF refinement**: Seeds are refined using a variant of constrained nonnegative matrix factorization, jointly optimizing spatial footprints and temporal components.
4. **Neural network quality control**: A trained classifier rejects artifact components.

**Strengths.** MIN1PIPE is purpose-built for the specific challenges of GRIN lens imaging with one-photon microscopy, including large background fluctuations and lower spatial resolution. Its automated end-to-end design requires minimal user intervention.

**Limitations.** MIN1PIPE was designed for one-photon microendoscopic imaging, and its background model may not be appropriate for two-photon data (where background is much lower). The tool has a smaller user community than Suite2p or CaImAn and is not as actively maintained. It does not provide confidence tiers or multi-method consensus.

### 3.5 EXTRACT

**Reference**: Inan et al. (2021). *EXTRACT: a robust cell extraction tool for calcium imaging.* bioRxiv.

EXTRACT (Extraction by Robust Association with Cell Types) is a matrix-factorization-based tool that incorporates explicit cell-shape priors.

**Algorithm.** EXTRACT models the data using a factorization similar to CNMF but adds a cell-shape prior (typically Gaussian or elliptical) to the spatial footprint estimation. This prior regularizes the spatial components to be smooth, compact, and cell-shaped, reducing the incidence of fragmented or diffuse footprints.

The key innovation is robust estimation: EXTRACT uses M-estimation (a generalization of least-squares that downweights outlier pixels) to handle contamination from artifacts, motion residuals, or nearby high-fluorescence objects. This makes the algorithm less sensitive to occasional bright frames or neuropil contamination.

**Strengths.** The explicit cell model improves spatial precision and reduces artifact contamination. EXTRACT handles both one-photon and two-photon data. The robust estimation framework gracefully handles outlier pixels that would bias standard NMF.

**Limitations.** The parametric cell-shape assumption (Gaussian/elliptical) may not fit irregular cell morphologies, particularly for non-neuronal cell types or cells with visible processes. The user community is smaller than Suite2p or CaImAn. Like other NMF-based methods, EXTRACT requires specification of the expected number of components and appropriate regularization parameters.

### 3.6 AQuA / AQuA2

**Reference**: Wang et al. (2019). *Accurate quantification of astrocyte and neurotransmitter fluorescence dynamics for single-cell and population-level physiology.* Nature Neuroscience.

AQuA (Activity Quantification and Analysis) represents a paradigm shift from ROI-based to event-based analysis.

**Algorithm.** Rather than first identifying fixed spatial ROIs and then extracting temporal traces, AQuA directly detects spatiotemporal events --- contiguous regions of space and time where fluorescence deviates significantly from baseline. Each event has a spatial extent (which pixels), a temporal extent (start and end frames), and propagation dynamics (how it spreads spatially over time).

This approach is particularly suited for astrocyte calcium signals, which manifest as spatially propagating waves rather than the point-source soma signals typical of neurons. AQuA2 improves upon the original with better handling of overlapping events and more robust baseline estimation.

**Strengths.** AQuA captures propagating signals that ROI-based methods fundamentally cannot represent. A calcium wave that traverses an astrocyte's territory would be split across multiple ROIs by conventional methods or missed entirely if the wave passes through regions not covered by any ROI. Event-based analysis preserves the spatiotemporal structure of the signal.

**Limitations.** The event-based paradigm produces fundamentally different output than ROI-based methods (events rather than traces), making it difficult to integrate with standard population-level analysis workflows that expect (n_rois, T) fluorescence matrices. Computational cost is high due to the need to search the full spatiotemporal volume. The method is primarily designed for astrocyte signals and may not provide advantages for neuronal point-source signals.

### 3.7 AstroCaST

**Reference**: Rupprecht et al. (2024). *AstroCaST: Automated detection of astrocyte calcium signals.*

AstroCaST is a specialized tool for detecting calcium signals in astrocytes, built around graph-based event detection.

**Algorithm.** AstroCaST constructs a spatial correlation network where each pixel is a node and edges represent temporal correlation between neighboring pixels. Events are detected as connected subgraphs that activate above a threshold. The graph structure naturally captures the spatially extended, irregularly shaped domains of astrocyte calcium activity.

**Strengths.** Purpose-built for astrocyte biology, AstroCaST handles the slow (seconds to tens of seconds), spatially propagating signals that are characteristic of astrocyte calcium dynamics. The graph-based representation does not assume compact, circular ROIs.

**Limitations.** AstroCaST is narrowly focused on astrocyte signals and is not designed for neuronal calcium imaging. Its graph-based detection paradigm does not produce the standard ROI masks and fluorescence traces expected by most downstream analysis pipelines.

### 3.8 STNeuroNet

**Reference**: Soltanian-Zadeh et al. (2019). *Fast and robust active neuron segmentation in two-photon calcium imaging using spatiotemporal deep learning.* PNAS.

STNeuroNet applies deep learning (U-Net architecture) to neuron segmentation in calcium imaging data.

**Algorithm.** STNeuroNet trains a U-Net semantic segmentation model on labels generated by Suite2p. The training data consists of mean/correlation projections as input and Suite2p-detected ROIs as ground-truth masks. At inference time, the trained model produces a pixel-level segmentation map that can be converted to individual ROI masks.

**Strengths.** Once trained, inference is very fast (orders of magnitude faster than iterative NMF methods). The model can potentially generalize across datasets once trained on a representative corpus.

**Limitations.** STNeuroNet inherits the biases of its training labels. Since it is trained on Suite2p outputs, it can only learn to replicate Suite2p's detection --- it cannot find cells that Suite2p misses. This creates a ceiling effect: the model's upper bound of performance is the performance of the tool that generated its training data. Additionally, the model requires retraining for new preparations (different brain regions, indicators, or imaging conditions), and it has no temporal component.

### 3.9 Mesmerize

**Reference**: Choudhary et al. (2022). *Mesmerize: a dynamically adaptable user interface platform for calcium imaging analysis.*

Mesmerize is a GUI-based platform that wraps CaImAn and other analysis tools, providing a unified interface for calcium imaging workflows.

**Algorithm.** Mesmerize does not introduce new detection algorithms. Instead, it provides a graphical interface for configuring and running CaImAn (CNMF, CNMF-E, OnACID), managing batch processing, and visualizing results. It also supports integration with other tools through a plugin system.

**Strengths.** Mesmerize lowers the barrier to entry for CaImAn, which has a steep learning curve due to its many parameters. The visualization tools are valuable for quality control and parameter tuning. Batch processing management simplifies large-scale analyses.

**Limitations.** Mesmerize's detection capabilities are entirely dependent on the underlying tools (primarily CaImAn). It does not provide multi-method consensus or novel detection algorithms. The Qt-based GUI is heavier to deploy than web-based alternatives.

### 3.10 ABLE

**Reference**: Reynolds et al. (2017). *ABLE: An activity-based level set segmentation algorithm for two-photon calcium imaging data.* eNeuro.

ABLE uses active contour (level set) methods for ROI segmentation in calcium imaging.

**Algorithm.** ABLE initializes contours at candidate cell locations (typically from a correlation or mean image) and iteratively refines the boundaries by minimizing an energy functional that balances internal contour smoothness against external image forces (intensity gradients). The "activity-based" component incorporates temporal information by weighting the energy functional with correlation or variance maps.

**Strengths.** Active contour methods produce smooth, anatomically meaningful boundaries that follow natural cell edges. The energy minimization framework provides a principled way to balance spatial precision against noise robustness.

**Limitations.** Active contour methods are sensitive to initialization --- poor seed locations lead to incorrect convergence. They are computationally slow for large FOVs with many cells, as each contour evolves independently. The sequential nature of level set evolution does not scale well to modern large-FOV datasets.

---

## 4. How roigbiv Integrates Third-Party Tools

roigbiv does not simply call Suite2p and Cellpose with default parameters. Each integration involves deliberate modifications to default behavior, custom parameter configurations, and novel ways of consuming each tool's output. This section details exactly what is used, what is changed, and why.

### 4.1 Suite2p Integration (Branch B)

**Location**: `roigbiv/suite2p.py`

roigbiv runs Suite2p in two independent passes, each producing a complete set of ROI candidates:

**Pass 1: Activity-based detection** (`anatomical_only=0`). This is Suite2p's default mode, detecting neurons by temporal correlation. Pixels that fluctuate together over time are clustered into candidate ROIs. This pass excels at finding phasic neurons with strong calcium transients.

**Pass 2: Anatomy-based detection** (`anatomical_only=1`). This mode operates on the time-averaged mean image, identifying cell-shaped structures by spatial morphology alone. This pass captures silent neurons that are visible in the mean image but produce no temporal signal.

The dual-pass design is motivated by the observation that these two modes have largely complementary blind spots: activity detection misses silent and tonic neurons, while anatomy detection misses dim or poorly shaped cells that are functionally active.

**Custom parameter configuration for GRIN lens imaging.** roigbiv's default ops differ from Suite2p's defaults in several key ways (configured in `configs/pipeline.yaml` and applied in `_build_ops()`):

| Parameter | Suite2p Default | roigbiv Value | Rationale |
|---|---|---|---|
| `threshold_scaling` | 1.0 | 0.5 | More permissive detection for lower-SNR GRIN data |
| `preclassify` | (varies) | 0.0 | Retain all candidates; consensus handles filtering |
| `allow_overlap` | True | False | Simplifies mask rasterization for IoU computation |
| `connected` | True | True | Enforces spatially contiguous ROIs |
| `delete_bin` | True | False | Retain `data.bin` for Branch C tonic detection and trace extraction |
| `tau` | 1.0 | 1.0 | GCaMP6s decay constant (unchanged) |
| `spatial_scale` | 0 | 0 | Auto-detect from data (unchanged) |

The critical change is `preclassify=0.0`, which disables Suite2p's built-in classifier from filtering any ROIs. In a standard Suite2p workflow, the classifier acts as a final quality gate, rejecting likely non-cell candidates. roigbiv deliberately bypasses this gate because the downstream consensus merge serves as a more powerful filter: an ROI confirmed by multiple independent methods is far more reliable than one accepted by a single classifier.

**Registration bypass.** For pre-motion-corrected data (filenames containing `_mc`), registration is disabled (`do_registration=0`). This avoids reprocessing already-corrected movies and sidesteps a version conflict where Suite2p's anatomy mode imports Cellpose 4.x internally, conflicting with the Cellpose 3.x used for roigbiv's fine-tuned models.

**Disk management.** Suite2p writes a `data.bin` file (a memory-mapped int16 array of the registered movie, typically ~500 MB per FOV). In a standard Suite2p workflow, this file can be deleted after processing. roigbiv retains it because it is consumed by both Branch C (tonic detection, which streams the movie through SVD) and Step 9 (trace extraction from merged ROI masks). The file is only deleted after all downstream stages complete.

**What is NOT changed.** roigbiv uses Suite2p's core SVD decomposition, sparse NMF source extraction, neuropil model, and registration algorithms without modification. The innovation is in running both detection modes and merging their outputs, not in modifying Suite2p's internal algorithms.

### 4.2 Cellpose Integration (Branch A)

**Location**: `roigbiv/union.py`, `scripts/train.py`

roigbiv uses Cellpose in two distinct capacities: as a fine-tuned segmentation model (Branch A), and as a probability scorer for consensus ROIs.

**Fine-tuned model training.** roigbiv fine-tunes from Cellpose's `cyto3` base model --- never from a previous checkpoint --- to avoid cumulative bias drift across training rounds. The training protocol (`scripts/train.py`) has several notable design decisions:

- **Dual-channel input**: Channel 1 = mean projection (anatomy), Channel 2 = Vcorr map (temporal correlation). This gives the model access to both spatial morphology and functional activity information in a single forward pass. The Vcorr channel encodes which pixels fluctuate together over time, providing a temporal activity signature that purely anatomical images lack.

- **Complementary projections**: Both mean and maximum-intensity projections are paired with the same ground-truth mask, effectively doubling the training set. Mean projections emphasize sustained brightness (anatomy), while maximum projections highlight the brightest moments (sparse activity peaks). Training on both exposes the model to different contrast regimes.

- **FOV-aware train/val split**: FOVs are shuffled as groups before splitting, preventing data leakage. If mean and max projections of the same FOV ended up in different splits, the model could effectively memorize cells from training by recognizing them in validation. The default validation fraction is 15% of FOVs.

- **Always from base**: The training script warns if a checkpoint is provided instead of `cyto3`, enforcing the principle that retraining from base produces more robust models than iterative fine-tuning from previous runs.

- **HITL workflow**: The intended workflow is iterative: run inference, correct errors in the Cellpose GUI, ingest corrections via `ingest_corrections.py`, retrain from `cyto3`, and repeat until fewer than 5% of ROIs change between rounds.

**Training results (run015).** The most recent training run used 99 image-mask pairs (85 train, 14 validation) with 300 epochs on an RTX 4060 GPU. Training completed in 11.6 minutes. Best validation loss was 0.1734 at epoch 280, with post-training AP metrics: AP@0.5 = 0.609, AP@0.75 = 0.288, AP@0.9 = 0.078.

**Inference parameter modifications.** Several inference parameters are adapted for GRIN lens imaging:

| Parameter | Cellpose Default | roigbiv Value | Rationale |
|---|---|---|---|
| `flow_threshold` | 0.4 | 0.6 | More permissive for optically aberrated boundaries |
| `cellprob_threshold` | 0.0 | -2.0 (inference) / -6.0 (scoring) | Accept dimmer cells; keep all candidates for consensus |
| `tile_norm_blocksize` | (auto) | 128 | Compensate for GRIN vignetting via tile-wise normalization |
| `diameter` | (varies) | 12 | Estimated cell diameter for GRIN lens neurons |
| `denoise` | False | False | Optional Cellpose 3.0 restoration (off by default) |

**Probability extraction mode.** In the union ROI building stage (`build_union()` in `roigbiv/union.py`), Cellpose is run with `cellprob_threshold=-6`, an extremely permissive setting that retains essentially all pixel-level candidates. The purpose is not to produce a hard segmentation but to extract the raw cell probability heatmap from Cellpose's `flows` output. This heatmap is then used to score each union ROI by its mean Cellpose probability over its pixel footprint --- transforming Cellpose from a hard segmenter into a soft probability scorer.

### 4.3 OASIS Integration (via Suite2p)

**Location**: `roigbiv/traces.py`

roigbiv uses the OASIS (Online Active Set method for Inference of Spikes) algorithm for spike deconvolution, accessed through Suite2p's implementation at `suite2p.extraction.dcnv.oasis`. OASIS models the calcium indicator fluorescence as a single-exponential decay convolved with a sparse spike train, and solves for the spike train using an efficient active-set algorithm.

The key adaptation is that OASIS is applied to roigbiv's own dF/F traces (computed from consensus-merged ROI masks with branch-aware baseline estimation) rather than to Suite2p's internally computed traces. This means the deconvolution benefits from roigbiv's improved neuropil correction and branch-aware baseline (Section 5.4).

The decay kernel parameter is computed as `g = exp(-1 / (tau * fs))`, where `tau` is the GCaMP decay time constant (default 1.0 s for GCaMP6s) and `fs` is the frame rate. The regularization parameter `lam` is set to 0 (no L1 penalty on spike amplitudes).

A fallback mechanism handles cases where Suite2p's OASIS implementation is unavailable (import errors, version conflicts): simple thresholding (rectified dF/F) is used instead, with a logged warning.

---

## 5. Novel Methods in roigbiv

This section describes algorithms and design decisions that are genuinely new --- not available in any single existing tool and not simply a reconfiguration of existing parameters.

### 5.1 Tonic Neuron Detection Algorithm (Branch C)

**Location**: `roigbiv/tonic.py`

**The problem.** Tonically active neurons fire near-continuously, producing elevated but relatively constant fluorescence. Their temporal variance is low compared to phasic neurons, which means they produce weak signals in both temporal correlation maps (Vcorr) and temporal variance maps. Suite2p's activity-based detection fundamentally relies on temporal variance to identify neurons, so tonic neurons are systematically missed. Suite2p's anatomy mode may catch them if they are sufficiently bright in the mean image, but dim tonic neurons fall through both nets.

This is not a hypothetical concern. Tonic firing is well-documented in brain regions commonly imaged through GRIN lenses, including the striatum (tonically active neurons, TANs), prefrontal cortex (persistent activity during working memory), and basal ganglia.

**The algorithm.** Branch C detects tonic neurons through a four-stage pipeline that identifies spatially compact regions of locally correlated activity in the calcium frequency band, operating entirely in compressed SVD space to maintain bounded memory usage.

#### Stage 1: Streaming Randomized SVD

The registered movie (`data.bin`, int16, shape T x Ly x Lx) is decomposed into a truncated SVD using a streaming randomized algorithm that requires only two passes over the data:

```
Pass 0: Compute temporal mean μ = (1/T) Σ_t X(t, :)
Pass 1: Random projection Y = (X - μ) @ Ω,  then QR(Y) → Q
Pass 2: B = Q^T @ (X - μ),  then SVD(B) → U_hat, S, V;  U = Q @ U_hat
```

where Ω is a Gaussian random matrix of shape (N, k) with k = n_components + n_oversamples (default 500 + 50 = 550). The result is a truncated SVD: U (T, 500) temporal components, S (500,) singular values, V (500, N) spatial components.

This approach is memory-bounded: the algorithm never loads the full movie into RAM. At any given time, only one chunk of frames (default 500 frames) is in memory, plus the running accumulation matrices. Peak RAM is approximately 2 GB for a 512x512 FOV, regardless of recording length.

#### Stage 2: Butterworth Bandpass Filtering

The temporal SVD components U are bandpass-filtered using a zero-phase Butterworth filter (`scipy.signal.sosfiltfilt`) to isolate the calcium dynamics frequency band:

- **Neuronal band**: 0.05--2.0 Hz (captures calcium transients from GCaMP6s at typical firing rates)
- **Astrocyte band**: 0.01--0.3 Hz (captures the slower calcium dynamics of astrocyte signals)

The filter order is 3 (default), providing 18 dB/octave rolloff. Zero-phase filtering (forward-backward application) ensures no phase distortion, preserving the temporal alignment of signals across spatial components.

The bandpass removes two types of contamination: low-frequency drift and photobleaching (below 0.05 Hz) and high-frequency noise (above 2.0 Hz). What remains is the neuronal calcium dynamics band, where tonic neurons produce weak but persistent fluctuations.

#### Stage 3: Local Correlation in SVD Space

This is the core novel computation. For each pixel p, the algorithm computes the Pearson correlation between p's bandpass-filtered timecourse and the mean timecourse of all pixels within a radius-8 neighborhood of p. Soma-sized islands of locally correlated activity produce high correlation scores; background noise, which is spatially uncorrelated, produces low scores.

The key innovation is that this computation is performed entirely in SVD space, without ever reconstructing the full movie. The pixel timecourse at location p is:

```
f(t, p) = U_bp(t, :) @ diag(S) @ V(:, p) = Σ_i U_bp(t,i) · S(i) · V(i, p)
```

Define weighted spatial components A(i, y, x) = S(i) * V(i, y*Lx + x). The local mean spatial components A_bar are computed by convolving each A(i) with a uniform kernel of diameter 2*radius+1 (approximating a disk neighborhood via `scipy.ndimage.uniform_filter`).

The temporal covariance matrix C = (1/T) U_bp^T @ U_bp is a small (k x k) matrix. The Pearson correlation is then:

```
cov(f, f_bar) = Σ_ij A(i,p) · C(i,j) · A_bar(j,p) - μ_f(p) · μ_f_bar(p)
var(f)        = Σ_ij A(i,p) · C(i,j) · A(j,p)   - μ_f(p)^2
var(f_bar)    = Σ_ij A_bar(i,p) · C(i,j) · A_bar(j,p) - μ_f_bar(p)^2
corr(p)       = cov / sqrt(var_f · var_fb)
```

This is computed for all N = Ly * Lx pixels simultaneously via vectorized matrix operations, producing a (Ly, Lx) local correlation map.

The result is a spatial map where soma-sized regions of correlated activity appear as bright islands. Tonic neurons, despite their low temporal variance, still exhibit local spatial correlation --- neighboring pixels within the same soma fluctuate together, just with lower amplitude than phasic neurons. By computing correlation rather than variance, the algorithm is sensitive to the synchrony of fluctuations regardless of their amplitude.

#### Stage 4: Morphological Filtering

The correlation map is thresholded at `corr_threshold=0.25` and processed through connected-component labeling (`scipy.ndimage.label`). Each connected component is evaluated using `skimage.measure.regionprops` with the following filters:

- **Area**: 80--350 pixels (rejects sub-cellular fragments and large artifacts)
- **Solidity**: ≥ 0.6 (area / convex hull area; rejects spindly, concave neuropil fragments)
- **Eccentricity**: ≤ 0.85 (eccentricity of equivalent ellipse; rejects elongated axon and process fragments)

Components passing all filters are assigned sequential uint16 labels, producing the Branch C ROI mask.

**Astrocyte extension.** The frequency band can be switched from neuronal (0.05--2.0 Hz) to astrocyte (0.01--0.3 Hz) via the `band` parameter, enabling detection of tonically active astrocytes using the same algorithmic framework. The morphological filter parameters would need recalibration for astrocyte-sized structures.

### 5.2 Three-Branch Consensus Merge

**Location**: `roigbiv/merge.py`

The consensus merge is the architectural centerpiece of roigbiv. It takes three independently generated ROI masks --- Branch A (Cellpose), Branch B (Suite2p, representing the earlier two-pass union), and Branch C (tonic detection) --- and produces a single merged mask with per-ROI confidence metadata.

#### Step 1: Pairwise IoU Matrices

For each pair of branches (A-B, A-C, B-C), the algorithm computes a pairwise Intersection-over-Union (IoU) matrix. For masks with n_a and n_b labeled ROIs respectively, the IoU matrix has shape (n_a, n_b):

```
IoU(i, j) = |pixels_i ∩ pixels_j| / |pixels_i ∪ pixels_j|
```

This quantifies the spatial overlap between every pair of ROIs across the two branches. An IoU of 1.0 means perfect overlap; 0.0 means no shared pixels.

#### Step 2: Hungarian Matching

For each pairwise IoU matrix, the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` applied to the negated IoU matrix) finds the globally optimal one-to-one assignment that maximizes total IoU. This is superior to greedy matching (which pairs the highest-IoU pair first, then the next, etc.) because it considers the global assignment problem --- it may sacrifice a slightly better individual match to achieve a better overall assignment.

Only assignments with IoU ≥ `iou_threshold` (default 0.3) are retained as genuine matches. The threshold of 0.3 was empirically calibrated to account for the geometric mismatch between Cellpose's smooth flow-derived contours and Suite2p's pixelated NMF footprints for neurons of approximately 12--30 pixel diameter.

#### Step 3: Union-Find Transitive Grouping

Matched ROIs are linked in a union-find (disjoint-set) data structure with path compression. The critical feature of union-find is transitive grouping: if Branch A ROI #5 matches Branch B ROI #12 (via A-B matching), and Branch B ROI #12 matches Branch C ROI #3 (via B-C matching), then all three are grouped into a single cluster --- even if A #5 and C #3 do not directly overlap (their IoU might be below threshold due to different spatial footprint definitions).

This transitive property is important because different detectors can produce different pixel footprints for the same underlying cell. A cell might be compact in Cellpose's output (tight boundary around the soma) but more diffuse in Suite2p's NMF footprint (including some surrounding pixels). The union-find ensures that indirect matches through a shared third branch are recognized.

The implementation uses path compression for efficient `find` operations:

```python
def find(self, x):
    while self.parent[x] != x:
        self.parent[x] = self.parent[self.parent[x]]  # path compression
        x = self.parent[x]
    return x
```

#### Step 4: Tier Assignment

Each cluster is assigned a tier based on which branches contributed ROIs to the cluster:

| Tier | Branches | Interpretation |
|---|---|---|
| **ABC** | All three | Highest confidence: all methods agree |
| **AB** | Cellpose + Suite2p | High confidence: spatial and temporal agreement |
| **AC** | Cellpose + Tonic | Cellpose confirms tonic detection |
| **BC** | Suite2p + Tonic | Suite2p confirms tonic detection |
| **A** | Cellpose only | Moderate: spatially neuron-shaped but not temporally detected |
| **B** | Suite2p only | Moderate: temporally active but not Cellpose-confirmed |
| **C** | Tonic only | Candidate tonic neuron; flagged for manual review |

The tier system is inherently interpretable: an ABC ROI was independently found by three methods using different algorithmic principles (deep learning, sparse NMF, local correlation). This multi-evidence approach is fundamentally more robust than any single classifier's accept/reject decision.

#### Step 5: Pixel Priority Ordering

When building the merged mask from the cluster's constituent ROIs, pixels are assigned from the highest-priority branch first:

```
Cellpose (A) > Suite2p (B) > Tonic (C)
```

This ordering reflects boundary quality: Cellpose's gradient-flow-derived contours are the smoothest and most anatomically accurate; Suite2p's sparse NMF footprints tend to be pixelated; and tonic detection's thresholded correlation blobs have the coarsest boundaries.

For an ABC-tier ROI, the merged mask uses Cellpose's pixel footprint, discarding Suite2p's and tonic's boundaries. This ensures that the final mask has the best available spatial precision.

#### Step 6: Spatial Support Gating

C-only (tonic-only) ROIs are subjected to a spatial support gate: if the ROI's centroid falls in a region of the FOV where the mean image intensity is below the 25th percentile, the ROI is discarded. This heuristic addresses a specific failure mode of correlation-based detection in dark regions of the FOV (often near the GRIN lens periphery, where vignetting reduces signal). In these regions, low-amplitude noise correlations can pass the correlation threshold, producing false-positive tonic detections. The spatial support gate requires that a tonic-only detection be located in a region with sufficient baseline fluorescence to plausibly contain a neuron.

### 5.3 Cellpose Probability Scoring

**Location**: `roigbiv/union.py`

roigbiv introduces a novel use of Cellpose: rather than accepting or rejecting ROIs based on Cellpose's segmentation, it uses Cellpose as a continuous probability scorer for ROIs detected by other methods.

In the union ROI building stage, Cellpose is run with `cellprob_threshold=-6` --- an extremely permissive setting that causes Cellpose to produce a cell probability value for essentially every pixel in the image, rather than a binary mask. The `flows` output contains a 2D cell probability heatmap where each pixel's value represents Cellpose's estimated probability that the pixel belongs to any cell.

Each union ROI (regardless of which branch detected it) is then scored by computing the mean Cellpose probability over its pixel footprint. This transforms a per-pixel probability into a per-ROI confidence score.

The scoring is asymmetric: a high Cellpose probability score for a Suite2p-detected ROI adds independent spatial evidence that the detection is a real cell. A low score does not necessarily invalidate the detection (Cellpose may not have learned to recognize this particular cell morphology), but it flags the ROI for closer inspection.

The probability heatmap is saved as a TIFF (`{stem}_roi_cellprob.tif`) and the per-ROI scores are included in `scored_rois_summary.csv`, enabling researchers to apply their own post-hoc probability thresholds without re-running the pipeline. This deferred-decision design avoids committing to a fixed cutoff during processing and preserves the full continuous confidence landscape for downstream analysis.

### 5.4 Branch-Aware Trace Extraction

**Location**: `roigbiv/traces.py`

roigbiv's trace extraction pipeline operates on the consensus-merged ROI masks rather than on individual branches' masks, and incorporates branch provenance into signal processing decisions.

#### Annular Neuropil Masks

For each ROI, a neuropil annulus is constructed by:

1. Expanding the ROI boundary outward by `inner_radius` pixels (default 2) using binary dilation.
2. Growing the outer boundary until at least `min_pixels` (default 350) neuropil pixels are included, up to `max_expansion` (default 15) pixels beyond the inner boundary.
3. Excluding all ROI pixels (from any ROI, not just the current one) from the neuropil annulus.

If the expansion reaches `max_expansion` without finding enough pixels (in densely packed regions), a fallback allows overlap with other ROIs' pixels but still excludes the current ROI's own pixels.

#### Iterative Alpha Estimation

The neuropil contamination coefficient alpha (which determines how much neuropil signal to subtract from the ROI signal: Fcorr = F - alpha * Fneu) is estimated per-ROI using iterative bisection:

```python
for _ in range(n_iter):  # 10 iterations
    mid = (lo + hi) / 2.0
    corrected = F - mid * Fneu
    baseline = np.percentile(corrected, 10)
    if baseline > 0:
        lo = mid  # alpha too low; increase subtraction
    else:
        hi = mid  # alpha too high; decrease subtraction
```

This finds the alpha value where the 10th percentile of the corrected trace is approximately zero, ensuring that the baseline fluorescence (after neuropil subtraction) is non-negative without being artificially elevated.

#### Branch-Aware Baseline Estimation

The key innovation in trace extraction is the **branch-aware dF/F baseline**. When computing dF/F = (Fcorr - F0) / F0, the baseline F0 is estimated as a rolling 10th percentile over a temporal window. For standard (phasic) neurons, the window is `baseline_window_seconds` (default 60 seconds). For tonic neurons (identified by source_branches containing "C"), the window is multiplied by `tonic_multiplier` (default 2.0), giving a 120-second window.

This is motivated by the biophysics of tonic firing: a tonically active neuron's fluorescence fluctuates slowly around an elevated mean. A narrow baseline window (60 s) captures these slow fluctuations and incorrectly interprets them as baseline, producing artificially low dF/F values. A wider window (120 s) provides a more stable estimate of the true baseline, preserving the tonic neuron's elevated signal.

The ROI's branch membership is extracted from the merge records:

```python
for rec in merge_records:
    if "C" in rec.get("source_branches", ""):
        is_tonic[rec["roi_id"] - 1] = True
```

This creates a feedback loop between detection and signal extraction: the merge stage identifies which ROIs are tonic, and the trace extraction stage uses this information to adjust its processing parameters accordingly.

### 5.5 Two-Stage Activity Classification

**Location**: `roigbiv/classify.py`

roigbiv implements a two-stage automated classification that assigns each ROI both a quality label (cell/not-cell) and an activity-type label.

#### Feature Extraction

Ten features are computed per ROI:

**Temporal features** (from dF/F traces):
- `std`: Standard deviation (overall signal variability)
- `skew`: Skewness (asymmetry; high skew indicates discrete transients on a quiet baseline)
- `pct_range`: 95th minus 5th percentile (dynamic range, robust to outliers)
- `snr`: max(dF/F) / std(dF/F) (peak signal relative to noise)
- `mean_F`: Mean raw fluorescence (baseline brightness)
- `cv`: Coefficient of variation of raw F (std/mean; low for tonic neurons)
- `n_transients`: Number of threshold crossings above 2*std (event rate)

**Spatial features** (from ROI mask):
- `area_px`: ROI area in pixels
- `compact`: Circularity = 4*pi*area / perimeter^2 (1.0 for a circle; low for irregular shapes)

**Branch provenance** (from merge records):
- `source_branch`: Bitmask encoding which branches detected this ROI (A=1, B=2, C=4)

#### Stage A: Cell/Not-Cell

A rule-based rejection filter removes ROIs that fail any of:
- SNR < 2.0 (insufficient signal above noise)
- Area outside [30, 500] pixels (too small to be a soma, or too large)
- Compactness < 0.15 (too irregular to be a cell body)

#### Stage B: Activity Type Classification

ROIs passing Stage A are classified by activity type using a priority rule chain:

1. **Phasic**: skewness ≥ 1.5 (sharp transients on a quiet baseline)
2. **Tonic**: Branch C source (bitmask & 4) OR coefficient of variation < 0.3
3. **Sparse**: fewer than 5 threshold crossings (rare events)
4. **Ambiguous**: none of the above criteria met

The tonic classification explicitly uses Branch C membership as evidence --- this is a deliberate feedback loop where the detection algorithm's output informs the classification. An ROI that was detected by the tonic detector (Branch C) is classified as tonic regardless of its temporal statistics, on the reasoning that the tonic detection algorithm has already validated the presence of local correlation in the calcium frequency band.

### 5.6 Interactive Curation Interface

**Location**: `roigbiv/curator.py`

roigbiv includes a Napari-based interactive curation interface that allows researchers to manually review and correct the automated detection results. While GUI-based ROI editing is not itself novel, several design decisions are specific to the consensus pipeline context.

**Full undo/redo with state snapshots.** The `CurationHistory` class maintains a stack of complete (mask, records) state snapshots, enabling arbitrary undo/redo of any operation. This is particularly important for merge and split operations, which are difficult to reverse analytically.

**Branch provenance preservation.** When ROIs are merged, the resulting ROI inherits the `source_branches` metadata from all constituent ROIs. When ROIs are split, the fragments retain the parent's branch provenance. New manually drawn ROIs receive a `MANUAL` tier and branch designation. This ensures that the tier and provenance information remains meaningful after curation.

**Split via drawn separators.** The split operation allows researchers to draw separator lines (polylines) across a touching-cell pair. The separator is rasterized using `scipy.ndimage.line`, dilated to ensure a clean cut, and the remaining connected components are assigned new ROI IDs. This is more ergonomic than the typical approach of erasing pixels and redrawing, because it preserves the original cell boundaries everywhere except at the division point.

**Curation log.** Every action (delete, merge, split, draw) is logged to a JSON file with timestamps and ROI IDs, creating an audit trail of manual corrections. This is valuable for reproducibility and for understanding how much manual intervention the automated pipeline required.

---

## 6. Comparative Analysis

### 6.1 Feature Comparison

| Feature | roigbiv | Suite2p | Cellpose | CaImAn | MIN1PIPE | EXTRACT | AQuA | STNeuroNet | ABLE |
|---|---|---|---|---|---|---|---|---|---|
| **Detection paradigm** | Multi-branch consensus | Temporal corr. + anatomy | Spatial (flow fields) | NMF (generative) | Correlation + NMF | Robust NMF | Event-based | Supervised DL | Active contour |
| **Handles phasic neurons** | Yes (Branch B) | Yes (activity mode) | Indirectly (bright soma) | Yes | Yes | Yes | Yes | Yes | Yes |
| **Handles tonic neurons** | Yes (Branch C) | No | No | Partially (if initialized) | No | No | No | No | No |
| **Handles silent neurons** | Yes (Branch B anatomy + A) | Yes (anatomy mode) | Yes | If initialized | Partially | If initialized | No | Partially | Partially |
| **GRIN lens optimized** | Yes (custom params) | Manual tuning | Requires fine-tuning | Manual tuning | Yes (1p only) | Manual tuning | No | No | No |
| **Confidence tiers** | Yes (ABC/AB/AC/BC/A/B/C) | Binary (iscell) | Binary (mask/no mask) | CNN quality score | Binary | No | N/A | Binary | No |
| **Boundary quality** | High (Cellpose priority) | Pixelated (NMF) | High (flow fields) | Diffuse (NMF) | Moderate | Moderate | N/A | Moderate | High (level sets) |
| **Temporal demixing** | No | Partial (NMF) | No | Yes (CNMF) | Yes (CNMF) | Yes | Yes | No | No |
| **Neuropil correction** | Yes (branch-aware) | Yes (annular) | No | Yes (background model) | Yes | Yes | N/A | No | No |
| **Spike deconvolution** | Yes (OASIS) | Yes (OASIS) | No | Yes (AR model) | Yes | No | N/A | No | No |
| **Activity classification** | Yes (phasic/tonic/sparse) | No | No | No | No | No | Event types | No | No |
| **Interactive curation** | Yes (Napari + undo/redo) | Yes (built-in GUI) | Yes (GUI) | No (script-based) | No | No | Yes | No | No |
| **Batch processing** | Yes (resumable) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Web interface** | Yes (Streamlit) | No (Qt GUI) | No (Qt GUI) | No | No | No | No | No | No |
| **GPU acceleration** | Yes (Suite2p + Cellpose) | Yes | Yes | Partial | Partial | No | No | Yes | No |
| **Primary modality** | Two-photon (GRIN) | Two-photon | Any (generalist) | Both 1p and 2p | One-photon (GRIN) | Both 1p and 2p | Both | Two-photon | Two-photon |

### 6.2 Detection Completeness

The central claim of roigbiv is that its multi-branch architecture captures neurons that no single tool finds alone. This section examines the mechanistic basis for that claim.

**Phasic neurons** are detected by Branch B (Suite2p activity mode), which identifies spatially connected clusters of temporally correlated pixels. This is the same algorithm used by standalone Suite2p and is well-validated for this purpose. Branch A (Cellpose) also detects phasic neurons indirectly, as they tend to be bright in the mean image and produce visible Vcorr signatures.

**Tonic neurons** are the primary beneficiaries of roigbiv's multi-branch design. Suite2p's activity mode misses them because their near-constant firing produces low temporal variance. Suite2p's anatomy mode may detect them if they are sufficiently bright in the mean image. Cellpose may detect them if their morphology is distinguishable from background. But Branch C (tonic detection) is the only method that specifically targets them, using local correlation in a bandpass-filtered SVD space where even weak but persistent fluctuations produce detectable spatial correlation patterns. No other tool in the landscape (Table, Section 6.1) has an equivalent detection mode.

**Sparse neurons** are challenging for all methods because their rare events provide limited statistical power. Suite2p's activity mode may detect them if the events are strong enough, but with only a handful of transients in a multi-minute recording, the pixel-level temporal correlations may be weak. roigbiv's approach of retaining all Suite2p candidates (`preclassify=0.0`) and filtering downstream via consensus improves recall: a sparse neuron that appears borderline in Suite2p's classifier may be confirmed by Cellpose's independent spatial assessment.

**Silent neurons** are detected by Branch B (Suite2p anatomy mode) and Branch A (Cellpose), both of which operate on spatial features of the mean image rather than temporal dynamics. The consensus between these two methods provides higher confidence than either alone: a GOLD-tier ROI was independently identified by both temporal correlation and spatial morphology.

### 6.3 Boundary Quality

The spatial precision of ROI boundaries matters for downstream signal extraction. When an ROI's boundary extends too far, it includes neuropil pixels that contaminate the trace. When it is too tight, it misses dimmer peripheral pixels that carry genuine signal.

**Cellpose** produces the highest-quality boundaries because its gradient-flow representation naturally follows cell morphology. Flow vectors point toward cell centers, and pixel grouping follows the flow field, producing smooth contours that respect the actual cell boundary.

**Suite2p's NMF footprints** tend to be pixelated and sometimes irregularly shaped, because sparse NMF does not enforce boundary smoothness. The spatial components are optimized for temporal demixing accuracy, not spatial fidelity.

**CaImAn's CNMF footprints** are similar to Suite2p's but can be even more diffuse, as NMF allows fractional pixel membership (each pixel has a continuous weight for each component rather than a binary assignment).

**ABLE's level set boundaries** are smooth by construction (the level set is a continuous function), but can be sensitive to initialization and may settle into local optima.

roigbiv leverages this hierarchy through pixel priority ordering (A > B > C): when a cell is detected by Cellpose, its Cellpose boundary is used in the merged mask regardless of what Suite2p or Branch C found. This ensures that the final mask has the best available spatial precision for every cell that Cellpose detected. For cells detected only by Suite2p or Branch C, those methods' coarser boundaries are used as the best available option.

### 6.4 Confidence and Interpretability

Most existing tools produce a single confidence measure per ROI:

- **Suite2p**: Binary iscell classification (with a continuous probability score from 0 to 1)
- **Cellpose**: Binary mask (pixel is inside a cell or not)
- **CaImAn**: CNN-based quality score (continuous, but a single scalar)
- **EXTRACT**: No explicit confidence score

roigbiv provides multi-dimensional confidence:

1. **Tier** (which branches agree): ABC > AB/AC/BC > A/B/C --- encodes the number and type of independent methods that found the cell.
2. **Pairwise IoU scores** (iou_ab, iou_ac, iou_bc): Quantify how strongly each pair of methods agreed on the cell's spatial extent.
3. **Cellpose mean probability**: Continuous probability that Cellpose assigns to the ROI's pixels, independent of whether Cellpose detected it as a discrete segment.
4. **Suite2p iscell probability**: Suite2p's built-in classifier score.
5. **Activity type** (phasic/tonic/sparse/ambiguous): Classifies the cell's functional behavior.
6. **Review flag**: Boolean flag for C-only ROIs, indicating that manual review is recommended.

This multi-dimensional confidence allows researchers to make nuanced inclusion decisions. For a conservative analysis, one might include only ABC and AB-tier ROIs with Cellpose probability > 0.5. For a discovery-oriented analysis, one might include all tiers but weight downstream analyses by confidence. The pipeline does not impose a single threshold --- it provides the information for researchers to make informed, application-specific choices.

---

## 7. Novelty Assessment

This section provides a clear categorization of what is genuinely new in roigbiv, what represents novel integration of existing components, and what is existing work used as-is.

### 7.1 Genuinely Novel Contributions

These algorithms and design decisions are not available in any existing tool and represent original work:

1. **Tonic neuron detection via SVD-space bandpass local correlation.** No existing calcium imaging tool specifically targets tonically active neurons using this algorithmic approach. The combination of streaming randomized SVD, bandpass filtering in the temporal component space, and local correlation computation without full-movie reconstruction is novel. The closest existing approach is Suite2p's Vcorr map, which computes pixel-level temporal correlations but does not apply bandpass filtering to isolate the calcium dynamics band or use morphological filtering to extract soma-shaped regions.

2. **Three-branch consensus merge with union-find transitive grouping.** The specific combination of pairwise Hungarian matching across three independent detection branches, with transitive grouping via union-find and tier assignment from branch membership, is new. Multi-method ROI detection has been discussed conceptually in the literature, but roigbiv provides a concrete, implemented algorithm with defined tier semantics and spatial support gating.

3. **Cellpose probability scoring of non-Cellpose ROIs.** The technique of running Cellpose in probability-extraction mode (`cellprob_threshold=-6`) and using the resulting probability heatmap to score ROIs detected by other methods (Suite2p, tonic detector) is a novel application of Cellpose. Standard Cellpose usage produces hard segmentation masks; repurposing it as a continuous soft scorer is an original contribution.

4. **Branch-aware dF/F baseline estimation.** Adjusting the baseline estimation window per ROI based on detected activity type (wider window for tonic ROIs identified by Branch C) is not implemented in any existing tool. Standard approaches use a fixed baseline window for all ROIs, which is suboptimal for cells with different temporal dynamics.

5. **Spatial support gating for single-branch ROIs.** The heuristic of rejecting C-only ROIs whose centroids fall below the FOV 25th-percentile intensity addresses a specific failure mode of correlation-based detection in low-signal regions. This quality control step is tailored to the consensus pipeline context and is not found in standalone detection tools.

6. **Dual-channel mean + Vcorr Cellpose fine-tuning for GRIN lens.** While Cellpose supports multi-channel input and fine-tuning is established, the specific choice of temporal variance correlation (Vcorr) as the second channel alongside the mean projection, targeted at GRIN lens neuron segmentation, is a novel application. This channel pairing provides the model with both anatomical (mean) and functional (Vcorr) information, a combination not used in any published Cellpose fine-tuning study for calcium imaging.

### 7.2 Novel Integration (Existing Components, New Combination)

These contributions combine existing tools or techniques in ways not previously implemented:

1. **Suite2p dual-pass (activity + anatomy) merger.** Suite2p's activity and anatomy modes exist individually, but running both and merging their outputs via Hungarian IoU matching with GOLD/SILVER/BRONZE tier assignment is roigbiv's contribution. Standard Suite2p usage runs one mode and accepts its classifier output.

2. **Annular neuropil subtraction on consensus-merged ROIs.** Neuropil subtraction is a standard technique (implemented in both Suite2p and CaImAn), but applying it to ROIs defined by a multi-branch consensus mask --- where the spatial footprint was selected based on pixel priority ordering across branches --- is a new application context.

3. **Activity-type classification with branch provenance as a feature.** Rule-based classification of ROI activity types exists in various forms, but incorporating the detection source (which branches found the ROI) as a classification feature is novel. Specifically, using Branch C membership as evidence for tonic classification creates a deliberate feedback loop between detection and characterization.

4. **Training Cellpose on complementary projections (mean + max).** Pairing both mean and maximum projections with the same ground-truth mask to double the effective training set exposes the model to different contrast regimes from the same underlying data, a simple but effective data augmentation strategy.

### 7.3 Existing Work Used As-Is

These components are used without modification from their original implementations:

1. **Suite2p's core algorithms**: SVD decomposition, sparse nonnegative matrix factorization, iterative source extraction, rigid and nonrigid registration, neuropil annulus model, random forest classifier, and OASIS spike deconvolution are all used as provided by the Suite2p package.

2. **Cellpose's architecture and training API**: The gradient-flow segmentation model, U-Net backbone, style vector mechanism, and `train.train_seg()` training loop are used as provided by the Cellpose package.

3. **Hungarian algorithm**: `scipy.optimize.linear_sum_assignment` is a standard implementation of the Kuhn-Munkres algorithm for optimal bipartite matching.

4. **Butterworth filter design**: `scipy.signal.butter` and `scipy.signal.sosfiltfilt` implement standard digital filter design and zero-phase filtering.

5. **Connected component labeling and regionprops**: `scipy.ndimage.label` and `skimage.measure.regionprops` are standard image processing primitives.

6. **Randomized SVD**: The two-pass streaming algorithm follows the Halko-Martinsson-Tropp framework for randomized matrix approximation, adapted for memory-mapped I/O.

---

## 8. Limitations and Future Directions

### 8.1 Current Limitations

**No temporal demixing for spatially overlapping sources.** roigbiv enforces `allow_overlap=False` in Suite2p, producing non-overlapping ROI footprints. This simplifies IoU computation but prevents recovery of signals from spatially overlapping neurons. CaImAn's CNMF framework can demix overlapping sources because it models each pixel as a weighted combination of multiple components. In regions with dense cell packing (e.g., hippocampal CA1), non-overlapping masks may assign contested pixels to the wrong neuron.

**IoU sensitivity to spatial registration errors.** The consensus merge relies on IoU to match ROIs across branches. If the spatial registration between branches is imperfect (e.g., due to slight offsets in how Suite2p and Cellpose define cell boundaries), genuine matches may fall below the IoU threshold. The current threshold of 0.3 provides some margin, but severe registration errors could fragment clusters.

**Rule-based classification thresholds.** The cell/not-cell and activity-type classifications use fixed thresholds (SNR ≥ 2.0, area 30--500, compactness ≥ 0.15, skew ≥ 1.5, etc.) that were empirically tuned for ventral striatum neurons imaged with GCaMP6s through GRIN lenses. These thresholds may not generalize to other brain regions, indicators, or imaging conditions without recalibration. A learned classifier (e.g., random forest or gradient-boosted trees trained on the QC features) could adapt to new contexts more robustly.

**Computational cost.** Running three full detection passes (Suite2p activity, Suite2p anatomy, Cellpose inference) plus tonic detection, merge, trace extraction, and classification is significantly more expensive than running a single tool. For large-scale datasets (hundreds of FOVs), the total processing time may be substantial. The per-stage resumability (skipping FOVs with existing outputs) mitigates this for iterative workflows but does not reduce the initial cost.

**Single-plane, single-channel assumption.** The current pipeline assumes single-plane, single-channel (GCaMP) recordings. Multi-plane (volumetric) imaging would require registration across planes and 3D merging. Dual-color indicators (e.g., GCaMP + jRGECO) would require channel-specific processing pathways.

### 8.2 Future Directions

**Astrocyte adaptation.** The pipeline architecture supports astrocyte detection with parameter recalibration (documented in `docs/ASTROCYTE_PLAN.md`). Key changes include: switching Branch C's bandpass to 0.01--0.3 Hz, increasing the Cellpose diameter, adjusting or removing the activity-mode Suite2p pass (which is unreliable for astrocyte morphologies), and potentially substituting the Vcorr second channel with a standard deviation map.

**Additional detection branches.** The merge system (`merge_three_branches`) is designed to accept any combination of branch masks --- any branch can be None. This modularity enables future extensions: a CaImAn branch could provide CNMF-based detections; an event-based branch (AQuA-style) could contribute astrocyte territory masks; a deep-learning branch (trained on consensus labels) could provide fast inference.

**Learned classification.** Replacing the rule-based classifier with a trained model (using the QC features as inputs and manually curated cell/not-cell labels as targets) would enable adaptation to new imaging contexts without manual threshold tuning.

**Quantitative benchmarking.** A systematic comparison of roigbiv's detection sensitivity and specificity against Suite2p, Cellpose, and CaImAn on ground-truth-annotated datasets would provide rigorous evidence for the consensus approach's advantages. The training infrastructure already supports AP evaluation; extending this to multi-method comparison on held-out FOVs is a natural next step.

---

## 9. Conclusion

roigbiv addresses a specific and well-motivated gap in the calcium imaging analysis landscape: no single existing tool reliably detects the full heterogeneity of neuronal activity types present in a typical two-photon recording. Phasic, tonic, sparse, and silent neurons each have distinct statistical signatures that favor different algorithmic approaches, and a pipeline that only runs one algorithm necessarily has blind spots.

The consensus approach --- running three independent detection branches and merging their outputs via IoU matching with transitive grouping --- provides two key advantages over single-tool workflows:

1. **Completeness**: Each branch captures a population of neurons that the others miss. Suite2p's activity mode finds phasic neurons; its anatomy mode finds silent neurons; Cellpose provides high-quality boundaries for morphologically distinctive cells; and the tonic detection branch identifies the near-constant-firing neurons that all other tools systematically miss.

2. **Interpretability**: The tier system provides a built-in confidence measure based on multi-method agreement. An ABC-tier ROI (found by all three branches) carries more evidence than a B-only ROI (found only by Suite2p's activity mode). This multi-dimensional confidence allows researchers to make informed, application-specific inclusion decisions rather than accepting a single classifier's binary judgment.

The pipeline's architecture is modular and extensible. Any branch can be omitted (the merge handles None masks) or replaced. The tier system naturally accommodates additional branches. And the downstream processing (traces, classification, curation) operates on the merged output regardless of how many or which branches contributed.

The genuinely novel contributions --- tonic neuron detection via SVD-space local correlation, three-branch consensus merging with union-find transitive grouping, Cellpose probability scoring of non-Cellpose ROIs, and branch-aware trace extraction --- are each motivated by a specific gap in the existing tool landscape and are implemented in a way that builds on, rather than replaces, the established tools that the neuroscience community already trusts.
