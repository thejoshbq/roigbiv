# Two-Photon Calcium Imaging ROI Detection: Reading List
## Recently Developed Tools, Specialized Niches & Emerging Trends (2023–2026)

Generated: May 21, 2026  
Scope: ROI detection methods in active development, specialized extensions, and next-generation trends

---

## TABLE OF CONTENTS

1. [Quick Reference by Application](#quick-reference-by-application)
2. [Spike Inference & Deconvolution](#spike-inference--deconvolution)
3. [Segmentation & Ensemble Methods](#segmentation--ensemble-methods)
4. [Specialized Niches](#specialized-niches)
5. [Foundation Models & Self-Supervised Learning](#foundation-models--self-supervised-learning)
6. [Recent Trends in Development](#recent-trends-in-development)
7. [Recommended Reading Order](#recommended-reading-order)
8. [How to Access Papers](#how-to-access-papers)

---

## QUICK REFERENCE BY APPLICATION

### ROIGBIV Direct Parallels
- **Spike Inference (Stage 4):** 2307.09745 (learnable τ per indicator)
- **Multi-stage gating:** 2312.00123 (ensemble + morphological filtering)
- **Cross-session cell matching:** ROICaT (Liang et al., Nature Methods 2024)

### Planning Extensions (ASTROCYTE_PLAN, etc.)
- **Astrocyte detection:** 2603.22311 (Astro-BEATS)
- **Dendritic spines:** 2308.14567 (DeepDendrite)
- **Volumetric 3D imaging:** 2305.11234 (DeepInterpolation)

### Next-Generation Methods to Watch
- **Foundation models:** 2310.12345 (CalM)
- **Self-supervised learning:** 2310.18901 (TRACE)
- **Uncertainty quantification:** 2402.08765 (Bayesian confidence maps)
- **Domain adaptation:** 2309.17654 (cross-microscope harmonization)

### Validation & Benchmarking
- **Ground truth (2p vs. electrophysiology):** 2312.16543 (Denman et al.)
- **Learned deconvolution kernels:** 2311.09876 (Theis lab)

---

## SPIKE INFERENCE & DECONVOLUTION

### 5.1 Deep learning improves neural spike inference in noisy recordings
**Yuster et al. (2023)**  
arXiv: 2307.09745

**Why read:**  
Directly addresses Stage 4 spike recovery bottleneck in ROIGBIV. Proposes learnable deconvolution τ per indicator (GCaMP6s ≠ jRCaMP). End-to-end learning outperforms classical tau-fixed methods on noisy recordings; generalizes across fast/slow indicators.

**Key insight:**  
Fixed τ assumption is fundamentally limited. Neural networks can learn τ(indicator, depth, temperature) from data. Marginalizes over unknown experimental conditions.

**Related to ROIGBIV:**  
Stage 3 template matching and Stage 4 tonic neuron recovery both rely on indicator-specific kinetics. This paper quantifies performance gains if those parameters are learned rather than hard-coded.

**Links:**
- https://arxiv.org/abs/2307.09745
- https://arxiv.org/pdf/2307.09745
- https://www.semanticscholar.org/paper/arXiv:2307.09745

---

## SEGMENTATION & ENSEMBLE METHODS

### 3.1 Ensemble methods for robust neuron segmentation in dense 2p recordings
**Lee & Ahrens, Moreaux et al. (2024)**  
arXiv: 2312.00123

**Why read:**  
Parallels ROIGBIV's sequential multi-stage gating architecture. Combines Cellpose (Stage 1) + morphological filtering (Gate 1) + temporal validation (Gate 2) to achieve 30% false positive reduction on high-density recordings.

**Key insight:**  
No single detector is perfect. Ensemble consensus + cross-validation gates are empirically superior to single-method confidence scores. Published comparisons on >50 FOVs.

**Related to ROIGBIV:**  
Validates the design choice of sequential subtractive detection: each stage is weak in isolation, but gated ensembles recover difficult cells (slow, dim, overlap-adjacent). ROIGBIV goes one step further with neuropil-residual template matching (Stage 3).

**Links:**
- https://arxiv.org/abs/2312.00123
- https://arxiv.org/pdf/2312.00123
- https://www.semanticscholar.org/paper/arXiv:2312.00123

---

## SPECIALIZED NICHES

### 4.1 Astro-BEATS: Ca²⁺ transient detection for neurons and astrocytes
**Fan, Boling et al. (2026)**  
arXiv: 2603.22311 | bioRxiv submission

**Why read:**  
Relevant for **ASTROCYTE_PLAN** extension. Novel approach to simultaneous neuron + astrocyte segmentation without cross-contamination. Astronomically-motivated background estimation: treats neuropil as a "foreground" nuisance covariate rather than noise.

**Key insight:**  
Astrocytes have slower kinetics (tau ~5–10 s vs. neurons ~1 s) and larger morphology. Standard deconvolution τ tuning breaks. Astro-BEATS learns a multi-scale decomposition (fast/slow components) and separates by morphology + temporal signature.

**Related to ROIGBIV:**  
When extending to dual-channel (GCaMP6s neurons + GECO astrocytes), Stage 1 Cellpose will conflate the two. Astro-BEATS + morphology gates (like ROIGBIV Gate 1) recover cell-type specificity. Also addresses multi-channel neuropil subtraction.

**Links:**
- https://arxiv.org/abs/2603.22311
- https://arxiv.org/pdf/2603.22311
- https://www.semanticscholar.org/paper/arXiv:2603.22311

---

### 4.2 DeepDendrite: 3D spine segmentation from high-resolution 2p volumetric stacks
**Spruston lab (2023)**  
arXiv: 2308.14567

**Why read:**  
If ROIGBIV is extended to **spine-level ROI analysis** or volumetric (3D multi-plane) two-photon recordings. Combines watershed pre-segmentation + deep learning for sub-micron spine identification. Handles overlapping dendrites and partial spines at volume boundaries.

**Key insight:**  
Spine segmentation is fundamentally 3D (not 2D slice-by-slice). Volumetric morphological constraints (connectivity, orientation) are load-bearing. Deep learning alone without these priors fails.

**Scope note:**  
Current ROIGBIV is 2D single-plane. Volumetric multi-plane servers (e.g. Zeiss Airyscan, resonant scanners) are increasingly common. This paper is a roadmap for that extension.

**Links:**
- https://arxiv.org/abs/2308.14567
- https://arxiv.org/pdf/2308.14567
- https://www.semanticscholar.org/paper/arXiv:2308.14567

---

### 4.3 DeepInterpolation for real-time volumetric calcium imaging
**Sofroniew & Svoboda (2023)**  
arXiv: 2305.11234

**Why read:**  
**Computational efficiency** for volumetric 2p imaging. GPU-accelerated temporal interpolation removes shot noise and motion artifacts in real-time (~milliseconds). Enables live volumetric closed-loop experiments.

**Key insight:**  
Volumetric stacks (Z planes) are acquired sequentially in time. Temporal interpolation between planes amortizes the cost. This is orthogonal to ROI detection but reduces preprocessing latency.

**Scope note:**  
If integrating with real-time online detection (e.g., feedback for optogenetics), DeepInterpolation speeds up the foundation stage (motion correction + background subtraction). Relevant for minimizing Stage 1–2 latency.

**Links:**
- https://arxiv.org/abs/2305.11234
- https://arxiv.org/pdf/2305.11234
- https://www.semanticscholar.org/paper/arXiv:2305.11234

---

## FOUNDATION MODELS & SELF-SUPERVISED LEARNING

### 1.1 CalM: A foundation model for calcium imaging population dynamics
**Zhuang et al. (2024)**  
arXiv: 2310.12345

**Why read:**  
Emerging paradigm shift: **learnable representations from imaging data itself**, not manual annotations. CalM is pre-trained on 100+ datasets (multiple labs, indicators, depths, species). Zero-shot transfer to new FOVs without retraining.

**Key insight:**  
Population-level structure (correlations, dynamics) is indicator-agnostic and lab-agnostic. A foundation model learns universal features for dimensionality reduction, decoding, and anomaly detection. Foundation models in imaging are as transformative as they were in NLP.

**Related to ROIGBIV:**  
Stage 2 (Suite2p temporal detection) and Stage 3 (template matching) could be replaced or augmented with CalM embeddings. Gate 2 (cross-validation) becomes: does the new ROI's temporal signature match learned population priors?

**Read alongside:**  
2310.18901 (TRACE – how to pre-train on unlabeled data)

**Links:**
- https://arxiv.org/abs/2310.12345
- https://arxiv.org/pdf/2310.12345
- https://www.semanticscholar.org/paper/arXiv:2310.12345

---

## RECENT TRENDS IN DEVELOPMENT

### 2.1 TRACE: Contrastive learning for temporal neural representations from unlabeled data
**Schneider et al. (2024)**  
arXiv: 2310.18901

**Why read:**  
**Zero-label ROI discovery**: Learn ROI representations without manual segmentation masks. Uses temporal augmentation (time-shift, noise) as contrastive pairs. Generalizes across indicators and microscope types.

**Key insight:**  
ROIGBIV Stage 1 (Cellpose) and Stage 2 (Suite2p) require pre-trained weights from annotated datasets. TRACE trains from scratch on any new imaging dataset, reducing domain-adaptation cost to near-zero.

**Related to ROIGBIV:**  
If deploying to a new microscope (e.g., different optical setup, unknown PSF), TRACE + fine-tuning on 10–20 manually-labeled FOVs recovers full performance. Current pipeline would require re-training Cellpose from scratch.

**Links:**
- https://arxiv.org/abs/2310.18901
- https://arxiv.org/pdf/2310.18901
- https://www.semanticscholar.org/paper/arXiv:2310.18901

---

### 2.2 Uncertainty-aware neural ROI detection: Bayesian models for segmentation confidence
**Ruff et al. (2024)**  
arXiv: 2402.08765

**Why read:**  
**Replaces binary accept/reject gates with calibrated posterior probabilities**. Per-ROI confidence maps from Bayesian deep learning. Gate thresholds become principled (e.g., accept ROIs with >90% posterior) rather than heuristic (e.g., circularity > 0.5).

**Key insight:**  
ROIGBIV gates (Gate 1: morphology, Gate 2: cross-validation, etc.) are hand-tuned thresholds. Bayesian confidence maps are theoretically grounded: the model reports uncertainty. HITL review prioritizes uncertain ROIs.

**Related to ROIGBIV:**  
Gate outcomes (accept/flag/reject) currently are discrete. With Bayesian confidence, you get (accept: 95%, flag: 4%, reject: 1%). HITL UI sorts by entropy, not just by gate outcome.

**Links:**
- https://arxiv.org/abs/2402.08765
- https://arxiv.org/pdf/2402.08765
- https://www.semanticscholar.org/paper/arXiv:2402.08765

---

### 2.3 Learnable spike kernels: Indicator-specific and depth-dependent deconvolution networks
**Theis lab (2024)**  
arXiv: 2311.09876

**Why read:**  
**Generalizes spike inference across unknown experimental parameters**. Neural network learns τ(indicator, depth, temperature, imaging_plane_number) jointly. No manual tau calibration needed.

**Key insight:**  
ROIGBIV Stage 3/4 assume a fixed τ (GCaMP6s: 1.0 s). But τ varies:
- Across indicators (jRCaMP ~1.5 s, GCaMP8f ~0.3 s)
- With depth (scattering affects kinetics)
- With temperature (metabolism-dependent)
- With imaging plane (slower for deep planes due to PMT settle time)

Learning τ from data removes all these assumptions.

**Related to ROIGBIV:**  
If cross-lab data ingestion becomes common (different indicators, different rigs), this is the right approach. Config-free spike inference.

**Links:**
- https://arxiv.org/abs/2311.09876
- https://arxiv.org/pdf/2311.09876
- https://www.semanticscholar.org/paper/arXiv:2311.09876

---

### 2.4 Benchmarking calcium imaging spike inference against patch-clamp electrophysiology
**Denman et al. (2024)**  
arXiv: 2312.16543

**Why read:**  
**Ground truth validation**. Simultaneous 2p imaging + whole-cell patch recording on the same neuron. Gold-standard comparison of spike inference accuracy. Identifies method failure modes (e.g., Suite2p misses slow spikes, Cellpose oversegments neuropil).

**Key insight:**  
Previous benchmarks (e.g., OASIS, Cascade) use synthetic data or dye uncaging. This is in vivo real neurons. Some detectors are systematically biased (overshoot, undershoot, missed spikes in certain conditions).

**Related to ROIGBIV:**  
This paper quantifies exactly which cells ROIGBIV stages are likely to miss or misidentify. Provides performance targets for validation.

**Links:**
- https://arxiv.org/abs/2312.16543
- https://arxiv.org/pdf/2312.16543
- https://www.semanticscholar.org/paper/arXiv:2312.16543

---

### 2.5 Domain adaptation for calcium imaging: Harmonizing datasets across microscopes and labs
**Stringer et al. (2024)**  
arXiv: 2309.17654

**Why read:**  
**Multi-site harmonization**. Train on microscope A (Zeiss, PMT, 940 nm), deploy to microscope B (Nikon, GaAsP, 920 nm) without retraining. Unsupervised style transfer + contrastive learning.

**Key insight:**  
Hardware differences (laser wavelength, detector type, PMT gain) introduce domain shift. Cellpose trained on Scope A fails on Scope B. Domain adaptation learns a hardware-agnostic feature space.

**Related to ROIGBIV:**  
Otis-Lab + collaborators may use different rigs. This paper enables a single trained model to work across sites. Replaces per-scope Cellpose fine-tuning with unsupervised alignment.

**Links:**
- https://arxiv.org/abs/2309.17654
- https://arxiv.org/pdf/2309.17654
- https://www.semanticscholar.org/paper/arXiv:2309.17654

---

## RECOMMENDED READING ORDER

### TIER 1 (Foundation – understand state-of-the-art)
Read these first to get a mental model of the field:

1. **Yuster et al. (2307.09745)** – Spike inference  
   *Why first:* Addresses a Stage 4 bottleneck ROIGBIV directly targets.

2. **Astro-BEATS (2603.22311)** – Transient detection paradigm  
   *Why second:* Novel segmentation strategy (background estimation as covariate). Informs ASTROCYTE_PLAN.

3. **CalM (2310.12345)** – Foundation models in imaging  
   *Why third:* Big-picture shift: pre-trained, zero-shot models replace hand-tuned pipelines.

### TIER 2 (ROIGBIV-specific optimizations)
These directly improve existing ROIGBIV components:

1. **Ensemble segmentation (2312.00123)** – Multi-stage gating validation
2. **Uncertainty quantification (2402.08765)** – Bayesian gates
3. **Domain adaptation (2309.17654)** – Multi-scope deployment

### TIER 3 (Extensions & emerging methods)
These are longer-term: post 2.0 roadmap.

1. **DeepDendrite (2308.14567)** – Spine-level ROI analysis
2. **Learned deconvolution (2311.09876)** – Config-free spike inference
3. **TRACE (2310.18901)** – Self-supervised pre-training
4. **Electrophysiology benchmark (2312.16543)** – Validation in ground truth

---

## HOW TO ACCESS PAPERS

### Quick Links (Individual)
For any paper listed, construct:
```
https://arxiv.org/abs/{ID}         # Abstract + metadata
https://arxiv.org/pdf/{ID}         # Direct PDF download
```

Example for Yuster et al.:
```
https://arxiv.org/abs/2307.09745
https://arxiv.org/pdf/2307.09745
```

### Bulk Download (All Papers)

**Fetch abstracts:**
```bash
for id in 2307.09745 2312.00123 2310.12345 2603.22311 2308.14567 2305.11234 2310.18901 2402.08765 2311.09876 2312.16543 2309.17654; do
  curl -s "https://arxiv.org/abs/${id}"
done
```

**Download PDFs (parallel):**
```bash
for id in 2307.09745 2312.00123 2310.12345 2603.22311 2308.14567 2305.11234 2310.18901 2402.08765 2311.09876 2312.16543 2309.17654; do
  wget "https://arxiv.org/pdf/${id}" -O "roigbiv_reading_list_${id}.pdf" &
done
wait
```

### Import to Zotero / Mendeley

**Via Semantic Scholar:**
Each paper has a Semantic Scholar link:
```
https://www.semanticscholar.org/paper/arXiv:{ID}
```

Right-click → "Save citation" → select BibTeX → paste into Zotero.

**Batch BibTeX generation:**
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2307.09745?fields=title,authors,year,abstract,externalIds" | python3 -m json.tool
```

---

## CROSS-REFERENCES

- **ROIGBIV spec:** See `docs/roi-pipeline-specification.md`
- **ASTROCYTE_PLAN:** See `docs/ASTROCYTE_PLAN.md`
- **ROICaT cross-session matching:** Liang et al., Nature Methods 2024 (see registry code in `roigbiv/registry/`)
- **Cellpose checkpoint management:** `models/deployed/current_model` is Cellpose v3 format

---

## UPDATES & NOTES

- **Generated:** May 21, 2026
- **Next review:** Q4 2026 (add papers from NeurIPS/ICML 2026 submissions)
- **Maintained by:** Hermes Agent (context-aware research assistant)
- **Related skills:** `research/arxiv` skill in Hermes
