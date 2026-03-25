# ROI G. Biv — Algorithm Breakdown & Astrocyte Viability Analysis

## What Is This System?

ROI G. Biv is a **consensus cell detection pipeline** for two-photon calcium imaging data. It combines two independent detectors — Suite2p and Cellpose — and merges their outputs using spatial overlap scoring. The goal is to identify neurons in field-of-view (FOV) images with higher confidence than either tool alone could provide.

The pipeline is organized into six sequential stages:

```
Raw TIF stacks
     │
     ├─ Stage 1: Suite2p (activity-based detection)
     ├─ Stage 2: Suite2p (anatomy-based detection)
     ├─ Stage 3: Vcorr extraction (temporal features)
     │
     ├─ Stage 4: Cellpose training  ◄── only done once
     ├─ Stage 5: Cellpose inference
     │
     └─ Stage 6: Union ROI building + consensus scoring
                  │
                  └─ GOLD / SILVER / BRONZE tiered output
```

---

## Stage 1 — Suite2p Activity-Based Detection

**Script:** `scripts/run_suite2p.py`
**Key parameter:** `anatomical_only=0`

Suite2p is a well-established fluorescence segmentation tool developed for calcium imaging. In its default mode, it detects cells by looking for **pixels that are temporally correlated** — i.e., groups of pixels that tend to brighten and dim together over time.

**How it works, step by step:**

1. **Load the raw TIF stack.** Each file is a 3D array: `[frames × height × width]`. For this dataset, each FOV is ~30 Hz, pre-motion-corrected (`_mc.tif`), single-channel GCaMP.

2. **Registration is skipped** (`do_registration=0`). The stacks are already motion-corrected, so this step is turned off to save compute time and avoid a version conflict with Cellpose (Suite2p's anatomical mode imports Cellpose 4.x, which conflicts with the Cellpose 3.x used for training).

3. **Build a temporal activity map.** Suite2p computes a pixel-level temporal correlation map (Vcorr) — each pixel's value represents how strongly it co-varies with its neighbors over time. Neurons show up as bright islands.

4. **Detect ROI candidates** by finding spatially connected clusters of correlated pixels. The `spatial_scale` parameter (set to 0 = auto-detect) controls roughly what size structures are considered cell-sized. With `threshold_scaling=1.0`, only clusters above a default significance threshold are kept.

5. **Neuropil subtraction.** For each ROI, Suite2p estimates a background "neuropil" signal from a surrounding annulus (`inner_neuropil_radius=2`, `min_neuropil_pixels=350`) and subtracts it. This corrects for diffuse background fluorescence that bleeds into the ROI.

6. **Classify ROIs** using a built-in classifier. `preclassify=0.0` means all candidates are retained (no pre-filtering), because the downstream consensus layer handles filtering. Each ROI gets an `iscell` probability score (stored in `iscell.npy`).

7. **Spike deconvolution** (`spikedetect=True`) converts the raw ΔF/F fluorescence trace into an estimated spike train using the OASIS algorithm and the GCaMP decay time constant (`tau=1.0 s`).

**Output:** `suite2p_workspace/output/{stem}/suite2p/plane0/`
- `stat.npy` — per-ROI pixel footprints
- `iscell.npy` — classifier probability scores
- `ops.npy` — run metadata including the Vcorr map
- `F.npy`, `Fneu.npy`, `spks.npy` — fluorescence traces, neuropil, and deconvolved spikes

**What this pass captures:** Neurons that were **active during the recording** — enough to generate detectable correlated fluctuations. Silent cells are missed.

---

## Stage 2 — Suite2p Anatomy-Based Detection

**Script:** `scripts/run_suite2p.py` with `--anatomical_only 1`
**Output directory:** `suite2p_workspace/output_anatomy_batch/`

The anatomy pass runs the same Suite2p pipeline but instead of using temporal correlations, it detects ROIs from the **mean image alone** — the time-averaged brightness of each pixel.

**Key difference:** `anatomical_only=1` tells Suite2p to treat the mean projection as a static image and use spatial gradient/morphology cues to find cell-shaped regions, the same way you might manually draw cells by eye.

**What this pass captures:** All cell-shaped structures visible in the mean image, regardless of whether they were active. This catches:
- **Silent neurons** that did not fire during the recording window
- Neurons with low SNR that the activity pass missed

**Disk management:** Each FOV writes a `data.bin` file (~500 MB) that Suite2p needs during processing. With `do_registration=0`, this file is never used again after detection completes. The pipeline deletes it immediately after each FOV to keep peak disk usage bounded to ~1 FOV at a time rather than accumulating across all 80 FOVs.

---

## Stage 3 — Vcorr Extraction (Temporal Features)

**Script:** `scripts/extract_vcorr.py`

The Vcorr map is a 2D image where each pixel's value represents how strongly it fluctuates with its immediate spatial neighbors over time. Suite2p computes this internally during the activity pass; this script extracts it from `ops.npy` and saves it as a TIFF.

**Why Vcorr matters:** A plain mean projection image shows anatomy — where fluorescence is bright on average. Vcorr shows **functional activity** — where fluorescence is flickering in a correlated way. Neurons light up in both; neuropil and background look different in each.

By stacking mean + Vcorr as a **two-channel image**, the Cellpose model gets complementary information:
- Channel 1 (mean): cell body shape, morphology
- Channel 2 (Vcorr): temporal activity signature

This is what `use_vcorr=True` enables in both training and inference.

---

## Stage 4 — Cellpose Fine-Tuning (Training)

**Script:** `scripts/train.py`

This stage trains a custom Cellpose model specialized for this specific neuron population, imaging setup, and GCaMP indicator. It starts from Cellpose's pre-trained `cyto3` model (trained on a large corpus of cell images) and **fine-tunes** it on manually annotated FOVs from this dataset.

### The Training Data

Training images are mean/max projections of each FOV:
- `data/annotated/{stem}_mean.tif` — time-averaged image
- `data/annotated/{stem}_max.tif` — maximum-projection image (brighter for sparse activity)
- `data/annotated/{stem}_vcorr.tif` — Vcorr map
- `data/masks/{stem}_masks.tif` — manually drawn ground-truth ROI labels

Both mean and max projections are included, paired with the same ground-truth mask, to double the effective training set and expose the model to different contrast regimes.

### FOV-Aware Train/Val Split

FOVs are shuffled as **groups** before splitting, preventing data leakage — if mean and max projections of the same FOV ended up in different splits, the model could effectively memorize cells from training by recognizing them in validation. The default validation fraction is 15% of FOVs.

### What Cellpose Is Learning

Cellpose is a **flow-based segmentation model**. Rather than directly predicting which pixels belong to cells, it learns to predict:

1. **A distance gradient field** pointing from each cell pixel toward the cell's center
2. **A cell probability map** — how likely each pixel is to be inside any cell

At inference time, these gradient flows are integrated forward to find where each cell's center of mass is, then pixels that converge to the same center are grouped into one ROI.

Fine-tuning adjusts the model weights so the flow fields and cell probabilities match what cells look like in this specific context (GCaMP6s, ~30px neuron diameter, two-photon imaging, ventral striatum/prefrontal neurons).

### Training Loop

```python
model_path, train_losses, test_losses = train.train_seg(
    model.net,
    train_data=train_imgs,
    train_labels=train_masks,
    channels=[1, 2],           # use both mean + Vcorr channels
    normalize=True,
    learning_rate=0.001,
    weight_decay=1e-4,
    n_epochs=300,
    batch_size=8,
    save_every=10,              # checkpoint every 10 epochs
)
```

The model is optimized with a combination of flow-field regression loss and binary cross-entropy on the cell probability mask, via Cellpose's internal `train_seg` function.

### Post-Training Evaluation

After training, the best checkpoint (by validation loss) is evaluated on held-out FOVs using **Average Precision (AP)** at three overlap thresholds:

- **AP@0.5** — a predicted ROI counts as correct if it overlaps the ground-truth by ≥50%
- **AP@0.75** — stricter; 75% overlap required
- **AP@0.9** — very strict; nearly perfect overlap required

---

## Stage 5 — Cellpose Inference

**Script:** `scripts/run_inference.py`

The trained model is applied to every FOV's mean projection (+ Vcorr if available). Output is a **uint16 label image** where each unique integer identifies one detected cell body.

Key inference parameters:
- `diameter=30` — tells Cellpose what size cells to expect, in pixels; used for internal rescaling
- `flow_threshold=0.4` — how clean a predicted flow field must be to accept the ROI; higher is more conservative
- `cellprob_threshold=0.0` — at consensus time this is set to `-6` in `build_union_rois.py` to keep all candidates, with consensus handling downstream filtering

---

## Stage 6 — Union ROI Building + Consensus Scoring

**Script:** `scripts/build_union_rois.py` + `scripts/match_rois.py`

This is the core novelty of the pipeline. Instead of trusting either Suite2p or Cellpose alone, all three candidate pools are merged and scored.

### Step 1: Merge the Two Suite2p Passes

For each FOV, ROIs from the activity pass and anatomy pass are combined into a **union set** using IoU (Intersection over Union) matching via the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`).

The Hungarian algorithm finds the globally optimal one-to-one pairing between anatomy ROIs and activity ROIs that maximizes total IoU. Pairs with IoU ≥ 0.3 are retained as genuine matches.

### Step 2: Tier Assignment

Each ROI in the union set receives a confidence tier:

| Tier | Condition | Interpretation |
|------|-----------|----------------|
| **GOLD** | Found by both activity AND anatomy pass (IoU ≥ 0.3) | High confidence — both morphology and activity agree |
| **SILVER** | Found by anatomy pass only | Morphologically neuron-shaped, but silent in this recording |
| **BRONZE** | Found by activity pass only | Was active, but not anatomically prominent in mean image |

### Step 3: Cellpose Probability Scoring

The fine-tuned Cellpose model is run on the mean image in **probability extraction mode** (`cellprob_threshold=-6` to retain all candidates). This produces a per-pixel probability heatmap. Each union ROI is assigned a score equal to the **mean Cellpose probability over its pixel footprint**.

This score is stored in `scored_rois_summary.csv` and the `_roi_cellprob.tif` heatmap for downstream analysis. Researchers can threshold on it post-hoc.

### Step 4: Mask Construction

For GOLD and SILVER ROIs, the **Cellpose boundary** is used (better spatial precision — Cellpose tends to produce smoother, more anatomically accurate cell contours). For BRONZE ROIs, Suite2p's pixelated footprint is used.

**Default output:** GOLD + SILVER (controlled by `default_tiers: [gold, silver]` in `configs/pipeline.yaml`).

---

## Full Data Flow Summary

```
Raw TIF stacks (80 FOVs)
        │
        ├──► Suite2p (activity)     → stat.npy, iscell.npy, ops.npy
        │                                (suite2p_workspace/output/)
        │
        ├──► Suite2p (anatomy)      → stat.npy, iscell.npy, ops.npy
        │                                (suite2p_workspace/output_anatomy_batch/)
        │
        ├──► extract_vcorr.py       → {stem}_mc_vcorr.tif
        │                                (data/annotated/)
        │
        ├──► Mean/Max projections   → {stem}_{mean,max}.tif
        │                                (data/annotated/)
        │
        ├──► Manual annotation      → {stem}_masks.tif
        │    (done in napari/FIJI)       (data/masks/)
        │
        ├──► train.py               → Cellpose model checkpoint
        │    (fine-tuning cyto3)         (models/checkpoints/)
        │
        ├──► build_union_rois.py    → {stem}_all_s2p_masks.tif
        │    (combines all above)        {stem}_roi_cellprob.tif
        │                                scored_rois_summary.csv
        │
        └──► extract_traces.py      → per-ROI ΔF/F traces
```

---

## Future Viability: Training a New Model for Astrocytes

### What Would Need to Change

Adapting this pipeline for astrocyte detection is feasible but requires several meaningful adjustments, since astrocytes differ from neurons in every dimension the pipeline is tuned around.

---

### 1. Morphology

Neurons in this dataset are compact, roughly circular cell bodies (~17–30 px diameter). Astrocytes have a fundamentally different morphology:

- **Star-shaped** with long, thin processes radiating outward
- No compact soma in the same sense — the body is often diffuse
- Processes can intermingle between cells, making boundaries ambiguous
- Overall **size is larger** and **less uniform**

**Impact on the code:**
- `diameter` in both training (`--diameter 17`) and inference would need to be re-estimated. Cellpose's diameter parameter controls rescaling; getting it wrong systematically degrades detection.
- `allow_overlap=false` in Suite2p may need to be relaxed — astrocyte processes frequently overlap.
- `connected=True` is likely fine, but process fragmentation could be an issue.

---

### 2. Functional Signal

GCaMP-based astrocyte calcium signals are fundamentally different from neuronal signals:

- Astrocyte calcium events are **slower** (seconds to tens of seconds, not sub-second spikes)
- Signals often propagate spatially across processes (wave-like), not a point-source soma signal
- **Vcorr** (the temporal correlation map) may look very different — potentially weaker or diffuse for astrocytes since they do not fire action potentials

**Impact on the code:**
- `tau` (decay time constant) would need adjustment — GCaMP6s in astrocytes decays more slowly
- `fs` is recording-specific and would not change
- `high_pass=100` for neuropil subtraction may interact poorly with slow astrocyte signals (it is a temporal filter that removes fluctuations slower than ~100 frames)
- The **Vcorr second channel** may be less informative or may need replacing with a different functional feature (e.g., standard deviation map, event frequency map)

---

### 3. Training Data Requirements

The current model was fine-tuned on manually annotated neurons. A new astrocyte model would require:

- New ground-truth masks drawn on astrocyte images (time-consuming; astrocyte boundaries are less crisp)
- A decision on what counts as "one astrocyte" — soma-only? soma + processes? territory?
- Potentially fewer training FOVs available (astrocyte imaging is less common)

The `train.py` script is fully general — it only needs new `*_mean.tif`, `*_max.tif`, and `*_masks.tif` files with a new `--run_id`. The training loop, validation split, and AP evaluation infrastructure are all reusable without modification.

---

### 4. Suite2p Detection

Suite2p's activity-based detector was designed and validated on neurons. Its assumptions (compact circular ROIs, neuropil annulus model, OASIS spike deconvolution) break down for astrocytes.

**Activity pass (`anatomical_only=0`):** Likely to perform poorly. The neuropil model assumes a smooth background around a compact soma; astrocyte processes would contaminate their own neuropil annulus.

**Anatomy pass (`anatomical_only=1`):** More viable — this looks for bright structures in the mean image and does not assume spike-like temporal dynamics. Astrocytes can be bright in GCaMP mean images.

**Practical recommendation:** For astrocytes, drop the Suite2p activity pass and rely on anatomy-only Suite2p + fine-tuned Cellpose, then run consensus between those two.

---

### 5. Consensus Thresholds

The IoU threshold (`iou_threshold=0.3`) was set based on the expected geometric mismatch between Suite2p's irregular pixel footprints and Cellpose's smooth contours for ~30px circular cells. For irregular, process-bearing astrocytes, this threshold would need empirical re-calibration — the geometric mismatch between detectors will be larger.

---

### Summary Table: Neuron vs. Astrocyte Adaptation

| Component | Neurons (current) | Astrocytes (required change) |
|---|---|---|
| Cellpose `diameter` | 17–30 px | Re-estimate (likely larger, more variable) |
| `tau` (decay) | 1.0 s | Likely 2–5 s |
| `high_pass` filter | 100 frames | May need to increase |
| Vcorr 2nd channel | Highly informative | Likely weaker; consider SD map instead |
| Suite2p activity pass | Core detector | Likely unreliable; anatomy-only preferred |
| Training masks | Compact soma | Process-inclusive territory; harder to annotate |
| `iou_threshold` | 0.3 | Re-calibrate; likely needs to be lower |
| `allow_overlap` | False | May need to be True |
| `train.py` script | Works as-is | Works as-is — just needs new data |

---

### Bottom Line

The pipeline architecture is well-suited for adaptation. The training infrastructure (`train.py`), consensus logic (`match_rois.py`, `build_union_rois.py`), and inference pipeline (`run_inference.py`) are all data-agnostic. The primary work for an astrocyte model would be:

1. Re-annotating training data with astrocyte-aware masks
2. Re-calibrating `diameter`, `tau`, and `iou_threshold` empirically
3. Deciding whether to retain the two-channel Vcorr input or substitute a different functional feature
4. Evaluating whether Suite2p's activity pass adds value or introduces more noise than signal for astrocyte morphologies

The anatomy pass + fine-tuned Cellpose combination is the most promising starting point for astrocytes, as it does not depend on the spike-detection assumptions baked into Suite2p's default mode.
