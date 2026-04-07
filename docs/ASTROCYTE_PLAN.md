# Astrocyte Model Plan
*roigbiv project — Otis Lab, MUSC*

## Context

The neuron pilot (run011, AP@0.5=0.652) demonstrated the Vcorr-channel fusion approach works. The long-term goal is astrocyte detection, where temporal features are essential: astrocytes have slow calcium dynamics (τ≈2–3s vs ~1s for neurons) and irregular star-shaped morphology (~50–80px diameter vs ~17px for neurons).

**Starting conditions:**
- Astrocytes express GCaMP in the existing 80 FOVs — annotation can begin now
- Future recordings will include a structural astrocyte marker (2nd channel, e.g. GFAP-tdTomato or SR101)
- Current Suite2p ran with `nchannels=1, tau=1.0` — astrocytes are underdetected by Suite2p
- Cellpose `channels=[1, 2]` with Vcorr as 2nd channel is the established pattern

**Biological differences driving design choices:**

| Property | Neurons (run011) | Astrocytes (target) |
|----------|-----------------|---------------------|
| Diameter | ~17 px | ~50–80 px |
| Morphology | Round, compact | Star-shaped, irregular |
| GCaMP τ | ~1s | **2–3s** |
| Current annotations | 80 FOVs | **0 — must create** |
| 2nd channel (now) | Vcorr | Vcorr (proxy until marker available) |
| 2nd channel (future) | — | Astrocyte structural marker |

---

## Phase 1: τ-map extraction

**New script: `scripts/extract_tau_map.py`**

Computes per-pixel temporal decay time constant (τ, in seconds) from Suite2p fluorescence traces and projects it onto a 2D image. Mirrors the pattern of `extract_vcorr.py`.

```
For each FOV in suite2p_workspace/output/{stem}/suite2p/plane0/:
  1. Load F.npy (n_rois × n_frames), Fneu.npy, stat.npy
  2. Neuropil-correct: F_corr = F - 0.7 × Fneu
  3. For each ROI i:
       normalize trace → compute autocorrelation → fit exp(-t/τ) → get τ_i
  4. Paint τ_i onto 2D image at stat[i]['ypix'], stat[i]['xpix']
     (background pixels = 0.0)
  5. Save → data/annotated/{stem}_mc_tau.tif
```

**Interpretation**: τ > 1.5s = candidate astrocyte ROI; τ < 1.5s = neuron. The τ-map guides visual annotation even if Suite2p missed most astrocytes.

**Run**:
```bash
conda activate roigbiv
cd ~/Otis-Lab/Projects/roigbiv/scripts
python3 extract_tau_map.py
```

---

## Phase 2: Annotation

**Goal**: 15–20 FOVs annotated with astrocyte masks → `data/astrocyte_masks/`

**Cellpose GUI — visual guides to use**:
- `{stem}_mean.tif` — primary image
- `{stem}_mc_vcorr.tif` — 2nd channel: large correlated blobs = candidate astrocytes
- `{stem}_mc_tau.tif` — overlay: high-τ (slow) regions highlight astrocytes

**What to draw**:
- Large, diffuse, irregular blobs (not compact bright somas)
- Star-shaped spread in max projection
- High Vcorr regions without a corresponding compact neuron

**Output format**: `data/astrocyte_masks/{stem}_astro_masks.tif`
- Same as neuron masks: uint16 label image, 0=background, 1..N=astrocyte instances
- Use `_mc_mean` stems (matches the annotated dir naming convention)

**Minimum**: 15 FOVs to train. Use `--val_frac 0.20` (12 train / 3 val minimum).

---

## Phase 3: Code changes + training

### 3a. Modify `scripts/train.py` (1-line fix)

Add `--diameter` CLI argument — currently hardcoded to 30 in the eval call at line 187.

```python
# Add to argparse:
ap.add_argument('--diameter', type=float, default=30)

# Change line 187:
pred_masks, _, _ = model_eval.eval(val_imgs, diameter=args.diameter, ...)
```

### 3b. New config: `configs/astrocyte_pipeline.yaml`

```yaml
cellpose:
  model_path: models/deployed/astrocyte_model
  diameter: 60
  channels: [1, 2]
  flow_threshold: 0.4
  cellprob_threshold: 0.0
  normalize: true

suite2p:
  anatomical_only: 0
  nplanes: 1
  nchannels: 1
  functional_chan: 1
  tau: 2.5            # astrocyte GCaMP decay (neurons use 1.0)
  fs: 30.0
  do_registration: 0
  nimg_init: 300
  batch_size: 250
  smooth_sigma: 1.15
  spatial_scale: 2    # larger scale for astrocytes (neurons use 0)
  allow_overlap: true # astrocyte territories overlap
  high_pass: 50       # lower cutoff for slow signals (neurons use 100)
  connected: true
  min_neuropil_pixels: 350
  preclassify: 0.0

paths:
  raw_dir: data/raw
  annotated_dir: data/annotated
  masks_dir: data/astrocyte_masks
  s2p_output: suite2p_workspace/output
  inference_output: inference/astrocyte_output
  traces_output: inference/astrocyte_traces
```

### 3c. Train astro001

```bash
python3 train.py \
  --run_id astro001 \
  --base_model cyto3 \
  --epochs 300 \
  --lr 0.001 \
  --diameter 60 \
  --masks_dir data/astrocyte_masks \
  --data_dir data/annotated \
  --use_vcorr
```

**Why cyto3 base (not run011)?** Neuron model weights are optimized for 17px cells. Astrocytes at 60px are a fundamentally different spatial scale — starting fresh from cyto3 avoids that bias.

### 3d. Future: retrain with structural marker (astro002)

When dual-channel recordings are available, the structural marker (GFAP-tdTomato, SR101, etc.) replaces Vcorr as the 2nd channel — a much stronger signal because it directly encodes cell identity:

```bash
python3 train.py \
  --run_id astro002 \
  --base_model models/checkpoints/models/astro001_epoch_XXXX \
  --diameter 60 \
  --masks_dir data/astrocyte_masks \
  --data_dir data/annotated_dual
  # channels=[1,2] = [GCaMP, astrocyte structural marker]
```

---

## Phase 4: Evaluate & deploy

```bash
# Evaluate
python3 eval_model.py \
  --model astro001 \
  --diameter 60 \
  --masks_dir data/astrocyte_masks

# Deploy if AP@0.5 > 0.5
ln -sf ~/Otis-Lab/Projects/roigbiv/models/checkpoints/models/astro001_epoch_XXXX \
        ~/Otis-Lab/Projects/roigbiv/models/deployed/astrocyte_model
```

**Success criterion**: AP@0.5 > 0.5. Lower initial bar than neurons (0.65) because the training dataset will be smaller and annotations harder.

---

## Phase 5: Suite2p re-run (optional, parallel with annotation)

Re-run Suite2p on astrocyte-rich FOVs with the new `astrocyte_pipeline.yaml` config to detect astrocyte ROIs that the current neuron-optimized run missed. This improves τ-map coverage.

```bash
python3 run_suite2p.py \
  --config configs/astrocyte_pipeline.yaml \
  --extract_vcorr
```

---

## Files summary

| File | Action | Notes |
|------|--------|-------|
| `scripts/extract_tau_map.py` | **Create** | New — τ from F.npy |
| `scripts/train.py` | **Modify** | Add `--diameter` arg, line 187 |
| `configs/astrocyte_pipeline.yaml` | **Create** | New config |
| `data/astrocyte_masks/` | **Create dir** | Filled during annotation |
| `models/deployed/astrocyte_model` | **Create symlink** | After training |

Reused unchanged:
- `scripts/run_inference.py` — use `--diameter 60 --vcorr_dir data/annotated`
- `scripts/eval_model.py` — use `--diameter 60 --masks_dir data/astrocyte_masks`
- `scripts/extract_vcorr.py` — Vcorr already computed for all 80 FOVs

---

## Execution order

```
1.  extract_tau_map.py              → data/annotated/*_mc_tau.tif
2.  Annotate 15–20 FOVs (GUI)       → data/astrocyte_masks/
3.  Modify train.py --diameter arg  → 1-line change
4.  Write astrocyte_pipeline.yaml
5.  train.py --run_id astro001 ...
6.  eval_model.py astro001 --diameter 60
7.  Deploy astrocyte_model symlink
──────────────────────────────────
8.  (optional) Suite2p re-run with astrocyte params
9.  (future)   astro002 with dual-channel structural marker
```
