# ROI G. Biv — Sequential Subtractive Cell Detection Pipeline

**Two-photon calcium imaging · Cellpose + Suite2p · four sequential detection stages with per-stage validation gates**

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/Otis-Lab-MUSC/roigbiv/releases)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![REACHER Suite](https://img.shields.io/badge/REACHER_Suite-member-orange)](https://github.com/Otis-Lab-MUSC)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Otis-Lab-MUSC/roigbiv/blob/main/notebooks/roigbiv.ipynb)

*Written by*: Joshua Boquiren

[![](https://img.shields.io/badge/@thejoshbq-grey?style=flat&logo=github)](https://github.com/thejoshbq)

---

## Quick Start

Upload your pre-motion-corrected TIF stacks to Google Drive and open the notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Otis-Lab-MUSC/roigbiv/blob/main/notebooks/roigbiv.ipynb)

The notebook handles all installation and processing. The only cell you need to edit sets your Drive path and frame rate.

### Install as a Python package

```bash
pip install git+https://github.com/Otis-Lab-MUSC/roigbiv.git
```

Or from a specific release:

```bash
pip install https://github.com/Otis-Lab-MUSC/roigbiv/releases/latest/download/roigbiv-0.1.0-py3-none-any.whl
```

> **Note:** `suite2p` and `cellpose` must be installed separately with correct ordering
> (suite2p first, then `cellpose==4.0.9 --upgrade`). See `notebooks/roigbiv.ipynb` Step 0.

---

## Overview

ROI G. Biv is a **sequential subtractive** ROI-detection pipeline for two-photon calcium
imaging. A shared Foundation prepares the movie (motion correction → truncated-SVD
low-rank/sparse background split → summary images), then **four detection stages run in
order**, each operating on the *residual* left after the previous stages subtract the
sources they found — so the detectors are complementary rather than redundant:

| Stage | Detector | Finds |
|------|----------|-------|
| **1** | Cellpose (fine-tuned CP3) | soma-shaped objects on the morphological image |
| **2** | Suite2p classifier | active neurons Stage 1 missed morphologically |
| **3** | GCaMP matched-filter template sweep | isolated transients too sparse for Suite2p |
| **4** | tonic-neuron search (bandpass + correlation contrast) | steady/tonic firers with no discrete transients |

Each stage is paired with a **validation gate** (morphology → temporal cross-validation →
waveform → correlation-contrast) that accepts, flags, or rejects every candidate before it
is subtracted. Every ROI carries full provenance: which stage found it, its gate outcome,
a confidence level, and the stage-specific score behind it.

> **Note:** this replaces the older parallel three-branch **GOLD/SILVER/BRONZE consensus**
> design. That architecture is retired.

Key capabilities:

- **Sequential subtractive detection** — four complementary detectors on a shared residual
- **Per-stage validation gates** with per-ROI provenance (`source_stage`, `gate_outcome`, `confidence`)
- **Human-in-the-loop review package** — prioritized review queue + additive corrections that retrain Cellpose
- **Cross-session FOV & cell registry** (ROICaT embeddings, `roigbiv-registry`)
- **Resumable processing** — `--resume` skips completed stages/FOVs after interruptions
- **Dash web app** (`roigbiv-ui`) and **end-to-end Colab notebook**

Output masks + traces feed downstream to [pynapse](https://github.com/Otis-Lab-MUSC/pynapse)
for calcium signal extraction and peri-event analysis.

**→ For the full purpose statement and a comprehensive breakdown of the pipeline and every
integrated algorithm, see [`docs/design/OVERVIEW.md`](docs/design/OVERVIEW.md).**

**→ Architecture direction:** the pipeline is pivoting from the destructive subtractive cascade
toward a non-destructive candidate union + joint validation — see
[ADR-0001](docs/adr/0001-non-destructive-candidate-union.md).

**→ Limitations:** for what ROI G. Biv does *not* claim — detection scope, denoising
validation, calibration scope, and more — see [`docs/limitations.md`](docs/limitations.md).

---

## Role in Ecosystem

```
Raw TIFFs ──► roigbiv ──► ROI masks ──► pynapse ──► axplorer
               (segmentation)          (signal extraction)  (visualization)
```

ROI G. Biv sits at the front of the analysis pipeline: it takes raw two-photon TIFF stacks, segments them into labeled ROI masks, and passes those masks downstream to pynapse for fluorescence trace extraction and behavioral alignment.

---

## Project Structure

```
roigbiv/
├── roigbiv/                  # Python package (pip-installable)
│   ├── pipeline/             # Sequential subtractive pipeline (Foundation → Stage 1–4)
│   │   └── run.py            # Pipeline + CLI entry point (roigbiv-pipeline)
│   ├── registry/             # Cross-session FOV + cell registry (SQLAlchemy + ROICaT)
│   ├── ui/                   # Dash + Plotly web app (roigbiv-ui)
│   ├── cli_registry.py       # Registry CLI (roigbiv-registry)
│   ├── cli_reingest.py       # External-mask ingest CLI (roigbiv-reingest)
│   ├── io.py                 # TIF discovery + validation
│   ├── suite2p.py            # Suite2p batch runner (used by Foundation)
│   └── overlay.py            # ROI overlay rendering for reports
├── scripts/                  # Training + data-prep utilities
│   ├── train.py              # Cellpose fine-tuning
│   ├── ingest_corrections.py # Ingest Cellpose GUI corrections
│   ├── roigbiv_to_pynapse.py # Export traces to pynapse
│   └── ...
├── configs/
│   └── pipeline.yaml         # All tunable parameters
├── models/
│   ├── checkpoints/          # Training checkpoints (Git LFS)
│   └── deployed/             # Deployed model (Git LFS)
├── data/
│   ├── raw/                  # Raw two-photon TIFF stacks
│   ├── annotated/            # Mean/max projections + Vcorr maps
│   └── masks/                # Ground-truth segmentation masks
├── pyproject.toml            # Package definition
└── .github/workflows/
    └── release.yml           # Build wheel + attach model on tag push
```

---

## Training

### Usage

```bash
python scripts/train.py --run_id run001 [--epochs 100] [--lr 0.1]
```

### Arguments

| Argument | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `--run_id` | Yes | — | Unique identifier for the training run |
| `--epochs` | No | 100 | Number of training epochs |
| `--lr` | No | 0.1 | Learning rate |
| `--batch_size` | No | 4 | Training batch size |
| `--base_model` | No | `cyto3` | Cellpose base model name or path to checkpoint |

### Dataset Format

- **Images**: `*_mean.tif` files in `data/annotated/` (mean projection TIFFs, single-channel)
- **Masks**: `*_masks.tif` files in `data/masks/` (labeled ROI masks, matching stems)

Each image file must have a corresponding mask file with the same stem prefix (e.g., `sample01_mean.tif` pairs with `sample01_masks.tif`). At least 3 image/mask pairs are required.

### Train/Val Split

Data is split 90/10 (training/validation). With fewer than 10 pairs, 1 pair is reserved for validation.

### Checkpoints

Model checkpoints are saved every 50 epochs to `models/checkpoints/`. Training logs are written to `logs/<run_id>.log`.

---

## Cellpose GUI — Interactive Segmentation & Correction

Use the Cellpose GUI to visually inspect model output, correct masks, and feed corrections back into training.

### Launch commands

**Option A — Load a specific image at launch**
```bash
conda activate roigbiv
cellpose \
  --pretrained_model ~/Otis-Lab/Projects/Phoxel-Workbench/roigbiv/models/deployed/current_model \
  --image_path ~/Otis-Lab/Projects/Phoxel-Workbench/roigbiv/data/annotated/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc_mean.tif
```

**Option B — Launch GUI and open image manually (recommended for exploration)**
```bash
conda activate roigbiv
cellpose --pretrained_model ~/Otis-Lab/Projects/Phoxel-Workbench/roigbiv/models/deployed/current_model
```
Then: File → Open image → navigate to `data/annotated/` and pick any `*_mean.tif`.

### Correction workflow

1. Load image → set diameter to **30 px** (matches training config)
2. Run segmentation (Ctrl+R or Run button)
3. Inspect overlaid masks; use brush/erase tools to correct
4. Save: File → Save masks as `*_seg.npy` (same directory as the input image)
5. Ingest corrections:
   ```bash
   conda run -n roigbiv python scripts/ingest_corrections.py
   ```
6. Corrected masks appear in `data/masks/` ready for the next retraining run

---

## Dependencies

| Package | Purpose |
|:--------|:--------|
| numpy | Array operations |
| tifffile | TIFF I/O |
| cellpose | Base segmentation models and training API |
| torch (PyTorch) | Deep learning backend |
| CUDA 11.8+ (optional) | GPU-accelerated training and inference |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Joshua Boquiren — [thejoshbq@proton.me](mailto:thejoshbq@proton.me)

[GitHub: Otis-Lab-MUSC/roigbiv](https://github.com/Otis-Lab-MUSC/roigbiv)
