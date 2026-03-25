# ROI G. Biv — Consensus Cell Detection Pipeline

**Two-photon calcium imaging · Suite2p + Cellpose · GOLD / SILVER / BRONZE confidence tiers**

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

ROI G. Biv is a **consensus cell-detection pipeline** for two-photon calcium imaging.
It combines two independent Suite2p detection modes (activity-based and anatomy-based)
with a fine-tuned Cellpose model, merging their outputs via spatial IoU matching to
produce three-tier confidence ROI masks.

| Tier | Condition | Interpretation |
|------|-----------|----------------|
| **GOLD** | Both Suite2p passes agree (IoU ≥ 0.3) | Highest confidence — morphology and activity agree |
| **SILVER** | Anatomy pass only | Morphologically neuron-shaped; silent during recording |
| **BRONZE** | Activity pass only | Was active; not anatomically prominent in mean image |

Key capabilities:

- **End-to-end Colab notebook** — upload TIFs to Drive, run everything in Colab
- **Consensus detection** combining temporal correlation + mean-image morphology
- **Cellpose probability scoring** per ROI for post-hoc thresholding
- **Resumable batch processing** — skips completed FOVs after disconnects
- **Interactive viewer** — ipywidgets FOV explorer with tier + probability filters
- **pip-installable package** distributed via GitHub releases

Output masks feed directly into [pynapse](https://github.com/Otis-Lab-MUSC/pynapse) for calcium signal extraction and peri-event analysis.

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
│   ├── io.py                 # TIF discovery, archive extraction, projection extraction
│   ├── suite2p.py            # Suite2p activity + anatomy batch runner
│   ├── match.py              # IoU matching and Hungarian assignment
│   ├── union.py              # Union ROI building + Cellpose probability scoring
│   └── viz.py                # ipywidgets interactive Colab viewer
├── notebooks/
│   └── roigbiv.ipynb         # End-to-end Google Colab notebook
├── scripts/                  # Local development scripts (advanced use)
│   ├── train.py              # Cellpose fine-tuning
│   ├── run_suite2p.py        # Suite2p runner
│   ├── build_union_rois.py   # Union ROI building (local)
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
  --pretrained_model ~/Otis-Lab/Projects/roigbiv/models/deployed/current_model \
  --image_path ~/Otis-Lab/Projects/roigbiv/data/annotated/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc_mean.tif
```

**Option B — Launch GUI and open image manually (recommended for exploration)**
```bash
conda activate roigbiv
cellpose --pretrained_model ~/Otis-Lab/Projects/roigbiv/models/deployed/current_model
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
