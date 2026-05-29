# Researcher Data Guide — What to Share for Cellpose Training

## Overview

The roigbiv pipeline fine-tunes a Cellpose segmentation model to detect calcium-imaging ROIs from two-photon data. To add a new animal's sessions to the training set, the pipeline needs only **two files per imaging session**: the motion-corrected stack and the ImageJ ROI annotation. Everything else your acquisition and analysis software generates (SIMA outputs, HDF5 containers, raw frame directories) is not used and does not need to be copied to the shared drive.

---

## Files Required

Copy **both** of these files for every session you want included:

| File Pattern | What it is | Typical Size |
|---|---|---|
| `*_mc.tif` | Motion-corrected 3D TIFF stack (frames × height × width). This is the imaging data after motion correction — the file your SIMA/PrairieView pipeline saves at the end of the MC step. | 1–4 GB |
| `*_RoiSet.zip` | ImageJ ROI Manager export for that session (`Analyze > Tools > ROI Manager > More > Save...`). This contains the neuron outlines you drew, which become the Cellpose training labels. | 3–20 KB |

**A session is only usable if it has both files.** If one is missing, that session will be skipped.

---

## Files NOT Required

You do not need to copy any of the following. Leave them on your analysis drive:

| File / Folder Pattern | What it is | Why it's not needed |
|---|---|---|
| `*_AVG.tif` | SIMA time-average projection | Wrong format — pipeline generates its own mean projection from the mc.tif |
| `*_STD.tif` | SIMA standard-deviation image | Not used at any pipeline stage |
| `*.h5` | SIMA / HDF5 data container | Redundant — all imaging data is in the `*_mc.tif` |
| `*_extractedsignals_raw.npy` | SIMA ΔF/F fluorescence traces | Pipeline extracts its own traces via Suite2p |
| `*.sima/` | SIMA dataset metadata directory | Not read by the pipeline |
| `*_mc.sima/` | SIMA motion-correction metadata | Not read by the pipeline |
| `extras/` | PrairieView XML references, `.env` files | Acquisition metadata — not used |
| Raw frame directories | Folders of individual `.ome.tif` frame files (pre-MC) | Only the final `*_mc.tif` is needed, not the source frames |

---

## Folder Organization

Keep the same animal → session folder hierarchy you already have. The pipeline walks the tree looking for `*_mc.tif` files and matches each one with its `*_RoiSet.zip` by filename stem.

**Expected structure on the shared drive:**

```
<AnimalID>/
├── YYMMDD_<AnimalID>_<Condition>_<FOV>/
│   ├── T1_..._BEH-001_mc.tif          ← copy this
│   ├── T1_..._BEH-001_RoiSet.zip      ← copy this
│   └── PRE Files/                     ← include this subdirectory if present
│       ├── T1_..._PRE-000_mc.tif      ← copy this
│       └── T1_..._PRE-000_RoiSet.zip  ← copy this
├── YYMMDD_<AnimalID>_<Condition>_<FOV>/
│   └── ...
└── CELLPOSE_TRAINING_AUDIT.md         ← include if present (optional, helpful)
```

**Key points:**
- Include `PRE Files/` subdirectories — the pre-session baseline recordings are valid training data.
- For multi-part sessions (BEH_PT2, BEH_PT3, etc.), include whichever parts have both `*_mc.tif` and `*_RoiSet.zip`.
- Session folder names and file names do not need to be renamed — the pipeline handles common naming variations.

---

## Quick-Copy Script

To copy only the required files from your analysis drive to the shared drive while preserving folder structure, run:

```bash
rsync -av \
  --include='*/' \
  --include='*_mc.tif' \
  --include='*RoiSet.zip' \
  --include='CELLPOSE_TRAINING_AUDIT.md' \
  --exclude='*' \
  /path/to/your/animal/folder/ \
  /path/to/shared/drive/animal/folder/
```

Replace `/path/to/your/animal/folder/` with the source (e.g., your SIMA analysis directory for one animal) and `/path/to/shared/drive/animal/folder/` with the destination on the external drive.

Run once per animal, or adjust the source path to cover multiple animals at once.

> **Note:** The trailing `/` on the source path is required — it tells rsync to copy the *contents* of the folder, not the folder itself.

---

## What Happens Next

Once the drive is handed off, Josh runs the pipeline:

1. Each `*_mc.tif` is processed through Suite2p to extract a mean projection and vertical-correlation map.
2. Each `*_RoiSet.zip` is converted into a Cellpose-format instance mask.
3. The image–mask pairs are added to the training set and used to fine-tune the Cellpose model.

Researchers do not need to run any processing steps — just copy the two file types above.
