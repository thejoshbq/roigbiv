# ROIGBIV

*ROI-Guided Boundary Identification and Validation*

**ROI detection, review, and cross-session cell tracking for two-photon calcium imaging.**

[![Version](https://img.shields.io/badge/version-0.1.10-blue)](https://github.com/thejoshbq/roigbiv/releases)
[![Language](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Changelog](https://img.shields.io/badge/changelog-CHANGELOG.md-orange)](docs/CHANGELOG.md)
[![Phoxel Workbench](https://img.shields.io/badge/Phoxel_Workbench-member-orange)](https://github.com/Otis-Lab-MUSC)

*Written by*: Joshua Boquiren

[![](https://img.shields.io/badge/@thejoshbq-grey?style=flat&logo=github)](https://github.com/thejoshbq)

---

## Overview

ROIGBIV turns two-photon movies into soma ROIs and per-ROI fluorescence traces, and links the same cells across imaging sessions. It has two entry points that share the same motion correction, Cellpose models, registry, and output format:

- **Annotation workflow (web app).** Cellpose proposes soma centroids on the anatomical mean image, a person confirms and corrects them, the registry links them across sessions, and boundaries are drawn from the confirmed centroids. This is the lab's day-to-day path.
- **Automated pipeline (CLI).** A sequential subtractive cascade of four detectors, each with a validation gate, for unattended batch runs.

Human corrections are stored as append-only JSONL edit logs under each FOV's `corrections/` directory and replayed on read; pipeline outputs are never overwritten.

## Annotation workflow

The Dash app (`roigbiv-ui`) has one page per operation, run in order on a workspace folder of TIFF stacks:

1. **Motion correction.** Suite2p rigid + non-rigid registration by default, with a live registration preview and per-FOV quality metrics. Writes `{stem}_mc.tif`. GPU row-wise phase correlation and a SIMA HMM backend are opt-in.
2. **Discovery.** Per-FOV Cellpose calibration, then centroid detection on the mean image (`mean_M`); each centroid records whether a Suite2p candidate corroborates it. Boundaries are seeded from the confirmed centroids: Cellpose's flow field defines cell extent, and a watershed on `-cellprob` with the centroids as markers assigns pixels to cells. Hand-drawn boundaries override the computed ones. A movie player streams the registered movie under the markers for finding transiently active cells. Traces can be extracted here without running Tracking.
3. **Tracking.** Set session order, stamp centroids as fixed-radius disks (`merged_masks.tif`), and register the sessions through the cross-session registry. A contact sheet shows each cell across sessions, with edits for missed centroids and wrong or missing links.

The registry receives disks, not real boundaries (ADR-0003): segmentation shape varies between sessions for the same cell and would leak into the matching embeddings. Real boundaries are written separately to `boundaries.tif` with the same label IDs.

## Automated pipeline

```
Foundation  motion correction -> truncated-SVD low-rank/sparse split -> summary images
Stage 1     Cellpose spatial detection    -> Gate 1 morphology        -> subtract
Stage 2     Suite2p temporal detection    -> Gate 2 cross-validation  -> subtract
Stage 3     GCaMP template sweep          -> Gate 3 waveform          -> subtract
Stage 4     tonic-neuron search           -> Gate 4 correlation contrast
            QC -> trace extraction -> activity classification -> review queue
```

Each stage detects on the residual left after earlier stages subtract their sources, so dim and sparsely firing cells are not masked by bright ones. The residual is reconstructed on demand from the registered movie and the SVD factors rather than written to disk. Every ROI carries its provenance: `source_stage`, `gate_outcome` (`accept` / `flag` / `reject`), `confidence`, and per-stage scores. Stage 4 candidates always route to review (see [Limitations](docs/limitations.md)).

A directory input runs in workspace mode: outputs go to `<input>/output/<stem>/`, and each session is registered in `<input>/registry.db` as it finishes. `--n-workers 2` processes two FOVs concurrently, with GPU sections serialized by a shared lock.

## Cross-session registry

`roigbiv/registry/` matches FOVs and cells across sessions with ROICaT (RoMa alignment, ROInet embeddings, clustering). A calibrated logistic posterior on embedding similarity classifies each FOV as `hash_match`, `auto_match`, `review`, or `new_fov` (accept at 0.9, review at 0.5). Storage is SQLAlchemy (SQLite by default, Postgres optional) plus a filesystem blob store, with Alembic migrations. Without the `embeddings` extra, matching falls back to phase correlation.

## Outputs

Per FOV: neuropil-corrected traces, dF/F and deconvolved spikes, per-stage ROI masks, summary images (`mean_M`, `std_S`, `vcorr_S`, ...), `roi_metadata.json` with the per-ROI schema, and `registry_match.json`. The `traces/` bundle and `roigbiv-export` (HDF5, one column per cell keyed by global cell ID) feed [pynapse](https://github.com/Otis-Lab-MUSC/pynapse).

## Getting started

Requires conda. An NVIDIA GPU is strongly recommended: Cellpose, ROICaT, and source subtraction use CUDA and fall back to CPU. Suite2p is CPU-only.

```bash
conda env create -f environment.yml
conda activate roigbiv
pip install -e ".[ui,embeddings]"     # web app + ROICaT matching
roigbiv-ui                            # http://127.0.0.1:8050
```

The `ui` extra currently depends on a local checkout of `phoxel-tokens` (shared Phoxel Workbench design tokens), which is not yet published.

| Command | Purpose |
|---|---|
| `roigbiv-pipeline --input PATH --fs HZ` | Run the pipeline on a TIFF or a workspace folder |
| `roigbiv-registry {list,show,match,track,backfill,migrate}` | Inspect and maintain the registry |
| `roigbiv-export` | Export traces to HDF5 |
| `roigbiv-reingest` | Ingest masks edited in Fiji/ImageJ |
| `roigbiv-benchmark` | Validate a benchmark FOV manifest |

`--fs` must be the effective frame rate of the stack. For frame-averaged stacks, pass the averaged rate and `--frame-averaging N`; a wrong rate miscalibrates the Stage 3 templates, the Stage 4 bandpass, and deconvolution. Stages 2 to 4 can be skipped with `--no-stage-N`.

## Stack

| Component | Libraries |
|---|---|
| Detection | Cellpose 3, Suite2p, PyTorch, NumPy, SciPy, scikit-image |
| Registry | ROICaT, SQLAlchemy 2, Alembic |
| Web app | Dash, Plotly, dash-bootstrap-components |
| Desktop viewer (optional) | napari |
| QC and classification | scikit-learn, pandas, PyTables, OpenCV |

## Documentation

- [Design overview](docs/design/OVERVIEW.md): architecture and per-stage behavior
- [Limitations](docs/limitations.md): what ROIGBIV does not claim
- [Decision records](docs/adr/)
- [Changelog](docs/CHANGELOG.md)

## License

MIT. See [LICENSE](LICENSE).

## Contact

Joshua Boquiren ([thejoshbq@proton.me](mailto:thejoshbq@proton.me)) · [GitHub: thejoshbq/roigbiv](https://github.com/thejoshbq/roigbiv)
