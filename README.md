# ROI G. Biv — Cell Detection Pipeline

**Sequential subtractive ROI detection for two-photon calcium imaging, with cross-session cell tracking and a browser-based review interface.**

[![Version](https://img.shields.io/badge/version-0.1.10-blue)](https://github.com/thejoshbq/roigbiv/releases)
[![Language](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Changelog](https://img.shields.io/badge/changelog-CHANGELOG.md-orange)](docs/CHANGELOG.md)
[![Phoxel Workbench](https://img.shields.io/badge/Phoxel_Workbench-member-orange)](https://github.com/Otis-Lab-MUSC)

*Written by*: Joshua Boquiren

[![](https://img.shields.io/badge/@thejoshbq-grey?style=flat&logo=github)](https://github.com/thejoshbq)

---

## Overview

ROI G. Biv turns raw two-photon calcium imaging stacks into curated cell masks and fluorescence traces. Rather than accepting a single detector's output, it applies four detection passes in sequence — Cellpose morphology, Suite2p temporal activity, a GCaMP template sweep, and a tonic-neuron search — each operating on the residual left after the previous pass subtracts what it found, so dim and slow-firing cells are recovered instead of being masked by bright ones. Every ROI carries its own provenance: which stage found it, which quality gate it passed, and a confidence score you can filter on downstream.

Across imaging sessions, the cross-session registry matches the same field of view and the same individual cells over days of recording, so longitudinal analyses follow real cells rather than re-detected approximations. Curation happens in a browser-based interface for motion correction, centroid review, session tracking, and boundary editing, with all human corrections stored additively so pipeline outputs are never overwritten. Results export to [pynapse](https://github.com/Otis-Lab-MUSC/pynapse) for signal analysis and to axplorer for visualization.

---

## Getting Started

Install order matters for Suite2p and Cellpose, so use the provided conda environment:

```bash
conda env create -f environment.yml
conda activate roigbiv
```

Reference documentation lives in [`docs/`](docs/) — pipeline behavior, tunable parameters (`configs/pipeline.yaml`), and the researcher data guide.

---

## Architecture & Dependencies

| Component | Language | Framework / Libraries |
|---|---|---|
| Detection pipeline | Python 3.10 | Cellpose (<4.0), Suite2p, PyTorch, NumPy, SciPy, scikit-image, tifffile |
| Cross-session registry | Python 3.10 | SQLAlchemy 2, Alembic, ROICaT, PyTorch |
| Web interface | Python 3.10 | Dash, Plotly, dash-bootstrap-components |
| Desktop viewer (optional) | Python 3.10 | napari |
| Classification & QC | Python 3.10 | scikit-learn, pandas, PyTables, OpenCV |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Joshua Boquiren — [thejoshbq@proton.me](mailto:thejoshbq@proton.me)

[GitHub: thejoshbq/roigbiv](https://github.com/thejoshbq/roigbiv)
