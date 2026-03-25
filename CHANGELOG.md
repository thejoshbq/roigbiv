# Changelog

All notable changes to roigbiv are documented here.

## [0.1.0] — 2026-03-25

### Added

- `roigbiv` Python package: installable from GitHub releases via pip
- End-to-end Google Colab notebook (`notebooks/roigbiv.ipynb`) — users only need to
  upload pre-motion-corrected TIF files to Google Drive; all processing runs in Colab
- `roigbiv.io`: dynamic TIF file discovery (supports directories, archives, nested
  subdirectories), Suite2p projection extraction (meanImg + Vcorr from ops.npy),
  and model checkpoint download with caching
- `roigbiv.suite2p`: Suite2p batch runner with per-FOV resumability (skips completed
  FOVs), disk management (data.bin deletion), and progress timing
- `roigbiv.union`: Union ROI building — merges activity and anatomy Suite2p passes via
  Hungarian IoU matching, assigns GOLD/SILVER/BRONZE tiers, scores with Cellpose
  probability, writes per-FOV TIFFs and `scored_rois_summary.csv`
- `roigbiv.match`: IoU computation and Hungarian matching (ported from
  `scripts/match_rois.py`)
- `roigbiv.viz`: Interactive Colab viewer using ipywidgets + matplotlib — FOV dropdown,
  tier checkboxes, Cellpose probability threshold slider
- GitHub Actions release workflow (`.github/workflows/release.yml`): builds wheel and
  attaches deployed model checkpoint as release artifact on `v*` tag push
- Git LFS tracking for `models/deployed/current_model` and checkpoint directory
- `pyproject.toml` for standard pip-installable packaging

### Unchanged

- `scripts/` directory: all local development scripts retained without modification
- `configs/pipeline.yaml`: pipeline parameter configuration
