# ROI G. Biv

**Find, review, and track cells in two-photon calcium imaging recordings.**

[![Version](https://img.shields.io/badge/version-0.1.10-blue)](https://github.com/thejoshbq/roigbiv/releases)
[![Language](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Changelog](https://img.shields.io/badge/changelog-CHANGELOG.md-orange)](docs/CHANGELOG.md)
[![Phoxel Workbench](https://img.shields.io/badge/Phoxel_Workbench-member-orange)](https://github.com/Otis-Lab-MUSC)

*Written by*: Joshua Boquiren

[![](https://img.shields.io/badge/@thejoshbq-grey?style=flat&logo=github)](https://github.com/thejoshbq)

---

## What it does

ROI G. Biv takes two-photon calcium imaging movies and turns them into a set of cells (ROIs) with a fluorescence trace for each one. It suggests where the cells are, lets you check and correct those suggestions by eye, and follows the same cells across recording sessions so you can compare them day to day.

Your corrections are saved alongside the results rather than written over them, so you can always see what the software found and what a person changed.

## How it is used

Most work happens in a web app that runs on your own machine. Point it at a folder of recordings and work through three pages in order:

1. **Motion correction** — steady each movie and check how well it worked.
2. **Discovery** — find cell centres with Cellpose, then adjust the cells and their outlines. You can play the movie under the markers to spot cells that only light up now and then.
3. **Tracking** — put the sessions in order, match the same cells across them, and fix any missed or wrong matches.

Traces can then be exported to an HDF5 file for analysis in [pynapse](https://github.com/Otis-Lab-MUSC/pynapse).

A fully automated detection pipeline is also available from the command line for batch runs.

## Getting started

Requires conda. An NVIDIA GPU is strongly recommended; without one, cell detection runs much slower.

```bash
conda env create -f environment.yml
conda activate roigbiv
pip install -e ".[ui]"     # adds the web app
roigbiv-ui                 # then open http://127.0.0.1:8050
```

The web app's `ui` extra currently depends on a local checkout of `phoxel-tokens`, the shared Phoxel Workbench styles, which is not yet published.

Command-line tools:

| Command | Purpose |
|---|---|
| `roigbiv-pipeline --input FOLDER --fs HZ` | Run the automated pipeline (use the movie's actual frame rate) |
| `roigbiv-registry` | Inspect and maintain the cross-session cell registry |
| `roigbiv-export` | Export traces to HDF5 |
| `roigbiv-reingest` | Bring back cell masks edited in Fiji/ImageJ |

Run any command with `--help` for its options.

## Documentation

- [Design overview](docs/design/OVERVIEW.md) — how the pipeline works
- [Limitations](docs/limitations.md) — what ROI G. Biv does not claim
- [Decision records](docs/adr/) — why the design is the way it is
- [Changelog](docs/CHANGELOG.md)

## License

MIT. See [LICENSE](LICENSE).

## Contact

Joshua Boquiren — [thejoshbq@proton.me](mailto:thejoshbq@proton.me) · [GitHub: thejoshbq/roigbiv](https://github.com/thejoshbq/roigbiv)
