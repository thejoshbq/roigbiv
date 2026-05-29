# Changelog

All notable changes to roigbiv are documented here.

## Unreleased

### Changed

- **Overlay PNG draws every ROI by default.** `roigbiv/overlay.py`
  previously skipped `reject` ROIs silently; now `accept` (green),
  `flag` (orange), and `reject` (red) are all drawn, so gate-discard
  issues surface in the email overlay without opening napari. The
  annotation block always reports `N accept | N flag | N reject`.
  New `--overlay-outcomes` CLI flag (default `accept,flag,reject`)
  re-narrows the view; e.g. `--overlay-outcomes accept` for an
  accept-only image. (Trello: 69f4c594.)
- **Single CLI entry point.** `roigbiv-pipeline` now absorbs every
  feature that previously required `roigbiv-cli`: directory-input
  batching, parallel `--n-workers` workers, overlay PNG render, and
  email-on-done. `--input <dir>` implicitly triggers workspace mode
  (in-input `output/`, `registry.db`, auto-migrate, auto-backfill);
  `--input <file.tif>` runs classic single-FOV mode. The `--workspace`
  flag is gone — directory-vs-file shape of `--input` is the trigger.
- Exit-code contract on `roigbiv-pipeline`: `0` pipeline + email
  succeeded (or email not requested), `1` all FOVs failed, `2` bad
  input, `3` pipeline succeeded but SMTP delivery failed (overlays
  preserved on disk).
- **All four stages now run by default.** `enable_stage_2`,
  `enable_stage_3`, and `enable_stage_4` all default to `True` in
  `PipelineConfig` so the cheapest invocation (`roigbiv-pipeline
  --input <path> --fs 7.5`) gives full coverage. Power users who want
  the fast Foundation + Stage 1 + Stage 2 path drop `--no-stage-3
  --no-stage-4` (~10–25 min/FOV faster). Combine with `--resume` to
  toggle a stage on a prior workspace without re-running upstream
  stages — the resume fingerprint is intentionally insensitive to the
  stage opt-in flags.
- Stages 2 and 3 skip their subtraction step entirely when no downstream
  stage is enabled (no consumer for the residual). Saves ~1.5 GB float32
  disk write + 1–2 min per skipped subtraction.
- When intermediate stages are disabled, downstream stages walk back the
  residual chain via `_stage_input_residual` to find the deepest
  available residual on disk (e.g., Stage 3 reads `residual_S1.dat` if
  Stage 2 subtraction was skipped).

### Added

- `--n-workers` flag on `roigbiv-pipeline`: in workspace mode (directory
  input), > 1 fans heavy pipeline calls through `pipeline/batch.run_batch`
  with the existing 2-worker GPU lock. Light post-pipeline steps
  (registry registration, traces bundle write, backfill) stay in the
  parent process so SQLite writes remain serialized.
- `--diameter`, `--cellprob-threshold`, `--flow-threshold`, `--channels`
  on `roigbiv-pipeline` for Cellpose tuning (formerly only on
  `roigbiv-cli`).
- Test coverage for the email path
  (`roigbiv/pipeline/tests/test_pipeline_email.py`): asserts the SMTP
  wire sequence (`starttls` → `login` → `sendmail`), missing-password
  / auth-failure / `OSError` branches, attachment-cap downsampling, and
  the `roigbiv-pipeline` exit-code contract (0/1/3). Runs without
  binding sockets.
- `scripts/verify_email_smoke.py` — manual one-shot smoke test that
  routes a 1×1 PNG through `roigbiv.pipeline._email.send_email` to
  confirm Proton Bridge / Gmail App-Password auth + STARTTLS on a new
  machine.
- `docs/email-notifications.md` — flag reference, Gmail App-Password
  setup, smoke-test usage, exit-code semantics. Includes a headless /
  SSH-only Proton Bridge runbook (CLI login, openssl wire-extraction
  of Bridge's self-signed cert into the system trust store via
  `update-ca-certificates`, systemd-user persistence with `loginctl
  enable-linger`) for remote lab boxes where the Bridge GUI is
  unreachable. Bridge 3.24.2 cert-handling subcommands (`cert install`
  is absent; `cert export` rejects every path input) are documented
  alongside the wire-extraction workaround.
- `--stage-2` / `--no-stage-2`, `--stage-3` / `--no-stage-3`,
  `--stage-4` / `--no-stage-4` CLI flags on `roigbiv-pipeline` (also
  forwarded through workspace mode).
- `--resume` flag on `roigbiv-pipeline`, plus `resume` parameter on
  `run_with_workspace`: skip stages whose outputs already exist on
  disk. Refuses if the config or input has changed since those outputs
  were written. Recovers correctly from a run interrupted between a
  stage's detection and its subtraction. See spec §21 for full
  semantics; implementation in `roigbiv/pipeline/resume.py`.
- `update_manifest(..., status="skipped")` records stages that were
  intentionally bypassed, distinct from "completed." `plan_resume`
  tolerates skipped stages without refusing.

### Removed

- `roigbiv-cli` console script. Use `roigbiv-pipeline` with the same
  flags (`--input`, `--fs`, `--email-to`, `--smtp-*`, `--no-email`,
  `--n-workers`).
- `--workspace` flag on `roigbiv-pipeline`. Directory-vs-file shape of
  `--input` now triggers the corresponding mode automatically.
- `roigbiv/cli.py` module. Email + overlay helpers now live in
  `roigbiv/pipeline/_email.py`; the entry-point logic is in
  `roigbiv/pipeline/run.py::main`.

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
