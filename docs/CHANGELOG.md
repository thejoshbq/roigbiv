# Changelog

All notable changes to roigbiv are documented here.

## Unreleased

### Added

- **Live motion-correction view.** The Dash Pipeline page now shows the FOV
  being corrected *while* registration runs: raw and corrected panes of the
  same frame, an A/B blink that flips them in place, a difference pane, live
  rigid-shift and phase-correlation-confidence traces, and running quality
  metrics. Previously the only MC feedback was a mean projection and four
  numbers computed after a FOV finished, 10–40 min into a run.

  The pipeline side writes a sidecar to `{output_dir}/mc_preview/`
  (`roigbiv/pipeline/mc_preview.py`) on every run — CLI, batch, and UI alike —
  so a finished run leaves a scrubbable timeline for A/B-ing two backends on
  the same FOV. `phasecorr` is fed by wrapping Suite2p's per-batch
  `register_frames` (`mc_preview_s2p.py`), which also surfaces the
  phase-correlation peak `cmax` — a registration-confidence signal the pipeline
  did not previously expose; `rowwise-pcc` emits from its own batch loop;
  `legacy`/SIMA reports that it has no in-process hook rather than showing an
  empty card. The writer never raises and never touches the registered data:
  output is byte-identical with the preview on or off, and a full disk or a
  Suite2p API change degrades the view instead of failing the run.

  New knobs: `--no-mc-preview` plus `mc_preview_{enabled,max_dim,
  min_interval_s,max_records,metrics,diff}`. All are excluded from the resume
  fingerprint. Served to the browser over `/api/mc-preview/{list,state,image}`
  rather than Dash callbacks, so the ~2.5 Hz refresh costs no Python per tick.
  See `docs/design/OVERVIEW.md` §3.1a.

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

- `docs/adr/0001-non-destructive-candidate-union.md` — first Architecture
  Decision Record, establishing the `docs/adr/` convention. Records the
  pivot from the destructive subtractive cascade toward a non-destructive
  candidate union + joint validation (Stages 1–4 become proposal
  generators; inter-stage subtraction deprecated as the default control
  flow; L+S+T decomposition out of scope). Linked from `README.md` and
  `docs/design/OVERVIEW.md` §14; also fixes README's stale
  `docs/OVERVIEW.md` link (the file now lives at `docs/design/OVERVIEW.md`).
- `docs/adr/0002-cascade-default-deprecation-criteria.md` — operationalizes
  ADR-0001 with fixed, benchmark-checkable go/no-go criteria (detection F1,
  split/merge counts, false-transient rate, review burden, runtime) for
  when `candidate_union` becomes the default `pipeline_mode`, plus the
  migration path once that criteria set passes. Does **not** flip
  `DEFAULT_PIPELINE_MODE` or implement any benchmark tooling — planning doc
  only, per roadmap item H5. Linked from `README.md` and
  `docs/design/OVERVIEW.md` §14.
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
- **Single-frame TIF series discovery generalized.** `roigbiv.io.discover_tifs`
  previously only recognised PrairieView/Bruker sessions, and only in
  *immediate* subdirectories of the input root. Pointing `--input` at the
  session directory itself yielded thousands of one-page "FOVs", each failing
  `validate_tif`; any non-PrairieView numbering did the same. Discovery now
  walks the whole tree (including the root itself), tries an ordered list of
  conventions (`_SERIES_PATTERNS`: PrairieView first, then generic
  `*_NNN.tif` numbering), and verifies a spread sample of frames really are
  lone 2D images of matching shape and dtype before assembling. Directories
  whose frame positions collide (interleaved channels) or that mix two
  filename prefixes are refused rather than silently concatenated into a
  scrambled stack, and chunked multi-frame files that merely *look* numbered
  are left as independent inputs. Assembled stacks still cache under
  `{root}/_stacks/`; nested sessions join their path components
  (`mouse1_day1.tif`) instead of colliding on a shared leaf name.

  Only the files that actually matched a series pattern are consumed, so a
  genuine multi-frame stack sitting alongside the frames remains an input in
  its own right. Symlinked acquisition trees are followed (with a realpath set
  breaking cycles) — neither `os.walk` nor pathlib's `**` descends into them by
  default, which would otherwise make a symlinked session invisible to both the
  series scan and the flat scan.

  Also fixes multi-cycle ordering: PrairieView restarts the frame index each
  cycle, so sorting on the trailing number alone interleaved them. A two-cycle
  session assembled as `c1f1, c2f1, c1f2, …`; ordering is now `(cycle, index)`.

  `assemble_prairie_stack` is now a deprecated alias for
  `assemble_frame_series`.

