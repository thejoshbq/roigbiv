# Changelog

All notable changes to roigbiv are documented here.

## Unreleased

### Added

- **Seeded cell boundaries from confirmed centroids.** Since ADR-0003 the project has
  had no real cell boundary anywhere: `merged_masks.tif` stamps fixed-radius disks, and
  centroid discovery computed Cellpose's masks only to throw everything away but their
  centre of mass. `/cells` now renders a real boundary per confirmed cell, written to a
  new `boundaries.tif` (+ `boundaries.json`) alongside — not instead of — the disks the
  registry matches on. Both carry the same label ids over the same effective centroids,
  so `CellObservation.local_label_id` resolves in either and the registry needed no
  change.

  The boundary is Cellpose's own flow field, re-clustered against the confirmed
  centroids instead of its histogram peaks (`roigbiv/pipeline/seeded_masks.py`): the
  flow decides which pixels are cell material, a watershed seeded on the centroids
  decides which cell each one belongs to, and a seed that captures no basin falls back
  to the canonical disk so a confirmed cell can never disappear. The two-step split
  matters — where Cellpose merges two touching somata its flow field has a single
  attractor equidistant from both, so a nearest-seed rule assigns nothing; the watershed
  recovers both cells. Centroid discovery now caches `dP`/`cellprob` under `flows/` on
  the same recompute key `centroids.json` uses, so a HITL centroid edit redraws
  boundaries without re-running inference.

  Measured against 782 hand-drawn ImageJ somata across 5 cranial-window FOVs
  (`scripts/boundary_bakeoff/`): on the cells where a flow field exists, seeded
  boundaries score mean IoU 0.668 against fixed disks' 0.640, while recall stays at
  0.977 versus free Cellpose's 0.143 on the same data. The win is small because these
  somata are near-circular at ~18 px and a disk is already close to ideal; the prism
  FOVs this was built for have no boundary ground truth yet. See
  `docs/adr/0005-seeded-boundaries-parallel-geometry-track.md` for the full numbers and
  the traps in reading them.

  New config: `centroid_persist_flows` (default on, ~6 MB/FOV at 512²),
  `boundary_capture_px`, `boundary_min_area`, `boundary_max_area`.
  `centroids.json` schema 4 → 5.

- **Cross-session tracking HITL controls.** The `/cells` page gained an edit mode
  (off by default) for the errors that were previously visible but unfixable there: a
  missed or misplaced soma centroid, and a cell the matcher failed to link across
  sessions. Delete / add / move a centroid; link or unlink a cell across sessions;
  "place here" composes add + link in one gesture when a selected cell is missing from
  the clicked session (`roigbiv/ui/pages/cells.py`). Edits apply instantly with an
  "Undo last" spanning every log the FOV owns.

  Two new append-only JSONL logs carry the edits — `corrections/centroids.jsonl` per
  session, `corrections/matches/{fov_id}.jsonl` per FOV
  (`roigbiv/pipeline/centroid_edits.py`, `roigbiv/registry/cell_edits.py`) — replayed by
  a single materializer, `apply_tracking_edits`, that both `run_tracking` and the UI call
  through. No ROICaT match runs for an edit: labels became explicit rather than
  positional (`stamp_labeled_centroids`, byte-identical output for an unedited
  workspace), so a moved centroid keeps its cell, a deleted one drops its observation,
  and an added one gets a deterministic new cell. `run_tracking` replays both logs after
  every registration, since a centroid edit changes the FOV fingerprint and would
  otherwise be silently destroyed by the next re-match. See
  `docs/adr/0004-tracking-hitl-additive-edit-logs.md` for the full design and named
  tradeoffs.

- **Canonical fixed-radius ROI stamps.** Every accepted/flagged ROI, from all four
  detection stages, now has its detector-native boundary (irregular Cellpose/Suite2p mask,
  or a regionprops-derived Stage-4 blob) replaced post-gate with a fixed-radius disk
  centered on its own centroid (`roigbiv/pipeline/roi_stamp.py`, new
  `PipelineConfig.roi_stamp_radius`, default 8 px, auto-scaled per FOV like
  `spatial_pool_radius`). Gates 1/2/4 are unchanged — they still validate real detector
  geometry before this runs; `area`/`solidity`/`eccentricity` remain that gate-time record.
  Motivation: session-to-session segmentation shape variance was leaking into the
  registry's ROICaT cross-session matching embeddings as a confound; one canonical shape
  per ROI removes that confound without touching the registry/subtraction/trace code
  itself (all three already consumed `roi.mask` generically). A new crowding guard
  (`resolve_crowding`) demotes the weaker of any two heavily-overlapping stamps
  `accept → flag`, mirroring Gate 1's existing merge-peak convention. See
  `docs/adr/0003-centroid-canonical-roi-stamps.md` and `docs/design/OVERVIEW.md` §9 for the
  full rationale and named tradeoffs (subtraction/trace precision vs. real footprints).

- **Live motion-correction view.** The Dash Pipeline page now shows the FOV
  being corrected *while* registration runs: raw and corrected panes of the
  same frame, an A/B blink that flips them in place, a raw running-average
  pane (a cumulative mean of previewed raw frames, showing drift the
  uncorrected movie has accumulated so far — an earlier `corrected − raw`
  difference pane was tried first and dropped for not carrying an actionable
  signal), live rigid-shift and phase-correlation-confidence traces, and
  running quality metrics. Previously the only MC feedback was a mean
  projection and four numbers computed after a FOV finished, 10–40 min into a
  run.

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

