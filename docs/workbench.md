# Engineering workbench (`roigbiv/workbench`)

Experimental. Tools for asking why a run found or missed a cell, from the files a run saved. Every module reads
a run directory and never writes to it; nothing in the pipeline imports the workbench. The one pipeline-side
piece is the gate-time mask sidecar (`roigbiv/pipeline/gate_masks.py`, below), a record the pipeline writes and
never reads back.

Install with the `workbench` extra (`pyarrow`). Modules and their tests live in `roigbiv/workbench/` and
`roigbiv/workbench/tests/`. Schemas are versioned (`models.SCHEMA_VERSION`, `match.MATCHER_VERSION`) and may change
without notice until the tool contract (a future ADR) is written.

## What each module answers

| Module | Question | Main entry points |
|---|---|---|
| `models` | What is one candidate, and one run? | `CandidateRow`, `RunManifest`, `check_outcome_counts` |
| `table` | Every candidate of every stage, rejects included, with its gate quantities and footprint | `build_fov_table`, `write_fov_table`, `read_fov_table` |
| `manifest` | Which inputs, config and code produced a saved run | `backfill_manifest`, `write_manifest`, `read_manifest` |
| `annotations` | Hand-drawn cells in the run's movie frame | `read_roiset`, `best_shift`, `to_movie_frame` |
| `match` | Which candidate is each annotated cell, one to one, and what would chance give | `match`, `summarize`, `rotate_180`, `shift_wrap` |
| `ledger` | Per stage: how many annotated cells were proposed and kept, and where each miss went | `fov_ledger`, `unreadable_pools` |
| `regate` | What a changed gate threshold would have kept, on the saved candidate pool | `regate`, `check_overrides` |
| `attenuation` | How much of a cell's calcium event survives background removal and source subtraction | `cell_attenuation`, `detect_events`, `*_region_traces` |
| `validity` | Which pixels the rigid registration shift filled with wrapped content | `suite2p_crop`, `valid_mask`, `wrapped_fraction`, `wrapped_frames` |

## Candidate table

`build_fov_table(fov_dir, run_id=..., fov_id=...)` reads `stage{N}/stage{N}_report.json` for N = 1 to 4 and
makes one `CandidateRow` per reported candidate. The outcome counts must equal each report's `detected`,
`accepted`, `flagged` and `rejected`; anything that does not add up, or any file that disagrees with another, goes
into `FovTable.problems`. `write_fov_table` refuses a table with problems unless `allow_problems=True`.

`write_fov_table(table, dest)` writes `candidates.parquet` (one row per candidate; each gate quantity is a
`gate.<name>` column), `footprints.npz` (flat pixel indices per candidate, CSR layout) and `table_meta.json` to
`dest`, never into the run directory.

Each row names where its footprint came from (`CandidateRow.footprint`), best source first:

| Source | From | Used for |
|---|---|---|
| `detected` | `stage{N}/stage{N}_gate_masks.npz`, the mask the gate judged | every candidate of a stage, when the run saved the sidecar |
| `suite2p_stat` | Suite2p's `stat.npy` (`suite2p_stat_index`) | Stage 2, older runs |
| `pool_disk` | a disk of `spatial_pool_radius` at the report centroid | Stage 3, older runs |
| `stamp` | the stage mask TIFF (the fixed disk that replaced the detected shape) | kept Stage 1 and 4 candidates, older runs |
| `not_recorded` | nothing saved | Stage 1 and 4 rejects in older runs; never read as "no candidate" |

Footprints are checked against the gate's recorded area. A sidecar is also checked for missing or extra labels,
for a frame shape that differs from the run's, for being unreadable, and for a report centroid more than
`table.CENTROID_TOLERANCE_PX` (2 px) from every pixel of its mask, which is what a sidecar left by another run
looks like. A rejected sidecar falls back to the older sources and is listed as a problem.

## Gate-time mask sidecar (pipeline)

After each gate, `roi_stamp.canonicalize` replaces every kept mask with a fixed-radius disk, so the detected
shape (a Cellpose outline, a Suite2p footprint, a Stage 4 component) is gone from the run's outputs.
`gate_masks.save_gate_masks` is called just before each of the four `canonicalize` loops in `run.py` and writes
`stage{N}/stage{N}_gate_masks.npz`:

| Key | Content |
|---|---|
| `label_id` | int64, one per candidate in the stage's gate output, rejects included |
| `offsets` | int64, CSR offsets into `flat_idx` (length `len(label_id) + 1`) |
| `flat_idx` | int64, sorted flat pixel indices of each mask |
| `shape` | int64, the frame shape `(Ly, Lx)` |

Read it with `gate_masks.load_gate_masks(path)`, which returns `(shape, {label_id: flat_idx})`.

It never changes the ROIs and adds no config field, so masks, traces and the resume fingerprint are unchanged.
This was checked on a real run (CARL-1M, 450 frames, two runs without the writer and one with it): every file was
byte-identical apart from the timestamps and paths that also differ between two runs without it. Any earlier
sidecar is removed before writing, the write is atomic, and every error becomes a printed warning with no
sidecar left for that stage. On `--resume`, a skipped stage keeps both its old report and its old sidecar.

Stage 3's detected mask is its pooling disk, since that is what Gate 3 judged.

## Matching and the miss ledger

`match(anns, cands, radius=..., shape=...)` assigns annotations to candidates one to one (minimum total centroid
distance within `radius`) and returns one `Link` per annotation. Two measures:

- **localized**: some candidate centroid lies within `radius` (not one to one);
- **recoverable**: assigned, with footprint overlap of at least `MIN_OVERLAP` (0.5) and an area ratio in
  `AREA_RATIO` (0.5 to 2). Unknown, not a miss, when either footprint is missing.

A chance baseline repeats the join against the annotations rotated 180 degrees and shifted by a fixed offset.
`fov_ledger` runs this per stage on two pools (all proposed, and kept) and gives each annotated cell a fate:
`found`, `shared`, `rejected` (with the nearest in-reach reject's reason codes), `chance_only` or
`never_proposed`. A pool whose own chance baseline exceeds `ledger.STOP_BASELINE` (0.3) is chance-level and is not
read; a rejection found only through such a pool is listed in `Fate.chance_level`, not counted as a reason.

Until runs save their sidecars, recoverable recall for kept Stage 1 and 4 candidates measures the stamp, not the
detected shape. Localization is unaffected.

## regate

`regate(table, run_config, overrides)` applies each gate's own rule (`roigbiv/pipeline/gate{1..4}.py`) to the
saved gate quantities and replays crowding (`roi_stamp.resolve_crowding`) stage by stage. With no overrides it
must reproduce every saved outcome and reason code (`RegateResult.mismatches` empty). Results are labelled
"frozen pool, non-causal downstream": the pool is the saved one, and a changed outcome does not change what later
stages saw. Only a rerun answers that.

`check_overrides` refuses settings that change how a quantity is measured (annulus and search radii, windows,
percentiles) and thresholds past the point where a saved stand-in value would start failing. Overrides for checks
a run never recorded are listed in `not_applied`.

## Attenuation

For a footprint, with M the registered movie (`data.bin`), L the run's rank-k background (`svd_factors.npz`),
S = M - L, and R = S minus the Stage 1 and 2 sources (`residual_S{1,2}.sources.npz`, what Stage 3 searches):

- `movie_region_traces`, `background_region_traces` and `source_region_traces` give each layer's mean trace over
  many footprints at once, from a sparse averaging matrix, without building the dense residual.
- `detect_events` finds upward events above `z` robust SDs over a sliding-percentile baseline. Run it on the
  neuropil-corrected movie trace, so it favours no layer. The same detector on the sign-flipped trace gives the
  trace's noise floor (`n_negative_events`).
- `event_amplitudes` reads each event as the peak window (peak +-1 frame) minus a pre-event window 0.4 to 1.2 s
  before it; NaN where a window falls off the trace or would overlap the peak.
- `cell_attenuation` returns per-event medians: `retained_s` = amplitude in S over amplitude in M, `retained_r`
  the same for R, and `neuropil_share` for the surrounding ring. Ratios can fall below 0 or exceed 1 when the
  background swings against the event; they replace the rank-30 "variance share", which can exceed 1 for any
  trace.

A trace with a NaN sample is refused.

## Validity mask

Suite2p 0.14.5 applies the rigid shift with `np.roll`, so the band a shift exposes holds pixels from the opposite
edge, not zeros. From the per-frame offsets in `motion_trace.npz`:

- `wrapped_fraction` gives the share of frames in which each pixel holds wrapped content;
- `wrapped_frames` gives, for chosen pixels, the frames in which they are wrapped;
- `suite2p_crop` rebuilds Suite2p's valid range (`register.compute_crop`; pass the run's saved
  `ops["badframes"]`), optionally widened by a margin; `valid_mask` turns it into a boolean image and
  `outside_share` measures a footprint against it.

Shifts must be whole pixels (rigid Suite2p offsets are). Suite2p's nonrigid step clamps at the border instead of
wrapping, so it brings in no foreign content.

## Running on a saved run

```python
from pathlib import Path
from roigbiv.workbench.table import build_fov_table, write_fov_table

fov_dir = Path("inference/pipeline/my_fov")
table = build_fov_table(fov_dir, run_id="runA", fov_id="my_fov")
print(len(table.rows), table.problems)
write_fov_table(table, Path("wb/runA/my_fov"))  # never inside fov_dir
```

Scripts that drive these modules over a batch of runs (registration of annotations to a movie, the ledger, the
attenuation and edge-artifact surveys) are kept outside the repository for now.

## Limits

- Old runs hold no detected shape for Stage 1 and 4 candidates; their kept rows carry the stamp and their
  rejects read `not_recorded`.
- `regate` cannot answer how later stages would change; it says so on every result.
- The miss ledger needs annotations registered to the run's movie: `best_shift` finds the offset, and a cell whose
  centroid leaves the movie after the shift is dropped. Nothing checks that the registration is right; check the
  correlation `best_shift` returns before reading a ledger.
- The pipeline does not write `candidates.parquet` or `run_manifest.json` itself: the table is rebuilt from the
  reports in seconds, and the manifest hashes the whole input movie.
