# Review page — data dependencies

## What it does

The Review page (`roigbiv-ui` → "Review" tab) shows you a saved FOV across all sessions where it was imaged: a multi-session grid of mean projections with ROI outlines, a cross-session FOV mean trace, and a per-ROI cross-session trace drawer. **It is read-only.** Editing happens out-of-process by opening the FOV's output directory in Fiji/ImageJ and round-tripping through `roigbiv-reingest`. Input is the URL (`?fov_id=…` or one chosen from the dropdown). Output is rendered Plotly figures and metadata cards — nothing is written back.

## Short answer to your question

**No, the Review page never opens the raw TIF (`*_mc.tif`).** It needs `registry.db` *plus* the per-session pipeline output directories — but those are the lightweight artifacts (PNG-scale TIFs, JSON metadata, NPY trace matrices), not the raw movie or the heavy intermediates (`data.bin`, `residual_S.dat`, `svd_factors.npz`).

You need both halves:
- `registry.db` alone is not enough — it gives you the FOV/session list but not the masks, projections, or traces.
- The output directories alone are not enough — without `registry.db` the page can't enumerate which sessions belong to a `fov_id` or align ROIs across sessions by `global_cell_id`.

## The dependency map

| Source | What the Review page reads from it | Required? |
|---|---|---|
| `registry.db` (the SQLAlchemy store) | FOV list, session list per FOV, `local_label_id → global_cell_id` per session | **Yes** |
| `<output_dir>/pipeline_log.json` | shape, k_background, stage_counts | Yes |
| `<output_dir>/roi_metadata.json` | per-ROI fields (gate_outcome, source_stage, area, activity_type, features) | Yes |
| `<output_dir>/merged_masks.tif` | uint16 label image — turned into per-ROI bool masks + Plotly contours | Yes |
| `<output_dir>/summary/mean_M.tif` | the gray background you see behind the outlines | Yes (without it, a placeholder is shown) |
| `<output_dir>/summary/{mean_S,max_S,std_S,vcorr_S,…}.tif` | available to the loader, not currently shown on Review | Optional |
| `<output_dir>/traces/traces.npy` + `traces_meta.json` + `traces_neuropil.npy` + `traces_raw.npy` | per-ROI fluorescence (drawer + cross-session figure) | Yes when a trace is shown |
| `<output_dir>/dFF.npy` | per-ROI ΔF/F (drawer + cross-session figure when "dF/F" toggle is on) | Yes when "dF/F" is selected |
| `<output_dir>/corrections/*.jsonl` | HITL (human-in-the-loop) edits — append-only ops replayed on top of the pipeline ROIs at load time | Optional |
| `<output_dir>/registry_match.json` | sanity-check fallback for `global_cell_id` if the DB observation is missing | Optional |
| `<output_dir>/suite2p/plane0/data.bin` | — | **No, never read by Review** |
| `<output_dir>/residual_S.dat` | — | **No** |
| `<output_dir>/svd_factors.npz` | — | **No** |
| `data/raw/*_mc.tif` | — | **No** |

In other words: Review is a thin viewer over the *artifacts* the pipeline persisted. The "raw data" only matters when you go *back* to the pipeline (rerun, re-extract traces, or external-edit then re-ingest).

## Walkthrough on a small concrete example

Say you have one FOV imaged on three days. After running the pipeline three times, your tree looks like:

```
data/raw/
├── registry.db                                  ← the SQLite registry
├── T1_221209_…_mc.tif                           ← raw, never read by Review
├── T1_221215_…_mc.tif
├── T1_230116_…_mc.tif
└── output/
    ├── T1_221209_…/
    │   ├── pipeline_log.json
    │   ├── roi_metadata.json
    │   ├── merged_masks.tif
    │   ├── registry_match.json
    │   ├── summary/mean_M.tif        + 6 other summary TIFs
    │   ├── traces/{traces,traces_neuropil,traces_raw}.npy + traces_meta.json
    │   ├── dFF.npy
    │   ├── corrections/*.jsonl       (optional)
    │   └── suite2p/plane0/data.bin   (large; not read by Review)
    ├── T1_221215_…/   …
    └── T1_230116_…/   …
```

When you load `/review?fov_id=fov_a3f2…`:

1. **Sidebar populates** → `list_fovs()` (`roigbiv/ui/services/registry_service.py:27`) does `store.list_fovs()` against `registry.db`. **Only the DB is touched.**
2. **You pick the FOV** → `load_cross_session_bundle(fov_id)` (`roigbiv/ui/services/loaders.py:203`) does `store.list_sessions(fov_id)` to get `[(session_id, session_date, output_dir, fov_posterior), …]`. **Only the DB is touched.**
3. **For each session**, the loader walks into `output_dir` and:
   - calls `load_fov_from_output_dir(output_dir)` (`roigbiv/pipeline/loaders.py:27`) which reads `pipeline_log.json`, `roi_metadata.json`, `merged_masks.tif`, and the seven `summary/*.tif` files;
   - replays HITL ops with `load_corrections` + `apply_corrections`;
   - reads `registry_match.json` for `local_label_id → global_cell_id`;
   - then **overrides** those gcids with the authoritative ones from `store.list_observations_for_session(session_id)` (DB is source of truth — disk JSON is just a backup if the DB is unreachable, see `loaders.py:230–237`).
4. **You click an ROI** → `load_session_traces(output_dir)` (`roigbiv/ui/services/trace_viz.py:137`) reads `traces/{traces.npy, traces_meta.json}` (and `dFF.npy` from the parent or the bundle dir). This populates the right-hand cross-session figure.
5. **You hit "Open output folder"** → `resolve_mask_target(output_dir)` returns `corrections/corrected_masks.tif` if it exists, else `merged_masks.tif`. Fiji opens that file. **The raw TIF is never touched.**

State at each step:
- After (1): app state has `[FOVRow(fov_id="fov_a3f2…"), …]`. Disk reads: 0.
- After (2): app state has `[SessionRef(session_id, output_dir, fov_posterior), …]`. Disk reads: 0.
- After (3): `CrossSessionBundle.bundles[session_id] = FOVBundle(mean_M, [ROIRender, …], …)`, fully populated for all three days. Disk reads: ~10 small files per session.
- After (4): A `SessionTraces` cached in `app_state` per session. Disk reads: 2–4 small NPYs per session.

## The code path, annotated

```python
# roigbiv/ui/services/registry_service.py:27
def list_fovs() -> list[FOVRow]:
    from roigbiv.registry import build_store
    store = build_store()                 # opens registry.db (SQLAlchemy session)
    store.ensure_schema()
    rows: list[FOVRow] = []
    with store.session_factory() as s:
        for fov in store.list_fovs():     # one row per FOV in the registry
            …                              # DB only — no disk artifacts touched
    rows.sort(key=lambda r: (r.animal_id or "", r.region or "", r.fov_id))
    return rows
```

```python
# roigbiv/ui/services/loaders.py:203
def load_cross_session_bundle(fov_id: str) -> CrossSessionBundle:
    store = build_store()
    sessions_rows = sorted(store.list_sessions(fov_id), …)  # DB

    bundles: dict[str, FOVBundle] = {}
    for row in sessions_rows:
        out_dir = Path(row.output_dir)
        if not out_dir.exists():
            continue                                        # graceful — same as napari viewer
        bundle = load_fov_bundle(out_dir)                   # ← disk reads start here
        # The DB observation table wins over the on-disk JSON, since a
        # rematch updates the DB but not the JSON.
        gcids = {obs.local_label_id: obs.global_cell_id
                 for obs in store.list_observations_for_session(row.session_id)}
        for rr in bundle.rois:
            rr.global_cell_id = gcids.get(rr.label_id, rr.global_cell_id)
        bundles[row.session_id] = bundle
    return CrossSessionBundle(fov_id=fov_id, …, bundles=bundles)
```

```python
# roigbiv/pipeline/loaders.py:27 (pipeline-level loader, reused by the UI)
def load_fov_from_output_dir(output_dir: Path) -> tuple[FOVData, list]:
    log     = json.loads((output_dir / "pipeline_log.json").read_text())   # required
    meta    = json.loads((output_dir / "roi_metadata.json").read_text())   # required
    merged  = tifffile.imread(str(output_dir / "merged_masks.tif"))         # required
    mean_M  = _maybe_read_tif(output_dir / "summary" / "mean_M.tif")        # optional
    …                                                                       # rest of summary
    # FOVData also gets paths it never reads (data.bin, residual_S.dat) so
    # other consumers (re-extract, rerun) can plumb to them. The Review page
    # ignores those fields.
```

The "**why**" for each comment in the actual files: the DB is canonical for cell identity (so a re-match updates state without rewriting disk JSON); on-disk artifacts are canonical for ROI geometry and traces (the registry never duplicates them). The split is deliberate — it keeps `registry.db` small and lets you nuke a session's output dir without orphaning DB rows, or rebuild the DB without losing ROI work.

## Complexity

Trivial. The page does `O(n_sessions × n_rois_per_session)` light disk I/O on FOV selection (a handful of TIFs and JSONs per session) and `O(n_rois)` for trace loading on click. Nothing on this page is computationally interesting — the heavy lifting already happened in the pipeline.

## Edge cases and gotchas

- **Output directory deleted but row still in DB.** `load_cross_session_bundle` silently drops sessions whose `output_dir` doesn't exist (`loaders.py:225`). The session disappears from the Review grid; no error surfaces. Same behaviour as the legacy napari viewer.
- **Output directory exists but `pipeline_log.json` / `roi_metadata.json` / `merged_masks.tif` is missing.** `load_fov_from_output_dir` raises `FileNotFoundError`; the loader catches it (`loaders.py:228`) and again drops the session silently. If you don't see a session you expected, this is usually why.
- **`registry_match.json` is stale after a rematch.** The DB-side `list_observations_for_session` overrides the on-disk gcids (`loaders.py:230–237`) — the JSON is only consulted when the DB has no observation row. Don't trust the JSON in isolation.
- **HITL corrections are additive by design.** They replay on top of the pipeline ROIs at load time. If `corrections/*.jsonl` is corrupt, the *pipeline* ROIs still render — corrections fail open, never closed.
- **`data_bin_path` and `residual_S.dat` paths are populated on the FOVData shell** even though the Review page never reads them. They exist because the same `FOVData` is consumed by `roigbiv-reextract` and the napari viewer, which *do* need them. Keep this in mind if you copy an output dir to a different machine and prune the heavies — Review will still work, but a re-extract will fail at the missing-`data.bin` check.
- **Cache key for traces is `traces_meta.json` mtime** (`trace_viz.py:111`). If you write a new bundle without bumping the meta sidecar, the UI keeps serving the cached traces. Always rewrite the sidecar.
- **`traces/corrections-*/`** subdirs are preferred over the primary `traces/` when their meta sidecar is newer (`trace_viz.py:71`). Edit-then-reextract creates these. If you delete one mid-session, refresh the page.

## When to use this knowledge

- **Migrating a workspace to a new machine** — copy `registry.db` *and* each `<fov>/output/<stem>/` (the small artifacts; data.bin/residual_S/svd_factors are optional unless you'll re-extract or rerun). Skip the raw `_mc.tif`s if your only goal is browsing.
- **Disk-pressure cleanup** — safe to delete `data.bin`, `residual_S*.dat`, `svd_factors.npz` for a finished FOV; Review still works. Don't delete `merged_masks.tif`, `roi_metadata.json`, `pipeline_log.json`, `summary/mean_M.tif`, `traces/`, or `dFF.npy`.
- **Debugging "session missing from Review"** — first check the DB row's `output_dir` exists; then check the three required artifacts (`pipeline_log.json`, `roi_metadata.json`, `merged_masks.tif`) are present.
- **Sanity-checking a cross-session match** — open Review, color by "Cross-session", and confirm same-color outlines land on the same cells across days. If colors look wrong, the DB observation table is the source of truth — re-run `roigbiv-registry match` rather than editing `registry_match.json`.

Alternative paths and when to reach for them:
- **Pipeline rerun** (`roigbiv-pipeline`): when you need new ROIs or new traces. Needs the raw `_mc.tif`.
- **Re-extract** (`roigbiv-reextract`): when masks changed (Fiji edit) but you want to keep everything else. Needs `merged_masks.tif`/`corrected_masks.tif` and `data.bin`.
- **Registry CLI** (`roigbiv-registry list|show|match|backfill|migrate`): when you need to inspect or repair the DB outside the UI. The Review page is intentionally read-only on the registry.
