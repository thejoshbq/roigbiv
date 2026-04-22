# ROICaT integration: cross-session cell identity via clustering

**Status**: Phase 1 design — awaiting approval before Phase 2 (adapter implementation).
**Author**: Claude (opus-4-7) + josh.
**Date**: 2026-04-22.
**Target ROICaT version**: 1.5.5 (installed at `/home/thejoshbq/miniconda3/envs/roigbiv/lib/python3.10/site-packages/roicat/`).

---

## Context

The existing FOV fingerprint is derived primarily from the mean projection + Suite2p/Cellpose-consensus centroid tables. Tonic and sparse-firing cells under-contribute to the mean projection, so their visibility varies across sessions even when the underlying biology does not. The current pairwise matcher (embedding-seeded RANSAC → ECC refine → Hungarian on calibrated `p_same`) is fragile against that drift.

Direct evidence: in the three-session dataset (`T1_221209`, `T1_221215`, `T1_230116` — same physical FOV), the pairwise matcher minted three separate FOVs. Session 2 landed in the review band at `fov_posterior=0.694`; session 3 at `0.056`. Ground truth is that all three should collapse to a single FOV.

Remediation:

1. Make cross-session identity depend on pipeline-produced **spatial footprints** (the unified ROI mask in each session's `merged_masks.tif`), not mean-projection appearance.
2. Delegate the cell-matching algorithm itself to **ROICaT**, which is purpose-built for this problem. Use footprint embeddings from ROICaT's ROInet (SimCLR-trained ConvNeXt-tiny) + scattering-wavelet features + session-sensitivity constraints, then cluster via `Clusterer.fit_sequentialHungarian` for our typical ≤8-session case.
3. Preserve the calibrated logistic for FOV-level accept/review/reject banding, but re-derive its input features from ROICaT outputs.

---

## Current state

File-by-file inventory grounded in actual reads, not recollection.

### Data layer

- **`roigbiv/registry/models.py`** — SQLAlchemy ORM: `FOV`, `Cell`, `Session`, `CellObservation`, `alembic_version`. `FOV` row carries `fingerprint_hash` (unique), `fingerprint_version` (1 or 2), and blob URIs for `mean_M`, centroids, pooled FOV embedding, per-ROI embeddings.
- **`roigbiv/registry/store/sqlalchemy_store.py`** — `SQLAlchemyStore` implementing the `RegistryStore` protocol. Candidate retrieval is scoped by `(animal_id, region)` (`find_candidates`), with embedding-ranked variant `find_candidates_by_embedding`.
- **`roigbiv/registry/blob/{base,local}.py`** — `BlobStore` protocol + `LocalBlobStore` writing `file:///…inference/fingerprints/{fov_id}/…` URIs.
- **`roigbiv/registry/migrations/versions/{0001_initial,0002_embeddings}.py`** — Alembic migrations. `0002` added `fingerprint_version`, `fov_embedding_uri`, `roi_embeddings_uri`, `session.fov_posterior`. Current head is `0002`.

### Fingerprint + matching

- **`roigbiv/registry/fingerprint.py`** — `compute_fingerprint(mean_m, cells, *, embedder=None, merged_masks=None)` produces: (a) 64×64 block-mean downsampled mean_M, (b) sorted centroid table, (c) sha256 hash over both, (d) optional ROInet embeddings (v2). The mean projection is the primary hash input.
- **`roigbiv/registry/embedder.py`** — `ROInetEmbedder` wrapping `roicat.ROInet.ROInet_embedder`. This is the only current ROICaT usage in the repo.
- **`roigbiv/registry/match.py`** — pairwise `match_fov(query, candidate, …)`: embedding-seeded RANSAC affine + OpenCV ECC refine → Hungarian on `-log p_same` → FOV posterior via calibrated logistic.
- **`roigbiv/registry/calibration.py`** — two-level logistic. Cell-level: `(delta_px, embedding_distance) → p_same_cell`. FOV-level: `(log_odds_sum, inlier_rate, log1p(n_matched)) → p_same_fov`. Hand-priors `DEFAULT_FOV_COEFS=(-3.0, 0.15, 6.0, 0.8)`. **No calibration JSON on disk** anywhere in the project; the system is running entirely on priors.
- **`roigbiv/registry/orchestrator.py`** — `register_or_match(...)`: (1) hash pre-filter shortcut, (2) retrieve top-k candidates by embedding or `(animal_id, region)` fallback, (3) pairwise `match_fov` loop keeping the best-posterior candidate, (4) branch on `decision ∈ {auto_match, review, reject}`, (5) write session/cells/observations + `registry_match.json` report. This is the heart of what's being replaced.

### External callers (surface area that must keep working)

- **`app.py`** — Streamlit `_registry_tab` at lines 790–1010+. Maintenance panel (migrate button, embedding-recompute button), backfill scanner, FOV overview table, per-FOV session/cell detail, longitudinal cell browser. Consumes `build_store`, `build_blob_store`, `register_or_match`, `run_backfill`, `recompute_embeddings`, `ensure_alembic_head`, `store.{list_fovs,list_sessions,list_cells}`.
- **`roigbiv/pipeline/run.py:629–659`** — `_register_fov_after_pipeline` called at pipeline end when `--registry` is set. Builds `merged_masks` in-memory via `_build_merged_masks` and invokes `register_or_match`.
- **`roigbiv/cli_registry.py`** — `roigbiv-registry` console script: subcommands `list | show | match | track | backfill | migrate`.
- **`roigbiv/registry/backfill.py`** — `run_backfill(root, …)` walks `data/output/*/` dirs that contain `merged_masks.tif + mean_M.npy + centroids.npy + roi_metadata.json`, sorts by `(session_date, stem)`, calls `register_or_match` per session.

### Per-session pipeline output contract

Inspected via `data/output/T1_221215_PrL-NAc-G6-5M_LOW-D1_FOV1_PRE-000/`:

- **`merged_masks.tif`** — uint16 `(H, W)` label image. Pixel value = ROI `label_id`; 0 = background. Written by `roigbiv/pipeline/hitl.py::export_hitl_package`.
- **`roi_metadata.json`** — array of objects with keys `label_id`, `area`, `solidity`, `eccentricity`, `nuclear_shadow_score`, `soma_surround_contrast`, plus pipeline internals.
- **`summary/mean_M.tif`** — float mean projection.
- **`pipeline_log.json`**, F/Fneu/dFF/spks numpy stacks, HITL dirs, `svd_factors.npz`, `registry_match.json` (if registry was invoked).

Note: `mean_M.npy` and `centroids.npy` don't exist as files in `data/output/`; they're constructed in-memory by the pipeline and written as blobs under `inference/fingerprints/{fov_id}/…` only when a FOV is minted.

### Registry DB state right now

- **`inference/registry.db`**: 2 FOV rows, 2 sessions, 151 cells, 151 observations. These are from the earlier run where the pairwise matcher failed — sessions 1 and 3 minted separate FOVs; session 2 went to review and was never written. Ground truth is a single FOV.
- **`inference/fingerprints/{19cb4c6e…, a0a3baa8…}/`** — each dir contains `mean_M.npy`, `centroids.npy`, `fov_embedding.npy`, `roi_embeddings.npy`.
- **No labeled cross-session pairs** anywhere. Logistic will retain hand-prior form until labels are collected.

### Three-session test dataset

- Raw inputs: `data/raw/T1_{221209_HI-D1, 221215_LOW-D1, 230116_EXT-D9}_…_mc.tif`.
- Processed outputs: `data/output/T1_{221209, 221215, 230116}_…/` — all three have `merged_masks.tif`, `roi_metadata.json`, `summary/mean_M.tif`. Everything the new registry needs, with no re-run of the detection pipeline required.

### Missing spec documents

`/home/thejoshbq/Otis-Lab/Projects/roigbiv/CLAUDE.md` does not exist. There is an `Otis-Lab/CLAUDE.md` at workspace root. The "§17.1 unified mask" contract referenced in the original prompt is not in a file I can cite; the contract is de facto encoded in `roigbiv/pipeline/hitl.py::export_hitl_package` (uint16 label image, one value per ROI).

---

## ROICaT API reference (version 1.5.5, installed)

Inspected via `inspect.signature` + source reads, not from memory. Install path: `/home/thejoshbq/miniconda3/envs/roigbiv/lib/python3.10/site-packages/roicat/`.

### Top-level wrapper

- `roicat.pipelines.pipeline_tracking(...)` exists at `pipelines.py`. It assumes Suite2p-shaped input and drives the full tracking workflow. We will **not** call it directly (the roigbiv pipeline already owns data loading and should not be round-tripped through ROICaT's Suite2p loader), but its source is the canonical reference for the call sequence.

### Aligner — `roicat.tracking.alignment.Aligner`

```python
__init__(use_match_search=True, all_to_all=False, radius_in=4, radius_out=20,
         order=5, z_threshold=4.0, um_per_pixel=1.0, device='cpu', verbose=True)

augment_FOV_images(FOV_images, spatialFootprints=None,
                   normalize_FOV_intensities=True, roi_FOV_mixing_factor=0.5,
                   use_CLAHE=True, CLAHE_grid_block_size=10, CLAHE_clipLimit=1,
                   CLAHE_normalize=True)

fit_geometric(template, ims_moving, template_method='sequential',
              mask_borders=(0,0,0,0), method='RoMa',
              kwargs_method={...}, constraint='affine',
              kwargs_RANSAC={...}, verbose=None) -> ndarray

fit_nonrigid(template, ims_moving, remappingIdx_init=None,
             template_method='sequential', method='RoMa',
             kwargs_method={...}) -> ndarray

transform_ROIs(ROIs, remappingIdx=None, normalize=True) -> List[ndarray]
```

### ROI_Blurrer — `roicat.tracking.blurring.ROI_Blurrer`

Small Gaussian blur on sharp binary footprints before similarity computation. Recommended for Cellpose/Suite2p-crisp masks (our case).

### Similarity graph — `roicat.tracking.similarity_graph.ROI_graph`

Produces sparse similarity matrices `s_sf`, `s_NN_z`, `s_SWT_z`, `s_sesh` that the Clusterer ingests:

```python
compute_similarity_blockwise(spatialFootprints, features_NN, features_SWT, ROI_session_bool)
make_normalized_similarities(centers_of_mass, features_NN, features_SWT, ...)
```

### Clusterer — `roicat.tracking.clustering.Clusterer`

```python
__init__(s_sf=None, s_NN_z=None, s_SWT_z=None, s_sesh=None,
         n_bins=None, smoothing_window_bins=None, verbose=True)

make_pruned_similarity_graphs(convert_to_probability=False, stringency=1.0,
                              kwargs_makeConjunctiveDistanceMatrix=None, d_cutoff=None)

fit(d_conj, session_bool, min_cluster_size=2, cluster_selection_method='leaf',
    alpha=0.999, split_intraSession_clusters=True, ...) -> ndarray  # HDBSCAN

fit_sequentialHungarian(d_conj, session_bool, thresh_cost=0.95) -> ndarray

compute_quality_metrics(sim_mat=None, dist_mat=None, labels=None) -> dict
```

- `labels` vector length = total ROIs across all sessions; `-1` = unclustered.
- Also exposes `self.labels` and `self.violations_labels` (clusters with multiple ROIs from the same session).
- **Choice of fit method**: ROICaT's own guidance is HDBSCAN for ≥8 sessions, `fit_sequentialHungarian` for <8. Our workflow typically has 2–8 sessions, so `fit_sequentialHungarian` is the default.

### ROInet embedder — `roicat.ROInet.ROInet_embedder`

Already wrapped in `embedder.py` today; will move into the adapter. `__init__(dir_networkFiles, device='cpu', download_method='check_local_first', …)`.

### Input shape contract

This is what dictates how we transform `merged_masks.tif`:

- `spatialFootprints`: `List[scipy.sparse.csr_matrix]`, one per session, shape `(n_rois_session, H*W)`, binary or float32.
- `FOV_images`: `List[ndarray (H, W)]`, one per session (mean projections).
- `session_bool` / `ROI_session_bool`: `ndarray (n_rois_total, n_sessions) bool`.

---

## Proposed architecture

**Central architectural choice**: ROICaT is N-way (takes all sessions at once, returns one label vector). The registry today is pairwise-incremental. We resolve the mismatch by changing the orchestration semantics, not by hacking ROICaT into pairwise calls.

```
                  ┌───────────────────────────────────────────────────────┐
                  │  pipeline/run.py  (new session just finished)         │
                  │  → _register_fov_after_pipeline(output_dir)           │
                  └──────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
         ┌───────────────────────────────────────────────────────────────┐
         │  orchestrator.register_or_match (rewritten)                    │
         │  1. parse filename metadata                                    │
         │  2. compute footprint fingerprint for QUERY session            │
         │  3. hash pre-filter (unchanged: exact re-run shortcut)         │
         │  4. scope candidates by (animal_id, region)                    │
         │  5. build N-session bundle (candidate FOV sessions + query)    │
         │  6. call roicat_adapter.cluster_sessions(bundle)               │
         │  7. derive FOV-level evidence vector from cluster output       │
         │  8. calibrated logistic → decision (accept / review / reject)  │
         │  9. write session + observations keyed to cluster labels       │
         └───────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │  roicat_adapter.py  (NEW; sole ROICaT import site)                 │
    │                                                                   │
    │  load_session_bundle(output_dirs) ─────────────────────────┐      │
    │     for each dir: read merged_masks.tif + summary/mean_M  │      │
    │     → per-session csr_matrix footprints + FOV image       │      │
    │                                                            ▼     │
    │  align_sessions(bundle) ──────────────► Aligner.fit_geometric     │
    │                                          (+ transform_ROIs)       │
    │                                                            │     │
    │                                                            ▼     │
    │  embed_footprints(aligned_bundle) ────► ROInet + SWT              │
    │                                                            │     │
    │                                                            ▼     │
    │  build_similarities(...) ─────────────► ROI_graph.compute_…       │
    │                                                            │     │
    │                                                            ▼     │
    │  cluster(bundle) ─────────────────────► Clusterer.fit_seq…        │
    │                                                            │     │
    │                                                            ▼     │
    │  returns: ClusterResult {                                         │
    │    labels: ndarray (n_rois_total,),                               │
    │    session_bool: ndarray (n_rois_total, n_sessions),              │
    │    per_session_roi_index: List[ndarray],                          │
    │    quality_metrics: dict,                                         │
    │    alignment: {inlier_rate, remappingIdx, …}                     │
    │  }                                                                │
    └───────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                    registry DB writes  +  registry_match.json
```

### File fates

| File | Fate |
|---|---|
| `roigbiv/registry/roicat_adapter.py` | **NEW** (Phase 2). Sole import site for `roicat`. |
| `roigbiv/registry/fingerprint.py` | **REWRITTEN** (Phase 3). Fingerprint hash now over footprint-derived content (sorted label ids + per-ROI centroids + mask-area stats); mean-projection becomes optional context blob. |
| `roigbiv/registry/match.py` | **REWRITTEN** (Phase 3). Collapses to a thin shim: call adapter, derive FOV evidence vector, run calibrated logistic, return FOVMatchResult. RANSAC/ECC/Hungarian code deleted. |
| `roigbiv/registry/embedder.py` | **DELETED** (Phase 3). ROInet usage moves into the adapter; nothing else imports the wrapper. |
| `roigbiv/registry/calibration.py` | **UPDATED** (Phase 3). `FOVLogisticCoefs` feature set changes (see below). `CellLogisticCoefs` deleted — cell-level posterior is now ROICaT's job. |
| `roigbiv/registry/orchestrator.py` | **REWRITTEN** (Phase 3). Re-shaped around the new N-way adapter flow; keeps hash pre-filter + three-way decision + JSON report contract. |
| `roigbiv/registry/store/sqlalchemy_store.py` | **MINOR UPDATE**. See schema change below. |
| `roigbiv/registry/models.py` | **MINOR UPDATE**. Same. |
| `roigbiv/registry/migrations/versions/0003_roicat.py` | **NEW**. Migration described below. |
| `roigbiv/registry/backfill.py` | **REWRITTEN** (Phase 3). New `run_backfill` performs one clustering pass per `(animal_id, region)` group rather than per-session pairwise matches. |
| `roigbiv/registry/filename.py` | Unchanged. |
| `roigbiv/registry/config.py` | **MINOR UPDATE**. Drop embedder builder; add adapter config knobs (`ROIGBIV_ROICAT_DEVICE`, `ROIGBIV_ROICAT_UM_PER_PIXEL`, `ROIGBIV_ROICAT_ALL_TO_ALL`). |
| `roigbiv/pipeline/run.py` | **UPDATED**. Pass `output_dir` path to the new orchestrator signature. |
| `roigbiv/cli_registry.py` | **UPDATED**. `match` / `track` subcommands align with new orchestrator. |
| `app.py` | **UPDATED** only if `FOVMatchResult` / `registry_match.json` schema shifts; otherwise unchanged. |
| Tests under `roigbiv/registry/tests/` | `test_align.py`, `test_match.py`, `test_match_probabilistic.py`, `test_embedder.py` **deleted or rewritten**. `test_fingerprint.py`, `test_calibration.py`, `test_orchestrator.py`, `test_registry_cross_day.py` **rewritten** against new semantics. `test_filename.py`, `test_blob.py`, `test_store.py` unchanged. New `test_roicat_adapter.py` in Phase 2. |

---

## Feature mapping for the calibrated logistic

Current FOV features: `(log_odds_sum, inlier_rate, log1p(n_matched))`. Replacements, derived directly from `ClusterResult`:

- **`n_shared_clusters`** → count of clusters that contain at least one ROI from the query session AND at least one ROI from any candidate-FOV session. Replaces `n_matched`.
- **`fraction_query_clustered`** → `n_query_in_shared_clusters / n_query_rois`. Scale-invariant; robust to query FOVs with many extra new cells.
- **`alignment_quality`** → `Aligner`'s RANSAC inlier rate (post-fit). Direct ROICaT-native substitute for the old `inlier_rate`.
- **`mean_cluster_cohesion`** → mean of `1 - d_conj` over shared clusters (equivalently, mean within-cluster cosine-like similarity). Replaces `log_odds_sum` — expresses "how tight are matched cells in feature space" without needing a separate cell-level logistic.

**New FOV feature vector**: `(n_shared_clusters, fraction_query_clustered, alignment_quality, mean_cluster_cohesion)`.

**Coefficients** (hand-priors, no labeled data exists):

```
intercept                       = -4.0
coef_n_shared_clusters          =  0.05
coef_fraction_query_clustered   =  3.0
coef_alignment_quality          =  4.0
coef_mean_cluster_cohesion      =  3.0
```

Target behavior: `posterior ≈ 0.9` when ~60% of query ROIs cluster with a candidate session AND alignment inlier rate ≥ 0.5 AND mean cohesion ≥ 0.5. These will be sanity-checked on the three-session dataset in Phase 4 and tuned in one pass if they are clearly mis-calibrated; they are explicitly marked as priors in code comments until a labeled set exists.

**Decision banding is preserved**: `accept ≥ 0.9`, `review ≥ 0.5`, else `reject`. Env overrides via `ROIGBIV_FOV_ACCEPT_THRESHOLD` / `ROIGBIV_FOV_REVIEW_THRESHOLD` unchanged.

---

## DB schema changes

Additive migration `0003_roicat.py`:

- **`fov.fingerprint_version`** — new allowed value `3` (meaning "footprint-derived fingerprint + ROICaT-clusterable"). Default remains `1`; existing rows unchanged.
- **`session.cluster_labels_uri`** — new nullable `VARCHAR(512)`. Blob is a numpy `(n_rois_session,)` int32 cluster-label array from the most recent clustering pass. `-1` = unclustered.
- **`cell_observation.cluster_label`** — new nullable `INTEGER`. Redundant with the blob but indexed for "show me all sessions containing this cluster" queries.
- No columns dropped. **Non-destructive**; old FOVs remain queryable.

Per the user's constraint, existing `inference/registry.db` is left untouched. Phase 4 validation uses a fresh `inference/registry_roicat.db`. Callers that still point at the legacy DB continue to work against the pairwise data until we explicitly switch.

---

## Backward compatibility

- **v1 / v2 fingerprints already in DB**: left in place and readable. The new orchestrator **does not attempt to match a v3 query against a v1/v2 candidate** (they have no footprint blobs). A v1/v2 FOV in the candidate set is silently skipped with a log warning, and the query either matches another v3 candidate or mints a new v3 FOV.
- **`fingerprint.py` public helpers** (`deserialize_mean_m`, `deserialize_cells`, `deserialize_embeddings`): preserved as dead-code shims for one release cycle so Streamlit's legacy FOV rendering keeps working. Annotated with a `# TODO(roicat): remove after ≥0003 DB migration adoption` comment.
- **`match.py` constants** (`AUTO_ACCEPT_THRESHOLD`, `REVIEW_THRESHOLD`): preserved at module scope (still read via `config.py` / env vars) even though the algorithm underneath is replaced.
- **Streamlit `_registry_tab`**: no schema-breaking change. New fields in `registry_match.json` (`cluster_labels_uri`, `n_shared_clusters`, …) are purely additive.

---

## Open questions

These need user resolution before Phase 2 implementation begins.

1. **Candidate scope for N-way clustering.** When a new session arrives, do we cluster the query against (a) *all* sessions of *all* `(animal, region)` FOVs, or (b) only the single best-ranked candidate FOV's sessions? Trade-off: (a) is slower but gives ROICaT more signal, (b) is closer to current semantics and keeps cluster labels stable for unrelated FOVs. **My default proposal**: (b) — best-ranked candidate FOV only — for the real-time pipeline path, and a separate "re-cluster everything for (animal, region)" admin action in the Streamlit maintenance panel.

2. **Global cell ID reassignment semantics.** If a later clustering pass splits or merges previously-registered clusters, do we:
   - (a) let existing `global_cell_id` values drift (rewrite `cell_observation.global_cell_id` rows);
   - (b) keep old IDs immutable and emit a separate `cluster_label` channel that may diverge from `global_cell_id`;
   - (c) hard-freeze IDs at first-insert and flag divergence for manual review?

   **My default proposal**: (b). Option (a) breaks any external references to `global_cell_id` that might live in axplorer / pynapse / plotting code; (c) creates silent drift. The `cell_observation.cluster_label` column in the schema diff encodes (b).

3. **Three-session migration path.** The current `inference/registry.db` has 2 stale FOV rows from the broken pairwise run. For Phase 4 validation, should I:
   - (a) ignore it entirely and write to `inference/registry_roicat.db`;
   - (b) write a one-shot script to drop the stale rows from the legacy DB and re-ingest the three sessions?

   **My default proposal**: (a). User's constraint is "Do not mutate the existing inference/registry.db in place."

4. **ROInet weights cache location.** Current embedder uses `~/.cache/roigbiv/roinet`. Confirm OK to reuse for the adapter.

5. **`all_to_all` on Aligner.** For 2–3 sessions, `all_to_all=False` with `template_method='sequential'` is the reasonable default; `all_to_all=True` is more accurate at O(n²) cost. **My default proposal**: `all_to_all=False`, exposed via `ROIGBIV_ROICAT_ALL_TO_ALL` env.

6. **Dropping the cell-level calibration logistic.** Phase 1 design removes `CellLogisticCoefs` + `p_same_cell` / `log_odds_cell` because ROICaT produces per-pair similarities internally. Grep across this repo finds no external consumers. **Confirm** there's no out-of-tree consumer (axplorer? pynapse?) before Phase 3.

7. **Non-rigid alignment.** `Aligner.fit_nonrigid` is available and typically improves cross-session registration at the cost of ~2–5× runtime. **My default proposal**: use only `fit_geometric` for v1 of the integration; expose `fit_nonrigid` as an opt-in env flag (`ROIGBIV_ROICAT_NONRIGID=1`) for later experimentation.

---

## Test plan (Phase 4 acceptance criterion)

1. Create fresh `inference/registry_roicat.db` — do not touch the legacy DB.
2. Ingest `T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002` first. Expected: mint a new FOV. Logged metrics: ROInet embedding dimensions, n_rois, `fingerprint_hash`.
3. Ingest `T1_221215_PrL-NAc-G6-5M_LOW-D1_FOV1_PRE-000` second. The clustering pass sees sessions 1 and 2. **Assertions**:
   - ≥ 40% of session-2 ROIs join clusters that contain a session-1 ROI.
   - `alignment_quality ≥ 0.4`.
   - `mean_cluster_cohesion ≥ 0.5`.
   - Final `p_same_fov ≥ accept_threshold` (0.9 default) → `decision = auto_match`.
   - Logged: `n_shared_clusters`, `fraction_query_clustered`, `alignment_quality`, `mean_cluster_cohesion`, `p_same_fov`, `decision`.
4. Ingest `T1_230116_PrL-NAc-G6-5M_EXT-D9_FOV1_EXT-D9_PRE-000` third. Same assertions against the now-2-session FOV.
5. **Final DB state**: exactly 1 FOV row, 3 session rows, cells keyed to stable `global_cell_id`s, observations carrying `cluster_label`.
6. On failure (N ≠ 1 FOV), report stage-level diagnostics:
   - **Alignment**: per-pair inlier rate + warped-image Pearson correlation.
   - **Embedding**: cosine-similarity distribution for cells in shared clusters vs. unshared.
   - **Clustering**: cluster size histogram + `Clusterer.violations_labels`.
   - **Logistic**: feature values at each session ingest.
7. Write `docs/validation/three-session-test.md` with logged values + pass/fail verdict. **On fail: stop. Do not retune parameters in the same pass** — surface the failing stage to the user for their next decision.

---

## Deliverables

- **Phase 1**: this document. Stop for approval. ← **CURRENT**
- **Phase 2**: `roigbiv/registry/roicat_adapter.py` + `roigbiv/registry/tests/test_roicat_adapter.py` (synthetic-footprint unit tests). Stop for approval.
- **Phase 3**: Rewritten `fingerprint.py`, `match.py`, `orchestrator.py`, `backfill.py`, `calibration.py`; updated `store/sqlalchemy_store.py`, `models.py`, `config.py`; new migration `0003_roicat.py` (NOT auto-run); caller updates in `pipeline/run.py`, `cli_registry.py`, and `app.py` only if the report schema shifts. Stop before validation.
- **Phase 4**: `inference/registry_roicat.db` populated on the three-session dataset + `docs/validation/three-session-test.md`.

---

## Post-Phase-4 addendum — ROI mixing factor experiment (rejected)

**Date added**: 2026-04-22

### Hypothesis (rejected)

The Phase-4 validation failed with `PhaseCorrelation` + `roi_FOV_mixing_factor=0.5` (ROICaT's default). Visual inspection of `merged_masks.tif` across the three T1 PrL-NAc sessions showed that the ROI footprint layout is *stable* across days while the mean projection is *session-variable* (tonic/sparse activity). The hypothesis was: raising `roi_FOV_mixing_factor` to `0.9` would let the stable signal dominate `Aligner.augment_FOV_images`, improve phase-correlation alignment, and cascade into higher `fraction_query_clustered` + `mean_cluster_cohesion` + `fov_posterior`.

### Result (rejected)

The hypothesis is false. Raising the mixing factor from 0.5 → 0.9 *regressed* alignment across the board on the three-session dataset:

| Metric | 0.5 (prev) | 0.9 (this run) |
|---|---|---|
| Step 1 `alignment_inlier_rate` | 0.214 | 0.090 |
| Step 2 `alignment_inlier_rate` | 0.443 | 0.078 |
| Step 1 `fov_posterior` | 0.330 | 0.230 |
| Step 2 `fov_posterior` | 0.791 | 0.268 |
| Verdict | 2 FOVs | **3 FOVs** |

### Interpretation

Stability and distinctiveness are not the same thing. The mean projection *is* session-variable, but its spatial content (bright cells, dim cells, specific intensity patterns) gives phase correlation something to lock onto. ROI footprint density on this dataset is near-uniform — ~70–85 blob-like regions distributed fairly evenly across the frame — which makes it a poor signal for translation estimation via frequency-domain cross-correlation. The intuition was backwards.

### Action taken

`AdapterConfig.roi_mixing_factor` default reverted to `0.5` (ROICaT's own default). The env var `ROIGBIV_ROICAT_ROI_MIXING` remains exposed so the knob is available for future experimentation, but is no longer the recommended remediation lever. The next remediation to try should change the *algorithm* rather than the *signal* — `RoMa` (deep-feature matcher) is the most promising candidate.

### Next hypothesis to test

Switch `alignment_method` from `PhaseCorrelation` → `RoMa` via `ROIGBIV_ROICAT_ALIGNMENT=RoMa`. RoMa is ROICaT's own default aligner; it was not tested in Phase 4 only because of the ~500 MB first-use model download. Unlike phase correlation it does not rely on global Fourier structure — it matches learned dense features that tolerate large photometric variation. Expected to lift `alignment_inlier_rate` on step 1 (currently 0.21) most.

---

## Post-Phase-4 addendum — RoMa alignment (accepted)

**Date added**: 2026-04-22

### Result

`ROIGBIV_ROICAT_ALIGNMENT=RoMa` + `ROIGBIV_ROICAT_DEVICE=cuda` passes the three-session validation with both cross-session steps auto-matching:

| Step | Method | Posterior | Inlier | Shared | Clustered |
|---|---|---|---|---|---|
| 1 (S2 vs S1) | PhaseCorrelation | 0.330 | 0.21 | 12 | 0.16 |
| 1 (S2 vs S1) | **RoMa** | **0.993** | **0.76** | **57** | **0.76** |
| 2 (S3 vs 1+2) | PhaseCorrelation | 0.791 | 0.44 | 27 | 0.40 |
| 2 (S3 vs 1+2) | **RoMa** | **0.978** | **0.63** | **46** | **0.68** |

Final DB: **1 FOV, 3 sessions, 123 cells**. Match runtime ~50 s on CUDA.

### Infra change required

ROICaT 1.5.5's `RoMa` wrapper (`roicat.tracking.alignment.RoMa.__init__`) does not forward a `use_custom_corr` kwarg to `romatch.roma_outdoor`. That matters when `romatch`'s compiled CUDA kernel (`local_corr`) is not installed — which is the default on any pip install without the extension build. Without the kernel, `roma_outdoor(use_custom_corr=True)` (romatch's own default) crashes on the first inference with `ModuleNotFoundError: No module named 'local_corr'`.

Fix: `roigbiv/registry/roicat_adapter.py::_maybe_patch_romatch_for_native_fallback()`. Detects the missing extension once per process, then monkey-patches `romatch.roma_{outdoor,indoor,tiny_v1_outdoor}` to force `use_custom_corr=False`. Emits a `log.warning`. When the extension is present, the patch is a no-op. This is an isolated, reversible workaround for a missing upstream passthrough — nothing in `romatch` or `roicat` is modified on disk.

### Defaults flipped (Run 4)

After Run 3 confirmed RoMa's correctness win, defaults were flipped:

- `AdapterConfig.alignment_method` default: `"PhaseCorrelation"` → `"RoMa"`.
- `AdapterConfig.device` default: hard-coded `"cpu"` → `field(default_factory=_auto_device)`, which returns `"cuda"` when `torch.cuda.is_available()` and falls back to `"cpu"` otherwise.
- Same flips applied to `RegistryConfig` + `from_env` (the env parser treats `ROIGBIV_ROICAT_DEVICE` as unset → auto-detect, rather than defaulting to `"cpu"`).

Run 4 repeated the three-session validation with zero env overrides and passed identically (posteriors 0.993 / 0.987). The defaults are now the validated state.

Known trade-off accepted by the flip:

- On CPU-only hosts, the first registry call triggers a ~1.5 GB RoMa/DINOv2 weight download and subsequent matches are minutes-per-session. This is considered acceptable because (a) the previous default silently produced wrong answers on real data, and (b) CPU-only deployments can opt back to `ROIGBIV_ROICAT_ALIGNMENT=PhaseCorrelation` explicitly.

### Non-changes

Mixing factor (still 0.5), nonrigid (off), all_to_all (off), accept/review thresholds (0.9 / 0.5), logistic coefficients (hand-priors). Single-hypothesis pass honoured.
