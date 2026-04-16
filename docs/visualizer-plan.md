# ROI G. Biv — Real-Time Pipeline Visualizer

## Context

The ROIGBIV pipeline (4-stage subtractive ROI detection on 2P calcium imaging stacks) currently emits progress only as stdout text captured by `app.py`'s Streamlit UI. There is no structured way to see *what* the pipeline is finding as it works — masks land in `roi_metadata.json` only at the end of a ~15-minute run.

We are building a browser-based visualizer that:
1. Watches the pipeline "paint" detected ROIs onto a persistent FOV canvas, stage-by-stage, with per-feature gate animations.
2. Records the full run as an append-only NDJSON event log that can be replayed at variable speed.
3. Adds zero overhead when disabled and <5% overhead when enabled.

The pipeline's scientific outputs must remain byte-identical with the visualizer on or off — instrumentation is purely additive.

**Build cadence:** phase-by-phase. After each phase I write `visualizer/BUILD_LOG.md` and stop for review.

---

## Discovery findings (Phase 0 — already mapped)

- **Orchestrator exists**: `roigbiv/pipeline/run.py:run_pipeline(tif_path, cfg, gpu_lock=None) → FOVData` (lines 169–~550). Calls Foundation → Stage 1+Gate 1+Subtraction → Stage 2+Gate 2+Subtraction → Stage 3+Gate 3+Subtraction → Stage 4+Gate 4 → unified QC/classification/HITL → save outputs.
- **Stage modules**: `foundation.py`, `stage1.py`, `gate1.py`, `stage2.py`, `gate2.py`, `stage3.py`, `gate3.py`, `stage4.py`, `gate4.py`, `subtraction.py`, `outputs.py` — all clean entry points, no ad-hoc structure.
- **No web framework installed**: no Flask/FastAPI/aiohttp/websockets. Streamlit + tornado are present (Streamlit's blocking handlers are unsuitable for low-latency event push).
- **Existing structured logs**: `pipeline_log.json` (run summary, written at end), `roi_metadata.json` (full per-ROI features, written at end), per-stage `stage{N}_report.json`.
- **Typical FOV**: H=505, W=493, T=8624 frames, fs=7.5 Hz, neuron diameter ≈ 12 px (test case).
- **Streamlit `app.py`** stays untouched — it remains the launcher for users who don't want the browser viewer. The new visualizer is a parallel option.

---

## Directory layout (all new code)

```
~/Otis-Lab/Projects/roigbiv/visualizer/
├── DISCOVERY.md            # Phase 0 deliverable — persist findings above
├── BUILD_LOG.md            # Per-phase status appendix
├── __init__.py
├── events.py               # Phase 1: singleton emitter + helpers
├── server.py               # Phase 2: websockets + aiohttp relay
├── test_synthetic.py       # Phase 5: T3 synthetic event generator
├── tests/
│   ├── test_emitter.py     # T2 event log completeness
│   └── test_identity.py    # T1 byte-identity check
├── frontend/               # Phase 3: Vite + React source
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── lib/
│       │   ├── ws.js               # auto-reconnect + buffered queue
│       │   └── canvasLayers.js     # offscreen canvas helpers
│       └── components/
│           ├── FOVCanvas.jsx       # layered canvas (5 layers)
│           ├── StageProgress.jsx   # right rail
│           ├── CurrentROI.jsx      # gate-check pulse panel
│           ├── Timeline.jsx        # bottom bar + scrubber
│           └── TransportControls.jsx
└── static/                 # Vite build output, served by server.py
```

Files modified outside `visualizer/` (Phase 4 only — additive emit calls, no logic changes): `pyproject.toml`, `roigbiv/pipeline/{run,foundation,stage1,gate1,stage2,gate2,stage3,gate3,stage4,gate4,subtraction}.py`.

---

## Phase 0 — Discovery (deliverable only)

Write `visualizer/DISCOVERY.md` capturing the orchestrator/stage map above, including: line numbers for each `run_*` and `evaluate_gate*` call site in `run.py`, the FOVData fields available at each stage boundary (`fov.mean_S`, `fov.vcorr_S`, `fov.shape`, `fov.rois`), and a table of injection points → event types.

---

## Phase 1 — Event System (`visualizer/events.py`)

**`PipelineEventEmitter` singleton.** Module-global instance accessed via `from visualizer.events import emitter`.

State:
- `enabled: bool` — set from `ROIGBIV_VIS_ENABLED` env var or `cfg.visualizer_enabled` at `init_run()` time. Once False, every `emit*` exits on a single attribute check.
- `t0: float` — `time.monotonic()` at `pipeline_start`.
- `log_path: Path` — `{fov_output_dir}/pipeline_events.ndjson`, opened with `buffering=1` (line-buffered) and `os.fsync` after each write to satisfy C5 (valid up to crash).
- `queue: asyncio.Queue(maxsize=1000)` — shared with the WebSocket server via the file-tail mechanism (queue itself is only used internally; the file is the IPC). Drop-oldest on overflow.
- `executor: ThreadPoolExecutor(max_workers=2)` — runs PNG encoding off the pipeline thread.

API:
- `init_run(fov_output_dir, fov_id, tif_path, dimensions, parameters, base_image)` — opens log file, emits `pipeline_start`.
- `emit(event_type, payload, stage=None)` — common path. Builds the event envelope, appends one JSON line, fsyncs.
- `emit_roi_candidate(stage, roi_id, mask_label, label_image, features)` — extracts contour via `skimage.measure.find_contours(label_image == mask_label, 0.5)` then `approximate_polygon(tol=1.0)` capped at 100 points, computes centroid & area, emits.
- `emit_roi_verdict(roi_id, verdict, **kwargs)` — emits accepted/flagged/rejected.
- `emit_image(event_type, image_array, **extra)` — submits encoding to executor; the worker emits when the future resolves.
- `stage(stage_id, name)` — context manager. Counts `roi_accepted`/`flagged`/`rejected` events emitted within scope and emits `stage_complete` on exit. Tracks counts via a per-stage tally dict reset on entry.
- `transient_batcher` — internal coalescer for Stage 3. Buffers `transient_event`s, flushes as `transient_event_batch` when `len(buf) >= 16` or `time.monotonic() - last_flush > 0.05` (≤ 60/s rate).

**Encoding helper.** `_encode_png(arr, max_dim=512)` — normalizes float arrays to uint8, downsamples if longest edge > 512, returns base64 string. Used by `emit_image`.

**Zero-cost guard.** Every callsite uses the pattern `if emitter.enabled: emitter.emit_*(...)`. The emitter's own methods *also* check `enabled` as a safety net, but the call-site guard prevents argument evaluation when disabled.

Adds to `pyproject.toml`: no new runtime deps for Phase 1 (uses stdlib + skimage which is already present).

---

## Phase 2 — WebSocket Server (`visualizer/server.py`)

**Stack:** `websockets` (WS endpoint) + `aiohttp` (static file server). No Flask/FastAPI per C1.

Components:
- `EventTailer` — async task that polls `pipeline_events.ndjson` every 100 ms, reads new lines, broadcasts JSON to all connected `/ws` clients. Tracks file position per file; handles file rotation (new run = new file in a new fov dir) by watching the configured output root.
- `LiveBroadcaster` — set of `websockets` connections. `broadcast(msg)` does `asyncio.gather(*[ws.send(msg) for ws in clients], return_exceptions=True)`; drops disconnected clients.
- `ReplaySession` — per-client task on `/ws/replay?file=<path>`. Reads NDJSON, emits events respecting `elapsed_s` timestamps modulated by `speed`. Handles control messages: `set_speed` (including 0.0 = pause), `seek`, `step`.
- `aiohttp` static handler — serves `visualizer/static/` at `/`. In dev mode, proxies to `http://localhost:5173` (Vite dev server) when `--dev` flag is set.

Launch:
```
python -m visualizer.server [--port 9876] [--watch <output_root>] [--replay <ndjson>] [--dev]
```

Default port `9876`, override via `ROIGBIV_VIS_PORT`.

Adds to `pyproject.toml` (optional `[project.optional-dependencies] viewer`): `websockets>=12`, `aiohttp>=3.9`.

---

## Phase 3 — React Frontend (`visualizer/frontend/`)

**Stack:** React 18 + Vite. No state library (useReducer is enough). No CDN deps (C1, F5). Bundle target <500 KB gzipped (F6).

Key implementation choices:
- **Layered canvas (`FOVCanvas.jsx`)**: 5 stacked `<canvas>` elements absolutely positioned, sharing a transform (zoom/pan) applied by CSS `transform`. Layer 1 (accepted ROIs) is an offscreen canvas drawn once per accept, then blitted onto the visible layer per frame — avoids re-stroking 100+ paths per `requestAnimationFrame` (F3).
- **Event reducer**: single `dispatch` that updates a normalized state shape: `{ baseImage, stages: {1..4: {accepted: [], flagged: [], rejected: []}}, currentRoi, transients: [], heatmaps: {}, timeline: {...} }`. Ephemeral state (transient flashes, current ROI pulse) lives in refs to avoid React re-renders during animation.
- **WebSocket client (`lib/ws.js`)**: auto-reconnect with backoff (1s → 2s → 4s → 8s → 16s → 30s cap), bounded inbound buffer (100 events; drop-oldest if rendering falls behind per F2).
- **Stage colors (per spec)**: `#00ff88`, `#4488ff`, `#ff8844`, `#ff44aa`. Flagged = dashed stroke. Rejected = red flash + 500 ms fade (CSS keyframe on a temporary canvas overlay).
- **Gate-check animation**: each `roi_gate_check` event appends a row in `CurrentROI` panel and triggers a brief brightness pulse on the candidate contour. Implemented as a 200 ms `globalAlpha` pulse on layer 2.
- **Theme**: dark `#0a0a0f` background, monospace (JetBrains Mono via system fallback `ui-monospace`), bright overlays. No rounded corners. Sharp scientific aesthetic per spec.
- **Transport**: live mode hides playback controls; replay mode reveals scrubber/play/pause/step/speed. Mode switch via header dropdown with file picker.

Build: `npm run build` writes to `../static/`. Served by `server.py` aiohttp handler. Dev: `npm run dev` (Vite on :5173) + `python -m visualizer.server --dev`.

---

## Phase 4 — Pipeline Integration (additive `emit()` calls only)

All call sites use `if emitter.enabled: emitter.emit_*(...)`. No logic moves, no signature changes, no refactors.

| File | Location | Event(s) emitted |
|---|---|---|
| `roigbiv/pipeline/run.py` | top of `run_pipeline` (line ~213) | `init_run` + `pipeline_start` (after foundation completes so we have `mean_S` for base image) |
| `run.py` | around `run_foundation` call (line 215) | `foundation_start` per sub-step (passed via callback or wrap), `foundation_complete` after |
| `run.py` | wrap each stage (lines 234, 324, 389, 460) | `with emitter.stage(N, name):` context manager |
| `run.py` | after `run_source_subtraction` (lines 291, 364, 446) | `subtraction_start` + `subtraction_complete` with new mean residual image |
| `run.py` | end of `run_pipeline` | `pipeline_complete` |
| `roigbiv/pipeline/foundation.py` | end of motion correction, SVD, L+S, summary images | `foundation_start` per sub-step |
| `stage1.py` | after Cellpose returns label image, loop labels | `roi_candidate` per mask |
| `gate1.py` | inside `evaluate_gate1` per feature threshold check (area, solidity, eccentricity, contrast, nuclear shadow) | `roi_gate_check` per feature, then `roi_accepted`/`flagged`/`rejected` per ROI |
| `stage2.py` | after Suite2p candidates filtered by IoU | `roi_candidate` per kept candidate |
| `gate2.py` | per-feature threshold (correlation, morphology) | `roi_gate_check` + verdict |
| `stage3.py` | inside `_process_chunk` per detected pixel event passing spatial coherence | `transient_event` (auto-batched by emitter); `roi_candidate` after clustering produces candidate |
| `gate3.py` | per-feature (waveform R², rise/decay, anticorr) | `roi_gate_check` + verdict |
| `stage4.py` | after each bandpass window's correlation map | `correlation_heatmap` per window; `roi_candidate` per cluster |
| `gate4.py` | per-feature (corr_contrast, motion_corr, intensity, anticorr) | `roi_gate_check` + verdict |
| `subtraction.py` | (none directly — orchestrator emits before/after) | — |

**Stage 3 caveat**: `_process_chunk` runs on GPU. Emitting per-pixel events from inside the GPU loop would force `.cpu()` syncs. Strategy: collect events in the chunk's existing return tuple (no change to loop), then emit from the chunk's caller in `stage3.py`. This is post-GPU and adds no sync overhead.

**Single new param**: `run_pipeline(tif_path, cfg, gpu_lock=None, fov_output_dir=None)` — `fov_output_dir` is already implicit in `cfg.output_dir` resolution; pass to `emitter.init_run`. Both `cli.py` and `app.py` call sites unchanged (defaults preserved).

---

## Phase 5 — Testing & Validation (`visualizer/tests/`)

| Test | Implementation |
|---|---|
| **T1 Pipeline identity** | `test_identity.py`: run pipeline twice on `test_raw/` (once with `ROIGBIV_VIS_ENABLED=0`, once `=1`). `filecmp.cmp` on `F.npy`, `dFF.npy`, `merged_masks.tif`, `roi_metadata.json` (with floating-point tolerance via numpy for npy). Skip `pipeline_events.ndjson`. |
| **T2 Event log completeness** | `test_emitter.py`: parse the NDJSON, assert exactly 1 `pipeline_start`/`pipeline_complete`, balanced stage_start/complete pairs, every `roi_candidate` has a verdict, accepted+flagged sums match `pipeline_complete.total_rois`. |
| **T3 Synthetic stream** | `test_synthetic.py`: generates an NDJSON with ~100 ROIs across 4 stages (circles + squares as contours), gaussian-noise base image with bright blobs, includes transient_event_batch and correlation_heatmap. Run via `python -m visualizer.server --replay synthetic.ndjson` and verify in browser. |
| **T4 Performance** | Time `run_pipeline` on `test_raw/` enabled vs disabled with `time` module. Assert overhead < 5%. If exceeded, profile and reduce image encoding frequency. |
| **T5 Replay fidelity** | Manual: live screenshot at end → replay at 50× → end screenshot. Diff via PIL `ImageChops.difference`; fail if any non-zero pixel in ROI regions. |

---

## Critical files to modify

**New (Phase 0–3, 5):**
- `visualizer/{events.py, server.py, test_synthetic.py, DISCOVERY.md, BUILD_LOG.md, __init__.py}`
- `visualizer/tests/{test_emitter.py, test_identity.py}`
- `visualizer/frontend/**` (full Vite project)

**Modified additively (Phase 4):**
- `pyproject.toml` — add `[project.optional-dependencies] viewer = ["websockets>=12", "aiohttp>=3.9"]`
- `roigbiv/pipeline/run.py` — orchestrator instrumentation (~10 emit calls)
- `roigbiv/pipeline/foundation.py` — 4 sub-step emits + final image emit
- `roigbiv/pipeline/{stage1,stage2,stage3,stage4}.py` — `roi_candidate` emits + Stage 3/4 specific events
- `roigbiv/pipeline/{gate1,gate2,gate3,gate4}.py` — per-feature `roi_gate_check` + verdict emits

**Reuse, do not duplicate:**
- `skimage.measure.find_contours` + `approximate_polygon` for contour extraction (already a dep).
- Mean projection / vcorr arrays already on `FOVData` after `run_foundation` — no recompute.
- Existing `stage{N}_report.json` writing logic — emitter is independent of this; both can run.

---

## Verification (end-to-end)

After Phase 5 completes:

1. **Backend smoke**: `conda activate roigbiv && python -m visualizer.server --replay visualizer/tests/synthetic.ndjson` → open `http://localhost:9876` → confirm ROIs paint stage-by-stage with correct colors and gate animations.
2. **Live run**: in one terminal `python -m visualizer.server --watch test_output/`; in another `ROIGBIV_VIS_ENABLED=1 roigbiv-cli --input test_raw/<tif>` → browser shows real-time progress.
3. **Identity**: `pytest visualizer/tests/test_identity.py` → green.
4. **Log completeness**: `pytest visualizer/tests/test_emitter.py` → green.
5. **Bundle size**: `du -sh visualizer/static/assets/*.gz` (after Vite build with gzip plugin) → confirm <500 KB total.
6. **Pipeline overhead**: compare wall-clock from T4 → confirm <5%.

After each phase, `visualizer/BUILD_LOG.md` gains a section with what landed, deviations, and concerns. I stop after each phase for review.
