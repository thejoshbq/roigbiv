"""Flask routes for the browser-based ROI editor.

Routes
------
GET  /roi-editor/<fov_stem>      — editor HTML page
GET  /api/fov-image/<fov_stem>   — mean projection as PNG
GET  /api/fov-meta/<fov_stem>    — JSON: {n_frames, height, width, fps}
GET  /api/frame/<fov_stem>       — single registered-movie frame as PNG (?n=<int>)
GET  /api/annotations/<fov_stem> — W3C annotation JSON for current ROIs
POST /api/corrections/<fov_stem> — accept edited annotations, write corrections

All routes accept a ``?dir=<base64-encoded-path>`` query parameter that
resolves to the FOV output directory.  The decoded path must contain a
``pipeline_log.json`` to be accepted (prevents arbitrary path traversal).
"""
from __future__ import annotations

import base64
import functools
import io
import threading
from pathlib import Path

import numpy as np
import tifffile
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory

from roigbiv.pipeline.corrections import apply_corrections, load_corrections
from roigbiv.pipeline.loaders import load_fov_from_output_dir
from roigbiv.pipeline.web_reingest import _mask_to_polygon_yx, reingest_from_annotations


# ── Concurrency protection for concurrent browser-tab writes ─────────────────────

_write_locks: dict[str, threading.Lock] = {}
_write_locks_meta: threading.Lock = threading.Lock()


def _get_write_lock(output_dir: Path) -> threading.Lock:
    key = str(output_dir.resolve())
    with _write_locks_meta:
        if key not in _write_locks:
            _write_locks[key] = threading.Lock()
        return _write_locks[key]


# ── FOV metadata cache (shape + dtype only, never the memmap) ────────────────

_fov_meta_cache: dict[str, dict] = {}
_fov_meta_lock: threading.Lock = threading.Lock()

# ── Persistent memmap pool ────────────────────────────────────────────────────
_memmap_pool: dict[str, np.memmap] = {}
_memmap_pool_lock: threading.Lock = threading.Lock()


def _get_memmap(data_bin: Path, T: int, Ly: int, Lx: int) -> np.memmap:
    """Return a cached read-only memmap for *data_bin* (thread-safe for reads)."""
    key = str(data_bin.resolve())
    mm = _memmap_pool.get(key)
    if mm is not None:
        return mm
    with _memmap_pool_lock:
        if key not in _memmap_pool:
            _memmap_pool[key] = np.memmap(
                str(data_bin), dtype=np.int16, mode="r", shape=(T, Ly, Lx)
            )
        return _memmap_pool[key]


# ── LRU PNG byte cache ────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=128)
def _encode_frame_png(fov_key: str, n: int, T: int, Ly: int, Lx: int) -> bytes:
    """Read frame *n* and return PNG bytes. Cached by (fov_key, n, T, Ly, Lx)."""
    from PIL import Image

    data_bin = Path(fov_key) / Path(fov_key).name / "suite2p" / "plane0" / "data.bin"
    mm = _get_memmap(data_bin, T, Ly, Lx)
    frame = np.asarray(mm[n], dtype=np.float32)
    p1, p99 = np.percentile(frame, [1, 99])
    if p99 > p1:
        frame = (frame - p1) / (p99 - p1)
    arr_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tif_dims(tif_path: Path) -> tuple[int, int]:
    arr = tifffile.imread(str(tif_path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.shape  # (H, W)


def _get_fov_meta(output_dir: Path) -> dict | None:
    """Read T, Ly, Lx, fps from ops.npy (or infer from file size). Returns None if data.bin absent."""
    key = str(output_dir.resolve())
    with _fov_meta_lock:
        if key in _fov_meta_cache:
            return _fov_meta_cache[key]

    s2p_plane = output_dir / output_dir.name / "suite2p" / "plane0"
    data_bin  = s2p_plane / "data.bin"
    ops_path  = s2p_plane / "ops.npy"
    if not data_bin.exists():
        return None

    if ops_path.exists():
        ops = np.load(str(ops_path), allow_pickle=True).item()
        T   = int(ops["nframes"])
        Ly  = int(ops["Ly"])
        Lx  = int(ops["Lx"])
        fps = float(ops.get("fs", 0.0))
    else:
        try:
            tif_path = _resolve_projection(output_dir)
            Ly, Lx   = _tif_dims(tif_path)
        except Exception:
            return None
        T   = data_bin.stat().st_size // (Ly * Lx * 2)  # int16 = 2 bytes
        fps = 0.0

    meta = {"n_frames": T, "height": Ly, "width": Lx, "fps": fps}
    with _fov_meta_lock:
        _fov_meta_cache[key] = meta
    return meta


# ── Security ────────────────────────────────────────────────────────────────────


def _decode_output_dir(encoded: str) -> Path:
    """Decode the ``dir`` query param and validate it is a pipeline output dir."""
    try:
        raw = base64.urlsafe_b64decode(encoded.encode()).decode()
    except Exception:
        raise ValueError("invalid dir parameter")
    p = Path(raw).resolve()
    if not (p / "pipeline_log.json").exists():
        raise ValueError(f"not a valid pipeline output directory: {p}")
    return p


# ── Image serving ───────────────────────────────────────────────────────────────


def _tif_to_png_bytes(tif_path: Path) -> tuple[bytes, int, int]:
    """Load a float32 TIF, stretch p1→p99 to uint8, return (png_bytes, W, H)."""
    from PIL import Image

    arr = tifffile.imread(str(tif_path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    H, W = arr.shape
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), W, H


def _resolve_projection(output_dir: Path) -> Path:
    candidates = [
        output_dir / "hitl_staging" / "images",
        output_dir / "summary",
        output_dir / "summary",
    ]
    names = [None, "mean_M.tif", "mean_S.tif"]
    for dirp, name in zip(candidates, names):
        if name is None:
            # hitl_staging/images — take first .tif
            if dirp.exists():
                tifs = sorted(dirp.glob("*.tif"))
                if tifs:
                    return tifs[0]
        else:
            p = dirp / name
            if p.exists():
                return p
    raise FileNotFoundError(f"No projection image found under {output_dir}")


# ── Annotation serialisation ────────────────────────────────────────────────────


def _rois_to_annotations(output_dir: Path) -> list[dict]:
    """Load current ROI set (pipeline + corrections) → W3C annotation list."""
    fov, _ = load_fov_from_output_dir(output_dir)
    base_rois = list(fov.rois)
    if base_rois and base_rois[0].mask is not None:
        H, W = base_rois[0].mask.shape
    elif fov.mean_M is not None:
        H, W = fov.mean_M.shape
    else:
        labels = tifffile.imread(str(output_dir / "merged_masks.tif"))
        H, W = labels.shape

    ops = load_corrections(output_dir)
    current_rois = apply_corrections(base_rois, ops, (H, W))

    annotations = []
    for roi in current_rois:
        if roi.mask is None or not roi.mask.any():
            continue
        contour = _mask_to_polygon_yx(roi.mask)
        if not contour:
            continue
        # Annotorious polygon: "x,y x,y ..." (x=col, y=row)
        points_str = " ".join(f"{x},{y}" for y, x in contour)
        svg_value = f'<svg><polygon points="{points_str}" /></svg>'
        annotations.append({
            "type": "Annotation",
            "id": f"roi-{roi.label_id}",
            "body": [{"type": "TextualBody", "value": str(roi.label_id),
                      "purpose": "tagging"}],
            "target": {
                "selector": {"type": "SvgSelector", "value": svg_value}
            },
        })
    return annotations


# ── HTML template ───────────────────────────────────────────────────────────────

_EDITOR_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ROI Editor · {{ fov_stem }}</title>
  <meta name="color-scheme" content="dark">
  <!-- Vendored assets served by Flask /roi-assets/ route, not Dash auto-include. -->
  <script src="/roi-assets/openseadragon.min.js"></script>
  <link rel="stylesheet" href="/roi-assets/annotorious.min.css">
  <script src="/roi-assets/openseadragon-annotorious.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { display: flex; flex-direction: column; height: 100vh;
           background: #0f1117; color: #e0e0e0; font-family: monospace; }
    #header { padding: 8px 16px; background: #1a1d27;
              display: flex; align-items: center; gap: 12px; flex-shrink: 0;
              border-bottom: 1px solid #2d2d4e; }
    h1 { font-size: 13px; font-weight: normal; color: #8ecae6; white-space: nowrap; }
    #viewer-container { }
    #osd { width: 100%; height: 100%; background: #0f1117; }
    .toolbar { display: flex; gap: 6px; }
    .sep { width: 1px; background: #2d2d4e; margin: 0 4px; }
    button { padding: 4px 10px; font-size: 11px; border: 1px solid #3d3d5e;
             background: #1e2030; color: #d0d0e0; cursor: pointer; border-radius: 3px; }
    button:hover { background: #2d2d4e; }
    button.active { background: #0077b6; border-color: #0077b6; color: #fff; }
    #btn-save { background: #0d6efd; border-color: #0d6efd; color: #fff; }
    #btn-save:hover { background: #0b5ed7; }
    #status { font-size: 11px; color: #888; margin-left: auto; min-width: 200px; text-align: right; }
    .a9s-annotation .a9s-inner { stroke: #00d4ff !important; stroke-width: 1.5px !important;
                                  fill: rgba(0,212,255,0.12) !important; }
    .a9s-annotation.selected .a9s-inner { stroke: #ff6b6b !important;
                                           fill: rgba(255,107,107,0.18) !important; }
    /* ── Scrubber ─────────────────────────────────────────────────────── */
    #scrubber-bar { display: flex; align-items: center; gap: 8px;
                    padding: 6px 16px; background: #12141f; flex-shrink: 0;
                    border-top: 1px solid #2d2d4e; font-size: 11px; color: #8ecae6; }
    #scrubber-bar.hidden { display: none; }
    #scrubber-track { flex: 1; height: 6px; background: #2d2d4e; border-radius: 3px;
                      position: relative; cursor: pointer; }
    #scrubber-fill  { height: 100%; background: #0077b6; border-radius: 3px;
                      width: 0%; pointer-events: none; }
    #scrubber-thumb { position: absolute; top: -5px; width: 16px; height: 16px;
                      background: #8ecae6; border-radius: 50%; transform: translateX(-50%);
                      cursor: grab; left: 0%; }
    #scrubber-thumb:active { cursor: grabbing; }
    #frame-label { min-width: 130px; text-align: right; white-space: nowrap; }
    #image-mode  { font-size: 10px; color: #555; min-width: 80px; }
  </style>
</head>
<body>
  <div id="header">
    <h1>ROI Editor &middot; {{ fov_stem }}</h1>
    <div class="sep"></div>
    <div class="toolbar">
      <button id="btn-draw" onclick="setMode('draw')">+ Draw</button>
      <button id="btn-select" onclick="setMode('select')" class="active">Select</button>
      <button id="btn-delete" onclick="deleteSelected()">Delete</button>
    </div>
    <div class="sep"></div>
    <div class="toolbar">
      <button id="btn-save" onclick="saveAnnotations()">Save</button>
      <button onclick="window.close()">Cancel</button>
    </div>
    <div class="sep"></div>
    <div class="toolbar">
      <button id="btn-zoomin"  onclick="zoomIn()"    title="Zoom in (+)">+</button>
      <button id="btn-zoomout" onclick="zoomOut()"   title="Zoom out (-)">&#x2212;</button>
      <button id="btn-reset"   onclick="resetZoom()" title="Reset zoom (0)">&#x2302;</button>
      <span id="zoom-indicator" style="font-size:11px;color:#8ecae6;min-width:40px;text-align:right;line-height:1.8;">1&#xD7;</span>
    </div>
    <span id="status">Loading&hellip;</span>
  </div>
  <div id="viewer-container" style="display:flex;flex-direction:column;flex:1;min-height:0;">
    <div id="osd" style="flex:1;min-height:0;"></div>
    <div id="scrubber-bar" class="hidden">
      <span id="image-mode">projection</span>
      <button id="btn-proj" onclick="_returnToProjection()"
              title="Return to mean projection (Home)"
              style="display:none;padding:2px 7px;font-size:10px;">&#x21A9; proj</button>
      <div id="scrubber-track">
        <div id="scrubber-fill"></div>
        <div id="scrubber-thumb"></div>
      </div>
      <span id="frame-label">&mdash; / &mdash;</span>
    </div>
  </div>

<script>
const IMAGE_URL      = {{ image_url | tojson }};
const ANNOT_URL      = {{ annotations_url | tojson }};
const SAVE_URL       = {{ save_url | tojson }};
const FRAME_URL_BASE = {{ frame_url | tojson }};
const META_URL       = {{ meta_url | tojson }};

let anno = null;
let selectedAnnotation = null;

function setStatus(msg) { document.getElementById('status').textContent = msg; }

const viewer = OpenSeadragon({
  id: 'osd',
  backgroundColor: '#0f1117',
  tileSources: { type: 'image', url: IMAGE_URL },
  // Disable toolbar buttons — no image assets needed.
  // Navigate with scroll-to-zoom, drag-to-pan, double-click-to-zoom.
  showZoomControl:      false,
  showHomeControl:      false,
  showFullPageControl:  false,
  showRotationControl:  false,
  showNavigator:        false,
  gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
  // Pixel-level zoom: allow up to 40× image-relative zoom.
  maxZoomLevel:          40,
  minZoomLevel:          0.3,
  visibilityRatio:       0.5,
  imageSmoothingEnabled: false,
});

let _viewerInitialized = false;

viewer.addHandler('open', function() {
  if (_viewerInitialized) return;
  _viewerInitialized = true;

  anno = OpenSeadragon.Annotorious(viewer, {
    drawOnSingleClick: true,
    disableEditor: true,
    widgets: [],
  });

  anno.on('selectAnnotation',  (a) => { selectedAnnotation = a; setStatus('ROI selected — press Delete to remove'); });
  anno.on('cancelSelected',    ()  => { selectedAnnotation = null; setStatus('Ready'); });
  anno.on('createAnnotation',  (a) => { setMode('select'); setStatus('ROI added — saving…'); _scheduleSave(); });
  anno.on('updateAnnotation',  ()  => { setStatus('ROI updated — saving…'); _scheduleSave(); });
  anno.on('deleteAnnotation',  ()  => { selectedAnnotation = null; setStatus('ROI deleted — saving…'); _scheduleSave(); });

  // Keyboard shortcuts: Delete/Backspace removes selection; +/-/0 for zoom.
  document.addEventListener('keydown', (e) => {
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnotation) {
      anno.removeAnnotation(selectedAnnotation);
      selectedAnnotation = null;
      setStatus('ROI deleted');
      return;
    }
    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
      if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomIn(); }
      if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomOut(); }
      if (e.key === '0')                  { e.preventDefault(); resetZoom(); }
    }
  });

  viewer.addHandler('viewport-change', _updateZoomIndicator);
  _updateZoomIndicator();

  fetch(ANNOT_URL)
    .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
    .then(data => {
      if (!Array.isArray(data)) {
        setStatus('Failed to load ROIs: ' + (data.error || JSON.stringify(data)));
        return;
      }
      anno.setAnnotations(data);
      setStatus(data.length + ' ROI(s) loaded');
    })
    .catch(e  => setStatus('Failed to load ROIs: ' + e))
    .finally(() => _initScrubber());
});

function setMode(mode) {
  if (!anno) return;
  const drawBtn = document.getElementById('btn-draw');
  const selBtn  = document.getElementById('btn-select');
  if (mode === 'draw') {
    anno.setDrawingTool('polygon');
    anno.setDrawingEnabled(true);
    drawBtn.classList.add('active'); selBtn.classList.remove('active');
    setStatus('Click to add vertices — double-click to close polygon');
  } else {
    anno.setDrawingEnabled(false);
    selBtn.classList.add('active'); drawBtn.classList.remove('active');
    setStatus('Ready');
  }
}

function deleteSelected() {
  if (!anno || !selectedAnnotation) { setStatus('No ROI selected'); return; }
  anno.removeAnnotation(selectedAnnotation);
  selectedAnnotation = null;
}

// ── Zoom controls ──────────────────────────────────────────────────────────────

function zoomIn()    { if (viewer) { viewer.viewport.zoomBy(1.5); viewer.viewport.applyConstraints(); } }
function zoomOut()   { if (viewer) { viewer.viewport.zoomBy(1 / 1.5); viewer.viewport.applyConstraints(); } }
function resetZoom() { if (viewer) viewer.viewport.goHome(); }

function _updateZoomIndicator() {
  if (!viewer || !viewer.viewport) return;
  const z = viewer.viewport.getZoom(true);
  document.getElementById('zoom-indicator').textContent =
    (z >= 10 ? Math.round(z) : z.toFixed(1)) + '\xD7';
}

// ── Auto-save state ────────────────────────────────────────────────────────────
let _saveTimer = null;
let _saveInFlight = false;

function _scheduleSave() {
  if (_saveTimer !== null) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    _saveTimer = null;
    if (!_saveInFlight) saveAnnotations();
  }, 800);
}

function saveAnnotations() {
  if (!anno || _saveInFlight) return;
  const annotations = anno.getAnnotations();
  _saveInFlight = true;
  setStatus('Saving…');
  document.getElementById('btn-save').disabled = true;
  fetch(SAVE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ annotations }),
  })
    .then(r => r.json())
    .then(data => {
      _saveInFlight = false;
      document.getElementById('btn-save').disabled = false;
      if (data.ok) {
        const ts = new Date().toLocaleTimeString();
        setStatus('✓ Saved ' + ts + ' — ' + data.summary);
      } else {
        setStatus('✗ Error: ' + (data.error || 'unknown'));
      }
    })
    .catch(e => {
      _saveInFlight = false;
      document.getElementById('btn-save').disabled = false;
      setStatus('Save failed: ' + e);
    });
}
// ── Scrubber ──────────────────────────────────────────────────────────────────
let _nFrames      = 0;
let _currentFrame = -1;   // -1 = showing mean projection
let _currentFps   = 0;
let _scrubDebounce = null;

function _setFrameDisplay(n) {
  const pct = _nFrames > 1 ? (n / (_nFrames - 1)) * 100 : 0;
  document.getElementById('scrubber-fill').style.width = pct + '%';
  document.getElementById('scrubber-thumb').style.left  = pct + '%';
  const secs = (_currentFps > 0) ? ' \xB7 ' + (n / _currentFps).toFixed(1) + ' s' : '';
  document.getElementById('frame-label').textContent = (n + 1) + ' / ' + _nFrames + secs;
  document.getElementById('image-mode').textContent  = 'frame ' + n;
  document.getElementById('btn-proj').style.display  = '';
}

function _scrubToFrame(n) {
  n = Math.max(0, Math.min(n, _nFrames - 1));
  if (n === _currentFrame) return;
  _currentFrame = n;
  _setFrameDisplay(n);
  if (_scrubDebounce) clearTimeout(_scrubDebounce);
  _scrubDebounce = setTimeout(() => {
    _scrubDebounce = null;
    const savedBounds = viewer.viewport.getBounds(true);
    const restoreVp = function() {
      viewer.viewport.fitBounds(savedBounds, true);
      viewer.removeHandler('open', restoreVp);
    };
    viewer.addHandler('open', restoreVp);
    viewer.open({ type: 'image', url: FRAME_URL_BASE + '&n=' + n });
  }, 80);
}

function _returnToProjection() {
  if (_currentFrame < 0) return;
  _currentFrame = -1;
  document.getElementById('scrubber-fill').style.width = '0%';
  document.getElementById('scrubber-thumb').style.left  = '0%';
  document.getElementById('frame-label').textContent = '— / ' + _nFrames;
  document.getElementById('image-mode').textContent  = 'projection';
  document.getElementById('btn-proj').style.display  = 'none';
  if (_scrubDebounce) { clearTimeout(_scrubDebounce); _scrubDebounce = null; }
  const savedBounds = viewer.viewport.getBounds(true);
  const restoreVp = function() {
    viewer.viewport.fitBounds(savedBounds, true);
    viewer.removeHandler('open', restoreVp);
  };
  viewer.addHandler('open', restoreVp);
  viewer.open({ type: 'image', url: IMAGE_URL });
}

function _initScrubber() {
  if (_nFrames > 0) return;
  fetch(META_URL)
    .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
    .then(meta => {
      if (!meta.n_frames) return;
      _nFrames    = meta.n_frames;
      _currentFps = meta.fps || 0;
      document.getElementById('scrubber-bar').classList.remove('hidden');
      document.getElementById('frame-label').textContent = '— / ' + _nFrames;

      const track = document.getElementById('scrubber-track');
      let _dragging = false;
      function _posToFrame(clientX) {
        const rect = track.getBoundingClientRect();
        const pct  = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        return Math.round(pct * (_nFrames - 1));
      }
      track.addEventListener('pointerdown', (e) => {
        e.preventDefault(); _dragging = true; track.setPointerCapture(e.pointerId);
        _scrubToFrame(_posToFrame(e.clientX));
      });
      track.addEventListener('pointermove',   (e) => { if (_dragging) _scrubToFrame(_posToFrame(e.clientX)); });
      track.addEventListener('pointerup',     ()  => { _dragging = false; });
      track.addEventListener('pointercancel', ()  => { _dragging = false; });

      document.addEventListener('keydown', (e) => {
        if (!_nFrames || e.ctrlKey || e.metaKey || e.altKey) return;
        const step = e.shiftKey ? 10 : 1;
        if (e.key === 'ArrowRight') { e.preventDefault(); _scrubToFrame(_currentFrame < 0 ? 0 : _currentFrame + step); }
        if (e.key === 'ArrowLeft')  { e.preventDefault(); if (_currentFrame >= 0) _scrubToFrame(_currentFrame - step); }
        if (e.key === 'Home')       { e.preventDefault(); _returnToProjection(); }
      });
    })
    .catch(e => console.warn('FOV meta fetch failed:', e));
}

// ── Tab-close fallback: synchronous XHR saves pending changes on navigation ──
window.addEventListener('beforeunload', () => {
  if (!anno) return;
  if (_saveTimer !== null) { clearTimeout(_saveTimer); _saveTimer = null; }
  if (_saveInFlight) return;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', SAVE_URL, false);  // false = synchronous (allowed in beforeunload)
  xhr.setRequestHeader('Content-Type', 'application/json');
  try { xhr.send(JSON.stringify({ annotations: anno.getAnnotations() })); } catch (_) {}
});
</script>
</body>
</html>
"""


# ── Route registration ──────────────────────────────────────────────────────────


_VENDOR_DIR: Path = Path(__file__).resolve().parents[1] / "vendor"


def register_flask_routes(server: Flask) -> None:
    """Register the ROI editor routes on the Dash app's underlying Flask server."""

    @server.route("/roi-assets/<path:filename>")
    def roi_vendor_assets(filename: str):
        return send_from_directory(_VENDOR_DIR, filename)

    @server.route("/roi-editor/<fov_stem>")
    def roi_editor_page(fov_stem: str):
        dir_b64 = request.args.get("dir", "")
        if not dir_b64:
            return "Missing ?dir= parameter", 400
        try:
            output_dir = _decode_output_dir(dir_b64)
        except ValueError as exc:
            return str(exc), 400
        qs = f"?dir={dir_b64}"
        return render_template_string(
            _EDITOR_HTML,
            fov_stem=fov_stem,
            image_url=f"/api/fov-image/{fov_stem}{qs}",
            annotations_url=f"/api/annotations/{fov_stem}{qs}",
            save_url=f"/api/corrections/{fov_stem}{qs}",
            frame_url=f"/api/frame/{fov_stem}{qs}",
            meta_url=f"/api/fov-meta/{fov_stem}{qs}",
        )

    @server.route("/api/fov-image/<fov_stem>")
    def fov_image(fov_stem: str):
        dir_b64 = request.args.get("dir", "")
        if not dir_b64:
            return "Missing ?dir= parameter", 400
        try:
            output_dir = _decode_output_dir(dir_b64)
            tif_path = _resolve_projection(output_dir)
        except (ValueError, FileNotFoundError) as exc:
            return str(exc), 404
        try:
            png_bytes, img_w, img_h = _tif_to_png_bytes(tif_path)
        except Exception as exc:
            return f"Image conversion failed: {exc}", 500
        resp = Response(png_bytes, mimetype="image/png")
        resp.headers["X-Image-Width"] = str(img_w)
        resp.headers["X-Image-Height"] = str(img_h)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @server.route("/api/fov-meta/<fov_stem>")
    def fov_meta(fov_stem: str):
        dir_b64 = request.args.get("dir", "")
        if not dir_b64:
            return jsonify({"error": "Missing ?dir= parameter"}), 400
        try:
            output_dir = _decode_output_dir(dir_b64)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        meta = _get_fov_meta(output_dir)
        if meta is None:
            return jsonify({"n_frames": 0, "height": 0, "width": 0, "fps": 0.0})
        return jsonify({"n_frames": meta["n_frames"], "height": meta["height"],
                        "width": meta["width"], "fps": meta["fps"]})

    @server.route("/api/frame/<fov_stem>")
    def fov_frame(fov_stem: str):
        dir_b64 = request.args.get("dir", "")
        n_str   = request.args.get("n", "0")
        if not dir_b64:
            return "Missing ?dir= parameter", 400
        try:
            output_dir = _decode_output_dir(dir_b64)
        except ValueError as exc:
            return str(exc), 400
        try:
            n = int(n_str)
        except ValueError:
            return "n must be an integer", 400
        meta = _get_fov_meta(output_dir)
        if meta is None:
            return "data.bin not found", 404
        T, Ly, Lx = meta["n_frames"], meta["height"], meta["width"]
        n = max(0, min(n, T - 1))
        fov_key = str(output_dir.resolve())
        try:
            png_bytes = _encode_frame_png(fov_key, n, T, Ly, Lx)
        except Exception as exc:
            return f"Frame read failed: {exc}", 500
        resp = Response(png_bytes, mimetype="image/png")
        resp.headers["X-Frame-Index"] = str(n)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @server.route("/api/annotations/<fov_stem>")
    def get_annotations(fov_stem: str):
        dir_b64 = request.args.get("dir", "")
        if not dir_b64:
            return jsonify([])
        try:
            output_dir = _decode_output_dir(dir_b64)
        except ValueError as exc:
            return str(exc), 400
        try:
            annotations = _rois_to_annotations(output_dir)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify(annotations)

    @server.route("/api/corrections/<fov_stem>", methods=["POST"])
    def post_corrections(fov_stem: str):
        dir_b64 = request.args.get("dir", "")
        if not dir_b64:
            return jsonify({"ok": False, "error": "Missing ?dir= parameter"}), 400
        try:
            output_dir = _decode_output_dir(dir_b64)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        payload = request.get_json(silent=True) or {}
        annotations = payload.get("annotations", [])
        if not isinstance(annotations, list):
            return jsonify({"ok": False, "error": "annotations must be a list"}), 400

        lock = _get_write_lock(output_dir)
        with lock:
            try:
                result = reingest_from_annotations(output_dir, annotations)
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        return jsonify({"ok": True, "summary": result.summary()})
