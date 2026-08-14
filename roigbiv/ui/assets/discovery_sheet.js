/* The /discovery page's viewer: one OpenSeadragon panel over one FOV, with an
 * SVG overlay carrying editable centroid markers and a read-only boundary
 * preview.
 *
 * A single-panel cousin of assets/cells_sheet.js (the /tracking contact
 * sheet), deliberately not sharing code with it. Two reasons:
 *
 *   1. cells_sheet.js is being edited concurrently by another task in a
 *      separate worktree — importing from it here would be a guaranteed merge
 *      conflict on a file this task does not own.
 *   2. The state shapes genuinely differ. Discovery has one FOV, no sessions,
 *      no ghosts, no cross-session selection or shift-click-link — a gesture
 *      here needs no "selected cell" at all, since add/delete/move each name
 *      their own target directly.
 *
 * Server contract: roigbiv/ui/routes/discovery_api.py. Gesture semantics live
 * in roigbiv/ui/services/discovery_edit_ops.py — this file only decides which
 * gesture a pointer made.
 *
 * Styles are injected here rather than added to assets/roigbiv.css, for the
 * same merge-conflict-avoidance reason as (1) above.
 */
(function () {
  "use strict";

  var SHEET_ID = "roigbiv-discovery-sheet";
  var MSG_ID = "roigbiv-discovery-edit-msg";

  // Below this, a pointerup is a click; above it, a drag.
  var DRAG_THRESHOLD_PX = 3;
  var SVG_NS = "http://www.w3.org/2000/svg";

  var state = {
    stem: null,
    editOn: false,
    diameterPx: null,
    data: null,            // last /api/discovery/<stem> payload
    boundaries: null,      // last {contours: {label: {origin, rings}}} payload
    viewer: null,
    svg: null,
    centroidGroup: null,
    boundaryGroup: null,
    calibCircle: null,
    shapes: {},
    generation: 0,          // bumped on every rebuild; stale async work checks it
    osdPromise: null,
  };

  // ── styles ────────────────────────────────────────────────────────────────

  function injectStyles() {
    if (document.getElementById("roigbiv-discovery-styles")) { return; }
    var style = document.createElement("style");
    style.id = "roigbiv-discovery-styles";
    style.textContent = [
      ".roigbiv-discovery-view { width: 100%; aspect-ratio: 1 / 1;",
      "  background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.12);",
      "  border-radius: 4px; overflow: hidden; min-height: 420px; }",
      ".roigbiv-discovery-overlay-holder { width: 100%; height: 100%; pointer-events: none; }",
      ".roigbiv-discovery-overlay { width: 100%; height: 100%; overflow: visible; }",
      ".roigbiv-discovery-centroid { fill: rgba(0,229,255,0.16); stroke: #00e5ff;",
      "  stroke-width: 2px; vector-effect: non-scaling-stroke; pointer-events: none; }",
      ".roigbiv-discovery-centroids.is-editing .roigbiv-discovery-centroid {",
      "  pointer-events: auto; cursor: grab; }",
      ".roigbiv-discovery-centroid.is-dragging { cursor: grabbing; stroke-width: 3px;",
      "  stroke-dasharray: 4 3; }",
      ".roigbiv-discovery-boundary { fill: none; stroke-width: 2px;",
      "  vector-effect: non-scaling-stroke; pointer-events: none; }",
      ".roigbiv-discovery-calib-circle { fill: none; stroke: #FFD400; stroke-width: 2px;",
      "  stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; pointer-events: none; }",
      ".roigbiv-discovery-sheet.is-editing .roigbiv-discovery-view { cursor: crosshair; }",
    ].join("\n");
    document.head.appendChild(style);
  }

  // ── OpenSeadragon loading ────────────────────────────────────────────────
  // Vendored and served by the ROI editor's /roi-assets route — see
  // cells_sheet.js's identical rationale for not using Dash's assets folder.

  function ensureOSD() {
    if (window.OpenSeadragon) { return Promise.resolve(); }
    if (state.osdPromise) { return state.osdPromise; }
    state.osdPromise = new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = "/roi-assets/openseadragon.min.js";
      script.onload = function () { resolve(); };
      script.onerror = function () {
        reject(new Error("could not load OpenSeadragon"));
      };
      document.head.appendChild(script);
    });
    return state.osdPromise;
  }

  function say(message) {
    var el = document.getElementById(MSG_ID);
    if (el) { el.textContent = message || ""; }
  }

  // ── fetch ────────────────────────────────────────────────────────────────

  function fetchState(stem) {
    return fetch("/api/discovery/" + encodeURIComponent(stem), {
      headers: { Accept: "application/json" },
    }).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.error || ("HTTP " + r.status));
        });
      }
      return r.json();
    });
  }

  function postGesture(gesture) {
    if (!state.stem) { return Promise.resolve(); }
    say("working…");
    return fetch("/api/discovery/" + encodeURIComponent(state.stem) + "/gesture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(gesture),
    }).then(function (r) {
      return r.json().catch(function () { return {}; });
    }).then(function (body) {
      say(body.message || body.error || "edit failed");
      if (body.ok && body.state) {
        // Geometry only — the viewer itself is never touched, which is what
        // keeps zoom and pan across an edit.
        state.data = body.state;
        drawCentroids();
      }
    }).catch(function (err) {
      say("edit failed: " + err.message);
    });
  }

  // ── viewer construction ──────────────────────────────────────────────────

  function mountViewer(root) {
    var generation = state.generation;
    root.innerHTML = "";
    root.classList.add("roigbiv-discovery-sheet");

    var viewEl = document.createElement("div");
    viewEl.className = "roigbiv-discovery-view";
    if (state.data.width && state.data.height) {
      viewEl.style.aspectRatio = state.data.width + " / " + state.data.height;
    }
    root.appendChild(viewEl);

    var viewer = window.OpenSeadragon({
      element: viewEl,
      tileSources: { type: "image", url: state.data.image_url },
      backgroundColor: "transparent",
      showNavigationControl: false,
      showNavigator: false,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
      imageSmoothingEnabled: false,
      maxZoomLevel: 40,
      minZoomLevel: 0.5,
      visibilityRatio: 0.6,
      animationTime: 0.4,
    });
    state.viewer = viewer;

    viewer.addHandler("open", function () {
      if (generation !== state.generation) { return; }
      attachOverlay(viewer, root);
      drawCentroids();
      drawCalibrationCircle();
      drawBoundaries();
    });

    // A click that is not a drag, landing on the background — the centroid
    // markers stop propagation on pointerdown, so anything OSD sees here
    // missed every marker. `quick` is OSD's own click/pan discrimination.
    viewer.addHandler("canvas-click", function (event) {
      if (!event.quick || !state.editOn) { return; }
      var point = viewer.viewport.viewerElementToImageCoordinates(event.position);
      postGesture({ kind: "add", y: point.y, x: point.x });
    });
  }

  function attachOverlay(viewer, root) {
    var width = state.data.width || 1;
    var height = state.data.height || 1;

    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.classList.add("roigbiv-discovery-overlay");
    state.svg = svg;

    var holder = document.createElement("div");
    holder.className = "roigbiv-discovery-overlay-holder";
    holder.appendChild(svg);

    viewer.addOverlay({
      element: holder,
      location: new window.OpenSeadragon.Rect(0, 0, 1, height / width),
    });

    var boundaryGroup = document.createElementNS(SVG_NS, "g");
    boundaryGroup.setAttribute("class", "roigbiv-discovery-boundaries");
    svg.appendChild(boundaryGroup);
    state.boundaryGroup = boundaryGroup;

    var centroidGroup = document.createElementNS(SVG_NS, "g");
    centroidGroup.setAttribute("class", "roigbiv-discovery-centroids");
    svg.appendChild(centroidGroup);
    state.centroidGroup = centroidGroup;

    root.classList.toggle("is-editing", state.editOn);
  }

  // ── centroid layer (editable) ────────────────────────────────────────────

  function drawCentroids() {
    var group = state.centroidGroup;
    if (!group) { return; }
    while (group.firstChild) { group.removeChild(group.firstChild); }
    state.shapes = {};

    var radius = (state.data && state.data.radius) || 8;
    (state.data.centroids || []).forEach(function (c) {
      var circle = document.createElementNS(SVG_NS, "circle");
      circle.setAttribute("cx", c.x);
      circle.setAttribute("cy", c.y);
      circle.setAttribute("r", radius);
      circle.setAttribute("class", "roigbiv-discovery-centroid");
      circle.dataset.labelId = String(c.label_id);
      wireCentroid(circle, c);
      group.appendChild(circle);
      state.shapes[c.label_id] = circle;
    });
    group.classList.toggle("is-editing", state.editOn);
    var sheet = document.getElementById(SHEET_ID);
    if (sheet) { sheet.classList.toggle("is-editing", state.editOn); }
  }

  function wireCentroid(circle, c) {
    circle.addEventListener("pointerdown", function (event) {
      if (event.button !== 0 || !state.editOn) { return; }
      // OSD's MouseTracker sits on the canvas this overlay lives inside;
      // without this it would pan the image under the drag.
      event.stopPropagation();
      event.preventDefault();
      beginDrag(circle, c, event);
    });

    circle.addEventListener("contextmenu", function (event) {
      event.preventDefault();
      event.stopPropagation();
      if (!state.editOn) { return; }
      postGesture({ kind: "delete", label: c.label_id });
    });
  }

  function beginDrag(circle, c, downEvent) {
    var origin = { x: downEvent.clientX, y: downEvent.clientY };
    var moved = false;
    var pointerId = downEvent.pointerId;

    try { circle.setPointerCapture(pointerId); } catch (e) { /* mouse */ }
    circle.classList.add("is-dragging");

    function onMove(event) {
      var dx = event.clientX - origin.x;
      var dy = event.clientY - origin.y;
      if (!moved && Math.abs(dx) + Math.abs(dy) < DRAG_THRESHOLD_PX) { return; }
      moved = true;
      // Live feedback in screen pixels — no coordinate conversion until drop.
      circle.setAttribute("transform", screenDeltaToImage(dx, dy));
    }

    function onUp(event) {
      circle.removeEventListener("pointermove", onMove);
      circle.removeEventListener("pointerup", onUp);
      circle.removeEventListener("pointercancel", onUp);
      circle.classList.remove("is-dragging");
      circle.removeAttribute("transform");
      try { circle.releasePointerCapture(pointerId); } catch (e) { /* mouse */ }

      if (!moved) { return; }   // a plain click on a marker does nothing here
      var point = imagePointOf(event);
      // The exact drop coordinate — nothing snaps it to a centroid, so a
      // two-pixel correction lands where it was put.
      postGesture({ kind: "move", label: c.label_id, y: point.y, x: point.x });
    }

    circle.addEventListener("pointermove", onMove);
    circle.addEventListener("pointerup", onUp);
    circle.addEventListener("pointercancel", onUp);
  }

  function screenDeltaToImage(dx, dy) {
    var viewer = state.viewer;
    var zoom = viewer.viewport.getZoom(true);
    var containerWidth = viewer.viewport.getContainerSize().x;
    var imageWidth = state.data.width || 1;
    var scale = (containerWidth * zoom) / imageWidth;
    if (!scale) { return ""; }
    return "translate(" + (dx / scale) + " " + (dy / scale) + ")";
  }

  function imagePointOf(event) {
    var viewer = state.viewer;
    var rect = viewer.element.getBoundingClientRect();
    return viewer.viewport.viewerElementToImageCoordinates(
      new window.OpenSeadragon.Point(
        event.clientX - rect.left, event.clientY - rect.top));
  }

  // ── calibration reference circle (read-only) ─────────────────────────────

  function drawCalibrationCircle() {
    var svg = state.svg;
    if (!svg) { return; }
    if (state.calibCircle) {
      svg.removeChild(state.calibCircle);
      state.calibCircle = null;
    }
    var d = state.diameterPx;
    if (!d || !state.data || !state.data.width || !state.data.height) { return; }
    var cx = state.data.width / 2;
    var cy = state.data.height / 2;
    var circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", d / 2);
    circle.setAttribute("class", "roigbiv-discovery-calib-circle");
    svg.appendChild(circle);
    state.calibCircle = circle;
  }

  // ── boundary preview layer (read-only) ───────────────────────────────────

  function drawBoundaries() {
    var group = state.boundaryGroup;
    if (!group) { return; }
    while (group.firstChild) { group.removeChild(group.firstChild); }
    var payload = state.boundaries;
    if (!payload || !payload.contours) { return; }
    Object.keys(payload.contours).forEach(function (label) {
      var entry = payload.contours[label] || {};
      var color = entry.origin === "disk_fallback" ? "#FF7A45" : "#3FC1C9";
      (entry.rings || []).forEach(function (ring) {
        if (!ring.length) { return; }
        var path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("d", ringPath(ring));
        path.setAttribute("class", "roigbiv-discovery-boundary");
        path.setAttribute("stroke", color);
        group.appendChild(path);
      });
    });
  }

  function ringPath(ring) {
    var d = "";
    for (var i = 0; i < ring.length; i++) {
      d += (i === 0 ? "M" : "L") + ring[i][0] + " " + ring[i][1];
    }
    return d + "Z";
  }

  // ── entry points ──────────────────────────────────────────────────────────

  function destroy() {
    state.generation += 1;
    if (state.viewer) {
      try { state.viewer.destroy(); } catch (e) { /* already gone */ }
    }
    state.viewer = null;
    state.svg = null;
    state.centroidGroup = null;
    state.boundaryGroup = null;
    state.calibCircle = null;
    state.shapes = {};
  }

  function render(config) {
    injectStyles();
    var root = document.getElementById(SHEET_ID);
    if (!root) { return; }
    config = config || {};

    state.editOn = !!config.edit_on;
    state.diameterPx = config.diameter_px || null;

    if (!config.stem) {
      destroy();
      state.stem = null;
      state.data = null;
      root.innerHTML = '<div class="text-muted small p-3">Nothing to preview '
        + 'yet — run motion correction, or pick a FOV that already has '
        + 'one, to see it here.</div>';
      return;
    }

    // Selection-free by design: the edit switch and the diameter field never
    // rebuild anything, which is what keeps zoom and pan stable while tuning.
    if (config.stem === state.stem && state.data) {
      drawCentroids();
      drawCalibrationCircle();
      return;
    }

    destroy();
    state.stem = config.stem;
    var generation = state.generation;

    root.innerHTML = '<div class="text-muted small p-3">Loading…</div>';

    Promise.all([ensureOSD(), fetchState(config.stem)])
      .then(function (results) {
        if (generation !== state.generation || state.stem !== config.stem) {
          return;   // a different FOV was picked while this was in flight
        }
        state.data = results[1];
        mountViewer(root);
      })
      .catch(function (err) {
        if (generation !== state.generation) { return; }
        root.innerHTML = "";
        var alert = document.createElement("div");
        alert.className = "alert alert-danger";
        alert.textContent = "Could not load this FOV: " + err.message;
        root.appendChild(alert);
      });
  }

  function setBoundaries(payload) {
    state.boundaries = payload || null;
    drawBoundaries();
  }

  function resize() {
    if (state.viewer && state.viewer.isOpen()) {
      state.viewer.viewport.resize();
      state.viewer.forceRedraw();
    }
  }

  window.roigbivDiscovery = {
    render: render, destroy: destroy, setBoundaries: setBoundaries, resize: resize,
  };
})();
