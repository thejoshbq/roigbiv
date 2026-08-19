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
    editBoundaryOn: false,
    showCentroids: true,
    showBoundaries: true,
    diameterPx: null,
    data: null,            // last /api/discovery/<stem> payload
    boundaries: null,      // last {contours: {label: {origin, rings}}} payload
    viewer: null,
    svg: null,
    centroidGroup: null,
    boundaryGroup: null,
    drawGroup: null,       // in-progress hand-drawn ring, boundary-edit mode
    calibCircle: null,
    shapes: {},
    draw: null,             // {label, points: [[y, x], ...]} while drawing
    generation: 0,          // bumped on every rebuild; stale async work checks it
    osdPromise: null,
    movieOn: false,
    frameCanvas: null,
    frameHolder: null,
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
      ".roigbiv-discovery-centroid { fill: rgba(0,229,255,0.5); stroke: #00e5ff;",
      "  stroke-width: 1.5px; vector-effect: non-scaling-stroke; pointer-events: none; }",
      ".roigbiv-discovery-centroids.is-editing .roigbiv-discovery-centroid {",
      "  pointer-events: auto; cursor: grab; }",
      ".roigbiv-discovery-centroid.is-dragging { cursor: grabbing; stroke-width: 3px;",
      "  stroke-dasharray: 4 3; }",
      ".roigbiv-discovery-boundary { fill: none; stroke-width: 2px;",
      "  vector-effect: non-scaling-stroke; pointer-events: none; }",
      ".roigbiv-discovery-boundaries.is-boundary-editing .roigbiv-discovery-boundary.is-manual {",
      "  pointer-events: auto; cursor: pointer; }",
      ".roigbiv-discovery-boundaries.is-boundary-editing .roigbiv-discovery-boundary.is-manual:hover {",
      "  stroke-width: 3px; }",
      ".roigbiv-discovery-centroids.is-boundary-editing .roigbiv-discovery-centroid {",
      "  pointer-events: auto; cursor: cell; }",
      ".roigbiv-discovery-draw-ring { fill: rgba(179,136,255,0.12); stroke: #B388FF;",
      "  stroke-width: 2px; stroke-dasharray: 3 3; vector-effect: non-scaling-stroke;",
      "  pointer-events: none; }",
      ".roigbiv-discovery-draw-vertex { fill: #B388FF; stroke: none; pointer-events: none; }",
      ".roigbiv-discovery-calib-circle { fill: none; stroke: #FFD400; stroke-width: 2px;",
      "  stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; pointer-events: none; }",
      ".roigbiv-discovery-sheet.is-editing .roigbiv-discovery-view { cursor: crosshair; }",
      ".roigbiv-discovery-sheet.is-boundary-editing .roigbiv-discovery-view { cursor: cell; }",
      ".roigbiv-discovery-centroids.is-hidden, .roigbiv-discovery-boundaries.is-hidden {",
      "  display: none; }",
      // Per-FOV label_id, plain (no "#") — cells_sheet.js's "#"+cell_index
      // badge means something else (a cross-session number this page has no
      // way to know yet), so this stays visually distinct from it.
      ".roigbiv-discovery-badge { font-family: var(--roigbiv-font-mono, monospace);",
      "  font-size: 10px; text-anchor: middle; dominant-baseline: middle;",
      "  pointer-events: none; paint-order: stroke; stroke: rgba(0,0,0,0.75);",
      "  stroke-width: 2.5px; vector-effect: non-scaling-stroke; fill: #00e5ff; }",
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
      if (body.ok && body.boundaries) {
        state.boundaries = body.boundaries;
        drawBoundaries();
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
      if (!event.quick) { return; }
      var point = viewer.viewport.viewerElementToImageCoordinates(event.position);
      if (state.editBoundaryOn && state.draw) {
        addDrawVertex(point.y, point.x);
        return;
      }
      if (!state.editOn) { return; }
      postGesture({ kind: "add", y: point.y, x: point.x });
    });

    // Closes the in-progress ring. Also drops one last vertex at the
    // double-click point first — harmless if a preceding single click already
    // placed a near-identical one, and means the ring always ends where the
    // pointer actually is.
    viewer.addHandler("canvas-double-click", function (event) {
      if (!state.editBoundaryOn || !state.draw) { return; }
      event.preventDefaultAction = true;
      var point = viewer.viewport.viewerElementToImageCoordinates(event.position);
      state.draw.points.push([point.y, point.x]);
      closeDrawRing();
    });
  }

  function attachOverlay(viewer, root) {
    var width = state.data.width || 1;
    var height = state.data.height || 1;

    // Added before the SVG so DOM order puts the movie underneath the markers.
    attachFrameCanvas(viewer);

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

    var drawGroup = document.createElementNS(SVG_NS, "g");
    drawGroup.setAttribute("class", "roigbiv-discovery-draw");
    svg.appendChild(drawGroup);
    state.drawGroup = drawGroup;

    var centroidGroup = document.createElementNS(SVG_NS, "g");
    centroidGroup.setAttribute("class", "roigbiv-discovery-centroids");
    svg.appendChild(centroidGroup);
    state.centroidGroup = centroidGroup;

    applyEditModeClasses(root);
  }

  function applyEditModeClasses(root) {
    root.classList.toggle("is-editing", state.editOn);
    root.classList.toggle("is-boundary-editing", state.editBoundaryOn);
    if (state.centroidGroup) {
      state.centroidGroup.classList.toggle("is-editing", state.editOn);
      state.centroidGroup.classList.toggle(
        "is-boundary-editing", state.editBoundaryOn);
      state.centroidGroup.classList.toggle("is-hidden", !state.showCentroids);
    }
    if (state.boundaryGroup) {
      state.boundaryGroup.classList.toggle(
        "is-boundary-editing", state.editBoundaryOn);
      state.boundaryGroup.classList.toggle("is-hidden", !state.showBoundaries);
    }
  }

  // ── movie layer (read-only, under the markers) ───────────────────────────
  //
  // The movie is painted into an OSD overlay rather than swapped into the tile
  // source. OSD keeps owning zoom and pan, the SVG above keeps owning
  // gestures, and this canvas only ever receives pixels — which is why editing
  // and zooming keep working during playback without either knowing about it.
  //
  // Transport, buffering and the controls live in assets/discovery_player.js;
  // everything here is the OSD-coordinate half it deliberately does not know.

  // Extra image pixels fetched beyond the visible rect, so a small pan is
  // served from the buffer instead of invalidating it.
  var RECT_PAD_PX = 24;

  function attachFrameCanvas(viewer) {
    var canvas = document.createElement("canvas");
    canvas.className = "roigbiv-player-canvas is-hidden";

    var holder = document.createElement("div");
    holder.className = "roigbiv-discovery-overlay-holder";
    holder.appendChild(canvas);

    // Placed over the whole image to start; visibleRect() re-places it on the
    // crop as soon as the player asks for one.
    viewer.addOverlay({
      element: holder,
      location: new window.OpenSeadragon.Rect(
        0, 0, 1, (state.data.height || 1) / (state.data.width || 1)),
    });

    state.frameCanvas = canvas;
    state.frameHolder = holder;

    if (!window.roigbivDiscoveryPlayer) { return; }
    window.roigbivDiscoveryPlayer.mount({
      canvas: canvas,
      controlsMount: document.getElementById(SHEET_ID),
      getRect: visibleRect,
      // The player must not take Escape/Enter/arrows out from under an
      // in-progress boundary ring.
      isDrawing: function () { return !!state.draw; },
      onRectApplied: placeFrameOverlay,
    });
    window.roigbivDiscoveryPlayer.setStem(state.stem).then(function () {
      window.roigbivDiscoveryPlayer.setEnabled(state.movieOn);
    });

    viewer.addHandler("animation-finish", onViewportSettled);
    viewer.addHandler("resize", onViewportSettled);
  }

  function onViewportSettled() {
    if (window.roigbivDiscoveryPlayer) {
      window.roigbivDiscoveryPlayer.setRect();
    }
  }

  /* The image rectangle currently on screen, padded, plus the decimation the
   * screen can actually resolve. Returned to the player, which turns it into a
   * crop request — this is what makes playback while zoomed in cost a fraction
   * of a full frame. */
  function visibleRect() {
    var viewer = state.viewer;
    if (!viewer || !viewer.isOpen() || !state.data) { return null; }
    var imgW = state.data.width || 1;
    var imgH = state.data.height || 1;

    var bounds = viewer.viewport.getBounds(true);
    var tl = viewer.viewport.viewportToImageCoordinates(bounds.getTopLeft());
    var br = viewer.viewport.viewportToImageCoordinates(bounds.getBottomRight());

    var x0 = clamp(Math.floor(tl.x) - RECT_PAD_PX, 0, imgW - 1);
    var y0 = clamp(Math.floor(tl.y) - RECT_PAD_PX, 0, imgH - 1);
    var x1 = clamp(Math.ceil(br.x) + RECT_PAD_PX, x0 + 1, imgW);
    var y1 = clamp(Math.ceil(br.y) + RECT_PAD_PX, y0 + 1, imgH);
    var rect = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };

    // How many image pixels each screen pixel covers, snapped down to a power
    // of two: a continuous stride would re-key the buffer on every small zoom
    // nudge, and the visible difference between ds=3 and ds=4 is nil.
    var containerW = viewer.viewport.getContainerSize().x || 1;
    var screenPerImage = (containerW * viewer.viewport.getZoom(true)) / imgW;
    var imagePerScreen = screenPerImage > 0 ? 1 / screenPerImage : 1;
    var ds = Math.pow(2, Math.floor(Math.log2(Math.max(1, imagePerScreen))));
    rect.ds = clamp(ds, 1, 8);

    return rect;
  }

  /* Move the canvas overlay onto a crop, in OSD's normalized coordinates (both
   * axes divided by image *width* — see attachOverlay). Called by the player
   * once it has actually painted a frame of that crop, never when the crop is
   * merely chosen: repositioning first would stretch the previous crop's
   * pixels over the new rectangle for as long as the fetch takes. */
  function placeFrameOverlay(rect) {
    if (!state.viewer || !state.frameHolder || !state.data) { return; }
    var imgW = state.data.width || 1;
    state.viewer.updateOverlay(state.frameHolder, new window.OpenSeadragon.Rect(
      rect.x / imgW, rect.y / imgW, rect.w / imgW, rect.h / imgW));
  }

  function clamp(value, lo, hi) {
    return Math.max(lo, Math.min(value, hi));
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

      var badge = document.createElementNS(SVG_NS, "text");
      badge.setAttribute("x", c.x);
      badge.setAttribute("y", c.y - radius - 4);
      badge.setAttribute("class", "roigbiv-discovery-badge");
      badge.textContent = String(c.label_id);
      group.appendChild(badge);
    });
    var sheet = document.getElementById(SHEET_ID);
    if (sheet) { applyEditModeClasses(sheet); }
  }

  function wireCentroid(circle, c) {
    circle.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) { return; }
      // OSD's MouseTracker sits on the canvas this overlay lives inside;
      // without this it would pan the image under the drag/click.
      if (state.editBoundaryOn) {
        event.stopPropagation();
        event.preventDefault();
        // A click on a centroid marker in boundary-edit mode always (re)starts
        // a draw session for that label — "edit an existing boundary" is just
        // drawing a new one, which fully replaces it (see boundary_edits.py).
        startDrawForLabel(c);
        return;
      }
      if (!state.editOn) { return; }
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

  // ── boundary layer (preview; manual outlines are delete-able) ────────────

  function drawBoundaries() {
    var group = state.boundaryGroup;
    if (!group) { return; }
    while (group.firstChild) { group.removeChild(group.firstChild); }
    var payload = state.boundaries;
    if (!payload || !payload.contours) { return; }
    Object.keys(payload.contours).forEach(function (label) {
      var entry = payload.contours[label] || {};
      var isManual = entry.origin === "manual";
      var color = isManual ? "#B388FF"
        : entry.origin === "disk_fallback" ? "#FF7A45" : "#3FC1C9";
      (entry.rings || []).forEach(function (ring) {
        if (!ring.length) { return; }
        var path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("d", ringPath(ring));
        path.setAttribute("class", "roigbiv-discovery-boundary"
          + (isManual ? " is-manual" : ""));
        path.setAttribute("stroke", color);
        if (isManual) {
          // Right-click a hand-drawn outline to revert that label to its
          // auto-computed shape. CSS gates pointer-events to boundary-edit
          // mode; this check is the same defense-in-depth wireCentroid uses.
          path.addEventListener("contextmenu", function (event) {
            event.preventDefault();
            event.stopPropagation();
            if (!state.editBoundaryOn) { return; }
            postGesture({ kind: "delete_boundary", label: Number(label) });
          });
        }
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

  // ── boundary drawing (hand-drawn ring, boundary-edit mode) ───────────────
  //
  // A draw session names its target the moment it starts — clicking a
  // centroid marker (wireCentroid) — so, like every other Discovery gesture,
  // there is nothing to select first. click-to-place-vertex, double-click or
  // Enter to close, Escape to cancel.

  function startDrawForLabel(c) {
    if (state.draw) { cancelDraw(); }
    state.draw = { label: c.label_id, points: [[c.y, c.x]] };
    document.addEventListener("keydown", onDrawKeydown);
    renderDrawPreview();
    say("drawing a boundary for cell " + c.label_id
      + " — click to add points, double-click or Enter to close, Escape to cancel");
  }

  function addDrawVertex(y, x) {
    if (!state.draw) { return; }
    state.draw.points.push([y, x]);
    renderDrawPreview();
  }

  function onDrawKeydown(event) {
    if (!state.draw) { return; }
    if (event.key === "Escape") {
      event.preventDefault();
      cancelDraw();
    } else if (event.key === "Enter") {
      event.preventDefault();
      closeDrawRing();
    }
  }

  function cancelDraw() {
    if (!state.draw) { return; }
    state.draw = null;
    document.removeEventListener("keydown", onDrawKeydown);
    renderDrawPreview();
    say("boundary draw cancelled");
  }

  function closeDrawRing() {
    if (!state.draw) { return; }
    document.removeEventListener("keydown", onDrawKeydown);
    if (state.draw.points.length < 3) {
      say("need at least 3 points to close a boundary — draw cancelled");
      state.draw = null;
      renderDrawPreview();
      return;
    }
    var label = state.draw.label;
    var ring = state.draw.points.slice();
    state.draw = null;
    renderDrawPreview();
    postGesture({ kind: "draw_boundary", label: label, ring: ring });
  }

  function renderDrawPreview() {
    var group = state.drawGroup;
    if (!group) { return; }
    while (group.firstChild) { group.removeChild(group.firstChild); }
    if (!state.draw || !state.draw.points.length) { return; }

    var points = state.draw.points;
    var path = document.createElementNS(SVG_NS, "path");
    var d = "";
    for (var i = 0; i < points.length; i++) {
      d += (i === 0 ? "M" : "L") + points[i][1] + " " + points[i][0];
    }
    path.setAttribute("d", d);
    path.setAttribute("class", "roigbiv-discovery-draw-ring");
    group.appendChild(path);

    points.forEach(function (p) {
      var dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", p[1]);
      dot.setAttribute("cy", p[0]);
      dot.setAttribute("r", 2.5);
      dot.setAttribute("class", "roigbiv-discovery-draw-vertex");
      group.appendChild(dot);
    });
  }

  // ── entry points ──────────────────────────────────────────────────────────

  function destroy() {
    state.generation += 1;
    cancelDraw();
    if (window.roigbivDiscoveryPlayer) {
      window.roigbivDiscoveryPlayer.destroy();
    }
    state.frameCanvas = null;
    state.frameHolder = null;
    if (state.viewer) {
      try { state.viewer.destroy(); } catch (e) { /* already gone */ }
    }
    state.viewer = null;
    state.svg = null;
    state.centroidGroup = null;
    state.boundaryGroup = null;
    state.drawGroup = null;
    state.calibCircle = null;
    state.shapes = {};
  }

  function render(config) {
    injectStyles();
    var root = document.getElementById(SHEET_ID);
    if (!root) { return; }
    config = config || {};

    var boundaryEditWas = state.editBoundaryOn;
    state.editOn = !!config.edit_on;
    state.editBoundaryOn = !!config.edit_boundary_on;
    state.diameterPx = config.diameter_px || null;
    state.showCentroids = config.show_centroids !== false;
    state.showBoundaries = config.show_boundaries !== false;
    state.movieOn = !!config.movie_on;
    if (boundaryEditWas && !state.editBoundaryOn) { cancelDraw(); }

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
      // Toggling the movie is a visibility change on a live player, never a
      // rebuild — same reason the edit switches take this path.
      if (window.roigbivDiscoveryPlayer) {
        window.roigbivDiscoveryPlayer.setEnabled(state.movieOn);
      }
      var sheet = document.getElementById(SHEET_ID);
      if (sheet) { applyEditModeClasses(sheet); }
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
