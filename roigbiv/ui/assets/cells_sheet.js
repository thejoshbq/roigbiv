/* The /cells contact sheet: one OpenSeadragon viewer per session, with an SVG
 * overlay carrying the ROI outlines and every edit gesture.
 *
 * Replaces a Plotly heatmap-per-panel. Three things that layer could not do
 * shaped this one:
 *
 *   1. A drag. Plotly reports clicks on chart axes, so "move" had to be two
 *      clicks with a nearest-centroid snap between them, which silently ate any
 *      nudge shorter than the stamp radius.
 *   2. Keeping the view. Applying an edit meant rebuilding the figure, which
 *      reset zoom on every single correction.
 *   3. Staying cheap. A quantised 1024x1024 background is ~1.9 MB of JSON per
 *      panel; here it is a PNG the browser caches by ETag.
 *
 * Server contract: roigbiv/ui/routes/cells_api.py. Gesture semantics live in
 * roigbiv/ui/services/cell_edit_ops.py — this file decides *which* gesture a
 * pointer made, never what it means.
 */
(function () {
  "use strict";

  var SHEET_ID = "roigbiv-cells-sheet";
  var SELECTED_ID = "roigbiv-cells-selected";
  var MSG_ID = "roigbiv-cells-edit-msg";
  var BODY_ID = "roigbiv-cells-body";
  var RAIL_STATE_ID = "roigbiv-cells-rail-state";

  // Below this, a pointerup is a click; above it, a drag. Generous enough to
  // survive the hand tremor of a click on a trackpad.
  var DRAG_THRESHOLD_PX = 3;

  var SVG_NS = "http://www.w3.org/2000/svg";

  var state = {
    fovId: null,
    data: null,          // the last /api/cells payload
    panels: [],          // {index, session, viewer, svg, shapes, el, viewEl}
    selected: null,
    showNumbers: true,
    editOn: false,
    showBoundaries: false,
    focusIndex: null,
    syncing: false,
    osdPromise: null,
    generation: 0,       // bumped on every rebuild; stale async work checks it
  };

  // ── OpenSeadragon loading ────────────────────────────────────────────────
  // Vendored, and served by the ROI editor's /roi-assets route rather than
  // Dash's assets folder — dropping a 400 KB library into assets/ would make
  // every page in the app load it.

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

  // ── Dash bridge ──────────────────────────────────────────────────────────

  function setProps(id, props) {
    var dc = window.dash_clientside;
    if (dc && typeof dc.set_props === "function") {
      try { dc.set_props(id, props); } catch (e) { /* page torn down */ }
    }
  }

  function say(message) {
    setProps(MSG_ID, { children: message || "" });
  }

  function selectCell(gcid) {
    if (state.selected === gcid) { return; }
    state.selected = gcid;
    restyle();
    // The rail, the filmstrip drawer and the header are still server-rendered;
    // this is what wakes them.
    setProps(SELECTED_ID, { data: gcid });
  }

  // ── fetch ────────────────────────────────────────────────────────────────

  function fetchState(fovId, showBoundaries) {
    var qs = showBoundaries ? "?show_boundaries=1" : "";
    return fetch("/api/cells/" + encodeURIComponent(fovId) + qs, {
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
    if (!state.fovId) { return Promise.resolve(); }
    say("working…");
    // show_boundaries rides along so the response's re-serialized state
    // matches the geometry already on screen — see cells_api.py's gesture
    // route, which re-fetches with this rather than trusting the gesture's
    // own (always disk-geometry) reload.
    var payload = {};
    for (var key in gesture) { if (gesture.hasOwnProperty(key)) { payload[key] = gesture[key]; } }
    payload.show_boundaries = state.showBoundaries;
    return fetch("/api/cells/" + encodeURIComponent(state.fovId) + "/gesture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (r) {
      return r.json().catch(function () { return {}; });
    }).then(function (body) {
      say(body.message || body.error || "edit failed");
      if (body.state) {
        // Geometry only. The viewers are never touched, which is what keeps
        // zoom and pan across an edit.
        state.data = body.state;
        state.selected = body.selected_gcid || null;
        redrawOverlays();
        setProps(SELECTED_ID, { data: state.selected });
      }
    }).catch(function (err) {
      say("edit failed: " + err.message);
    });
  }

  // ── panel construction ───────────────────────────────────────────────────

  function buildSheet(root) {
    root.innerHTML = "";

    var sheet = document.createElement("div");
    sheet.className = "roigbiv-cells-sheet";
    root.appendChild(sheet);

    var note = document.createElement("div");
    note.className = "text-muted small mt-1";
    note.textContent = "Panels share one zoom window, but sessions are not "
      + "co-registered — the same cell sits tens of pixels apart between them.";
    root.appendChild(note);

    state.panels = state.data.sessions.map(function (session, i) {
      return buildPanel(sheet, session, i);
    });
  }

  function buildPanel(sheet, session, i) {
    var el = document.createElement("div");
    el.className = "roigbiv-cells-panel";
    el.dataset.index = String(i);

    var head = document.createElement("div");
    head.className = "roigbiv-cells-panel-head";
    head.title = "click to focus this session";
    head.appendChild(spanWith("roigbiv-track-seq me-1", (i + 1) + "."));
    head.appendChild(spanWith("font-monospace", session.short_label));
    head.appendChild(spanWith("text-muted small ms-2", session.label));
    head.appendChild(spanWith("text-muted small ms-auto", session.counts));
    head.addEventListener("click", function () { toggleFocus(i); });
    el.appendChild(head);

    if (session.stale) {
      var warn = document.createElement("div");
      warn.className = "alert alert-warning py-1 px-2 small mt-1 mb-0";
      warn.textContent = "This session's registry_match.json names a different "
        + "session than the registry does — the counts above come from the "
        + "registry. Re-run tracking to reconcile them.";
      el.appendChild(warn);
    }

    var viewEl = document.createElement("div");
    viewEl.className = "roigbiv-cells-panel-view";
    // The panel takes the projection's own shape. The Plotly version pinned a
    // viewport-height box and let the square image letterbox inside it.
    if (session.width && session.height) {
      viewEl.style.aspectRatio = session.width + " / " + session.height;
    }
    el.appendChild(viewEl);
    sheet.appendChild(el);

    var panel = {
      index: i, session: session, el: el, viewEl: viewEl,
      viewer: null, svg: null, shapes: {},
    };
    mountViewer(panel);
    return panel;
  }

  function spanWith(className, text) {
    var span = document.createElement("span");
    span.className = className;
    span.textContent = text || "";
    return span;
  }

  function mountViewer(panel) {
    var generation = state.generation;
    var viewer = window.OpenSeadragon({
      element: panel.viewEl,
      tileSources: { type: "image", url: panel.session.image_url },
      backgroundColor: "transparent",
      showNavigationControl: false,
      showNavigator: false,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
      // Matches the ROI editor: these are 1024px frames being inspected at the
      // pixel level, so nearest-neighbour and deep zoom both matter.
      imageSmoothingEnabled: false,
      maxZoomLevel: 40,
      minZoomLevel: 0.5,
      visibilityRatio: 0.6,
      animationTime: 0.4,
    });
    panel.viewer = viewer;

    viewer.addHandler("open", function () {
      if (generation !== state.generation) { return; }
      attachOverlay(panel);
      drawOverlay(panel);
    });

    // A click that is not a drag, landing on the background — the shapes stop
    // propagation before OSD ever sees them, so anything arriving here missed
    // every cell. `quick` is OSD's own click/pan discrimination.
    viewer.addHandler("canvas-click", function (event) {
      if (!event.quick || !state.editOn) { return; }
      var point = viewer.viewport.viewerElementToImageCoordinates(event.position);
      postGesture({
        kind: "add",
        stem: panel.session.stem,
        y: point.y,
        x: point.x,
        selected_gcid: state.selected,
      });
    });

    viewer.addHandler("zoom", function () { syncFrom(panel); });
    viewer.addHandler("pan", function () { syncFrom(panel); });
  }

  function attachOverlay(panel) {
    var session = panel.session;
    var width = session.width || 1;
    var height = session.height || 1;

    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.classList.add("roigbiv-cells-overlay");
    panel.svg = svg;

    var holder = document.createElement("div");
    holder.className = "roigbiv-cells-overlay-holder";
    holder.appendChild(svg);

    // OSD's viewport coordinates put the image at x∈[0,1], y∈[0,H/W]; the
    // overlay tracks it through zoom and pan for free.
    panel.viewer.addOverlay({
      element: holder,
      location: new window.OpenSeadragon.Rect(0, 0, 1, height / width),
    });
  }

  // ── overlay drawing ──────────────────────────────────────────────────────

  function redrawOverlays() {
    state.panels.forEach(function (panel) {
      var fresh = state.data.sessions[panel.index];
      if (fresh) { panel.session = fresh; }
      var head = panel.el.querySelector(".roigbiv-cells-panel-head .ms-auto");
      if (head && fresh) { head.textContent = fresh.counts; }
      if (panel.svg) { drawOverlay(panel); }
    });
  }

  function drawOverlay(panel) {
    var svg = panel.svg;
    if (!svg) { return; }
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    panel.shapes = {};

    var palette = (state.data && state.data.palette) || {};
    panel.session.rois.forEach(function (roi) {
      var color = palette[roi.match_status] || "rgba(160,160,160,0.8)";
      var group = document.createElementNS(SVG_NS, "g");
      group.setAttribute("class", shapeClass(roi));
      group.dataset.labelId = String(roi.label_id);
      group.dataset.gcid = roi.gcid || "";

      roi.contours.forEach(function (ring) {
        var path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("d", ringPath(ring));
        path.setAttribute("stroke", color);
        // A translucent fill is the click target — asking a researcher to hit a
        // 3px ring forty times a session is most of what "too many clicks"
        // meant. Ghosts stay unfilled: they own no footprint here.
        path.setAttribute("fill", roi.ghost ? "none" : fillFrom(color));
        group.appendChild(path);
      });

      // A small fixed-size location marker, independent of the outline above
      // (disk stamp or seeded boundary, whichever `boundaries` has picked) —
      // that outline stays exactly what it was, this just says "here". Ghosts
      // are skipped: they own no footprint here, and a solid dot would claim
      // a detection that does not exist.
      if (!roi.ghost) {
        var dot = document.createElementNS(SVG_NS, "circle");
        dot.setAttribute("cx", roi.centroid[1]);
        dot.setAttribute("cy", roi.centroid[0]);
        dot.setAttribute("r", 5);
        dot.setAttribute("fill", color);
        dot.setAttribute("class", "roigbiv-cells-centroid-dot");
        group.appendChild(dot);
      }

      if (roi.cell_index != null) {
        var text = document.createElementNS(SVG_NS, "text");
        text.setAttribute("x", roi.centroid[1]);
        text.setAttribute("y", roi.centroid[0]);
        text.setAttribute("fill", color);
        text.setAttribute("class", "roigbiv-cells-badge");
        text.textContent = "#" + roi.cell_index;
        group.appendChild(text);
      }

      wireShape(panel, group, roi);
      svg.appendChild(group);
      panel.shapes[roi.label_id] = group;
    });

    restylePanel(panel);
  }

  function ringPath(ring) {
    var d = "";
    for (var i = 0; i < ring.length; i++) {
      d += (i === 0 ? "M" : "L") + ring[i][0] + " " + ring[i][1];
    }
    return d + "Z";
  }

  function fillFrom(color) {
    var match = /rgba?\(([^)]+)\)/.exec(color || "");
    if (!match) { return "rgba(0,229,255,0.12)"; }
    var parts = match[1].split(",");
    return "rgba(" + parts[0] + "," + parts[1] + "," + parts[2] + ",0.14)";
  }

  function shapeClass(roi) {
    var classes = ["roigbiv-roi", "status-" + (roi.match_status || "unknown")];
    if (roi.ghost) { classes.push("is-ghost"); }
    return classes.join(" ");
  }

  function restyle() {
    state.panels.forEach(restylePanel);
  }

  function restylePanel(panel) {
    if (!panel.svg) { return; }
    Object.keys(panel.shapes).forEach(function (labelId) {
      var group = panel.shapes[labelId];
      var lit = !!state.selected && group.dataset.gcid === state.selected;
      group.classList.toggle("is-selected", lit);
      // A selected cell keeps its number with the switch off: the switch
      // controls the resting field of numbers, and hiding the one just clicked
      // would leave no identity cue at all.
      group.classList.toggle("hide-badge", !state.showNumbers && !lit);
    });
    panel.el.classList.toggle("roigbiv-cells-panel-editing", state.editOn);
  }

  // ── gestures ─────────────────────────────────────────────────────────────

  function wireShape(panel, group, roi) {
    group.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) { return; }
      // OSD's MouseTracker sits on the canvas this overlay lives inside;
      // without this it would pan the image under the drag.
      event.stopPropagation();
      event.preventDefault();

      // Ctrl-click a ghost: "this cell is here". The ghost names its own cell,
      // so no prior selection is needed — that is the click this saves over
      // select-then-shift-click. The server decides whether that means placing
      // a centroid or adopting an outline already under the ghost.
      if (event.ctrlKey && state.editOn && roi.ghost) {
        postGesture({
          kind: "confirm",
          stem: panel.session.stem,
          y: roi.centroid[0],
          x: roi.centroid[1],
          selected_gcid: roi.gcid,
        });
        return;
      }

      if (event.shiftKey && state.editOn && !roi.ghost) {
        postGesture({
          kind: "link",
          stem: panel.session.stem,
          label: roi.label_id,
          selected_gcid: state.selected,
        });
        return;
      }

      // A ghost has no footprint to drag, and a read-only page has none
      // either — both still select.
      if (!state.editOn || roi.ghost) {
        selectCell(roi.gcid);
        return;
      }
      beginDrag(panel, group, roi, event);
    });

    group.addEventListener("contextmenu", function (event) {
      event.preventDefault();
      event.stopPropagation();
      if (!state.editOn || roi.ghost) { return; }
      postGesture({
        kind: "delete",
        stem: panel.session.stem,
        label: roi.label_id,
        selected_gcid: state.selected,
      });
    });
  }

  function beginDrag(panel, group, roi, downEvent) {
    var origin = { x: downEvent.clientX, y: downEvent.clientY };
    var moved = false;
    var pointerId = downEvent.pointerId;

    try { group.setPointerCapture(pointerId); } catch (e) { /* mouse */ }
    group.classList.add("is-dragging");

    function onMove(event) {
      var dx = event.clientX - origin.x;
      var dy = event.clientY - origin.y;
      if (!moved && Math.abs(dx) + Math.abs(dy) < DRAG_THRESHOLD_PX) { return; }
      moved = true;
      // Live feedback in screen pixels — the shape follows the pointer without
      // any coordinate conversion until the drop.
      group.setAttribute("transform", screenDeltaToImage(panel, dx, dy));
    }

    function onUp(event) {
      group.removeEventListener("pointermove", onMove);
      group.removeEventListener("pointerup", onUp);
      group.removeEventListener("pointercancel", onUp);
      group.classList.remove("is-dragging");
      group.removeAttribute("transform");
      try { group.releasePointerCapture(pointerId); } catch (e) { /* mouse */ }

      if (!moved) {
        selectCell(roi.gcid);
        return;
      }
      var point = imagePointOf(panel, event);
      // The exact drop coordinate. Nothing snaps it to a centroid, which is
      // what makes a two-pixel correction land where it was put.
      postGesture({
        kind: "move",
        stem: panel.session.stem,
        label: roi.label_id,
        y: point.y,
        x: point.x,
        selected_gcid: state.selected,
      });
    }

    group.addEventListener("pointermove", onMove);
    group.addEventListener("pointerup", onUp);
    group.addEventListener("pointercancel", onUp);
  }

  function screenDeltaToImage(panel, dx, dy) {
    var zoom = panel.viewer.viewport.getZoom(true);
    var containerWidth = panel.viewer.viewport.getContainerSize().x;
    var imageWidth = panel.session.width || 1;
    var scale = (containerWidth * zoom) / imageWidth;   // screen px per image px
    if (!scale) { return ""; }
    return "translate(" + (dx / scale) + " " + (dy / scale) + ")";
  }

  function imagePointOf(panel, event) {
    var rect = panel.viewer.element.getBoundingClientRect();
    return panel.viewer.viewport.viewerElementToImageCoordinates(
      new window.OpenSeadragon.Point(
        event.clientX - rect.left, event.clientY - rect.top));
  }

  // ── viewport sync ────────────────────────────────────────────────────────

  function syncFrom(source) {
    if (state.syncing || !source.viewer.isOpen()) { return; }
    state.syncing = true;
    try {
      var center = source.viewer.viewport.getCenter(true);
      var zoom = source.viewer.viewport.getZoom(true);
      state.panels.forEach(function (panel) {
        if (panel === source || !panel.viewer || !panel.viewer.isOpen()) {
          return;
        }
        panel.viewer.viewport.zoomTo(zoom, null, true);
        panel.viewer.viewport.panTo(center, true);
      });
    } finally {
      // Each zoomTo/panTo re-emits zoom/pan on its own viewer; releasing on the
      // next tick lets that settle instead of ping-ponging between panels.
      window.setTimeout(function () { state.syncing = false; }, 0);
    }
  }

  // ── focus ────────────────────────────────────────────────────────────────

  function toggleFocus(index) {
    state.focusIndex = (state.focusIndex === index) ? null : index;
    applyFocus();
  }

  function applyFocus() {
    state.panels.forEach(function (panel) {
      panel.el.classList.toggle(
        "roigbiv-cells-panel-focus", state.focusIndex === panel.index);
    });
    // The grid reflows first; OSD only re-reads its container on demand.
    window.setTimeout(function () {
      state.panels.forEach(function (panel) {
        if (panel.viewer && panel.viewer.isOpen()) {
          panel.viewer.viewport.resize();
          panel.viewer.viewport.applyConstraints();
          panel.viewer.forceRedraw();
        }
      });
    }, 60);
  }

  // ── keyboard ─────────────────────────────────────────────────────────────

  document.addEventListener("keydown", function (event) {
    var tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") { return; }

    // Ahead of the FOV guard: the rail is page furniture, not sheet state, and
    // hiding it before picking a FOV is a reasonable thing to want. The class
    // on the wrapper is the rendered truth, so read it rather than mirroring it.
    if (event.key === "[" && !event.ctrlKey && !event.metaKey && !event.altKey) {
      var body = document.getElementById(BODY_ID);
      if (!body) { return; }
      event.preventDefault();
      setProps(RAIL_STATE_ID, {
        data: !body.classList.contains("is-collapsed"),
      });
      return;
    }

    if (!state.fovId || !state.panels.length) { return; }

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      if (!state.editOn) { return; }
      event.preventDefault();
      postGesture({ kind: "undo", selected_gcid: state.selected });
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) { return; }

    if (event.key === "Delete" || event.key === "Backspace") {
      if (!state.editOn || !state.selected) { return; }
      event.preventDefault();
      deleteSelectedInFocus();
      return;
    }
    if (event.key === "Escape" && state.focusIndex !== null) {
      event.preventDefault();
      state.focusIndex = null;
      applyFocus();
    }
  });

  function deleteSelectedInFocus() {
    // Ambiguous by nature — the selected cell exists in several sessions. The
    // focused panel disambiguates; without one, the keyboard has no way to say
    // *which* session's centroid to drop, so it declines rather than guessing.
    if (state.focusIndex === null) {
      say("focus a session first (click its header) — Delete acts on one session");
      return;
    }
    var panel = state.panels[state.focusIndex];
    if (!panel) { return; }
    var roi = panel.session.rois.filter(function (r) {
      return r.gcid === state.selected && !r.ghost;
    })[0];
    if (!roi) {
      say("that cell has no centroid in the focused session");
      return;
    }
    postGesture({
      kind: "delete",
      stem: panel.session.stem,
      label: roi.label_id,
      selected_gcid: state.selected,
    });
  }

  // ── entry point ──────────────────────────────────────────────────────────

  function destroy() {
    state.generation += 1;
    state.panels.forEach(function (panel) {
      if (panel.viewer) {
        try { panel.viewer.destroy(); } catch (e) { /* already gone */ }
      }
    });
    state.panels = [];
  }

  function render(config) {
    var root = document.getElementById(SHEET_ID);
    if (!root) { return; }
    config = config || {};

    state.showNumbers = config.show_numbers !== false;
    state.editOn = !!config.edit_on;
    var showBoundaries = !!config.show_boundaries;

    if (!config.fov_id) {
      destroy();
      state.fovId = null;
      state.showBoundaries = showBoundaries;
      root.innerHTML = "";
      return;
    }

    // Selection and the display switches never rebuild anything: repainting
    // the sheet for them is what used to throw the zoom away.
    if (config.fov_id === state.fovId && state.panels.length) {
      if (config.selected_gcid !== undefined
          && config.selected_gcid !== state.selected) {
        state.selected = config.selected_gcid || null;
      }
      if (showBoundaries !== state.showBoundaries) {
        // A geometry-track change needs fresh contours from the server, but
        // still no viewer teardown — same contract as a gesture's response.
        reloadGeometry(config.fov_id, showBoundaries);
        return;
      }
      restyle();
      return;
    }

    destroy();
    state.fovId = config.fov_id;
    state.selected = config.selected_gcid || null;
    state.showBoundaries = showBoundaries;
    state.focusIndex = null;
    var generation = state.generation;

    root.innerHTML = '<div class="text-muted small p-3">Loading sessions…</div>';

    Promise.all([ensureOSD(), fetchState(config.fov_id, showBoundaries)])
      .then(function (results) {
        if (generation !== state.generation || state.fovId !== config.fov_id) {
          return;   // a different FOV was picked while this was in flight
        }
        state.data = results[1];
        buildSheet(root);
      })
      .catch(function (err) {
        if (generation !== state.generation) { return; }
        root.innerHTML = "";
        var alert = document.createElement("div");
        alert.className = "alert alert-danger";
        alert.textContent = "Could not load this FOV's sessions: " + err.message;
        root.appendChild(alert);
      });
  }

  function reloadGeometry(fovId, showBoundaries) {
    var generation = state.generation;
    state.showBoundaries = showBoundaries;
    fetchState(fovId, showBoundaries)
      .then(function (data) {
        // A further toggle, a FOV change, or a destroy() could all have
        // landed while this was in flight; only the most recent request's
        // response is allowed to paint.
        if (generation !== state.generation || state.fovId !== fovId
            || state.showBoundaries !== showBoundaries) {
          return;
        }
        state.data = data;
        redrawOverlays();
      })
      .catch(function (err) {
        if (generation !== state.generation || state.fovId !== fovId) { return; }
        say("could not load boundaries: " + err.message);
      });
  }

  function undo() {
    if (!state.editOn) { return; }
    postGesture({ kind: "undo", selected_gcid: state.selected });
  }

  function resize() {
    state.panels.forEach(function (panel) {
      if (panel.viewer && panel.viewer.isOpen()) {
        panel.viewer.viewport.resize();
        panel.viewer.forceRedraw();
      }
    });
  }

  window.roigbivCells = {
    render: render, destroy: destroy, undo: undo, resize: resize,
  };
})();
