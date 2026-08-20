/* Movie playback for the /discovery viewer: a ring buffer, a render loop, and
 * a control bar.
 *
 * Deliberately knows nothing about OpenSeadragon. It is handed a canvas to
 * paint, a mount point for its controls, and a `getRect()` the caller
 * implements against whatever viewer it owns — assets/discovery_sheet.js wires
 * those to an OSD viewport. Keeping that seam means the zoom-tracking logic
 * (which is entirely about OSD's coordinate systems) stays in the file that
 * already handles OSD's coordinate systems, and the transport logic here stays
 * testable by inspection.
 *
 * Server contract: the /movie/meta and /movie/chunk routes in
 * roigbiv/ui/routes/discovery_api.py.
 *
 * The frame-exactness contract
 * ----------------------------
 * Two rules, and everything else here follows from them:
 *
 *   1. The scrubber sets a *target* frame; the loop draws that frame or draws
 *      nothing. It never draws a nearby frame it happens to have.
 *   2. On underrun, playback holds and does NOT accumulate elapsed time. A
 *      player that keeps counting while it waits jumps forward when the data
 *      lands, which is the frame-skipping this whole design exists to avoid.
 *
 * Styles are injected here for the same reason discovery_sheet.js injects its
 * own — see that file's header.
 */
(function () {
  "use strict";

  // Frames to keep behind and ahead of the playhead. Ahead dominates because
  // playback and forward scrubbing are the common motions; behind exists so a
  // small backward correction does not force a refetch.
  var LOOKAHEAD = 96;
  var LOOKBEHIND = 24;

  // Ring-buffer ceiling in bytes, not frames: at ds=1 over a full field a
  // frame is 256 KB and a frame-counted buffer would balloon, while zoomed in
  // it is 16 KB and the same count would be pointlessly small.
  var MAX_BUFFER_BYTES = 48 << 20;

  // Frames per request. Held under the server's own MAX_COUNT; meta reports
  // the real ceiling and we take the smaller.
  var CHUNK = 32;
  var MAX_INFLIGHT = 2;

  // Enough frames to redraw immediately after a zoom or pan, before the wider
  // prefetch for the new rect has had time to arrive.
  var URGENT_CHUNK = 4;

  var SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

  var state = null;

  // ── styles ────────────────────────────────────────────────────────────────

  function injectStyles() {
    if (document.getElementById("roigbiv-player-styles")) { return; }
    var style = document.createElement("style");
    style.id = "roigbiv-player-styles";
    style.textContent = [
      ".roigbiv-player { display: flex; align-items: center; gap: 8px;",
      "  flex-wrap: wrap; margin-top: 6px; font-size: var(--fs-micro); }",
      ".roigbiv-player.is-hidden { display: none; }",
      ".roigbiv-player button { background: var(--surface-2);",
      "  border: 1px solid var(--border-control); border-radius: var(--radius);",
      "  color: inherit; padding: 1px 8px; line-height: 1.5; cursor: pointer; }",
      ".roigbiv-player button:hover { background: var(--accent-soft);",
      "  border-color: var(--accent); color: var(--accent); }",
      ".roigbiv-player button:disabled { opacity: 0.4; cursor: default; }",
      ".roigbiv-player input[type=range] { flex: 1 1 160px; min-width: 120px; }",
      ".roigbiv-player-scrub { flex: 1 1 260px; }",
      ".roigbiv-player-readout { font-family: var(--roigbiv-font-mono, monospace);",
      "  white-space: nowrap; color: var(--text-muted); }",
      ".roigbiv-player-stall { color: var(--warn); white-space: nowrap; }",
      ".roigbiv-player-stall.is-hidden { visibility: hidden; }",
      ".roigbiv-player-adjust { display: flex; align-items: center; gap: 4px;",
      "  white-space: nowrap; color: var(--text-muted); }",
      ".roigbiv-player-adjust input[type=range] { flex: 0 0 68px; min-width: 68px; }",
      // The frame canvas is a sibling under the same OSD overlay container as
      // the SVG; pointer-events must stay off it or it would eat the gestures
      // the SVG layer above it is there to receive.
      ".roigbiv-player-canvas { width: 100%; height: 100%; display: block;",
      "  pointer-events: none; image-rendering: pixelated;",
      "  image-rendering: crisp-edges; }",
      ".roigbiv-player-canvas.is-hidden { display: none; }",
    ].join("\n");
    document.head.appendChild(style);
  }

  // ── ring buffer ───────────────────────────────────────────────────────────
  //
  // Keyed by frame index within one "generation". A generation is a
  // (stem, rect, ds) triple — change any of them and every buffered frame is
  // the wrong shape, so the whole buffer is dropped rather than reconciled.

  function newBuffer() {
    return { gen: 0, frames: new Map(), bytes: 0, cols: 0, rows: 0 };
  }

  function bufferClear(buf) {
    buf.frames.clear();
    buf.bytes = 0;
  }

  function bufferPut(buf, index, bytes) {
    if (buf.frames.has(index)) { return; }
    buf.frames.set(index, bytes);
    buf.bytes += bytes.length;
  }

  // Evicts whatever is furthest from the playhead. Distance rather than
  // insertion order: after a scrub backwards, the most recently fetched frames
  // are the ones worth keeping, and an LRU would throw exactly those away.
  function bufferTrim(buf, center) {
    if (buf.bytes <= MAX_BUFFER_BYTES) { return; }
    var keys = Array.from(buf.frames.keys());
    keys.sort(function (a, b) {
      return Math.abs(b - center) - Math.abs(a - center);
    });
    for (var i = 0; i < keys.length && buf.bytes > MAX_BUFFER_BYTES; i++) {
      var bytes = buf.frames.get(keys[i]);
      buf.frames.delete(keys[i]);
      buf.bytes -= bytes.length;
    }
  }

  // ── fetching ──────────────────────────────────────────────────────────────

  function rectHeader(rect) {
    return rect.x + "," + rect.y + "," + rect.w + "," + rect.h;
  }

  // The next run of unbuffered frames at or after `from`, capped at `limit`.
  // Returns null when everything in the window is already resident.
  function nextGap(buf, from, until, limit) {
    for (var i = from; i <= until; i++) {
      if (buf.frames.has(i)) { continue; }
      var count = 0;
      while (count < limit && i + count <= until
             && !buf.frames.has(i + count)) {
        count++;
      }
      return { start: i, count: count };
    }
    return null;
  }

  function pump() {
    if (!state || !state.meta || !state.meta.available) { return; }
    var buf = state.buffer;
    if (state.inflight >= MAX_INFLIGHT) { return; }

    var center = Math.round(state.target);
    var last = state.meta.n_frames - 1;
    var chunk = Math.min(CHUNK, state.meta.max_count || CHUNK);

    // Ahead of the playhead first, then behind it: a stalled loop is waiting on
    // a frame ahead, and a backfill request queued in front of it would extend
    // the stall.
    var gap = nextGap(buf, center, Math.min(last, center + LOOKAHEAD), chunk);
    if (gap === null) {
      gap = nextGap(buf, Math.max(0, center - LOOKBEHIND), Math.max(0, center - 1),
                    chunk);
    }
    if (gap === null) { return; }
    request(gap.start, gap.count, buf.gen);
  }

  function request(start, count, gen) {
    var rect = state.rect;
    var ds = state.ds;
    var url = "/api/discovery/" + encodeURIComponent(state.stem)
      + "/movie/chunk?start=" + start + "&count=" + count
      + "&x=" + rect.x + "&y=" + rect.y + "&w=" + rect.w + "&h=" + rect.h
      + "&ds=" + ds;

    state.inflight += 1;
    var controller = new AbortController();
    state.controllers.add(controller);

    fetch(url, { signal: controller.signal })
      .then(function (r) {
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        // Trust the response, not the request: the server clamps, and reading
        // the echo back is what keeps a trimmed request from writing
        // wrong-shaped frames into the buffer.
        var served = {
          start: parseInt(r.headers.get("X-Movie-Start"), 10),
          count: parseInt(r.headers.get("X-Movie-Count"), 10),
          cols: parseInt(r.headers.get("X-Movie-Cols"), 10),
          rows: parseInt(r.headers.get("X-Movie-Rows"), 10),
          rect: r.headers.get("X-Movie-Rect"),
          ds: parseInt(r.headers.get("X-Movie-Ds"), 10),
        };
        return r.arrayBuffer().then(function (bytes) {
          return { served: served, bytes: new Uint8Array(bytes) };
        });
      })
      .then(function (payload) {
        if (!state || state.buffer.gen !== gen) { return; }   // stale generation
        var served = payload.served;
        if (served.rect !== rectHeader(state.rect) || served.ds !== state.ds) {
          return;   // server clamped into a different crop than we now want
        }
        var per = served.cols * served.rows;
        state.buffer.cols = served.cols;
        state.buffer.rows = served.rows;
        for (var i = 0; i < served.count; i++) {
          // Copied, not subarray'd: a subarray keeps the whole chunk's
          // ArrayBuffer alive, so evicting 31 frames of a 32-frame chunk would
          // free nothing while the byte counter happily reported the drop.
          bufferPut(state.buffer, served.start + i,
                    new Uint8Array(payload.bytes.subarray(i * per, (i + 1) * per)));
        }
        bufferTrim(state.buffer, Math.round(state.target));
      })
      .catch(function (err) {
        if (err.name === "AbortError") { return; }
        say("movie: " + err.message);
      })
      .then(function () {
        state.controllers.delete(controller);
        if (state) { state.inflight -= 1; }
      });
  }

  function abortAll() {
    state.controllers.forEach(function (c) {
      try { c.abort(); } catch (e) { /* already settled */ }
    });
    state.controllers.clear();
  }

  // ── drawing ───────────────────────────────────────────────────────────────

  function draw(index) {
    var buf = state.buffer;
    var gray = buf.frames.get(index);
    if (!gray || !buf.cols || !buf.rows) { return false; }

    var canvas = state.canvas;
    if (canvas.width !== buf.cols || canvas.height !== buf.rows) {
      canvas.width = buf.cols;
      canvas.height = buf.rows;
      state.image = null;
    }
    var ctx = state.ctx || (state.ctx = canvas.getContext("2d"));
    if (!state.image || state.image.width !== buf.cols
        || state.image.height !== buf.rows) {
      state.image = ctx.createImageData(buf.cols, buf.rows);
      // Alpha is opaque for every pixel and never changes, so it is written
      // once here rather than once per frame.
      var data = state.image.data;
      for (var i = 3; i < data.length; i += 4) { data[i] = 255; }
    }
    var out = state.image.data;
    for (var p = 0, o = 0; p < gray.length; p++, o += 4) {
      out[o] = out[o + 1] = out[o + 2] = gray[p];
    }
    ctx.putImageData(state.image, 0, 0);
    state.drawn = index;
    // Only now is the new crop's imagery actually on screen. The caller places
    // the canvas here rather than when the crop was chosen, so a zoom cannot
    // stretch the previous crop's pixels over the new rectangle while the
    // first chunk is still in flight.
    if (state.rectPending) {
      state.rectPending = false;
      state.onRectApplied(state.rect, state.ds);
    }
    return true;
  }

  // ── render loop ───────────────────────────────────────────────────────────

  function tick(now) {
    if (!state) { return; }
    state.raf = window.requestAnimationFrame(tick);
    if (!state.meta || !state.meta.available) { return; }

    if (state.playing) {
      var last = state.meta.n_frames - 1;
      var next = Math.floor(state.target) + 1;
      var elapsed = (now - state.lastTs) / 1000;
      var due = elapsed >= 1 / (state.fps * state.speed);

      if (state.target >= last) {
        if (state.loop) { setTarget(0); } else { setPlaying(false); }
      } else if (due) {
        if (state.buffer.frames.has(next)) {
          state.target = next;
          setStalled(false);
          state.lastTs = now;
        } else {
          // Rule 2: hold, and reset the clock so the wait does not become
          // credit that gets spent skipping frames once data arrives.
          setStalled(true);
          state.lastTs = now;
        }
      }
    }

    var want = Math.round(state.target);
    if (want !== state.drawn) {
      if (draw(want)) { setStalled(false); }
      else if (!state.playing) { setStalled(true); }
    }
    pump();
    syncReadout();
  }

  // ── controls ──────────────────────────────────────────────────────────────

  function buildControls(mount) {
    var bar = document.createElement("div");
    bar.className = "roigbiv-player is-hidden";

    var play = button("▶", function () { setPlaying(!state.playing); });
    var back = button("◀❘", function () { step(-1); });
    var fwd = button("❘▶", function () { step(1); });

    var scrub = document.createElement("input");
    scrub.type = "range";
    scrub.min = "0";
    scrub.max = "0";
    scrub.step = "1";
    scrub.value = "0";
    scrub.className = "roigbiv-player-scrub";
    scrub.setAttribute("aria-label", "Frame");
    // `input`, not `change`: the frame under the thumb must follow the drag,
    // not appear when it is released.
    scrub.addEventListener("input", function () {
      setTarget(parseInt(scrub.value, 10));
    });

    var speed = document.createElement("select");
    SPEEDS.forEach(function (s) {
      var opt = document.createElement("option");
      opt.value = String(s);
      opt.textContent = s + "×";
      if (s === 1) { opt.selected = true; }
      speed.appendChild(opt);
    });
    speed.addEventListener("change", function () {
      state.speed = parseFloat(speed.value) || 1;
    });

    var readout = document.createElement("span");
    readout.className = "roigbiv-player-readout";

    var stall = document.createElement("span");
    stall.className = "roigbiv-player-stall is-hidden";
    stall.textContent = "buffering…";

    bar.appendChild(back);
    bar.appendChild(play);
    bar.appendChild(fwd);
    bar.appendChild(scrub);
    bar.appendChild(readout);
    bar.appendChild(speed);
    bar.appendChild(adjust("Brightness", "brightness", 0.2, 3, 1));
    bar.appendChild(adjust("Contrast", "contrast", 0.2, 3, 1));
    bar.appendChild(stall);
    mount.appendChild(bar);

    return { bar: bar, play: play, scrub: scrub, readout: readout, stall: stall };
  }

  function button(label, onClick) {
    var b = document.createElement("button");
    b.type = "button";
    b.innerHTML = label;
    b.addEventListener("click", onClick);
    return b;
  }

  // Brightness and contrast ride a CSS filter on the canvas rather than a
  // different server-side window: the pixels already in the ring buffer stay
  // valid, so adjusting is instant and costs no refetch.
  function adjust(label, prop, min, max, initial) {
    var wrap = document.createElement("span");
    wrap.className = "roigbiv-player-adjust";
    var text = document.createElement("span");
    text.textContent = label.slice(0, 1);
    text.title = label;
    var slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(min);
    slider.max = String(max);
    slider.step = "0.05";
    slider.value = String(initial);
    slider.setAttribute("aria-label", label);
    slider.addEventListener("input", function () {
      state.filters[prop] = parseFloat(slider.value);
      applyFilters();
    });
    wrap.appendChild(text);
    wrap.appendChild(slider);
    return wrap;
  }

  function applyFilters() {
    state.canvas.style.filter =
      "brightness(" + state.filters.brightness + ") "
      + "contrast(" + state.filters.contrast + ")";
  }

  function syncReadout() {
    var ui = state.ui;
    var frame = Math.round(state.target);
    if (ui.scrub.value !== String(frame) && document.activeElement !== ui.scrub) {
      ui.scrub.value = String(frame);
    }
    var secs = state.fps > 0 ? (frame / state.fps).toFixed(1) + "s" : "";
    ui.readout.textContent = frame + " / " + (state.meta.n_frames - 1)
      + (secs ? "  " + secs : "");
  }

  function setStalled(on) {
    if (state.stalled === on) { return; }
    state.stalled = on;
    state.ui.stall.classList.toggle("is-hidden", !on);
  }

  function setPlaying(on) {
    state.playing = !!on && !!state.meta && state.meta.available;
    state.lastTs = window.performance.now();
    state.ui.play.innerHTML = state.playing ? "❚❚" : "▶";
    if (!state.playing) { setStalled(false); }
  }

  function setTarget(frame) {
    if (!state.meta || !state.meta.available) { return; }
    state.target = Math.max(0, Math.min(frame, state.meta.n_frames - 1));
    state.lastTs = window.performance.now();
  }

  function step(delta) {
    setPlaying(false);
    setTarget(Math.round(state.target) + delta);
  }

  function say(message) {
    var el = document.getElementById("roigbiv-discovery-edit-msg");
    if (el) { el.textContent = message || ""; }
  }

  // ── keyboard ──────────────────────────────────────────────────────────────

  function onKeydown(event) {
    if (!state || !state.enabled || !state.meta || !state.meta.available) {
      return;
    }
    // Never steal a key from a form field, and never from an in-progress
    // boundary draw — that gesture already owns Enter and Escape, and its own
    // handler must see the rest of the keyboard uninterrupted.
    var active = document.activeElement;
    if (active && /^(INPUT|SELECT|TEXTAREA)$/.test(active.tagName)) { return; }
    if (state.isDrawing && state.isDrawing()) { return; }

    var jump = event.shiftKey ? 10 : 1;
    if (event.key === " ") {
      event.preventDefault();
      setPlaying(!state.playing);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      step(-jump);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      step(jump);
    }
  }

  // ── entry points ──────────────────────────────────────────────────────────

  /* mount({canvas, controlsMount, getRect, isDrawing}) — install the player
   * over a caller-owned canvas. Idempotent per canvas; call destroy() first to
   * move to a different one. */
  function mount(options) {
    injectStyles();
    destroy();
    state = {
      stem: null,
      enabled: false,
      meta: null,
      canvas: options.canvas,
      ctx: null,
      image: null,
      getRect: options.getRect,
      isDrawing: options.isDrawing || function () { return false; },
      onRectApplied: options.onRectApplied || function () {},
      rectPending: false,
      rect: { x: 0, y: 0, w: 1, h: 1 },
      ds: 1,
      buffer: newBuffer(),
      controllers: new Set(),
      inflight: 0,
      target: 0,
      drawn: -1,
      playing: false,
      stalled: false,
      loop: true,
      speed: 1,
      fps: 7.5,
      lastTs: 0,
      filters: { brightness: 1, contrast: 1 },
      raf: null,
      ui: null,
    };
    state.ui = buildControls(options.controlsMount);
    applyFilters();
    document.addEventListener("keydown", onKeydown);
    state.raf = window.requestAnimationFrame(tick);
  }

  /* setStem(stem) — load (or clear) a FOV's movie metadata. */
  function setStem(stem) {
    if (!state) { return Promise.resolve(null); }
    state.stem = stem;
    state.meta = null;
    state.target = 0;
    state.drawn = -1;
    setPlaying(false);
    resetBuffer();
    if (!stem) { return Promise.resolve(null); }

    var mine = stem;
    return fetch("/api/discovery/" + encodeURIComponent(stem) + "/movie/meta")
      .then(function (r) { return r.json(); })
      .then(function (meta) {
        if (!state || state.stem !== mine) { return null; }
        state.meta = meta;
        if (meta.available) {
          state.fps = meta.fps > 0 ? meta.fps : 7.5;
          state.ui.scrub.max = String(Math.max(0, meta.n_frames - 1));
        }
        syncVisibility();
        return meta;
      })
      .catch(function (err) {
        if (!state || state.stem !== mine) { return null; }
        state.meta = { available: false, reason: err.message };
        syncVisibility();
        return state.meta;
      });
  }

  /* setEnabled(on) — show/hide the player without discarding its metadata. */
  function setEnabled(on) {
    if (!state) { return; }
    state.enabled = !!on;
    if (!state.enabled) {
      setPlaying(false);
      abortAll();
      resetBuffer();
    }
    syncVisibility();
  }

  function syncVisibility() {
    if (!state) { return; }
    var live = state.enabled && !!state.meta && state.meta.available;
    state.ui.bar.classList.toggle("is-hidden", !live);
    state.canvas.classList.toggle("is-hidden", !live);
    if (state.enabled && state.meta && !state.meta.available) {
      say(state.meta.reason || "no movie for this FOV");
    }
    if (live) { refreshRect(); }
  }

  /* setRect() — the viewport moved; re-derive the crop from the caller. The
   * canvas keeps its current contents until the first frame for the new crop
   * lands, so a pan does not blank the view. */
  function refreshRect() {
    if (!state || !state.enabled || !state.meta || !state.meta.available) {
      return;
    }
    var next = clampRect(state.getRect());
    if (!next) { return; }
    if (next.ds === state.ds && next.x === state.rect.x && next.y === state.rect.y
        && next.w === state.rect.w && next.h === state.rect.h) {
      return;
    }
    state.rect = { x: next.x, y: next.y, w: next.w, h: next.h };
    state.ds = next.ds;
    state.rectPending = true;
    resetBuffer();
    // A short urgent read for the frame already on screen, ahead of the wide
    // prefetch, so the redraw after a zoom is one round trip rather than one
    // full chunk.
    request(Math.max(0, Math.round(state.target) - 1), URGENT_CHUNK,
            state.buffer.gen);
  }

  /* The server clamps anything out of bounds and says so, and the response
   * handler drops a chunk whose rect is not the one it is holding. Clamping
   * here too is what keeps those two from disagreeing: an unclamped ask would
   * come back trimmed, be discarded, and be re-asked identically forever. */
  function clampRect(rect) {
    if (!rect || !state.meta) { return null; }
    var w = state.meta.width;
    var h = state.meta.height;
    var x = Math.max(0, Math.min(Math.round(rect.x), w - 1));
    var y = Math.max(0, Math.min(Math.round(rect.y), h - 1));
    return {
      x: x, y: y,
      w: Math.max(1, Math.min(Math.round(rect.w), w - x)),
      h: Math.max(1, Math.min(Math.round(rect.h), h - y)),
      ds: Math.max(1, Math.min(Math.round(rect.ds) || 1, 32)),
    };
  }

  function resetBuffer() {
    abortAll();
    state.buffer.gen += 1;
    bufferClear(state.buffer);
    state.drawn = -1;
  }

  function destroy() {
    if (!state) { return; }
    abortAll();
    document.removeEventListener("keydown", onKeydown);
    if (state.raf) { window.cancelAnimationFrame(state.raf); }
    if (state.ui && state.ui.bar && state.ui.bar.parentNode) {
      state.ui.bar.parentNode.removeChild(state.ui.bar);
    }
    state = null;
  }

  window.roigbivDiscoveryPlayer = {
    mount: mount,
    setStem: setStem,
    setEnabled: setEnabled,
    setRect: refreshRect,
    destroy: destroy,
  };
})();
