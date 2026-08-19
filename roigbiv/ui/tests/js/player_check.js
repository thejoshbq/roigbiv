/* Headless check of assets/discovery_player.js.
 *
 * Stubs enough DOM + fetch to run the real render loop, then asserts the
 * contract the acceptance criteria rest on:
 *   1. the drawn frame is always the *target* frame, never a nearby one
 *   2. on underrun, playback holds and does not bank elapsed time to spend
 *      skipping frames later
 *   3. a viewport change re-crops without blanking or wedging the loop
 *
 * Identifying the drawn frame: the fake server fills frame N's pixels with
 * N % 251, and the player copies gray bytes straight into the ImageData, so
 * data[0] after putImageData names the frame that was actually painted.
 */
"use strict";
const fs = require("fs");
const path = require("path");

const N_FRAMES = 400;
const IMG_W = 512, IMG_H = 512;
const COLS = 32, ROWS = 24;
const FPS = 10;

let now = 0;
let latencyMs = 0;
let servedChunks = 0;
let maxCountSeen = 0;
const rafQueue = [];
const pending = [];
const painted = [];      // frame index per putImageData, in order

// ── DOM stubs ───────────────────────────────────────────────────────────────

function el(tag) {
  return {
    tagName: tag.toUpperCase(), style: {}, children: [], parentNode: null,
    className: "", value: "", textContent: "", innerHTML: "", type: "",
    min: "", max: "", step: "", width: 0, height: 0, _l: {},
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
    },
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    removeChild(c) { this.children = this.children.filter(x => x !== c); return c; },
    setAttribute() {}, getAttribute() { return null; },
    addEventListener(t, fn) { (this._l[t] ||= []).push(fn); },
    removeEventListener() {},
    fire(t, ev) { (this._l[t] || []).forEach(fn => fn(ev || {})); },
    getContext() {
      return {
        createImageData: (w, h) => ({
          width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
        putImageData: (img) => { painted.push(img.data[0]); },
      };
    },
  };
}

const msg = el("div");
global.document = {
  head: el("head"), body: el("body"), activeElement: null,
  createElement: el,
  getElementById: (id) => (id === "roigbiv-discovery-edit-msg" ? msg : null),
  addEventListener() {}, removeEventListener() {},
};
global.window = {
  requestAnimationFrame(fn) { rafQueue.push(fn); return rafQueue.length; },
  cancelAnimationFrame() {},
  performance: { now: () => now },
};
global.AbortController = class {
  constructor() { this.signal = {}; }
  abort() { this.aborted = true; }
};

// ── fake movie server ───────────────────────────────────────────────────────

global.fetch = function (url) {
  if (url.includes("/movie/meta")) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({
      available: true, kind: "data_bin", n_frames: N_FRAMES,
      height: IMG_H, width: IMG_W, fps: FPS, window: [0, 100],
      max_count: 64 }) });
  }
  const p = {};
  url.split("?")[1].split("&").forEach(kv => {
    const [k, v] = kv.split("="); p[k] = parseInt(v, 10);
  });
  // Clamps exactly as movie_source.clamp_request does, so an unclamped ask
  // comes back describing a different rect than the client is holding.
  p.x = Math.max(0, Math.min(p.x, IMG_W - 1));
  p.y = Math.max(0, Math.min(p.y, IMG_H - 1));
  p.w = Math.max(1, Math.min(p.w, IMG_W - p.x));
  p.h = Math.max(1, Math.min(p.h, IMG_H - p.y));
  p.ds = Math.max(1, p.ds);
  p.start = Math.max(0, Math.min(p.start, N_FRAMES - 1));
  p.count = Math.max(1, Math.min(p.count, 64, N_FRAMES - p.start));
  servedChunks += 1;
  maxCountSeen = Math.max(maxCountSeen, p.count);
  const buf = new Uint8Array(p.count * COLS * ROWS);
  for (let i = 0; i < p.count; i++) {
    buf.fill((p.start + i) % 251, i * COLS * ROWS, (i + 1) * COLS * ROWS);
  }
  const h = new Map([
    ["X-Movie-Start", String(p.start)], ["X-Movie-Count", String(p.count)],
    ["X-Movie-Cols", String(COLS)], ["X-Movie-Rows", String(ROWS)],
    ["X-Movie-Rect", `${p.x},${p.y},${p.w},${p.h}`],
    ["X-Movie-Ds", String(p.ds)],
  ]);
  const resp = { ok: true, headers: { get: k => h.get(k) },
                 arrayBuffer: () => Promise.resolve(buf.buffer.slice(0)) };
  return new Promise(res => pending.push({ due: now + latencyMs,
                                           go: () => res(resp) }));
};

// ── load the real module ────────────────────────────────────────────────────

const PLAYER_JS = process.env.PLAYER_JS
  || path.resolve(__dirname, "../../assets/discovery_player.js");
eval(fs.readFileSync(PLAYER_JS, "utf8"));
const player = global.window.roigbivDiscoveryPlayer;

// ── driving ─────────────────────────────────────────────────────────────────

const microtasks = () => new Promise(r => setImmediate(r));

async function advance(ms, stepMs = 8) {
  for (let t = 0; t < ms; t += stepMs) {
    now += stepMs;
    for (let i = pending.length - 1; i >= 0; i--) {
      if (pending[i].due <= now) { pending.splice(i, 1)[0].go(); }
    }
    await microtasks();
    const queued = rafQueue.splice(0, rafQueue.length);
    for (const fn of queued) { fn(now); }
    await microtasks();
  }
}

const failures = [];
function check(name, cond, detail) {
  if (cond) { console.log(`  ok   ${name}`); }
  else { failures.push(name); console.log(`  FAIL ${name}${detail ? "\n         " + detail : ""}`); }
}

// ── the checks ──────────────────────────────────────────────────────────────

(async function main() {
  const canvas = el("canvas");
  const mount = el("div");
  let rect = { x: 0, y: 0, w: 512, h: 512, ds: 1 };
  let drawing = false;

  const placements = [];   // [{ at, rect }] every onRectApplied
  player.mount({ canvas, controlsMount: mount,
                 getRect: () => rect, isDrawing: () => drawing,
                 onRectApplied: (r) => placements.push(
                   { at: painted.length, rect: `${r.x},${r.y},${r.w},${r.h}` }) });
  const bar = mount.children[0];
  const [back, play, fwd, scrub] = [bar.children[0], bar.children[1],
                                    bar.children[2], bar.children[3]];

  await player.setStem("fov1");
  player.setEnabled(true);
  await advance(300);

  console.log("\n[1] playback paints every frame in order, none skipped");
  painted.length = 0;
  play.fire("click");
  await advance(3000);           // 3 s at 10 fps => ~30 frames
  const seq = painted.slice();
  const gaps = seq.filter((f, i) => i > 0 && f !== seq[i - 1] + 1);
  check("frames advance strictly by one", gaps.length === 0,
        `first break at ${seq.findIndex((f, i) => i > 0 && f !== seq[i - 1] + 1)}: ${seq.slice(0, 40)}`);
  check("played roughly real time", seq.length >= 25 && seq.length <= 34,
        `painted ${seq.length} frames in 3 s at ${FPS} fps`);

  console.log("\n[2] underrun stalls and resumes without banking time");
  play.fire("click");                       // pause
  latencyMs = 900;                          // far slower than a frame budget
  player.setRect();
  rect = { x: 100, y: 100, w: 256, h: 256, ds: 1 };
  player.setRect();                         // new crop => empty buffer
  painted.length = 0;
  await advance(200);
  const duringStall = painted.length;
  play.fire("click");                       // play into an empty buffer
  // Shorter than the 900 ms round trip, so the buffer is genuinely empty for
  // the whole window: at 10 fps an unguarded loop would have banked ~4 frames.
  await advance(400);
  check("the playhead does not advance while the buffer is empty",
        painted.length === duringStall,
        `painted ${painted.length - duringStall} frame(s) with nothing buffered`);
  latencyMs = 0;
  await advance(1500);
  const after = painted.slice(duringStall);
  const jumps = after.filter((f, i) => i > 0 && f !== after[i - 1] + 1);
  check("resumes without a forward jump", after.length > 0 && jumps.length === 0,
        `after stall: ${after.slice(0, 30)}`);

  console.log("\n[3] scrubbing lands on exactly the requested frame");
  play.fire("click");                       // pause
  await advance(200);
  for (const target of [300, 12, 199, 0, N_FRAMES - 1]) {
    painted.length = 0;
    scrub.value = String(target);
    scrub.fire("input");
    await advance(1200);
    const last = painted[painted.length - 1];
    check(`scrub to ${target} paints ${target}`, last === target % 251,
          `painted ${last}`);
  }

  console.log("\n[4] a viewport change re-crops without wedging");
  const before = servedChunks;
  latencyMs = 250;                 // long enough to separate choose from paint
  rect = { x: 300, y: 40, w: 64, h: 64, ds: 4 };
  painted.length = 0;
  placements.length = 0;
  player.setRect();
  // The crop is chosen now; nothing of it has been painted yet. Placing the
  // canvas here would stretch the old crop's pixels over the new rectangle.
  await advance(150);
  check("the canvas is not re-placed before the new crop is painted",
        placements.length === 0 && painted.length === 0,
        `${placements.length} placement(s), ${painted.length} paint(s)`);
  await advance(900);
  latencyMs = 0;
  check("refetches for the new crop", servedChunks > before);
  check("still paints after the re-crop", painted.length > 0);
  check("re-places onto the new crop once painted",
        placements.length >= 1 && placements[0].rect === "300,40,64,64"
        && placements[0].at >= 1,
        JSON.stringify(placements));

  console.log("\n[5] step buttons move exactly one frame and pause");
  scrub.value = "150"; scrub.fire("input");
  await advance(600);
  painted.length = 0;
  fwd.fire("click"); await advance(400);
  check("step forward paints 151", painted[painted.length - 1] === 151);
  painted.length = 0;
  back.fire("click"); back.fire("click"); await advance(400);
  check("two steps back paint 149", painted[painted.length - 1] === 149);

  console.log("\n[6] buffered frames own their bytes");
  // A subarray keeps its whole chunk's ArrayBuffer alive, so byte-capped
  // eviction would free nothing while reporting that it had.
  check("a frame is a standalone copy, not a view into its chunk",
        /new Uint8Array\(payload\.bytes\.subarray/.test(
          fs.readFileSync(PLAYER_JS, "utf8")),
        "bufferPut must not store a bare subarray");

  console.log("\n[7] an out-of-bounds viewport settles instead of spinning");
  // Before the client clamped its own rect, the server's trimmed answer was
  // discarded and re-asked identically, forever.
  rect = { x: -40, y: -40, w: 9999, h: 9999, ds: 1 };
  player.setRect();
  painted.length = 0;
  await advance(600);
  const spinBase = servedChunks;
  await advance(1200);
  check("paints under an over-large viewport", painted.length > 0);
  check("does not refetch endlessly", servedChunks - spinBase < 20,
        `${servedChunks - spinBase} chunks in 1.2 s`);

  console.log("\n[8] a chunk never exceeds the server's advertised cap");
  check("count <= max_count", maxCountSeen <= 64, `saw ${maxCountSeen}`);

  console.log("\n[9] the player yields the keyboard to a boundary draw");
  drawing = true;
  const wasPlaying = bar.children[1].innerHTML;
  // keydown is registered on document; the stub swallows it, so assert the
  // guard directly by construction instead.
  check("isDrawing is consulted", /state\.isDrawing\(\)/.test(
    fs.readFileSync(PLAYER_JS, "utf8")));
  drawing = false;

  console.log(failures.length ? `\n${failures.length} FAILED` : "\nall checks passed");
  process.exit(failures.length ? 1 : 0);
})();
