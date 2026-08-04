"""Flask routes serving the live motion-correction preview sidecar.

Routes
------
GET /api/mc-preview/list                      — one entry per FOV with a sidecar
GET /api/mc-preview/state?stem=<s>            — that FOV's state.json (+ ``stale``)
GET /api/mc-preview/image?stem=<s>&kind=&seq= — a preview PNG

These sit outside Dash's callback machinery on purpose. The live pane refreshes
several times a second, and routing that through ``/_dash-update-component``
would run Python on every tick — in the UI the pipeline is a daemon thread of
this same process, so those ticks would compete with registration for the GIL.
Here the page fetches a few hundred bytes of JSON and lets the browser pull the
PNGs directly.

Path safety: the FOV directory is derived from the *requesting session's*
workspace, never from a client-supplied path, and must resolve under that
workspace's output root (see
:func:`roigbiv.ui.services.mc_preview.fov_preview_dir`).
"""
from __future__ import annotations

from flask import Flask, Response, jsonify, request

from roigbiv.ui.services.mc_preview import (
    fov_preview_dir,
    list_states,
    read_state,
)

_KINDS = {"raw": "raw", "corr": "corr", "corrected": "corr", "avg": "avg"}

# Keys exposed by /list — the fast poll only needs enough to build image URLs
# and a status line, so the traces and metrics in state.json stay out of it.
_LIST_KEYS = ("stem", "backend", "phase", "seq", "n_done", "n_total",
              "pass_index", "updated_at", "stale", "has_avg")


def _session_preview_dir(stem: str):
    from roigbiv.ui.services.app_state import get_app_state

    return fov_preview_dir(get_app_state().workspace, stem)


def register_flask_routes(server: Flask) -> None:
    """Register the MC-preview routes on the Dash app's Flask server."""

    @server.route("/api/mc-preview/list")
    def mc_preview_list():
        from roigbiv.ui.services.app_state import get_app_state

        states = list_states(get_app_state().workspace)
        resp = jsonify([{k: s.get(k) for k in _LIST_KEYS} for s in states])
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @server.route("/api/mc-preview/state")
    def mc_preview_state():
        try:
            pdir = _session_preview_dir(request.args.get("stem", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        state = read_state(pdir)
        if state is None:
            return jsonify({"error": "no preview for this FOV"}), 404
        resp = jsonify(state)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @server.route("/api/mc-preview/image")
    def mc_preview_image():
        try:
            pdir = _session_preview_dir(request.args.get("stem", ""))
        except ValueError as exc:
            return str(exc), 400
        kind = _KINDS.get(request.args.get("kind", "corr"))
        if kind is None:
            return "kind must be raw, corr, or avg", 400
        try:
            seq = int(request.args.get("seq", "-1"))
        except ValueError:
            return "seq must be an integer", 400
        if seq < 0:
            return "seq is required", 400
        try:
            payload = (pdir / f"{kind}_{seq:06d}.png").read_bytes()
        except OSError:
            return "no such preview frame", 404
        resp = Response(payload, mimetype="image/png")
        # Preview frames are seq-suffixed and never rewritten, so they are
        # immutable and the browser may keep them for the whole session.
        resp.headers["Cache-Control"] = "public, max-age=300, immutable"
        return resp
