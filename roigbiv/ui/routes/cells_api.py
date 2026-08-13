"""Flask routes backing the /cells contact sheet.

Routes
------
GET  /api/cells/<fov_id>                  — sessions + cells + ROI geometry as JSON
GET  /api/cells/<fov_id>/image/<stem>.png — one session's mean projection
POST /api/cells/<fov_id>/gesture          — apply one edit gesture, return fresh state

Why the sheet is fetched rather than pushed through Dash
--------------------------------------------------------
The page used to ship each panel's mean projection as a Plotly heatmap: a
quantised 1024x1024 uint8 array is ~1.9 MB of JSON *per panel*, re-sent on
every edit because applying one required rebuilding the figure. Here the
picture is a PNG the browser caches by ETag and never re-fetches, and geometry
is a few tens of KB of contours. An edit returns only the geometry.

No route takes a filesystem path. A session is addressed by ``fov_id`` plus
its ``stem``, both resolved through the registry — there is nothing for a
crafted path to traverse.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request

from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.cell_edit_ops import Gesture, apply_gesture
from roigbiv.ui.services.colors import MATCH_STATUS_PALETTE
from roigbiv.ui.services.tracked_cells import TrackedFOV, load_tracked_fov_cached

# Contours are the bulk of the payload; a tenth of a pixel is well past what a
# 40x-zoomed outline can show, and rounding there halves the JSON.
_COORD_DP = 1

_write_locks: dict[str, threading.Lock] = {}
_write_locks_meta = threading.Lock()


def _write_lock(fov_id: str) -> threading.Lock:
    with _write_locks_meta:
        return _write_locks.setdefault(fov_id, threading.Lock())


# ── serialisation ──────────────────────────────────────────────────────────


def serialize_fov(fov: TrackedFOV) -> dict:
    """The whole sheet as JSON — the browser's only source for what to draw.

    Deliberately flat: the client does no joining. Each ROI already carries
    the cell number and cross-session status the overlay colours by, so a
    repaint after an edit is a straight swap of this structure.
    """
    index_by_gcid = {c.global_cell_id: c.index for c in fov.cells}
    return {
        "fov_id": fov.fov_id,
        "animal_id": fov.animal_id,
        "region": fov.region,
        "ordering_is_confirmed": fov.ordering_is_confirmed,
        "palette": dict(MATCH_STATUS_PALETTE),
        "sessions": [
            _serialize_session(fov, session, i, index_by_gcid)
            for i, session in enumerate(fov.sessions)
        ],
        "cells": [
            {
                "gcid": cell.global_cell_id,
                "index": cell.index,
                "present": list(cell.present),
                "anomalies": list(cell.anomalies),
            }
            for cell in fov.cells
        ],
    }


def _serialize_session(fov: TrackedFOV, session, i: int,
                       index_by_gcid: dict) -> dict:
    height, width = _frame_shape(session)
    return {
        "index": i,
        "session_id": session.session_id,
        "stem": session.stem,
        "label": session.label,
        "short_label": session.short_label,
        "stale": bool(session.stale),
        "width": width,
        "height": height,
        "image_url": f"/api/cells/{fov.fov_id}/image/{session.stem}.png",
        "counts": _counts(session),
        "rois": [
            {
                "label_id": int(roi.label_id),
                "gcid": roi.global_cell_id,
                # The "#N" badge. Ghosts share their cell's number, which is
                # the point — it is how a dropout is recognised as *that* cell.
                "cell_index": index_by_gcid.get(roi.global_cell_id),
                "match_status": roi.match_status,
                # A ghost is drawn where the cell last was; it owns no
                # footprint here, so it is never a gesture target.
                "ghost": int(roi.label_id) < 0,
                "centroid": [round(float(roi.centroid_yx[0]), _COORD_DP),
                             round(float(roi.centroid_yx[1]), _COORD_DP)],
                "contours": [_ring(ys, xs) for ys, xs in roi.contours if ys],
            }
            for roi in session.rois
        ],
    }


def _ring(ys, xs) -> list:
    """One contour as ``[[x, y], ...]`` — SVG order, not array order."""
    return [[round(float(x), _COORD_DP), round(float(y), _COORD_DP)]
            for y, x in zip(ys, xs)]


def _counts(session) -> str:
    """What this panel is actually showing, counted off the panel itself.

    Deliberately *not* the session row's ``n_matched`` / ``n_new`` /
    ``n_missing``: those record what the matcher decided at registration time
    and can disagree with the observations on disk. A caption that
    contradicts the picture above it is worse than either number alone, so the
    caption follows the picture. The registration figures are still reported
    by ``roigbiv-registry show``.
    """
    counts: dict[str, int] = {}
    for roi in session.rois:
        if roi.match_status:
            counts[roi.match_status] = counts.get(roi.match_status, 0) + 1
    parts = [f"{counts.get('matched', 0)} matched"]
    if counts.get("new"):
        parts.append(f"{counts['new']} new")
    if counts.get("lost"):
        parts.append(f"{counts['lost']} not detected")
    return " · ".join(parts)


def _frame_shape(session) -> tuple[int, int]:
    if session.mean_M is not None:
        shape = np.asarray(session.mean_M).shape
        return int(shape[0]), int(shape[1])
    return 0, 0


# ── image encoding ─────────────────────────────────────────────────────────


def _projection_png(mean: np.ndarray) -> bytes:
    """*mean* as a PNG, windowed the way the Plotly sheet windowed it.

    The 1st/99.5th percentile stretch is carried over verbatim from the
    figure builder it replaces — researchers know these projections by sight,
    and a different window would look like the data changed.
    """
    arr = np.asarray(mean, dtype=np.float64)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99.5))
    if hi <= lo:
        flat = np.zeros(arr.shape, dtype=np.uint8)
        return _encode_png(flat)
    scaled = (arr - lo) / (hi - lo) * 255.0
    return _encode_png(np.clip(scaled, 0, 255).astype(np.uint8))


def _encode_png(arr_u8: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr_u8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _projection_etag(output_dir: Path) -> Optional[str]:
    """mtime of the projection on disk — it changes only on a pipeline re-run."""
    try:
        stat = (Path(output_dir) / "summary" / "mean_M.tif").stat()
    except OSError:
        return None
    return f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'


# ── route registration ─────────────────────────────────────────────────────


def register_flask_routes(server: Flask) -> None:
    """Register the /cells data + gesture routes on the Dash Flask server."""

    def _load(fov_id: str):
        """``(fov, state)`` for *fov_id*, or ``(None, error_response)``."""
        state = get_app_state()
        if state.workspace is None:
            return None, (jsonify({"error": "no workspace selected"}), 409)
        try:
            fov = load_tracked_fov_cached(fov_id, cfg=state.registry_config)
        except Exception as exc:  # noqa: BLE001 — store or disk, both the user's
            return None, (jsonify({"error": str(exc)}), 404)
        if not fov.sessions:
            return None, (jsonify({"error": "this FOV has no readable sessions"}),
                          404)
        return (fov, state), None

    @server.route("/api/cells/<fov_id>")
    def cells_state(fov_id: str):
        loaded, err = _load(fov_id)
        if err is not None:
            return err
        fov, _state = loaded
        return jsonify(serialize_fov(fov))

    @server.route("/api/cells/<fov_id>/image/<stem>.png")
    def cells_image(fov_id: str, stem: str):
        loaded, err = _load(fov_id)
        if err is not None:
            return err
        fov, _state = loaded

        session = next((s for s in fov.sessions if s.stem == stem), None)
        if session is None:
            return jsonify({"error": f"no session {stem!r} in this FOV"}), 404
        if session.mean_M is None:
            return jsonify({"error": "this session has no mean projection"}), 404

        etag = _projection_etag(session.output_dir) if session.output_dir else None
        if etag and request.headers.get("If-None-Match") == etag:
            return "", 304

        resp = Response(_projection_png(session.mean_M), mimetype="image/png")
        if etag:
            resp.headers["ETag"] = etag
            # The projection is a pipeline artefact; it only changes when the
            # pipeline re-runs, and the ETag catches that.
            resp.headers["Cache-Control"] = "private, max-age=86400"
        else:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @server.route("/api/cells/<fov_id>/gesture", methods=["POST"])
    def cells_gesture(fov_id: str):
        loaded, err = _load(fov_id)
        if err is not None:
            return err
        fov, state = loaded

        try:
            gesture = Gesture.from_payload(request.get_json(silent=True) or {})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        # Serialised per FOV: two tabs on one FOV would otherwise interleave
        # appends to the same log and replay them against each other's state.
        with _write_lock(fov_id):
            try:
                result = apply_gesture(
                    fov, gesture,
                    input_root=state.workspace.input_root,
                    registry_cfg=state.registry_config,
                )
            except Exception as exc:  # noqa: BLE001 — store or disk, the user's
                return jsonify({"ok": False, "error": str(exc)}), 500

        body = {
            "ok": result.ok,
            "message": result.message,
            "selected_gcid": result.selected_gcid,
            "warnings": result.warnings,
        }
        # Only a gesture that wrote something carries fresh state; a refusal
        # leaves the sheet exactly as the browser already has it.
        if result.fov is not None:
            body["state"] = serialize_fov(result.fov)
        return jsonify(body), result.status
