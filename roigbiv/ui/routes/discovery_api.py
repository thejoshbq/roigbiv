"""Flask routes backing the /discovery page's OpenSeadragon viewer.

Analogous to :mod:`roigbiv.ui.routes.cells_api`, scoped to one FOV / one
session rather than a cross-session ``TrackedFOV``: Discovery edits raw
per-FOV output (``centroids.json`` + ``corrections/centroids.jsonl``) before
— or entirely without — that FOV ever going through cross-session tracking.

Routes
------
GET  /api/discovery/<stem>              — centroid geometry + image URL
GET  /api/discovery/<stem>/image.png    — mean projection as PNG
POST /api/discovery/<stem>/gesture      — apply one add/delete/move/undo op

No route takes a filesystem path. *stem* resolves against the session's
``AppState.workspace.output_root`` and must name an existing, direct child of
that root — the same confinement
:func:`roigbiv.ui.components.fov_select.resolve_output_dir` already applies
to a dropdown value, enforced here at the HTTP boundary instead.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request

from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.discovery_edit_ops import Gesture, apply_gesture

_write_locks: dict[str, threading.Lock] = {}
_write_locks_meta = threading.Lock()


def _write_lock(stem: str) -> threading.Lock:
    with _write_locks_meta:
        return _write_locks.setdefault(stem, threading.Lock())


# ── FOV resolution ─────────────────────────────────────────────────────────


def _resolve_output_dir(stem: str) -> Optional[Path]:
    """``output_root / stem``, confined to an existing direct child of it."""
    state = get_app_state()
    workspace = state.workspace
    if workspace is None:
        return None
    root = Path(workspace.output_root).resolve()
    candidate = (root / stem).resolve()
    if candidate.parent != root or not candidate.is_dir():
        return None
    return candidate


def _radius(output_dir: Path) -> float:
    """The marker radius to draw centroids at — a measurement when there is one."""
    from roigbiv.pipeline.calibration import load_calibration

    calib = load_calibration(output_dir)
    if calib is not None and calib.diameter_px > 0:
        return float(calib.diameter_px) / 2.0
    return 8.0  # PipelineConfig.roi_stamp_radius default


# ── serialisation ──────────────────────────────────────────────────────────


def serialize_fov(stem: str, output_dir: Path) -> dict:
    """This FOV's whole viewer state — the browser's only source for what to draw."""
    import tifffile

    from roigbiv.pipeline.centroid_masks import load_effective_centroids

    height = width = 0
    mean_path = output_dir / "summary" / "mean_M.tif"
    if mean_path.exists():
        try:
            arr = np.asarray(tifffile.imread(str(mean_path)))
            height, width = int(arr.shape[0]), int(arr.shape[1])
        except (OSError, ValueError):
            pass

    centroids: list[dict] = []
    warnings: list[str] = []
    if (output_dir / "centroids.json").exists():
        effective, warnings = load_effective_centroids(output_dir)
        centroids = [
            {"label_id": label, "y": round(float(y), 1), "x": round(float(x), 1)}
            for label, (y, x) in sorted(effective.items())
        ]

    return {
        "stem": stem,
        "width": width,
        "height": height,
        "image_url": f"/api/discovery/{stem}/image.png",
        "radius": _radius(output_dir),
        "centroids": centroids,
        "warnings": warnings,
    }


# ── image encoding ─────────────────────────────────────────────────────────


def _projection_png(mean: np.ndarray) -> bytes:
    """*mean* as a PNG, windowed the same way every other projection in the
    UI is (1st/99.5th percentile stretch) — see ``cells_api._projection_png``."""
    arr = np.asarray(mean, dtype=np.float64)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99.5))
    if hi <= lo:
        return _encode_png(np.zeros(arr.shape, dtype=np.uint8))
    scaled = (arr - lo) / (hi - lo) * 255.0
    return _encode_png(np.clip(scaled, 0, 255).astype(np.uint8))


def _encode_png(arr_u8: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr_u8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _projection_etag(output_dir: Path) -> Optional[str]:
    try:
        stat = (output_dir / "summary" / "mean_M.tif").stat()
    except OSError:
        return None
    return f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'


# ── route registration ─────────────────────────────────────────────────────


def register_flask_routes(server: Flask) -> None:
    """Register the /discovery data + gesture routes on the Dash Flask server."""

    def _load(stem: str):
        output_dir = _resolve_output_dir(stem)
        if output_dir is None:
            return None, (jsonify(
                {"error": f"no FOV named {stem!r} in this workspace"}), 404)
        return output_dir, None

    @server.route("/api/discovery/<stem>")
    def discovery_state(stem: str):
        output_dir, err = _load(stem)
        if err is not None:
            return err
        return jsonify(serialize_fov(stem, output_dir))

    @server.route("/api/discovery/<stem>/image.png")
    def discovery_image(stem: str):
        output_dir, err = _load(stem)
        if err is not None:
            return err

        import tifffile

        mean_path = output_dir / "summary" / "mean_M.tif"
        if not mean_path.exists():
            return jsonify({"error": "no mean projection for this FOV"}), 404

        etag = _projection_etag(output_dir)
        if etag and request.headers.get("If-None-Match") == etag:
            return "", 304

        try:
            mean = np.asarray(tifffile.imread(str(mean_path)))
        except (OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 500

        resp = Response(_projection_png(mean), mimetype="image/png")
        if etag:
            resp.headers["ETag"] = etag
            resp.headers["Cache-Control"] = "private, max-age=86400"
        else:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @server.route("/api/discovery/<stem>/gesture", methods=["POST"])
    def discovery_gesture(stem: str):
        output_dir, err = _load(stem)
        if err is not None:
            return err

        try:
            gesture = Gesture.from_payload(request.get_json(silent=True) or {})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        # Serialised per FOV: two tabs editing the same FOV would otherwise
        # interleave appends to the same log.
        with _write_lock(stem):
            try:
                result = apply_gesture(output_dir, gesture)
            except Exception as exc:  # noqa: BLE001 — store or disk, the user's
                return jsonify({"ok": False, "error": str(exc)}), 500

        body = {"ok": result.ok, "message": result.message,
                "warnings": result.warnings}
        if result.ok:
            body["state"] = serialize_fov(stem, output_dir)
        return jsonify(body), result.status
