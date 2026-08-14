"""Recompute seeded boundaries fast enough to drag a slider against.

:func:`roigbiv.pipeline.boundaries.compute_boundaries` does two things and only
one of them depends on the controls the boundary page exposes:

``converge_pixels``   runs Cellpose's own pixel dynamics over the cached flow
                      field. ~0.5–2 s on CPU at 512². Depends on the flow cache
                      and ``cellprob_threshold`` — neither of which a slider
                      moves.
``seeded_labels``     partitions those converged pixels among the seeds. Pure
                      numpy plus one watershed, tens of milliseconds. This is
                      what ``capture_px`` and ``min_area`` change.

So the expensive half is cached per FOV and only the cheap half re-runs. Without
that split every slider tick would re-run the dynamics and the control would be
unusable — which is the whole reason the page exists, since ``capture_px`` is
the one real tuning knob seeded boundaries have.

The cache is keyed on the output dir plus the flow cache's own mtime, so a
re-run of centroid discovery invalidates it without anyone having to remember
to. It is bounded because the arrays are large: ~6 MB per 512² FOV, ~24 MB at
1024².
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.seeded_masks import SeededMasks, seeded_labels

#: How many FOVs' converged-pixel arrays to hold. Small on purpose — a user
#: tunes one FOV at a time and compares against a couple of neighbours.
_MAX_ENTRIES = 4

_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_lock = threading.Lock()


class NoFlowCache(Exception):
    """This FOV has no usable flow field, so no boundary can be seeded.

    Normal rather than exceptional: centroid discovery never ran, ran with
    ``centroid_persist_flows=False``, or ran under the cpsam backend, which
    returns no flow field at all. The page reports it as a state with a way out,
    not as an error.
    """


def default_settings(output_dir: Path, cfg) -> tuple[float, int]:
    """``(capture_px, min_area)`` this FOV would be drawn with right now."""
    from roigbiv.pipeline.boundaries import resolve_capture_px, resolve_min_area

    return (resolve_capture_px(output_dir, cfg),
            resolve_min_area(output_dir, cfg))


def preview(output_dir: Path, cfg, *, capture_px: float,
            min_area: int = 0) -> SeededMasks:
    """Seeded boundaries at these settings, reusing the converged pixels.

    Raises :class:`NoFlowCache` when the FOV cannot be drawn at all.
    """
    from roigbiv.pipeline.centroid_masks import (
        load_effective_centroids,
        resolve_stamp_radius,
    )

    output_dir = Path(output_dir)
    inds, converged, cellprob = _converged(output_dir, cfg)
    seeds, warnings = load_effective_centroids(output_dir)

    result = seeded_labels(
        seeds, tuple(int(v) for v in cellprob.shape[:2]),
        inds=inds, converged=converged, cellprob=cellprob,
        capture_px=float(capture_px),
        fallback_radius=resolve_stamp_radius(output_dir, cfg),
        min_area=int(min_area or 0),
        max_area=getattr(cfg, "boundary_max_area", None),
    )
    result.warnings = list(warnings) + result.warnings
    return result


def _converged(output_dir: Path, cfg):
    """``(inds, converged, cellprob)`` for this FOV, cached across slider moves."""
    import json

    from roigbiv.pipeline.centroids import load_flow_cache
    from roigbiv.pipeline.seeded_masks import converge_pixels

    centroids_json = output_dir / "centroids.json"
    if not centroids_json.exists():
        raise NoFlowCache(f"{output_dir.name} has no centroids.json")
    try:
        params = json.loads(centroids_json.read_text()).get("params")
    except (json.JSONDecodeError, OSError) as exc:
        raise NoFlowCache(f"{output_dir.name}: unreadable centroids.json") from exc

    key = (str(output_dir.resolve()), _flow_mtime(output_dir))
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    cached = load_flow_cache(output_dir, params)
    if cached is None:
        raise NoFlowCache(
            f"{output_dir.name} has no cached flow field — re-run centroid "
            f"discovery with 'Cache the flow field' enabled")

    cellprob = np.asarray(cached["cellprob"], dtype=np.float32)
    # Read the threshold off the recorded params rather than cfg: it is the one
    # that selected the dynamics pixels at inference time, and a config change
    # since then must not silently reinterpret the field.
    threshold = float((params or {}).get(
        "cellprob_threshold", getattr(cfg, "cellprob_threshold", -2.0)))
    inds, converged = converge_pixels(
        np.asarray(cached["dP"], dtype=np.float32), cellprob,
        cellprob_threshold=threshold,
        niter=cached["niter"], dp_scale=cached["dp_scale"],
    )

    value = (inds, converged, cellprob)
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)
    return value


def _flow_mtime(output_dir: Path) -> Optional[int]:
    try:
        return (output_dir / "flows" / "cellprob.npy").stat().st_mtime_ns
    except OSError:
        return None


def clear_cache() -> None:
    """Drop every cached array. For tests and for a workspace change."""
    with _lock:
        _cache.clear()
