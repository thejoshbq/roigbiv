"""Picking one FOV out of a workspace — the convention, shared.

Three pages let a user choose a FOV (motion correction, centroids, boundaries)
and they must agree on what a choice *means*, so the value convention and its
resolvers live here rather than being forked per page.

A dropdown value is self-describing, because the render callback receives only
the value and not the option it came from:

``summary:{output_dir}``  a FOV Foundation has already written a temporal mean
                          for. Has an output dir, so it may also have
                          ``centroids.json``, a flow cache, boundaries.
``input:{tif_path}``      a pre-corrected stack sitting in the workspace that
                          has not been run. Its mean is sampled on demand and
                          it has no output dir yet.

Enumeration itself stays in :func:`roigbiv.ui.services.loaders
.list_motion_corrected_fovs`, which is filesystem-based on purpose — the
registry only knows about fully-registered FOVs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash_bootstrap_components as dbc

from roigbiv.pipeline.loaders import _maybe_read_tif
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.loaders import list_motion_corrected_fovs, mc_input_mean


def options_and_value(workspace, current: Optional[str] = None):
    """Build a FOV dropdown's ``(options, value)`` from a workspace.

    Keeps ``current`` selected if it still exists, else defaults to the first
    FOV. Shared by every page's layout seed, tick refresh and scan handler so
    the three can't drift.
    """
    fovs = list_motion_corrected_fovs(workspace)
    options = [{"label": label, "value": value} for label, value in fovs]
    values = {opt["value"] for opt in options}
    if current in values:
        value = current
    elif options:
        value = options[0]["value"]
    else:
        value = None
    return options, value


def processed_options_and_value(workspace, current: Optional[str] = None):
    """Only FOVs that have been run — those with an output dir on disk.

    The boundaries page uses this: a seeded boundary needs a cached flow field,
    which only exists once centroid discovery has written one, so offering a
    not-yet-run input would be offering a dead end.
    """
    options, _ = options_and_value(workspace)
    options = [o for o in options if str(o["value"]).startswith("summary:")]
    values = {opt["value"] for opt in options}
    if current in values:
        return options, current
    return options, (options[0]["value"] if options else None)


def select(select_id: str, workspace, *, current: Optional[str] = None,
           processed_only: bool = False, **kwargs) -> dbc.Select:
    """The dropdown itself, persisted per workspace.

    ``persistence=False`` when no workspace is resolved: a constant key would
    leak one workspace's selection onto the next.
    """
    builder = processed_options_and_value if processed_only else options_and_value
    options, value = builder(workspace, current)
    key = str(workspace.input_root) if workspace is not None else False
    return dbc.Select(id=select_id, options=options, value=value,
                      persistence=key, persistence_type="local", **kwargs)


def mean_and_title(value: Optional[str]):
    """Resolve a dropdown ``value`` to ``(mean, title, output_dir)``.

    ``output_dir`` is ``None`` for an ``input:`` FOV — a stack that has not been
    run has no output directory, and therefore no centroids, flows or
    boundaries. ``None`` / unparseable returns ``(None, None, None)``.
    """
    if value and ":" in value:
        kind, payload = value.split(":", 1)
        if kind == "summary":
            return (_maybe_read_tif(Path(payload) / "summary" / "mean_M.tif"),
                    Path(payload).name, Path(payload))
        if kind == "input":
            return (mc_input_mean(Path(payload)),
                    f"{Path(payload).stem.replace('_mc', '')} (input)", None)
    return None, None, None


def resolve_output_dir(value: Optional[str]) -> Optional[Path]:
    """The FOV's output dir, resolved even for a not-yet-processed input.

    Unlike :func:`mean_and_title`, this answers for an ``input:`` FOV too,
    mirroring how :func:`roigbiv.pipeline.workspace._run_centroids_only`
    resolves its own ``out_dir`` — calibration is meant to work ahead of the
    first run, same as centroids-only mode itself.
    """
    workspace = get_app_state().workspace
    if not value or ":" not in value or workspace is None:
        return None
    kind, payload = value.split(":", 1)
    stem = (Path(payload).name if kind == "summary"
            else Path(payload).stem.replace("_mc", ""))
    return workspace.output_root / stem
