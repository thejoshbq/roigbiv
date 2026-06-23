"""Process page — scan a workspace, set pipeline params, run.

Flow
----
1. User pastes / types a path into the input field and clicks **Scan**.
2. Workspace summary card shows what was discovered (input / output /
   registry / TIF count + TIF list with validity ticks).
3. User sets ``fs`` + tunables in the form and clicks **Run pipeline**.
4. Background runner streams logs; interval polls render them live.
5. Per-FOV summary rows show up under the log as they complete, including
   the registry decision (``hash_match`` / ``auto_match`` / ``review`` /
   ``new_fov``) so no tab switch is required to see the full outcome.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from roigbiv.io import validate_tif
from roigbiv.pipeline.loaders import _maybe_read_tif
from roigbiv.pipeline.profiles import AUTO, get_profile, list_profiles
from roigbiv.pipeline.stage1 import list_available_models
from roigbiv.pipeline.types import PipelineConfig
from roigbiv.pipeline.workspace import WorkspacePaths, resolve_workspace
from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.components.forms import HELP_TEXT, help_icon, labeled_with_help
from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.loaders import list_motion_corrected_fovs, mc_input_mean
from roigbiv.ui.services.pipeline_runner import RunSnapshot, get_pipeline_runner


# ── layout ─────────────────────────────────────────────────────────────────


# ── acquisition/lens profile autofill ───────────────────────────────────────
# The Profile dropdown autofills every Stage-1 field below from the profile
# bundle (pipeline/profiles.py), falling back to the grin/dataclass baseline for
# keys the profile does not set. The form fields stay the source of truth at run
# time — the dropdown is a convenience that seeds them; the user may then tweak
# any field. (config field name, dash component id), in display order.
_PROFILE_AUTOFILL: list[tuple[str, str]] = [
    ("cellpose_model", "roigbiv-param-model"),
    ("channels", "roigbiv-param-channels"),
    ("flow_threshold", "roigbiv-param-flow-threshold"),
    ("cellprob_threshold", "roigbiv-param-cellprob-threshold"),
    ("diameter", "roigbiv-param-diameter"),
    ("use_denoise", "roigbiv-param-use-denoise"),
    ("min_area", "roigbiv-param-min-area"),
    ("max_area", "roigbiv-param-max-area"),
    ("min_solidity", "roigbiv-param-min-solidity"),
    ("max_eccentricity", "roigbiv-param-max-eccentricity"),
    ("tile_norm_blocksize", "roigbiv-param-tile-norm-blocksize"),
    ("mc_strip_height", "roigbiv-param-mc-strip-height"),
]

# UI-selectable profiles. ``auto`` (first, recommended) classifies the optics
# from the FOV and derives the gates per-FOV — the user uploads and the pipeline
# adapts, pausing for confirmation only when uncertain. The concrete profiles
# remain for users who want to pin the optics + tune fields by hand.
_UI_PROFILES: list[str] = [AUTO, *[p for p in list_profiles() if p != AUTO]]

# Concrete (non-auto) profiles, offered in the per-FOV optics confirmation card.
_CONCRETE_PROFILES: list[str] = [p for p in list_profiles() if p != AUTO]


def _profile_field_values(profile_name: str) -> dict:
    """Resolved Stage-1 field values for *profile_name*: the profile bundle laid
    over the grin/dataclass baseline. Keyed by PipelineConfig field name."""
    base = PipelineConfig()
    bundle = get_profile(profile_name)
    return {key: bundle.get(key, getattr(base, key)) for key, _ in _PROFILE_AUTOFILL}


def _channels_to_str(ch) -> str:
    """Tuple channels → the 'cyto,nucleus' string the dbc.Select carries."""
    return f"{int(ch[0])},{int(ch[1])}"


def _parse_channels_value(spec) -> tuple:
    """'cyto,nucleus' string → (int, int); falls back to the GRIN default."""
    try:
        a, b = (int(p.strip()) for p in str(spec).split(","))
        return (a, b)
    except (ValueError, AttributeError):
        return (1, 2)


def layout() -> html.Div:
    state = get_app_state()
    workspace = state.workspace
    # Rehydrate the run UI on refresh: the PipelineRunner persists per Flask
    # session, so a mid-run reload re-enables the interval and repaints the
    # last snapshot rather than dropping back to an empty Run-status panel.
    snap = get_pipeline_runner().snapshot()
    has_run = snap.started_at is not None
    return html.Div([
        dcc.Interval(id="roigbiv-process-tick", interval=1500,
                     disabled=not snap.active),
        # Benign sink for the TIF-selection sync callback (writes the user's
        # subset into server-side AppState); kept in the always-present layout
        # so its Output target exists before the first scan.
        dcc.Store(id="roigbiv-tif-select-sink"),
        dbc.Row([
            dbc.Col(_left_column(workspace, run_active=snap.active,
                                 diameter_default=state.calibrated_diameter() or 12),
                    md=5, lg=4, className="pe-md-4"),
            dbc.Col(_right_column(snap if has_run else None),
                    md=7, lg=8),
        ], className="g-3"),
    ])


def _left_column(workspace: Optional[WorkspacePaths],
                 run_active: bool = False,
                 diameter_default: int = 12) -> html.Div:
    return html.Div([
        html.H4("Workspace", className="mb-3"),
        dbc.InputGroup([
            dbc.Input(
                id="roigbiv-input-path",
                placeholder="Path to a .tif file or a directory of stacks",
                value=str(workspace.input_root) if workspace else "",
                type="text",
                readonly=workspace is not None,
            ),
            dbc.Button("Scan", id="roigbiv-scan-btn", color="primary",
                       n_clicks=0, disabled=workspace is not None),
        ], className="mb-3"),
        html.Div(
            id="roigbiv-scan-result",
            children=_workspace_summary(workspace) if workspace else None,
        ),
        html.Hr(className="roigbiv-h-line"),
        html.H5("Pipeline parameters", className="mb-2"),
        _params_form(diameter_default=diameter_default),
        dbc.Button("Run pipeline", id="roigbiv-run-btn",
                   color="primary", className="mt-3 w-100", n_clicks=0,
                   disabled=workspace is None or run_active),
        dbc.Button("Stop run", id="roigbiv-stop-btn",
                   color="danger", outline=True, className="mt-2 w-100",
                   n_clicks=0, disabled=not run_active),
    ])


def _field_row(label: str, target_id: str, control) -> dbc.Row:
    """Label (with hover-help) + control, two columns."""
    return dbc.Row([
        dbc.Col(labeled_with_help(label, target_id, HELP_TEXT[target_id]),
                md=6, className="d-flex align-items-center"),
        dbc.Col(control, md=6),
    ], className="mb-2")


def _switch_row(switch: dbc.Switch, target_id: str):
    """A ``dbc.Switch`` with a trailing hover-help icon."""
    return html.Div(
        [switch, *help_icon(target_id, HELP_TEXT[target_id])],
        className="d-flex align-items-center mb-1",
    )


def _stage_card(title: str, body: list) -> dbc.Card:
    return dbc.Card(dbc.CardBody([html.H6(title, className="mb-3"), *body]),
                    className="mb-3")


def _params_form(diameter_default: int = 12) -> html.Div:
    _model_opts = list_available_models()
    foundation = _stage_card("Foundation", [
        _field_row("fs (Hz)", "roigbiv-param-fs",
                   dbc.Input(id="roigbiv-param-fs", type="number",
                             value=7.5, step=0.5)),
        _field_row("tau (s)", "roigbiv-param-tau",
                   dbc.Input(id="roigbiv-param-tau", type="number",
                             value=1.0, step=0.1)),
        _field_row("k_background", "roigbiv-param-k",
                   dbc.Input(id="roigbiv-param-k", type="number",
                             value=30, step=1)),
        _field_row("Motion correction", "roigbiv-param-mc-backend",
                   dbc.Select(
                       id="roigbiv-param-mc-backend",
                       options=[
                           {"label": "Row-wise non-rigid (rowwise-pcc)",
                            "value": "rowwise-pcc"},
                           {"label": "Suite2p (phasecorr)",
                            "value": "phasecorr"},
                           {"label": "Legacy SIMA HMM2D (legacy)",
                            "value": "legacy"},
                       ],
                       value="phasecorr",
                   )),
        html.Small(
            "Legacy = genuine SIMA HiddenMarkov2D from the original "
            "notebook. CPU-only and slow (~16 min/session); requires "
            "the 'sima-legacy' conda env (build once with "
            "envs/build_sima_legacy.sh).",
            id="roigbiv-param-mc-backend-help",
            className="text-muted d-block mt-1",
        ),
        _field_row("mc_strip_height (px)", "roigbiv-param-mc-strip-height",
                   dbc.Input(id="roigbiv-param-mc-strip-height", type="number",
                             value=32, step=8, min=8, max=256)),
    ])
    stage1 = _stage_card("Stage 1 · Cellpose detection", [
        _field_row("Acquisition profile", "roigbiv-param-profile",
                   dbc.Select(
                       id="roigbiv-param-profile",
                       options=[{"label": p, "value": p} for p in _UI_PROFILES],
                       value=AUTO,
                   )),
        html.Small(
            "auto (recommended) classifies the optics per-FOV and derives the "
            "gates from the measured soma scale — the manual fields below are "
            "ignored, and the run pauses for confirmation only when uncertain. "
            "Selecting a concrete profile autofills the Stage-1 + Gate-1 fields "
            "below for that lens (grin = 512² GRIN baseline; prism = 1024² Prism). Tweak "
            "any field afterward — the fields are what actually run.",
            className="text-muted d-block mb-2",
        ),
        _field_row("Model", "roigbiv-param-model",
                   dbc.Select(
                       id="roigbiv-param-model",
                       options=_model_opts,
                       value=(_model_opts[0]["value"] if _model_opts
                              else "models/deployed/current_model"),
                   )),
        _field_row("channels (cyto,nucleus)", "roigbiv-param-channels",
                   dbc.Select(
                       id="roigbiv-param-channels",
                       options=[
                           {"label": "Single-channel — mean_M only (0,0)",
                            "value": "0,0"},
                           {"label": "Cyto + vcorr nucleus (1,2)",
                            "value": "1,2"},
                       ],
                       value="1,2",
                   )),
        _field_row("flow_threshold", "roigbiv-param-flow-threshold",
                   dbc.Input(id="roigbiv-param-flow-threshold", type="number",
                             value=0.4, step=0.05, min=0.0, max=3.0)),
        _field_row("cellprob_threshold", "roigbiv-param-cellprob-threshold",
                   dbc.Input(id="roigbiv-param-cellprob-threshold", type="number",
                             value=-2.0, step=0.5, min=-6.0, max=6.0)),
        _field_row("diameter (px)", "roigbiv-param-diameter",
                   dbc.Input(id="roigbiv-param-diameter", type="number",
                             value=diameter_default, step=1, min=3, max=200,
                             debounce=True)),
        html.Div([
            dbc.Button("Suggest from image", id="roigbiv-mc-suggest-btn",
                       size="sm", color="secondary", outline=True, n_clicks=0),
            html.Small(id="roigbiv-mc-diameter-readout",
                       className="text-muted ms-2"),
        ], className="d-flex align-items-center mt-1"),
        html.Small(
            "Soma diameter in pixels. Drag the cyan circle on the "
            "motion-correction preview to match a representative cell, or click "
            "Suggest. Applies to every FOV in the run.",
            className="text-muted d-block mt-1 mb-2",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-use-denoise",
                       label="Cellpose denoise (denoise_cyto3)", value=True),
            "roigbiv-param-use-denoise",
        ),
        _field_row("tile_norm_blocksize", "roigbiv-param-tile-norm-blocksize",
                   dbc.Input(id="roigbiv-param-tile-norm-blocksize", type="number",
                             value=128, step=8, min=0, max=512)),
        html.Hr(className="my-2"),
        html.Small("Gate 1 · morphology bounds", className="text-muted d-block mb-2"),
        _field_row("min_area (px²)", "roigbiv-param-min-area",
                   dbc.Input(id="roigbiv-param-min-area", type="number",
                             value=80, step=10, min=0)),
        _field_row("max_area (px²)", "roigbiv-param-max-area",
                   dbc.Input(id="roigbiv-param-max-area", type="number",
                             value=600, step=50, min=1)),
        _field_row("min_solidity", "roigbiv-param-min-solidity",
                   dbc.Input(id="roigbiv-param-min-solidity", type="number",
                             value=0.55, step=0.05, min=0.0, max=1.0)),
        _field_row("max_eccentricity", "roigbiv-param-max-eccentricity",
                   dbc.Input(id="roigbiv-param-max-eccentricity", type="number",
                             value=0.90, step=0.05, min=0.0, max=1.0)),
    ])
    stage_control = _stage_card("Stage control", [
        html.Small(
            "Foundation, Stage 1 (Cellpose), and gates 1–4 always run.",
            className="text-muted d-block mb-2",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-scout",
                       label="Scout mode — Cellpose-only (fast triage)",
                       value=False),
            "roigbiv-param-scout",
        ),
        html.Small(
            "Skip SVD/L+S/residual; run only Cellpose + Gate 1 for fast FOV-"
            "clarity and model checks. No traces/QC/registry — not analysis-"
            "grade. Overrides the stage toggles below.",
            className="text-muted d-block ms-4 mb-2",
        ),
        html.Hr(className="my-2"),
        _switch_row(
            dbc.Switch(id="roigbiv-param-foundation-only",
                       label="Foundation-only — dry run (motion correction, "
                             "then stop)",
                       value=False),
            "roigbiv-param-foundation-only",
        ),
        html.Small(
            "Run motion correction + SVD/L+S + summary images, then stop before "
            "ROI detection so you can inspect the corrected FOV first. View it in "
            "Review. Re-run with Resume (foundation-only off) to continue. "
            "Overrides the stage toggles below.",
            className="text-muted d-block ms-4 mb-2",
        ),
        html.Hr(className="my-2"),
        _switch_row(
            dbc.Switch(id="roigbiv-param-stage-2",
                       label="Stage 2 — Temporal Detection (Suite2p)",
                       value=True),
            "roigbiv-param-stage-2",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-stage-3",
                       label="Stage 3 — Template Sweep",
                       value=True),
            "roigbiv-param-stage-3",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-stage-4",
                       label="Stage 4 — Tonic Search",
                       value=True),
            "roigbiv-param-stage-4",
        ),
        html.Hr(className="my-2"),
        _switch_row(
            dbc.Switch(id="roigbiv-param-resume", label="Resume", value=False),
            "roigbiv-param-resume",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-override",
                       label="Override previous registry entry", value=False),
            "roigbiv-param-override",
        ),
    ])
    notifications = _stage_card("Notifications", [
        _field_row("Slack channel ID", "roigbiv-param-slack-channel",
                   dbc.Input(id="roigbiv-param-slack-channel", type="text",
                             placeholder="C0123ABCD (optional)")),
        html.Small(
            "Posts a run summary + overlay PNGs to this Slack channel when the "
            "run finishes. Requires ROIGBIV_SLACK_TOKEN exported in the "
            "environment that launched roigbiv-ui. See "
            "docs/slack-notifications.md.",
            id="roigbiv-param-slack-channel-help",
            className="text-muted d-block mt-1",
        ),
    ])
    form = html.Div([foundation, stage1, stage_control, notifications])
    _persist_param_controls(form)
    return form


def _persist_param_controls(tree) -> None:
    """Mark every ``roigbiv-param-*`` control for browser persistence in-place.

    Walks the built form tree and turns on native Dash persistence
    (``localStorage``, constant key — these tunables are workspace-independent)
    for each parameter control, so values survive page navigation / reload
    instead of resetting to the hardcoded ``value=`` defaults. Centralized here
    so new params are covered automatically; the ``_prop_names`` guard skips the
    ``roigbiv-param-*-help`` ``html.Small`` spans that share the id prefix but
    don't support persistence.
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        cid = getattr(node, "id", None)
        if (isinstance(cid, str) and cid.startswith("roigbiv-param-")
                and "persistence" in getattr(node, "_prop_names", ())):
            node.persistence = True
            node.persistence_type = "local"
        children = getattr(node, "children", None)
        if children is not None:
            stack.append(children)


def _stage_control_reactivity(scout, foundation_only) -> tuple:
    """Form state mirroring ``_on_run``'s early-stop precedence.

    Scout and Foundation-only both override the downstream stages; **scout takes
    precedence over foundation-only** (it stops even earlier). Returns, in the
    order the ``_sync_stage_controls`` callback emits them::

        (fo_disabled,
         s2_disabled, s2_value, s3_disabled, s3_value, s4_disabled, s4_value,
         resume_disabled, resume_value)

    Under an early-stop mode the stage 2/3/4 + resume switches go off+disabled;
    otherwise they restore to their on-defaults (resume default is off). The
    foundation-only switch is greyed (not unchecked) under scout — its ``value``
    must stay an Input-only of the callback or Dash flags a circular dependency;
    ``_on_run`` already forces it off under scout, so the run stays correct.
    """
    scout_on = bool(scout)
    foundation_only_on = bool(foundation_only) and not scout_on
    early_stop = scout_on or foundation_only_on
    return (
        scout_on,
        early_stop, not early_stop,
        early_stop, not early_stop,
        early_stop, not early_stop,
        early_stop, False,
    )


def _mc_mean_and_title(value: Optional[str]):
    """Resolve a self-describing dropdown ``value`` to ``(mean, title)``.

    The value is self-describing (see :func:`list_motion_corrected_fovs`):

    * ``summary:{output_dir}`` — a processed FOV; read its precomputed
      ``summary/mean_M.tif``.
    * ``input:{tif_path}`` — a pre-corrected input not yet run; compute a sampled
      temporal mean on demand (:func:`mc_input_mean`).

    ``None`` / unparseable returns ``(None, None)``.
    """
    if value and ":" in value:
        kind, payload = value.split(":", 1)
        if kind == "summary":
            return (_maybe_read_tif(Path(payload) / "summary" / "mean_M.tif"),
                    Path(payload).name)
        if kind == "input":
            return (mc_input_mean(Path(payload)),
                    f"{Path(payload).stem.replace('_mc', '')} (input)")
    return None, None


def _stem_for_value(value: Optional[str]) -> Optional[str]:
    """Light parse of a dropdown ``value`` to a FOV name (no pixel read)."""
    if value and ":" in value:
        _, payload = value.split(":", 1)
        return Path(payload).name
    return None


def _diameter_circle_shape(W: int, H: int, diameter_px: float) -> dict:
    """An editable Plotly circle of ``diameter_px`` centered on the image.

    Drawn in data (pixel) coordinates so its extent reads directly as a soma
    diameter. ``editable`` + the Graph's ``edits.shapePosition`` config give it
    drag handles; the user resizes it to a representative cell.
    """
    r = float(diameter_px) / 2.0
    cx, cy = W / 2.0, H / 2.0
    return dict(
        type="circle", xref="x", yref="y",
        x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
        line=dict(color="#00E5FF", width=2),
        fillcolor="rgba(0,229,255,0.10)",
        editable=True, layer="above",
    )


def _diameter_from_relayout(relayout) -> Optional[float]:
    """Extract the circle diameter (px) from a Graph ``relayoutData`` payload.

    Plotly emits either incremental keys (``shapes[0].x0`` …) or a whole
    ``shapes`` list when a shape is edited; pan/zoom emit axis-range keys
    instead, for which this returns ``None`` (the caller no-ops, breaking any
    figure→relayout feedback loop).

    Diameter is taken from the x-extent (``|x1 - x0|``); the y-axis is reversed
    on the image figure (``range=[H-1, 0]``), so ``y0 > y1`` after a drag — the
    ``abs`` on the y-fallback keeps the result orientation-independent.
    """
    if not isinstance(relayout, dict):
        return None

    sh = None
    if isinstance(relayout.get("shapes"), list) and relayout["shapes"]:
        sh = relayout["shapes"][0]

    def _coord(name):
        if sh is not None and name in sh:
            return sh[name]
        return relayout.get(f"shapes[0].{name}")

    x0, x1 = _coord("x0"), _coord("x1")
    if x0 is not None and x1 is not None:
        try:
            d = abs(float(x1) - float(x0))
            return d if d > 0 else None
        except (TypeError, ValueError):
            return None
    y0, y1 = _coord("y0"), _coord("y1")
    if y0 is not None and y1 is not None:
        try:
            d = abs(float(y1) - float(y0))
            return d if d > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def _coerce_diameter(d) -> Optional[float]:
    """Coerce a form value to a usable diameter (px), or ``None``."""
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    return d if d >= 3.0 else None


def _on_run_diameter(d) -> int:
    """Diameter (int px) for the run config; falls back to the cfg default 12.

    A blank/garbage/too-small field must not poison the run — it degrades to the
    same value as an untouched form, so behaviour matches the pre-feature
    default (diameter=12, diameter_auto off).
    """
    coerced = _coerce_diameter(d)
    return int(round(coerced)) if coerced is not None else 12


def _diameter_overrides(calibrated: Optional[int], field_value) -> dict:
    """Diameter-related pipeline overrides for a run.

    The AppState calibration (drag / type / Suggest, all funnelled there) wins
    over the raw form field, since a drag never writes the number input.
    ``diameter_auto`` is always forced off: stage1 silently overrides
    ``cfg.diameter`` when it's on (stage1.py:259), which would discard the
    user's measured value.
    """
    diam = int(calibrated) if calibrated is not None else _on_run_diameter(field_value)
    return {"diameter": diam, "diameter_auto": False}


def _mc_preview_figure(value: Optional[str], diameter_px: Optional[float] = None):
    """Render the MC preview for a dropdown ``value`` into an overlay-free figure.

    When a ``diameter_px`` is given and a mean image is available, an editable
    reference circle of that diameter is drawn so the user can size it against a
    real soma. ``show_overlay=False`` is the mode :func:`build_roi_figure`
    documents for inspecting MC quality.
    """
    mean, title = _mc_mean_and_title(value)
    fig = build_roi_figure(mean, [], show_overlay=False, title=title)
    if mean is not None and diameter_px:
        H, W = mean.shape
        fig.update_layout(shapes=[_diameter_circle_shape(W, H, diameter_px)])
    return fig


def _mc_options_and_value(workspace, current: Optional[str] = None):
    """Build the MC dropdown ``(options, value)`` from a workspace.

    Shared by the layout seed, the run-tick refresh, and the scan handler so the
    three can't drift. Keeps ``current`` selected if it still exists, else
    defaults to the first FOV. Values are the self-describing strings from
    :func:`list_motion_corrected_fovs`.
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


def _tif_options_and_values(workspace):
    """Build the TIF-selection checklist ``(options, all_values)``.

    One option per ``workspace.tifs`` entry; ``value`` is ``str(tif)`` (the
    resolved path already stored in the workspace — stable and unique, so the
    run can map a selection back to ``Path`` objects). Each label carries the
    validity tick + name + shape so the checklist doubles as the discovery
    summary. Shared by :func:`_workspace_summary` and the scan rebuild so the
    rendered options can't drift.
    """
    options: list[dict] = []
    values: list[str] = []
    if workspace is None:
        return options, values
    for tif in workspace.tifs:
        value = str(tif)
        values.append(value)
        try:
            _, shape = validate_tif(tif)
            label = html.Span([
                html.Span("OK ", className="text-success fw-bold"),
                html.Span(tif.name, className="me-2"),
                html.Span(f"{shape[0]}×{shape[1]}×{shape[2]}",
                          className="text-muted small"),
            ])
        except ValueError as exc:
            label = html.Span([
                html.Span("! ", className="text-danger fw-bold"),
                html.Span(tif.name, className="me-2"),
                html.Span(str(exc), className="text-danger small"),
            ])
        options.append({"label": label, "value": value})
    return options, values


def _sync_select_all_values(trigger, master_value, child_value, all_values):
    """Pure decision core for the Select-all ↔ checklist sync.

    Returns ``(child_value_out, master_value_out)``; either element may be
    :data:`no_update` to leave that control untouched. Breaks the master/child
    feedback loop: a master *uncheck* only clears the children when they are
    *currently* all-selected (a genuine user toggle) — not when it is the
    programmatic echo of a partial child selection that just drove the master
    to empty.
    """
    child_set = set(child_value or [])
    full = set(all_values)
    if trigger == "roigbiv-tif-select-all":
        checked = bool(master_value and "all" in master_value)
        if checked:
            return (no_update if child_set == full else list(all_values)), no_update
        if all_values and child_set == full:
            return [], no_update
        return no_update, no_update
    # Child changed → reflect all-or-not in the master, but only if it differs
    # (else the master update would re-trigger this callback needlessly).
    desired = ["all"] if (all_values and child_set == full) else []
    if list(master_value or []) == desired:
        return no_update, no_update
    return no_update, desired


def _selected_run_paths(workspace, selected):
    """Map the stored selection (set of path strings, or ``None`` = all) to the
    ordered ``Path`` subset of ``workspace.tifs`` to run."""
    return [t for t in workspace.tifs
            if selected is None or str(t) in selected]


def _mc_preview_section() -> html.Div:
    """Read-only motion-correction preview: mean projection per FOV.

    Seeded from the active workspace at render time so a page reload (or a
    workspace with prior outputs / pre-corrected inputs) shows FOVs immediately;
    the interval tick keeps the list fresh as Foundation finishes each FOV during
    a live run, and the scan handler seeds it after an interactive scan.
    """
    state = get_app_state()
    options, value = _mc_options_and_value(state.workspace)
    diam = state.calibrated_diameter() or 12
    # Persist the previewed FOV per workspace; False = no persistence when no
    # workspace is resolved yet (a constant key would leak across workspaces).
    mc_key = str(state.workspace.input_root) if state.workspace else False
    return html.Div([
        html.H5("Motion-correction preview", className="mb-2 mt-3"),
        dbc.Select(id="roigbiv-mc-fov-select", options=options, value=value,
                   className="mb-2",
                   persistence=mc_key, persistence_type="local"),
        dcc.Graph(id="roigbiv-mc-preview",
                  figure=_mc_preview_figure(value, diameter_px=diam),
                  config={"displaylogo": False, "scrollZoom": True,
                          "edits": {"shapePosition": True}},
                  style={"height": "720px"}),
    ])


def _launched_config_summary(snap: Optional["RunSnapshot"]):
    """Read-only echo of the overrides that actually launched the run.

    Rendered from the runner snapshot at page-load time, so it survives
    navigation / reload and — unlike the live, persisted parameter form — never
    misrepresents an in-progress run if the user edits the form afterward.
    Returns ``None`` before any run has started.
    """
    if snap is None or snap.started_at is None or not snap.overrides:
        return None
    ov = snap.overrides

    def _item(label: str, value) -> html.Div:
        return html.Div(
            [html.Span(f"{label}: ", className="text-muted"),
             html.Span(str(value), className="fw-semibold")],
            className="small me-3 d-inline-block",
        )

    if ov.get("scout_mode"):
        stages = "scout"
    elif ov.get("foundation_only"):
        stages = "foundation-only"
    else:
        on = [f"S{n}" for n in (2, 3, 4) if ov.get(f"enable_stage_{n}", True)]
        if ov.get("resume"):
            on.append("resume")
        stages = ", ".join(on) if on else "—"
    model = str(ov.get("cellpose_model", "")).rsplit("/", 1)[-1]
    diam = ov.get("diameter")
    return dbc.Card(dbc.CardBody([
        html.H6("Launched config", className="mb-2"),
        html.Div([
            _item("FOVs", snap.n_fovs),
            _item("fs", ov.get("fs")),
            _item("tau", ov.get("tau")),
            _item("model", model or "—"),
            _item("MC", ov.get("motion_correction_backend")),
            _item("diameter", diam if diam is not None else "auto"),
            _item("channels", ov.get("channels")),
            _item("stages", stages),
        ]),
    ]), className="roigbiv-card-accent mb-3")


def _right_column(snap: Optional["RunSnapshot"] = None) -> html.Div:
    progress, label = _progress_for(snap)
    config_card = _launched_config_summary(snap)
    return html.Div([
        html.H4("Run status", className="mb-3"),
        *([config_card] if config_card is not None else []),
        html.Div(id="roigbiv-run-timer", className="mb-2",
                 children=(_format_timer(snap.started_at, snap.completed_at)
                           if snap else "")),
        dbc.Progress(id="roigbiv-run-progress", value=progress, label=label,
                     striped=True, className="mb-3"),
        html.Div(id="roigbiv-run-banner", children=_render_banner(snap)),
        html.Div(id="roigbiv-run-log",
                 children=log_stream(snap.logs if snap else [])),
        _mc_preview_section(),
        html.Hr(),
        html.Div(id="roigbiv-run-confirm",
                 children=_render_confirm(snap.results_summary if snap else [])),
        html.H5("Per-FOV results", className="mb-2"),
        html.Div(id="roigbiv-run-results",
                 children=_render_results(snap.results_summary if snap else [])),
    ])


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-scan-result", "children"),
        Output("roigbiv-run-btn", "disabled"),
        Output("roigbiv-active-registry", "children", allow_duplicate=True),
        Output("roigbiv-mc-fov-select", "options", allow_duplicate=True),
        Output("roigbiv-mc-fov-select", "value", allow_duplicate=True),
        Input("roigbiv-scan-btn", "n_clicks"),
        State("roigbiv-input-path", "value"),
        prevent_initial_call=True,
    )
    def _on_scan(_n: int, path: Optional[str]):
        state = get_app_state()
        if not path:
            return (dbc.Alert("Enter a path first.", color="warning"), True,
                    no_update, no_update, no_update)
        try:
            workspace = resolve_workspace(Path(path))
        except FileNotFoundError as exc:
            return (dbc.Alert(str(exc), color="danger"), True,
                    no_update, no_update, no_update)
        state.set_workspace(workspace)
        # Seed the MC preview immediately: the run tick is disabled while idle,
        # so without this the list would only populate on page reload or a run.
        options, value = _mc_options_and_value(workspace)
        return (
            _workspace_summary(workspace),
            False,
            f"registry: {workspace.db_path}",
            options,
            value,
        )

    @app.callback(
        Output("roigbiv-active-registry", "children", allow_duplicate=True),
        Input("roigbiv-url", "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _sync_registry_label(_pathname: str):
        state = get_app_state()
        if state.workspace is not None:
            return f"registry: {state.workspace.db_path}"
        return no_update

    @app.callback(
        Output("roigbiv-process-tick", "disabled"),
        Output("roigbiv-run-banner", "children"),
        Input("roigbiv-run-btn", "n_clicks"),
        State("roigbiv-param-fs", "value"),
        State("roigbiv-param-tau", "value"),
        State("roigbiv-param-k", "value"),
        State("roigbiv-param-model", "value"),
        State("roigbiv-param-mc-backend", "value"),
        State("roigbiv-param-flow-threshold", "value"),
        State("roigbiv-param-diameter", "value"),
        State("roigbiv-param-scout", "value"),
        State("roigbiv-param-foundation-only", "value"),
        State("roigbiv-param-stage-2", "value"),
        State("roigbiv-param-stage-3", "value"),
        State("roigbiv-param-stage-4", "value"),
        State("roigbiv-param-resume", "value"),
        State("roigbiv-param-slack-channel", "value"),
        State("roigbiv-param-profile", "value"),
        State("roigbiv-param-channels", "value"),
        State("roigbiv-param-cellprob-threshold", "value"),
        State("roigbiv-param-use-denoise", "value"),
        State("roigbiv-param-min-area", "value"),
        State("roigbiv-param-max-area", "value"),
        State("roigbiv-param-min-solidity", "value"),
        State("roigbiv-param-max-eccentricity", "value"),
        State("roigbiv-param-tile-norm-blocksize", "value"),
        State("roigbiv-param-mc-strip-height", "value"),
        State("roigbiv-param-override", "value"),
        prevent_initial_call=True,
    )
    def _on_run(_n: int, fs, tau, k, model, mc_backend, flow_threshold,
                diameter, scout, foundation_only, stage_2, stage_3, stage_4,
                resume, slack_channel, profile, channels, cellprob_threshold,
                use_denoise, min_area, max_area, min_solidity, max_eccentricity,
                tile_norm_blocksize, mc_strip_height, override):
        state = get_app_state()
        if state.workspace is None:
            return True, dbc.Alert("Scan a workspace first.", color="warning")
        selected = state.selected_tifs
        if selected is not None and len(selected) == 0:
            return True, dbc.Alert("Select at least one TIF to run.",
                                   color="warning")
        scout_on = bool(scout)
        # Foundation-only is a dry run that stops before Stage 1; scout takes
        # precedence if both are toggled (it stops even earlier).
        foundation_only_on = bool(foundation_only) and not scout_on
        early_stop = scout_on or foundation_only_on
        selected_paths = _selected_run_paths(state.workspace, selected)
        # Foundation + control fields, independent of how the optics resolve.
        base = {
            "fs": float(fs or 7.5),
            "tau": float(tau or 1.0),
            "k_background": int(k or 30),
            "motion_correction_backend": mc_backend or "phasecorr",
            "mc_strip_height": int(mc_strip_height) if mc_strip_height is not None else 32,
            "scout_mode": scout_on,
            "foundation_only": foundation_only_on,
            # Scout / foundation-only stop early; the stage toggles are ignored
            # when either is on, and a foundation-only dry run is not resumable.
            "enable_stage_2": False if early_stop else (True if stage_2 is None else bool(stage_2)),
            "enable_stage_3": False if early_stop else (True if stage_3 is None else bool(stage_3)),
            "enable_stage_4": False if early_stop else (True if stage_4 is None else bool(stage_4)),
            "resume": False if (resume is None or early_stop) else bool(resume),
        }
        if (profile or AUTO) == AUTO:
            # Auto: classify the optics + derive gates per-FOV; ignore the manual
            # Stage-1 fields (they're disabled for auto). Pauses for confirmation
            # when uncertain — surfaced in the per-FOV confirmation card.
            from roigbiv.pipeline.run import build_auto_workspace_overrides
            run_tifs = selected_paths if selected_paths else list(state.workspace.tifs)
            overrides = build_auto_workspace_overrides(run_tifs, base)
        else:
            overrides = {
                **base,
                # Concrete profile: a provenance label; the per-field values
                # below are what actually run. auto_scale OFF so the user's
                # explicit form values are never overridden, and no pause.
                "profile": profile,
                "auto_scale": False,
                "assume_optics": True,
                "cellpose_model": model or "models/deployed/current_model",
                "channels": _parse_channels_value(channels),
                "flow_threshold": float(flow_threshold if flow_threshold is not None else 0.4),
                "cellprob_threshold": float(cellprob_threshold if cellprob_threshold is not None else -2.0),
                "use_denoise": bool(use_denoise),
                "tile_norm_blocksize": int(tile_norm_blocksize) if tile_norm_blocksize is not None else 128,
                "min_area": int(min_area) if min_area is not None else 80,
                "max_area": int(max_area) if max_area is not None else 600,
                "min_solidity": float(min_solidity) if min_solidity is not None else 0.55,
                "max_eccentricity": float(max_eccentricity) if max_eccentricity is not None else 0.90,
                # Diameter chosen on the MC preview (+ diameter_auto forced off).
                **_diameter_overrides(state.calibrated_diameter(), diameter),
            }
        # Opt-in: replace (not accumulate) the prior registry entry for each
        # re-run FOV. Set once here so it covers both the AUTO and concrete
        # branches. Popped before PipelineConfig is built (see workspace.py).
        overrides["override"] = bool(override)
        runner = get_pipeline_runner()
        slack_channel = (slack_channel or "").strip() or None
        result = runner.start(state.workspace, overrides,
                              registry_config=state.registry_config,
                              slack_channel=slack_channel,
                              selected_tifs=selected_paths)
        if result == "busy":
            return False, dbc.Alert(
                "Pipeline is running for another session — try again shortly.",
                color="warning",
            )
        if not result:
            return False, dbc.Alert(
                "A pipeline run is already active — wait for it to finish.",
                color="warning",
            )
        # Seed the banner from the fresh snapshot (current_stage = Foundation);
        # the interval tick takes over from here.
        return False, _render_banner(runner.snapshot())

    @app.callback(
        Output("roigbiv-run-log", "children"),
        Output("roigbiv-run-progress", "value"),
        Output("roigbiv-run-progress", "label"),
        Output("roigbiv-run-results", "children"),
        Output("roigbiv-process-tick", "disabled", allow_duplicate=True),
        Output("roigbiv-run-timer", "children"),
        Output("roigbiv-run-banner", "children", allow_duplicate=True),
        Output("roigbiv-run-btn", "disabled", allow_duplicate=True),
        Output("roigbiv-run-confirm", "children"),
        Output("roigbiv-stop-btn", "disabled", allow_duplicate=True),
        Input("roigbiv-process-tick", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def _on_tick(_n):
        runner = get_pipeline_runner()
        snap = runner.snapshot()
        progress, label = _progress_for(snap)
        state = get_app_state()
        return (
            log_stream(snap.logs),
            progress, label,
            _render_results(snap.results_summary),
            not snap.active,
            _format_timer(snap.started_at, snap.completed_at),
            _render_banner(snap),
            state.workspace is None or snap.active,
            _render_confirm(snap.results_summary),
            # Stop is actionable only while a run is in flight (and not already
            # stopping).
            (not snap.active) or snap.stopping,
        )

    @app.callback(
        Output("roigbiv-run-banner", "children", allow_duplicate=True),
        Output("roigbiv-stop-btn", "disabled", allow_duplicate=True),
        Input("roigbiv-stop-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_stop(_n: int):
        # Cooperative stop: flag the in-flight run to halt at the next stage
        # boundary. Disable the button once a stop is requested; the tick
        # refreshes the banner from "Stopping…" to "Run stopped." when it ends.
        runner = get_pipeline_runner()
        requested = runner.abort()
        snap = runner.snapshot()
        return _render_banner(snap), (not requested)

    @app.callback(
        Output("roigbiv-run-banner", "children", allow_duplicate=True),
        Output("roigbiv-process-tick", "disabled", allow_duplicate=True),
        Input({"type": "optics-confirm-btn", "stem": ALL}, "n_clicks"),
        State({"type": "optics-confirm-profile", "stem": ALL}, "value"),
        State({"type": "optics-confirm-profile", "stem": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _on_confirm_optics(clicks, profiles, ids):
        # Fire only on a real click (rendering the cards seeds n_clicks=0).
        if not any(c for c in (clicks or []) if c):
            raise PreventUpdate
        trig = dash.ctx.triggered_id
        if not trig or trig.get("type") != "optics-confirm-btn":
            raise PreventUpdate
        stem = trig.get("stem")
        sel = next((v for pid, v in zip(ids, profiles)
                    if pid.get("stem") == stem), None) or "generic"

        state = get_app_state()
        ws = state.workspace
        if ws is None:
            raise PreventUpdate
        tif = next((t for t in ws.tifs
                    if t.stem.replace("_mc", "") == stem), None)
        if tif is None:
            raise PreventUpdate

        # Resume from Stage 1 on the existing foundation with the confirmed
        # optics. Foundation-relevant fields come from the original run's
        # manifest snapshot so the (relaxed) resume stays consistent.
        from roigbiv.pipeline.optics import _AUTO_SCALE_PROFILES
        from roigbiv.pipeline.profiles import merged_overrides
        from roigbiv.pipeline.resume import read_manifest
        snap_cfg = (read_manifest(ws.output_root / stem) or {}).get(
            "cfg_snapshot", {}) or {}
        base = {
            "fs": snap_cfg.get("fs", 7.5),
            "tau": snap_cfg.get("tau", 1.0),
            "k_background": snap_cfg.get("k_background", 30),
            "motion_correction_backend":
                snap_cfg.get("motion_correction_backend", "phasecorr"),
            "resume": True,
            "assume_optics": True,          # the user just confirmed — don't re-pause
            "auto_scale": sel in _AUTO_SCALE_PROFILES,
            "explicit_fields": (),
            "auto_adapt": {},
            "enable_stage_2": snap_cfg.get("enable_stage_2", True),
            "enable_stage_3": snap_cfg.get("enable_stage_3", True),
            "enable_stage_4": snap_cfg.get("enable_stage_4", True),
            "no_viewer": True,
        }
        overrides = merged_overrides(sel, base, [])
        runner = get_pipeline_runner()
        res = runner.start(ws, overrides,
                           registry_config=state.registry_config,
                           selected_tifs=[tif])
        if res == "busy":
            return dbc.Alert(
                "Pipeline busy for another session — try again shortly.",
                color="warning"), no_update
        if not res:
            return dbc.Alert("A run is already active — wait for it to finish.",
                             color="warning"), no_update
        return _render_banner(runner.snapshot()), False   # re-enable the tick

    @app.callback(
        Output("roigbiv-mc-fov-select", "options", allow_duplicate=True),
        Output("roigbiv-mc-fov-select", "value", allow_duplicate=True),
        Input("roigbiv-process-tick", "n_intervals"),
        State("roigbiv-mc-fov-select", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _refresh_mc_fovs(_n, current):
        # Cheap: enumeration only globs summary names + reads _mc filename
        # suffixes, never pixels (the heavy mean read is in _render_mc_preview,
        # on selection change). Keeps the user's current selection if it still
        # exists. Rides the same 1.5 s run-status interval.
        return _mc_options_and_value(get_app_state().workspace, current)

    @app.callback(
        Output("roigbiv-tif-select", "value", allow_duplicate=True),
        Output("roigbiv-tif-select-all", "value", allow_duplicate=True),
        Input("roigbiv-tif-select-all", "value"),
        Input("roigbiv-tif-select", "value"),
        State("roigbiv-tif-select", "options"),
        prevent_initial_call=True,
    )
    def _on_select_all(master_value, child_value, options):
        # Single combined callback (master + child as inputs) keyed on the
        # trigger id: avoids the destructive feedback loop a two-callback master/
        # child pair hits when the programmatic master update echoes back. Pure
        # logic in _sync_select_all_values.
        all_values = [opt["value"] for opt in (options or [])]
        return _sync_select_all_values(dash.ctx.triggered_id, master_value,
                                       child_value, all_values)

    @app.callback(
        Output("roigbiv-tif-select-sink", "data"),
        Input("roigbiv-tif-select", "value"),
        prevent_initial_call=False,
    )
    def _sync_selected_tifs(value):
        # Mirror the checklist selection into server-side AppState so the run
        # path (_on_run, server-side) reads it without a new callback State.
        # Fires on the *restored* value too (persistence re-mounts the checklist
        # on nav/reload), keeping AppState in step with what's displayed. Guard
        # the workspace-less initial render: the sink lives in the always-present
        # layout, so this can fire before any scan.
        if get_app_state().workspace is None:
            return no_update
        get_app_state().set_selected_tifs(value or [])
        return len(value or [])

    @app.callback(
        Output("roigbiv-mc-preview", "figure"),
        Input("roigbiv-mc-fov-select", "value"),
        Input("roigbiv-param-diameter", "value"),
        prevent_initial_call=True,
    )
    def _render_mc_preview(value, diameter):
        # Single figure writer. Fires on FOV change or number-input change (the
        # input is debounced and Suggest writes it once — so the heavy mean read
        # happens at most once per deliberate change, never per keystroke/tick).
        #
        # Persist to AppState ONLY when the number input is the trigger: a bare
        # FOV switch must not overwrite a diameter the user set by dragging the
        # circle (drags live in AppState, not the box). Always draw at the live
        # AppState diameter so a FOV switch redraws the circle at the measured
        # size rather than the possibly-stale box value.
        state = get_app_state()
        d = _coerce_diameter(diameter)
        if dash.ctx.triggered_id == "roigbiv-param-diameter" and d is not None:
            state.set_calibration(d, _stem_for_value(value))
        draw_d = state.calibrated_diameter() or d
        return _mc_preview_figure(value, diameter_px=draw_d)

    @app.callback(
        Output("roigbiv-mc-diameter-readout", "children"),
        Input("roigbiv-mc-preview", "relayoutData"),
        State("roigbiv-mc-fov-select", "value"),
        prevent_initial_call=True,
    )
    def _on_circle_edit(relayout, value):
        # Drag/resize of the reference circle. We deliberately do NOT write the
        # number input or re-render the figure here: Plotly already moved/resized
        # the shape on-screen, so re-rendering would snap it back to center. We
        # only capture the measured diameter into AppState (read by _on_run) and
        # echo it in the readout. Pan/zoom relayouts carry no shape keys →
        # _diameter_from_relayout returns None → no_update.
        diam = _diameter_from_relayout(relayout)
        if diam is None:
            return no_update
        get_app_state().set_calibration(diam, _stem_for_value(value))
        return f"circle = {diam:.0f} px"

    @app.callback(
        Output("roigbiv-param-diameter", "value", allow_duplicate=True),
        Output("roigbiv-mc-diameter-readout", "children", allow_duplicate=True),
        Input("roigbiv-mc-suggest-btn", "n_clicks"),
        State("roigbiv-mc-fov-select", "value"),
        prevent_initial_call=True,
    )
    def _on_suggest_diameter(_n, value):
        # Estimate the soma diameter from the displayed mean image using the same
        # DoG-peak + Otsu estimator that backs the pipeline's diameter_auto, then
        # pre-fill the number input. Fast (a few hundred ms); runs inline.
        mean, _title = _mc_mean_and_title(value)
        if mean is None:
            return no_update, "Run foundation first to load a FOV."
        from roigbiv.pipeline.stage1 import _estimate_diameter_px
        est = _estimate_diameter_px(mean)
        if est is None or est <= 4.0:
            return no_update, "No estimate — adjust the circle manually."
        return int(round(est)), f"suggested = {est:.0f} px"

    @app.callback(
        Output("roigbiv-param-model", "value", allow_duplicate=True),
        Output("roigbiv-param-channels", "value", allow_duplicate=True),
        Output("roigbiv-param-flow-threshold", "value", allow_duplicate=True),
        Output("roigbiv-param-cellprob-threshold", "value", allow_duplicate=True),
        Output("roigbiv-param-diameter", "value", allow_duplicate=True),
        Output("roigbiv-param-use-denoise", "value", allow_duplicate=True),
        Output("roigbiv-param-min-area", "value", allow_duplicate=True),
        Output("roigbiv-param-max-area", "value", allow_duplicate=True),
        Output("roigbiv-param-min-solidity", "value", allow_duplicate=True),
        Output("roigbiv-param-max-eccentricity", "value", allow_duplicate=True),
        Output("roigbiv-param-tile-norm-blocksize", "value", allow_duplicate=True),
        Output("roigbiv-param-mc-strip-height", "value", allow_duplicate=True),
        Input("roigbiv-param-profile", "value"),
        prevent_initial_call=True,
    )
    def _on_profile_change(profile_name):
        # Autofill every Stage-1/Gate-1 field from the chosen profile bundle.
        # The fields remain user-editable and are the source of truth at run time.
        # ``auto`` resolves per-FOV at run time and ignores these manual fields,
        # so leave them untouched (no_update) when auto is selected.
        if (profile_name or "auto") == AUTO:
            return tuple(no_update for _ in range(12))
        v = _profile_field_values(profile_name)
        return (
            v["cellpose_model"],
            _channels_to_str(v["channels"]),
            v["flow_threshold"],
            v["cellprob_threshold"],
            v["diameter"],
            v["use_denoise"],
            v["min_area"],
            v["max_area"],
            v["min_solidity"],
            v["max_eccentricity"],
            v["tile_norm_blocksize"],
            v["mc_strip_height"],
        )

    @app.callback(
        Output("roigbiv-param-foundation-only", "disabled"),
        Output("roigbiv-param-stage-2", "disabled"),
        Output("roigbiv-param-stage-2", "value"),
        Output("roigbiv-param-stage-3", "disabled"),
        Output("roigbiv-param-stage-3", "value"),
        Output("roigbiv-param-stage-4", "disabled"),
        Output("roigbiv-param-stage-4", "value"),
        Output("roigbiv-param-resume", "disabled"),
        Output("roigbiv-param-resume", "value"),
        Input("roigbiv-param-scout", "value"),
        Input("roigbiv-param-foundation-only", "value"),
        prevent_initial_call=True,
    )
    def _sync_stage_controls(scout, foundation_only):
        # Make the form visibly mirror what _on_run already enforces: scout /
        # foundation-only override the downstream stages (scout precedence).
        # foundation-only.value is Input-only here — never an Output of this
        # callback — or Dash raises a circular dependency.
        return _stage_control_reactivity(scout, foundation_only)


# ── rendering helpers ──────────────────────────────────────────────────────


def _progress_for(snap: Optional["RunSnapshot"]) -> tuple[int, str]:
    """Progress-bar value (0–100) and ``done / total`` label from a snapshot."""
    if snap and snap.n_fovs > 0:
        done = snap.n_done + snap.n_failed
        return int(round(100 * done / snap.n_fovs)), f"{done} / {snap.n_fovs}"
    return 0, ""


def _render_banner(snap: Optional["RunSnapshot"]):
    """Live run-status banner: current stage while active, outcome when done."""
    if snap is None or snap.started_at is None:
        return None
    # Error wins over stopped: a crash on the post-stop path (e.g. backfill)
    # still sets the abort event, so guard stopped on the absence of an error
    # or the failure would be masked as a clean "Run stopped."
    if snap.error:
        return dbc.Alert("Run failed — see log below.",
                         color="danger", className="py-2 mb-2")
    if snap.stopped:
        return dbc.Alert("Run stopped.",
                         color="secondary", className="py-2 mb-2")
    if not snap.active:
        return dbc.Alert("Run complete.",
                         color="success", className="py-2 mb-2")
    if snap.stopping:
        stage = snap.current_stage or "current stage"
        return dbc.Alert(
            [html.Span("Stopping · ", className="fw-bold"),
             html.Span(f"finishing {stage}, then halting…")],
            color="warning", className="py-2 mb-2",
        )
    stage = snap.current_stage or "Pipeline run started"
    return dbc.Alert(
        [html.Span("Running · ", className="fw-bold"), html.Span(stage)],
        color="info", className="py-2 mb-2",
    )


def _workspace_summary(workspace: WorkspacePaths) -> html.Div:
    # The TIF list is a checklist: the user picks which detected stacks to run.
    # All are selected by default; the "Select all" master toggles the set. The
    # checklist lives inside roigbiv-scan-result, so a re-scan rebuilds it (and
    # AppState.set_workspace resets the stored selection to all) — no separate
    # seeding output is needed on the scan callback.
    options, values = _tif_options_and_values(workspace)
    # Persist the selection keyed to workspace identity so it survives reload
    # but never bleeds a stale selection onto a different workspace.
    ws_key = str(workspace.input_root)
    return dbc.Card(dbc.CardBody([
        html.H6("Workspace resolved", className="mb-2"),
        html.Small("Select which detected TIF stacks to run.",
                   className="text-muted d-block mb-2"),
        dbc.Checklist(
            id="roigbiv-tif-select-all",
            options=[{"label": "Select all", "value": "all"}],
            value=["all"] if values else [],
            className="fw-bold mb-1",
            persistence=ws_key, persistence_type="local",
        ),
        dbc.Checklist(
            id="roigbiv-tif-select",
            options=options,
            value=list(values),
            className="ms-3 mb-0",
            persistence=ws_key, persistence_type="local",
        ),
    ]), className="roigbiv-card-accent mt-2")


def _format_timer(
    started_at: Optional[float],
    completed_at: Optional[float],
) -> "str | html.Div":
    import time
    if started_at is None:
        return ""
    start_str = time.strftime("%H:%M:%S", time.localtime(started_at))
    end_ts = completed_at if completed_at is not None else time.time()
    elapsed_s = int(end_ts - started_at)
    h, rem = divmod(elapsed_s, 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
    return html.Div(
        [
            html.Span(f"Started: {start_str}", className="me-4"),
            html.Span(f"Elapsed: {elapsed_str}"),
        ],
        style={
            "fontFamily": "var(--roigbiv-font-mono)",
            "fontSize": "0.80rem",
            "color": "var(--roigbiv-accent)",
        },
    )


def _render_results(summaries: list[dict]) -> html.Div:
    if not summaries:
        return html.Div(html.Em("No FOV results yet.", className="text-muted"))
    rows = []
    for s in summaries:
        status = "FAILED" if s.get("error") else "OK"
        decision = s.get("registry_decision") or "—"
        counts = s.get("roi_counts") or {}
        duration = f"{s.get('duration_s', 0):.1f}s"
        rows.append(html.Tr([
            html.Td(status,
                    className=("text-danger fw-bold" if s.get("error")
                               else "text-success fw-bold")),
            html.Td(s.get("stem")),
            html.Td(duration),
            html.Td(f"A {counts.get('accept', 0)} / "
                    f"F {counts.get('flag', 0)} / "
                    f"R {counts.get('reject', 0)}"),
            html.Td(decision),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th(""), html.Th("FOV"), html.Th("Duration"),
            html.Th("ROIs"), html.Th("Registry"),
        ])), html.Tbody(rows)],
        size="sm", striped=True, borderless=False,
        className="mb-0",
    )


def _render_confirm(summaries: list[dict]) -> html.Div:
    """Per-FOV optics-confirmation cards for FOVs that paused after foundation."""
    awaiting = [s for s in (summaries or []) if s.get("awaiting_confirmation")]
    if not awaiting:
        return html.Div()
    cards = []
    for s in awaiting:
        stem = s.get("stem")
        p = s.get("awaiting_confirmation") or {}
        cand = p.get("candidate_profile", "generic")
        d = p.get("soma_diameter_med")
        reasons = p.get("reasons", []) or []
        cards.append(dbc.Card(dbc.CardBody([
            html.Div([
                html.Strong(stem),
                html.Span(f"  candidate: {cand} ({p.get('confidence', '?')})",
                          className="text-muted"),
            ]),
            html.Div(
                f"measured soma d≈{d if d is not None else '?'}px  "
                f"n={p.get('n_somata', 0)}",
                className="small text-muted mb-1"),
            (html.Ul([html.Li(r, className="small text-muted") for r in reasons],
                     className="mb-2") if reasons else None),
            dbc.InputGroup([
                dbc.InputGroupText("Optics"),
                dbc.Select(
                    id={"type": "optics-confirm-profile", "stem": stem},
                    options=[{"label": pp, "value": pp} for pp in _CONCRETE_PROFILES],
                    value=cand if cand in _CONCRETE_PROFILES else "generic",
                ),
                dbc.Button("Confirm & resume",
                           id={"type": "optics-confirm-btn", "stem": stem},
                           color="primary", n_clicks=0),
            ]),
        ]), color="warning", outline=True, className="mb-2"))
    return html.Div([
        html.H5("Optics confirmation needed", className="mb-2"),
        html.Div(html.Em(
            "Auto-adaptation was uncertain for these FOVs. Confirm the optics "
            "to resume from Stage 1 — foundation is already done, so it's fast.",
            className="text-muted small"), className="mb-2"),
        *cards,
        html.Hr(),
    ])
