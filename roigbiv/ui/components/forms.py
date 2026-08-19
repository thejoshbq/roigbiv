"""Form primitives shared across pages.

Currently: a tiny tooltip helper that pairs a label with a hover-help info
icon, plus the central :data:`HELP_TEXT` copy dictionary keyed by the exact
component id the tooltip describes. Both the Pipeline and Review pages pull
from the same dictionary so wording lives in one place.

Why an info-icon rather than wrapping the input itself in the tooltip target:
``dbc.Tooltip`` fires on hover of its ``target``. Targeting a number-spinner
input makes the tooltip flicker as the spinner gains/loses focus, so we hang
the tooltip off a dedicated ``<i>`` icon whose id is derived from the input id
(``f"{target_id}-help-icon"``).
"""
from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import html


def help_icon(target_id: str, help_text: str) -> list:
    """Return ``[<i info-icon>, dbc.Tooltip]`` for an existing control.

    Use this for controls that render their own label (e.g. ``dbc.Switch``,
    ``segmented``), where wrapping a ``dbc.Label`` is not appropriate. Append
    the returned list next to the control.
    """
    icon_id = f"{target_id}-help-icon"
    return [
        html.I(id=icon_id, className="bi bi-info-circle ms-1 text-muted",
               style={"cursor": "help"}),
        dbc.Tooltip(help_text, target=icon_id, placement="top"),
    ]


def button_tooltip(target_id: str, help_text: str) -> dbc.Tooltip:
    """A ``dbc.Tooltip`` triggered by hovering ``target_id`` directly.

    For buttons: the button's own label is already the hover target, so a
    separate info icon next to it is redundant clutter. Unlike a number
    input, a button doesn't flicker its hover state from an adjacent
    element gaining focus, so targeting it directly is safe here.
    """
    return dbc.Tooltip(help_text, target=target_id, placement="top")


def labeled_with_help(
    label_text: str,
    target_id: str,
    help_text: str,
    *,
    html_for: Optional[str] = None,
) -> html.Span:
    """A ``dbc.Label`` followed by a hover-help info icon + tooltip.

    Parameters
    ----------
    label_text:
        Visible label text.
    target_id:
        Id of the input this row labels; the tooltip icon id is derived from
        it as ``f"{target_id}-help-icon"`` and the label's ``html_for`` defaults
        to it.
    help_text:
        Tooltip copy — what the parameter is and how it affects the pipeline.
    html_for:
        Override the label's ``html_for`` if it differs from ``target_id``.
    """
    return html.Span(
        [
            dbc.Label(label_text, html_for=html_for or target_id,
                      className="mb-0"),
            *help_icon(target_id, help_text),
        ],
        className="d-inline-flex align-items-center",
    )


# ── tooltip copy ─────────────────────────────────────────────────────────────
# Keyed by the exact component id the tooltip describes. One home for wording
# shared across the Pipeline and Review pages.
HELP_TEXT: dict[str, str] = {
    # Pipeline · Foundation
    "roigbiv-param-fs":
        "Acquisition frame rate in Hz. Calibrates GCaMP templates, bandpass "
        "windows, and deconvolution τ. Use 7.5 for this lab's 4×-averaged "
        "(_mc) stacks; 30 only for un-averaged ones. A wrong value miscalibrates "
        "Stages 3–4.",
    "roigbiv-param-tau":
        "GCaMP decay timescale in seconds (GCaMP6s ≈ 1.0). Sets the calcium "
        "kernel used for deconvolution and temporal templates downstream.",
    "roigbiv-param-k":
        "Background SVD rank for Foundation's low-rank + sparse (L+S) "
        "separation. Higher captures more slow background structure but risks "
        "absorbing real signal; 20–50 is typical (default 30).",
    "roigbiv-param-mc-backend":
        "Motion-correction algorithm run in Foundation. phasecorr is the "
        "default — Suite2p rigid/non-rigid phase correlation, tuned to "
        "legacy-SIMA parity on dim Prism FOVs; rowwise-pcc is the opt-in GPU "
        "row-wise non-rigid fast path (can haze/band low-SNR FOVs); legacy is "
        "the genuine SIMA HiddenMarkov2D from the original notebook — CPU-"
        "only and slow (~16 min/session), needs the 'sima-legacy' conda env "
        "(build once with envs/build_sima_legacy.sh). Skipped for pre-"
        "corrected _mc stacks.",
    "roigbiv-param-mc-strip-height":
        "Height in rows of the horizontal strips the row-wise motion correction "
        "registers independently. Larger = steadier but coarser non-rigid "
        "correction (default 32).",
    "roigbiv-param-force-cpu":
        "Force CPU for the SVD/L+S background separation and rowwise-pcc GPU "
        "path, bypassing CUDA even when available. For debugging or GPU-"
        "contended hosts; much slower.",
    # Pipeline · Foundation · rowwise-pcc knobs
    "roigbiv-param-mc-max-displacement":
        "Pixel clamp on the estimated per-strip/frame shift. Shared by "
        "rowwise-pcc and legacy (SIMA). Default 50px.",
    "roigbiv-param-mc-n-template-iters":
        "Number of template-refinement iterations for rowwise-pcc registration "
        "(default 2). More iterations sharpen the reference template at extra "
        "compute cost.",
    "roigbiv-param-mc-subpixel-upsample":
        "Parabolic sub-pixel refinement factor for rowwise-pcc shift estimation "
        "(default 10). Higher gives finer sub-pixel precision.",
    "roigbiv-param-mc-frame-batch":
        "Frames processed per GPU batch for rowwise-pcc (default 256, auto-"
        "capped by available VRAM).",
    "roigbiv-param-mc-smooth-sigma-rows":
        "Per-row displacement-field smoothing (strip regularization) for "
        "rowwise-pcc. This — not the DoG prefilter — is what suppresses the "
        "noise-driven per-row warps on dim/low-SNR FOVs (default 6.0).",
    "roigbiv-param-mc-smooth-sigma-time":
        "Temporal smoothing of the displacement field across a rowwise-pcc "
        "frame batch (default 1.0).",
    "roigbiv-param-mc-strip-confidence-weight":
        "Median + confidence-weighted outlier rejection across strips in "
        "rowwise-pcc registration. On by default — part of the strip "
        "regularization that closes the quality gap vs. legacy SIMA.",
    "roigbiv-param-mc-prefilter":
        "DoG band-pass filter on rowwise-pcc's shift-estimation inputs. Off by "
        "default — it only helps when a structured background dominates; on "
        "white-noise-limited (shot-noise) frames it degrades the correlation "
        "peak. A/B it per dataset.",
    "roigbiv-param-mc-prefilter-sigma-low":
        "DoG prefilter's small-blur sigma (shot-noise suppression). Only used "
        "when the prefilter is enabled (default 1.0).",
    "roigbiv-param-mc-prefilter-sigma-high":
        "DoG prefilter's large-blur sigma (background high-pass). Only used "
        "when the prefilter is enabled (default 8.0).",
    # Pipeline · Foundation · legacy (SIMA) knobs
    "roigbiv-param-mc-sima-env":
        "Conda env hosting the legacy SIMA 1.3.2 sidecar (default "
        "'sima-legacy'; build once with envs/build_sima_legacy.sh).",
    "roigbiv-param-mc-granularity":
        "SIMA HiddenMarkov2D correction granularity: 'row' (matches the "
        "original notebook) or 'frame' (coarser, rigid-only).",
    # Pipeline · Foundation · phasecorr (Suite2p) knobs
    "roigbiv-param-mc-s2p-block-h":
        "Non-rigid registration block height (px). Tuned default 64 — the "
        "[64,64] + 1Preg combination reaches ~103% of legacy-SIMA sharpness on "
        "dim Prism FOVs vs. 58% for the old [128,128]/no-1Preg default. Try "
        "128 for bright, high-SNR 2P data.",
    "roigbiv-param-mc-s2p-block-w":
        "Non-rigid registration block width (px). See block height — tuned "
        "default 64.",
    "roigbiv-param-mc-s2p-smooth-sigma":
        "Spatial Gaussian blur (px) applied to the reference image before "
        "registration (default 1.15).",
    "roigbiv-param-mc-s2p-smooth-sigma-time":
        "Temporal smoothing applied before shift estimation (default 0.0 — "
        "off).",
    "roigbiv-param-mc-s2p-maxregshift":
        "Rigid shift clamp as a fraction of the frame size (default 0.1).",
    "roigbiv-param-mc-s2p-nonrigid":
        "Enable piecewise (non-rigid, block-based) registration on top of the "
        "rigid pass. On by default.",
    "roigbiv-param-mc-s2p-maxregshift-nr":
        "Max non-rigid per-block shift in pixels (default 5).",
    "roigbiv-param-mc-s2p-nimg-init":
        "Number of frames used to build the registration reference image "
        "(default 300).",
    "roigbiv-param-mc-s2p-two-step-registration":
        "Rigid registration pass, then a second non-rigid pass on the result. "
        "Needs the raw (unregistered) movie; off by default.",
    "roigbiv-param-mc-s2p-one-photon-reg":
        "1-photon-style high-pass filter before registration. Raises shift-"
        "estimation SNR on dim/low-contrast (GRIN/Prism) frames — load-bearing "
        "for legacy parity. On by default; turn off for bright, high-SNR 2P "
        "data.",
    "roigbiv-param-mc-s2p-spatial-hp-reg":
        "Spatial high-pass window (px) for the 1-photon-style registration "
        "prefilter (default 42).",
    "roigbiv-param-mc-s2p-pre-smooth":
        "Gaussian smoothing (px) applied before the high-pass prefilter "
        "(default 0.0 — off).",
    "roigbiv-param-mc-s2p-spatial-taper":
        "Edge pixels tapered out of the registration computation (default "
        "40.0).",
    # Pipeline · Notifications
    "roigbiv-param-slack-channel":
        "Posts a run summary and overlay PNGs to this Slack channel when the "
        "run finishes (foundation-only runs have no ROI overlays to attach). "
        "Leave blank to disable. Requires ROIGBIV_SLACK_TOKEN in the "
        "environment that launched roigbiv-ui. See "
        "docs/slack-notifications.md.",
    # Pipeline · Motion page-level
    "roigbiv-motion-run-btn":
        "Runs Foundation only — motion correction, background separation "
        "and summary images. Centroid detection is on its own page.",
    # Discovery · calibration & detection
    "roigbiv-discovery-diameter":
        "Per-FOV — a measured soma size, not a global setting.",
    "roigbiv-discovery-persist-flows":
        "The cached flow field is what boundary tuning below draws from. "
        "Turning it off saves ~6 MB per 512² FOV and makes seeded boundaries "
        "impossible for that FOV.",
    "roigbiv-discovery-run-btn":
        "Requires an already motion-corrected stack — a pre-corrected input, "
        "or a prior motion-correction run's output. It does not run motion "
        "correction first.",
    # Discovery · boundary tuning & extraction
    "roigbiv-discovery-capture":
        "How far a flow trajectory may land from a seed and still be that "
        "cell. Recomputes instantly against a cached flow field — nothing "
        "here touches the GPU.",
    "roigbiv-discovery-extract-stats":
        "Extracts every ROI's fluorescence trace for the whole session from "
        "this FOV's current merged_masks.tif. Mean is always extracted; "
        "median and mode are additional per-frame spatial statistics over "
        "the same masked pixels, each neuropil-corrected the same way as "
        "mean.",
    # Discovery · preview canvas gestures
    "roigbiv-discovery-edit":
        "drag to move · right-click to delete · click empty space to add",
    "roigbiv-discovery-edit-boundary":
        "click a cell to start a hand-drawn outline · click to add a point · "
        "double-click or Enter to close · Escape to cancel · right-click a "
        "hand-drawn outline to revert it to auto",
    # Tracking · setup
    "roigbiv-track-setup-toggle":
        "Drag sessions into the order they were recorded. This order decides "
        "which session owns each cell's identity, so it is worth getting "
        "right.",
    # Tracking · contact sheet
    "roigbiv-cells-edit":
        "drag to move · right-click to delete · click empty space to add · "
        "ctrl-click a dashed outline to confirm the cell is there · shift-"
        "click to link to the selected cell · Ctrl+Z to undo",
    "roigbiv-cells-list-heading":
        "Numbers are for reading this page only — they follow the session "
        "order and are not the cell's registry id.",
    # Review · view controls
    "roigbiv-review-kind":
        "Trace normalization. F corrected (neuropil-subtracted fluorescence) "
        "is the default; dF/F is the legacy baseline-normalized signal. Both "
        "are read from disk, not converted in-app.",
    "roigbiv-review-color":
        "How ROI outlines are colored: a single color, by detection Stage, by "
        "Feature class, or by Cross-session identity (matched cells share a "
        "color across days).",
    "roigbiv-review-overlay":
        "Toggle the ROI outline overlay on the editor canvas. Turn off to "
        "inspect the raw mean projection underneath.",
    # Review · fine-tune
    "roigbiv-trainer-epochs":
        "Number of Cellpose fine-tuning passes over the HITL-corrected masks. "
        "More epochs fit the corrections harder but cost time and can overfit "
        "small correction sets (default 200).",
    "roigbiv-trainer-lr":
        "Optimizer learning rate for fine-tuning. Larger adapts faster but "
        "risks diverging from the pretrained CP3 weights; 0.01–0.1 is typical "
        "(default 0.05).",
}
