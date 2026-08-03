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
        "the original SIMA HMM2D (CPU-only, slow, needs the sima-legacy env). "
        "Skipped for pre-corrected _mc stacks.",
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
        "run finishes. Leave blank to disable. Requires ROIGBIV_SLACK_TOKEN in "
        "the environment that launched roigbiv-ui. See "
        "docs/slack-notifications.md.",
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
