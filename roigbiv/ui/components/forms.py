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
        "correction (default 32; the prism profile uses 48 for its 1024² FOVs).",
    # Pipeline · Stage 1 Cellpose
    "roigbiv-param-profile":
        "Acquisition/lens profile. Selecting one autofills the Stage-1 and "
        "Gate-1 fields for that optic: grin = the 512² GRIN baseline (pipeline "
        "defaults); prism = dim, diffuse 1024² Prism FOVs (single-channel cyto3, "
        "no denoise, ~56 px somata, relaxed area/shape gates); generic = a "
        "conservative single-channel fallback. The fields stay editable — they, "
        "not the dropdown, are what actually run.",
    "roigbiv-param-channels":
        "Cellpose input channels (cyto, nucleus). Single-channel (0,0) segments "
        "the mean_M image alone — the dominant Prism fix. Cyto+vcorr (1,2) feeds "
        "the correlation map as a nucleus channel (the GRIN default), which "
        "suppresses detection on Prism's diffuse vcorr.",
    "roigbiv-param-cellprob-threshold":
        "Cellpose cell-probability threshold (−6…6). Lower admits dimmer, "
        "lower-confidence masks; higher is stricter. GRIN default −2.0; the "
        "prism profile uses 0.0.",
    "roigbiv-param-use-denoise":
        "Run Cellpose3 denoise_cyto3 before segmentation. Helps bright GRIN "
        "FOVs but suppresses dim Prism signal — the prism profile turns it off.",
    "roigbiv-param-tile-norm-blocksize":
        "Block size (px) for Cellpose tile normalization; 0 disables it. Larger "
        "blocks normalize over a wider area (default 128; prism uses 256 for its "
        "1024² FOVs).",
    "roigbiv-param-min-area":
        "Gate 1 minimum ROI area in px². Masks smaller than this are rejected. "
        "GRIN default 80 (512²); the prism profile raises it to 900 for ~56 px "
        "somata.",
    "roigbiv-param-max-area":
        "Gate 1 maximum ROI area in px². Masks larger than this are rejected as "
        "merged blobs. GRIN default 600 (512²); the prism profile raises it to "
        "9000 (large somata; the peak-count check still flags genuine merges).",
    "roigbiv-param-min-solidity":
        "Gate 1 minimum solidity (filled-area fraction, 0–1). Rejects ragged / "
        "non-convex masks. GRIN default 0.55; prism relaxes to 0.40 for cyto3's "
        "ugly-but-valid masks.",
    "roigbiv-param-max-eccentricity":
        "Gate 1 maximum eccentricity (0–1). Rejects elongated fiber/axon shapes. "
        "GRIN default 0.90; prism relaxes to 0.97.",
    "roigbiv-param-model":
        "Cellpose model checkpoint used for Stage 1 spatial detection. "
        "Defaults to the deployed CP3 model; fine-tuned checkpoints appear "
        "here once trained.",
    "roigbiv-param-flow-threshold":
        "Cellpose flow error tolerance (0–3). Lower is stricter — fewer, "
        "cleaner masks; higher admits more candidate ROIs. Default 0.4.",
    "roigbiv-param-diameter":
        "Expected soma diameter in pixels for Stage 1 Cellpose detection "
        "(default 12). Drag the reference circle on the motion-correction "
        "preview to match a representative cell, or click Suggest to estimate "
        "it from the image. Applies to every FOV in the run; setting it here "
        "disables automatic per-FOV diameter estimation.",
    # Pipeline · Stage control
    "roigbiv-param-scout":
        "Cellpose-only fast triage: runs Stage 1 + Gate 1 and stops. No "
        "SVD/L+S, traces, QC, or registry — for quick FOV-clarity and model "
        "checks, not analysis-grade output. Overrides the stage toggles.",
    "roigbiv-param-foundation-only":
        "Dry run: motion correction + SVD/L+S + summary images, then stop "
        "before ROI detection so you can inspect the corrected FOV in Review. "
        "Re-run with Resume to continue. Overrides the stage toggles.",
    "roigbiv-param-stage-2":
        "Stage 2 — Suite2p temporal detection. Catches burst-firing and "
        "task-locked neurons that Cellpose misses on the spatial pass.",
    "roigbiv-param-stage-3":
        "Stage 3 — matched-filter template sweep on the residual. Recovers "
        "sparse-firing neurons (≈1–15 events) below Cellpose/Suite2p "
        "sensitivity.",
    "roigbiv-param-stage-4":
        "Stage 4 — tonic-neuron search via correlation contrast. Finds "
        "sustained-firing cells (≈2–5 Hz) with little transient structure.",
    "roigbiv-param-resume":
        "Skip stages already completed when the config and input are "
        "unchanged, reusing prior outputs. Ignored under scout / "
        "foundation-only.",
    "roigbiv-param-override":
        "Replace this FOV's prior registry entry instead of accumulating "
        "rows. Drops the previous run's session (and the FOV + cells if that "
        "leaves it with no other sessions) before re-registering. Destructive: "
        "the delete and the re-register are separate steps, so interrupting an "
        "override re-run can leave the prior entry gone — just re-run to "
        "rebuild it. Off by default — normal re-runs are non-destructive.",
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
