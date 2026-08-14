"""Guards for the Motion-correction page.

The page runs Foundation only, and the CLI accepts three MC backends
(rowwise-pcc, phasecorr, legacy); the UI selector must offer the same set so
every CLI-supported backend is reachable from the web interface — and the
default must stay unchanged.

It also must carry *only* motion correction. The page it was split out of held
the workspace scanner and the Cellpose calibration too, which is how "did
motion correction finish" and "did detection work" became one question.
"""
from __future__ import annotations

import pytest

from roigbiv.ui.components.forms import HELP_TEXT
from roigbiv.ui.pages import motion
from roigbiv.ui.pages.motion import _params_form
from roigbiv.ui.tests._tree import find_by_id, h6_texts, ids, walk

MC_BACKEND_ID = "roigbiv-param-mc-backend"

# Every tunable that should carry a hover-help tooltip on this page.
PARAM_IDS = [
    "roigbiv-param-fs",
    "roigbiv-param-tau",
    "roigbiv-param-k",
    "roigbiv-param-mc-backend",
    "roigbiv-param-force-cpu",
    "roigbiv-param-mc-strip-height",
    "roigbiv-param-mc-max-displacement",
    "roigbiv-param-mc-n-template-iters",
    "roigbiv-param-mc-subpixel-upsample",
    "roigbiv-param-mc-frame-batch",
    "roigbiv-param-mc-smooth-sigma-rows",
    "roigbiv-param-mc-smooth-sigma-time",
    "roigbiv-param-mc-strip-confidence-weight",
    "roigbiv-param-mc-prefilter",
    "roigbiv-param-mc-prefilter-sigma-low",
    "roigbiv-param-mc-prefilter-sigma-high",
    "roigbiv-param-mc-sima-env",
    "roigbiv-param-mc-granularity",
    "roigbiv-param-mc-s2p-block-h",
    "roigbiv-param-mc-s2p-block-w",
    "roigbiv-param-mc-s2p-smooth-sigma",
    "roigbiv-param-mc-s2p-smooth-sigma-time",
    "roigbiv-param-mc-s2p-maxregshift",
    "roigbiv-param-mc-s2p-nonrigid",
    "roigbiv-param-mc-s2p-maxregshift-nr",
    "roigbiv-param-mc-s2p-nimg-init",
    "roigbiv-param-mc-s2p-two-step-registration",
    "roigbiv-param-mc-s2p-one-photon-reg",
    "roigbiv-param-mc-s2p-spatial-hp-reg",
    "roigbiv-param-mc-s2p-pre-smooth",
    "roigbiv-param-mc-s2p-spatial-taper",
    "roigbiv-param-slack-channel",
]


class _NoWorkspace:
    workspace = None
    registry_config = None


def _mc_backend_select():
    return find_by_id(_params_form(), MC_BACKEND_ID)


def _option_values(select):
    return [opt["value"] for opt in select.options]


# ── backend selector ───────────────────────────────────────────────────────


def test_mc_backend_options_include_legacy():
    values = _option_values(_mc_backend_select())
    assert set(values) == {"rowwise-pcc", "phasecorr", "legacy"}, (
        f"UI must offer all three CLI backends; got {values}"
    )


def test_mc_backend_default_is_phasecorr():
    # UI default must match the validated pipeline default (phasecorr — Suite2p
    # phase-correlation, tuned to legacy-SIMA parity). rowwise-pcc and legacy are
    # strictly opt-in; rowwise-pcc hazes/bands dim FOVs (the Logan/Prism
    # regression), legacy is CPU-only/slow and needs the sima-legacy env. Guards
    # against the UI silently diverging from PipelineConfig's phasecorr default.
    assert _mc_backend_select().value == "phasecorr"


def test_mc_backend_values_match_cli_choices():
    # The selector's values must be a subset of what
    # foundation.run_motion_correction accepts, or the run fails with a backend
    # ValueError.
    accepted = {"rowwise-pcc", "phasecorr", "legacy"}
    values = set(_option_values(_mc_backend_select()))
    assert values <= accepted, f"UI offers unaccepted backend(s): {values - accepted}"


def test_mc_backend_legacy_option_has_help_text():
    # A help note must accompany the selector so users know legacy needs the
    # sidecar env and is slow.
    help_small = find_by_id(_params_form(), "roigbiv-param-mc-backend-help")
    body = "".join(str(c) for c in
                   (help_small.children if isinstance(help_small.children, (list, tuple))
                    else [help_small.children]))
    assert "sima-legacy" in body and "slow" in body.lower()


# ── the params form ────────────────────────────────────────────────────────


def test_every_param_has_a_tooltip_icon():
    # Each tunable gets a hover-help info icon whose id derives from the param id.
    present = ids(_params_form())
    missing = [pid for pid in PARAM_IDS if f"{pid}-help-icon" not in present]
    assert not missing, f"params without a help-icon tooltip: {missing}"


def test_every_param_field_has_help_text():
    # _field_row / _switch_row hard-subscript HELP_TEXT[target_id]; a field added
    # without a HELP_TEXT entry 500s the whole page on render. Auto-discover the
    # field ids from the rendered form rather than trusting a hand-kept list —
    # the list is exactly what drifted last time.
    field_ids = {
        cid for cid in ids(_params_form())
        if isinstance(cid, str) and cid.startswith("roigbiv-param-")
        and not cid.endswith("-help-icon") and not cid.endswith("-help")
    }
    missing = sorted(cid for cid in field_ids if cid not in HELP_TEXT)
    assert not missing, f"form fields missing HELP_TEXT copy: {missing}"


def test_tooltips_target_their_icons():
    # A dbc.Tooltip must point at each help icon, else hover shows nothing.
    targets = {getattr(c, "target", None) for c in walk(_params_form())
               if type(c).__name__ == "Tooltip"}
    missing = [pid for pid in PARAM_IDS if f"{pid}-help-icon" not in targets]
    assert not missing, f"params without a Tooltip target: {missing}"


def test_params_grouped_into_cards():
    titles = h6_texts(_params_form())
    for expected in ("Foundation", "rowwise-pcc", "legacy (SIMA)",
                     "phasecorr (Suite2p)", "Notifications"):
        assert expected in titles, f"missing group header: {expected!r}"


def test_no_stage1_or_stage_control_params_remain():
    # Stage 1-4 / classification / registry-override controls were removed —
    # this page always runs a foundation-only pass.
    present = ids(_params_form())
    removed = (
        "roigbiv-param-profile", "roigbiv-param-model",
        "roigbiv-param-channels", "roigbiv-param-flow-threshold",
        "roigbiv-param-cellprob-threshold", "roigbiv-param-diameter",
        "roigbiv-param-use-denoise", "roigbiv-param-tile-norm-blocksize",
        "roigbiv-param-min-area", "roigbiv-param-max-area",
        "roigbiv-param-min-solidity", "roigbiv-param-max-eccentricity",
        "roigbiv-param-scout", "roigbiv-param-foundation-only",
        "roigbiv-param-stage-2", "roigbiv-param-stage-3",
        "roigbiv-param-stage-4", "roigbiv-param-resume",
        "roigbiv-param-override",
    )
    leftover = [pid for pid in removed if pid in present]
    assert not leftover, f"stage-1/stage-control ids should be gone: {leftover}"


def test_foundation_group_contains_core_params():
    foundation = next(c for c in walk(_params_form())
                      if type(c).__name__ == "Card"
                      and "Foundation" in h6_texts(c))
    fids = ids(foundation)
    for pid in ("roigbiv-param-fs", "roigbiv-param-tau", "roigbiv-param-k",
                "roigbiv-param-mc-backend", "roigbiv-param-force-cpu"):
        assert pid in fids, f"{pid} should be in the Foundation group"


def test_rowwise_group_contains_strip_regularization_knobs():
    rowwise = next(c for c in walk(_params_form())
                   if type(c).__name__ == "Card"
                   and "rowwise-pcc" in h6_texts(c))
    rids = ids(rowwise)
    for pid in ("roigbiv-param-mc-strip-height",
                "roigbiv-param-mc-smooth-sigma-rows",
                "roigbiv-param-mc-strip-confidence-weight",
                "roigbiv-param-mc-prefilter"):
        assert pid in rids


def test_phasecorr_group_contains_tuned_s2p_knobs():
    phasecorr = next(c for c in walk(_params_form())
                     if type(c).__name__ == "Card"
                     and "phasecorr (Suite2p)" in h6_texts(c))
    pids = ids(phasecorr)
    for pid in ("roigbiv-param-mc-s2p-block-h", "roigbiv-param-mc-s2p-block-w",
                "roigbiv-param-mc-s2p-one-photon-reg"):
        assert pid in pids


def test_legacy_group_contains_sima_knobs():
    legacy = next(c for c in walk(_params_form())
                  if type(c).__name__ == "Card"
                  and "legacy (SIMA)" in h6_texts(c))
    lids = ids(legacy)
    assert "roigbiv-param-mc-sima-env" in lids
    assert "roigbiv-param-mc-granularity" in lids


def test_param_controls_persist_to_localstorage():
    # Every roigbiv-param-* control is marked for localStorage persistence so
    # edits survive a page reload (the *-help spans are correctly skipped).
    form = _params_form()
    persisted = [c for c in walk(form)
                 if isinstance(getattr(c, "id", None), str)
                 and c.id.startswith("roigbiv-param-")
                 and getattr(c, "persistence", None) is True]
    assert len(persisted) >= 20
    for c in persisted:
        assert c.persistence_type == "local"
    fs = find_by_id(form, "roigbiv-param-fs")
    assert fs.persistence is True and fs.persistence_type == "local"


# ── isolation ──────────────────────────────────────────────────────────────


def test_the_page_carries_no_detection_or_workspace_controls(monkeypatch):
    """The whole point of the split.

    Calibration moved to the Centroids page and the scanner to the navbar; a
    control reappearing here would put two operations back on one page.
    """
    monkeypatch.setattr(motion, "get_app_state", lambda: _NoWorkspace())
    present = ids(motion.layout())
    for foreign in ("roigbiv-centroids-diameter", "roigbiv-centroids-threshold",
                    "roigbiv-centroids-model", "roigbiv-input-path",
                    "roigbiv-scan-btn", "roigbiv-tif-select",
                    "roigbiv-run-mode"):
        assert foreign not in present, f"{foreign} does not belong on this page"


def test_the_run_is_foundation_only():
    """No run-mode radio any more — the page *is* the mode."""
    overrides = motion.motion_overrides(
        7.5, 1.0, 30, "phasecorr", False, 32, 50, 2, 10, 256, 6.0, 1.0,
        True, False, 1.0, 8.0, "sima-legacy", "row", 64, 64, 1.15, 0.0, 0.1,
        True, 5, 300, False, True, 42, 0.0, 40.0)
    assert overrides["foundation_only"] is True
    assert overrides["run_centroids"] is False


def test_empty_numeric_inputs_fall_back_to_the_displayed_defaults():
    """A cleared field must not reach PipelineConfig as ``None``."""
    overrides = motion.motion_overrides(
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None)
    assert overrides["fs"] == 7.5
    assert overrides["tau"] == 1.0
    assert overrides["k_background"] == 30
    assert overrides["motion_correction_backend"] == "phasecorr"
    assert overrides["mc_s2p_block_size"] == [64, 64]


# ── preview ────────────────────────────────────────────────────────────────


def test_mc_fov_select_is_dbc_select(monkeypatch):
    # The FOV selector must be a dbc.Select (native <select>, themed by
    # .form-select) like every other selector — not an unstyled dcc.Dropdown.
    monkeypatch.setattr(motion, "get_app_state", lambda: _NoWorkspace())
    sel = find_by_id(motion._mc_preview_section(), motion.FOV_SELECT_ID)
    assert type(sel).__name__ == "Select", (
        f"FOV selector must be dbc.Select, got {type(sel).__name__}"
    )


def test_mc_preview_is_enlarged(monkeypatch):
    # The preview is aspect-locked (build_roi_figure sets scaleanchor="y"), so
    # for a roughly-square FOV the container height is what bounds the displayed
    # image size. Lock the enlarged viewer so it can't silently regress back to
    # a cramped preview.
    monkeypatch.setattr(motion, "get_app_state", lambda: _NoWorkspace())
    graph = find_by_id(motion._mc_preview_section(), motion.PREVIEW_ID)
    height = graph.style["height"]
    assert height.endswith("px"), f"expected a px height, got {height!r}"
    assert int(height[:-2]) >= 600, (
        f"MC preview height must stay enlarged (>=600px); got {height}"
    )


def test_the_preview_draws_no_overlay(monkeypatch):
    """Judging registration and judging detection are different questions.

    An ROI overlay here would answer the second one over a picture meant for
    the first — so the figure carries the heatmap and nothing else.
    """
    import numpy as np

    monkeypatch.setattr(
        motion.fov_select, "mean_and_title",
        lambda _v: (np.zeros((8, 8), dtype=np.float32), "sess01", None))
    fig = motion._preview_figure("summary:/ws/output/sess01")
    assert len(fig.data) == 1, "heatmap only — no ROI scatter traces"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
