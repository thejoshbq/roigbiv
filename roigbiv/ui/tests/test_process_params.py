"""Guards for the Process page motion-correction backend selector.

The CLI accepts three MC backends (rowwise-pcc, phasecorr, legacy); the UI
selector must offer the same set so every CLI-supported backend is reachable
from the web interface — and the default must stay unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from roigbiv.ui.components.forms import HELP_TEXT
from roigbiv.ui.pages.process import _params_form

MC_BACKEND_ID = "roigbiv-param-mc-backend"

# Every tunable that should carry a hover-help tooltip on the Pipeline page.
PARAM_IDS = [
    "roigbiv-param-fs",
    "roigbiv-param-tau",
    "roigbiv-param-k",
    "roigbiv-param-mc-backend",
    "roigbiv-param-mc-strip-height",
    "roigbiv-param-profile",
    "roigbiv-param-model",
    "roigbiv-param-channels",
    "roigbiv-param-flow-threshold",
    "roigbiv-param-cellprob-threshold",
    "roigbiv-param-diameter",
    "roigbiv-param-use-denoise",
    "roigbiv-param-tile-norm-blocksize",
    "roigbiv-param-min-area",
    "roigbiv-param-max-area",
    "roigbiv-param-min-solidity",
    "roigbiv-param-max-eccentricity",
    "roigbiv-param-scout",
    "roigbiv-param-foundation-only",
    "roigbiv-param-stage-2",
    "roigbiv-param-stage-3",
    "roigbiv-param-stage-4",
    "roigbiv-param-resume",
    "roigbiv-param-slack-channel",
]


def _walk(component):
    """Yield component and every descendant in its Dash children tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        # skip raw strings / numbers — only Dash components have children
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


def _ids(root):
    return {getattr(c, "id", None) for c in _walk(root)}


def _h6_texts(root):
    out = []
    for c in _walk(root):
        if type(c).__name__ == "H6":
            kids = c.children
            out.append(kids if isinstance(kids, str) else str(kids))
    return out


def _find_by_id(root, target_id):
    for comp in _walk(root):
        if getattr(comp, "id", None) == target_id:
            return comp
    raise AssertionError(f"component id={target_id!r} not found in layout")


def _mc_backend_select():
    return _find_by_id(_params_form(), MC_BACKEND_ID)


def _option_values(select):
    return [opt["value"] for opt in select.options]


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
    # The selector's values must be a subset of what foundation.run_motion_correction
    # accepts, or the run fails with a backend ValueError.
    accepted = {"rowwise-pcc", "phasecorr", "legacy"}
    values = set(_option_values(_mc_backend_select()))
    assert values <= accepted, f"UI offers unaccepted backend(s): {values - accepted}"


def test_mc_backend_legacy_option_has_help_text():
    # A help note must accompany the selector so users know legacy needs the
    # sidecar env and is slow.
    help_small = _find_by_id(_params_form(), "roigbiv-param-mc-backend-help")
    text = "".join(str(c) for c in
                    (help_small.children if isinstance(help_small.children, (list, tuple))
                     else [help_small.children]))
    assert "sima-legacy" in text and "slow" in text.lower()


def test_every_param_has_a_tooltip_icon():
    # Each tunable gets a hover-help info icon whose id derives from the param id.
    ids = _ids(_params_form())
    missing = [pid for pid in PARAM_IDS if f"{pid}-help-icon" not in ids]
    assert not missing, f"params without a help-icon tooltip: {missing}"


def test_every_param_field_has_help_text():
    # _field_row / _switch_row hard-subscript HELP_TEXT[target_id]; a field added
    # without a HELP_TEXT entry 500s the whole Process page on render (this is
    # what KeyError: 'roigbiv-param-slack-channel' was). Auto-discover the field
    # ids from the rendered form rather than trusting a hand-kept list — the list
    # is exactly what drifted last time.
    field_ids = {
        cid for cid in _ids(_params_form())
        if isinstance(cid, str) and cid.startswith("roigbiv-param-")
        and not cid.endswith("-help-icon") and not cid.endswith("-help")
    }
    missing = sorted(cid for cid in field_ids if cid not in HELP_TEXT)
    assert not missing, f"form fields missing HELP_TEXT copy: {missing}"


def test_tooltips_target_their_icons():
    # A dbc.Tooltip must point at each help icon, else hover shows nothing.
    form = _params_form()
    targets = {getattr(c, "target", None) for c in _walk(form)
               if type(c).__name__ == "Tooltip"}
    missing = [pid for pid in PARAM_IDS if f"{pid}-help-icon" not in targets]
    assert not missing, f"params without a Tooltip target: {missing}"


def test_params_grouped_into_stage_cards():
    # The flat form is now grouped under per-stage headers.
    titles = _h6_texts(_params_form())
    for expected in ("Foundation", "Stage 1 · Cellpose detection", "Stage control"):
        assert expected in titles, f"missing stage group header: {expected!r}"


def test_foundation_group_contains_calibration_params():
    # k_background lives under Foundation (background SVD rank), not Stage 1.
    from roigbiv.ui.pages.process import _stage_card  # noqa: F401 — sanity import
    foundation = next(c for c in _walk(_params_form())
                      if type(c).__name__ == "Card"
                      and "Foundation" in _h6_texts(c))
    fids = _ids(foundation)
    for pid in ("roigbiv-param-fs", "roigbiv-param-tau", "roigbiv-param-k",
                "roigbiv-param-mc-backend"):
        assert pid in fids, f"{pid} should be in the Foundation group"
    # Cellpose params must NOT be in Foundation.
    assert "roigbiv-param-flow-threshold" not in fids


def test_stage1_group_contains_cellpose_params():
    stage1 = next(c for c in _walk(_params_form())
                  if type(c).__name__ == "Card"
                  and any("Stage 1" in t for t in _h6_texts(c)))
    sids = _ids(stage1)
    assert "roigbiv-param-model" in sids
    assert "roigbiv-param-flow-threshold" in sids
    # diameter is the Cellpose soma-scale knob, set on the MC preview circle.
    assert "roigbiv-param-diameter" in sids
    assert "roigbiv-mc-suggest-btn" in sids


def test_profile_select_offers_concrete_profiles_only():
    # The Profile dropdown lists concrete profiles (no 'auto') and defaults to
    # grin — the no-op 512² baseline, so a fresh form == the prior behavior.
    sel = _find_by_id(_params_form(), "roigbiv-param-profile")
    values = _option_values(sel)
    assert set(values) == {"grin", "prism", "generic"}
    assert "auto" not in values
    assert sel.value == "grin"


def test_profile_field_values_prism_applies_the_levers():
    # The autofill resolver pulls every PRISM lever from the profile bundle,
    # falling back to the grin/dataclass baseline for unset keys.
    from roigbiv.ui.pages.process import _profile_field_values

    v = _profile_field_values("prism")
    assert v["channels"] == (0, 0)
    assert v["cellpose_model"] == "cyto3"
    assert v["use_denoise"] is False
    assert v["diameter"] == 56
    assert v["min_area"] == 900 and v["max_area"] == 9000
    assert v["cellprob_threshold"] == 0.0
    assert v["mc_strip_height"] == 48


def test_profile_field_values_grin_is_dataclass_baseline():
    from roigbiv.pipeline.types import PipelineConfig
    from roigbiv.ui.pages.process import _PROFILE_AUTOFILL, _profile_field_values

    base = PipelineConfig()
    v = _profile_field_values("grin")
    for key, _id in _PROFILE_AUTOFILL:
        assert v[key] == getattr(base, key), f"grin {key} must match the dataclass default"


def test_channels_str_round_trip():
    from roigbiv.ui.pages.process import _channels_to_str, _parse_channels_value

    assert _channels_to_str((0, 0)) == "0,0"
    assert _parse_channels_value("0,0") == (0, 0)
    assert _parse_channels_value("1,2") == (1, 2)
    assert _parse_channels_value(None) == (1, 2)        # bad input → GRIN default


def test_mc_fov_select_is_dbc_select(monkeypatch):
    # The MC-preview FOV selector must be a dbc.Select (native <select>, themed
    # by .form-select) like every other selector — not an unstyled dcc.Dropdown.
    # _mc_preview_section reads get_app_state().workspace, which needs a Flask
    # request context; stub it with an empty workspace so the section builds.
    import roigbiv.ui.pages.process as proc

    class _FakeState:
        workspace = None

        def calibrated_diameter(self):
            return None

    monkeypatch.setattr(proc, "get_app_state", lambda: _FakeState())
    sel = _find_by_id(proc._mc_preview_section(), "roigbiv-mc-fov-select")
    assert type(sel).__name__ == "Select", (
        f"FOV selector must be dbc.Select, got {type(sel).__name__}"
    )


def test_mc_preview_is_enlarged(monkeypatch):
    # The MC preview is aspect-locked (build_roi_figure sets scaleanchor="y"),
    # so for a roughly-square FOV the container height is what bounds the
    # displayed image size. Lock the enlarged viewer (~2x the original 360px)
    # so it can't silently regress back to a cramped preview. Non-brittle: any
    # height >= 600px satisfies the "displayed large" intent.
    import roigbiv.ui.pages.process as proc

    class _FakeState:
        workspace = None

        def calibrated_diameter(self):
            return None

    monkeypatch.setattr(proc, "get_app_state", lambda: _FakeState())
    graph = _find_by_id(proc._mc_preview_section(), "roigbiv-mc-preview")
    height = graph.style["height"]
    assert height.endswith("px"), f"expected a px height, got {height!r}"
    assert int(height[:-2]) >= 600, (
        f"MC preview height must stay enlarged (>=600px); got {height}"
    )


def test_stage_control_reactivity():
    # The form must mirror _on_run's early-stop precedence: scout / foundation-only
    # force stage 2/3/4 + resume off+disabled; scout takes precedence over
    # foundation-only. Tuple order:
    #   (fo_disabled, s2_disabled, s2_value, s3_disabled, s3_value,
    #    s4_disabled, s4_value, resume_disabled, resume_value)
    from roigbiv.ui.pages.process import _stage_control_reactivity as react

    neither = (False, False, True, False, True, False, True, False, False)
    scout   = (True,  True,  False, True,  False, True,  False, True,  False)
    found   = (False, True,  False, True,  False, True,  False, True,  False)

    assert react(False, False) == neither
    assert react(True, False) == scout
    assert react(False, True) == found
    assert react(True, True) == scout          # scout precedence over foundation-only
    assert react(None, None) == neither        # None (unset switch) coerces to off


def test_mc_preview_figure_input_branch(monkeypatch):
    # An "input:"-prefixed value renders via mc_input_mean (the on-demand mean
    # of a pre-corrected stack), not by reading a summary/mean_M.tif.
    import numpy as np

    import roigbiv.ui.pages.process as proc

    called = {}

    def _fake_mean(path):
        called["path"] = path
        return np.zeros((8, 8), dtype=np.float32)

    def _fail_read(_path):  # the summary reader must NOT be hit on this branch
        raise AssertionError("input branch must not read a summary tif")

    monkeypatch.setattr(proc, "mc_input_mean", _fake_mean)
    monkeypatch.setattr(proc, "_maybe_read_tif", _fail_read)
    fig = proc._mc_preview_figure("input:/data/session01_mc.tif")
    assert fig is not None
    assert str(called["path"]).endswith("session01_mc.tif")


def test_mc_preview_figure_summary_branch(monkeypatch):
    # A "summary:"-prefixed value reads the FOV's precomputed mean_M.tif.
    import numpy as np

    import roigbiv.ui.pages.process as proc

    seen = {}

    def _fake_read(path):
        seen["path"] = path
        return np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(proc, "_maybe_read_tif", _fake_read)
    fig = proc._mc_preview_figure("summary:/ws/output/session01")
    assert fig is not None
    assert str(seen["path"]).endswith("session01/summary/mean_M.tif")


def test_mc_options_and_value_keeps_current(monkeypatch):
    # The shared option-builder keeps the current selection if it still exists,
    # else defaults to the first FOV.
    import roigbiv.ui.pages.process as proc

    monkeypatch.setattr(proc, "list_motion_corrected_fovs",
                        lambda _ws: [("a", "summary:/o/a"), ("b (input)", "input:/i/b_mc.tif")])
    opts, value = proc._mc_options_and_value(object(), current="input:/i/b_mc.tif")
    assert [o["value"] for o in opts] == ["summary:/o/a", "input:/i/b_mc.tif"]
    assert value == "input:/i/b_mc.tif"            # preserved
    _, value2 = proc._mc_options_and_value(object(), current="gone")
    assert value2 == "summary:/o/a"                # falls back to first
    monkeypatch.setattr(proc, "list_motion_corrected_fovs", lambda _ws: [])
    assert proc._mc_options_and_value(object()) == ([], None)


# ── TIF-selection checklist ─────────────────────────────────────────────────


def test_workspace_summary_renders_checklist(monkeypatch):
    # The workspace summary lists detected TIFs as a checklist, all selected by
    # default, with a "Select all" master that starts checked. Option values are
    # str(tif) so the run can map a selection back to Path objects.
    import roigbiv.ui.pages.process as proc

    monkeypatch.setattr(proc, "validate_tif", lambda _t: (None, (10, 32, 32)))
    tifs = (Path("/ws/a_mc.tif"), Path("/ws/b_mc.tif"))
    summary = proc._workspace_summary(SimpleNamespace(tifs=tifs))

    child = _find_by_id(summary, "roigbiv-tif-select")
    assert type(child).__name__ == "Checklist"
    assert [o["value"] for o in child.options] == [str(t) for t in tifs]
    assert child.value == [str(t) for t in tifs]          # all selected

    master = _find_by_id(summary, "roigbiv-tif-select-all")
    assert type(master).__name__ == "Checklist"
    assert master.value == ["all"]


def test_select_all_pure_logic():
    # _sync_select_all_values is the loop-free decision core for the master ↔
    # child checklist sync. no_update means "leave this control untouched".
    from dash import no_update

    from roigbiv.ui.pages.process import _sync_select_all_values as sync

    allv = ["a", "b", "c"]
    M, C = "roigbiv-tif-select-all", "roigbiv-tif-select"

    # Master checked → drive children to all (when not already all).
    assert sync(M, ["all"], [], allv) == (allv, no_update)
    assert sync(M, ["all"], allv, allv) == (no_update, no_update)
    # Master unchecked from a full set → clear children.
    assert sync(M, [], allv, allv) == ([], no_update)
    # Master empty but children partial = programmatic echo → leave children.
    assert sync(M, [], ["a"], allv) == (no_update, no_update)
    # Child became full → reflect in master.
    assert sync(C, [], allv, allv) == (no_update, ["all"])
    # Child partial → master clears.
    assert sync(C, ["all"], ["a"], allv) == (no_update, [])
    # Child partial, master already empty → no master update (loop break).
    assert sync(C, [], ["a"], allv) == (no_update, no_update)


def test_selected_run_paths_maps_subset():
    from roigbiv.ui.pages.process import _selected_run_paths

    tifs = (Path("/ws/a_mc.tif"), Path("/ws/b_mc.tif"))
    ws = SimpleNamespace(tifs=tifs)
    assert _selected_run_paths(ws, None) == list(tifs)        # None → all
    assert _selected_run_paths(ws, {str(tifs[1])}) == [tifs[1]]
    assert _selected_run_paths(ws, set()) == []               # empty → none


def test_app_state_selection_round_trip():
    # set_workspace seeds the selection to all; set_selected_tifs replaces it;
    # an empty selection is the guarded "run nothing" state.
    from roigbiv.ui.services.app_state import AppState

    ws = SimpleNamespace(
        tifs=(Path("/ws/a_mc.tif"), Path("/ws/b_mc.tif")),
        db_dsn="sqlite:///x.db", blob_root=Path("/b"),
        calibration_path=Path("/c.json"), db_path=Path("/x.db"),
    )
    st = AppState()
    st.set_workspace(ws)
    assert st.selected_tifs == {str(t) for t in ws.tifs}
    st.set_selected_tifs([str(ws.tifs[0])])
    assert st.selected_tifs == {str(ws.tifs[0])}
    st.set_selected_tifs([])
    assert st.selected_tifs == set()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
