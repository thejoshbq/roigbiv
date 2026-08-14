"""Guards for the workspace disclosure in the navbar.

Scanning used to belong to the Pipeline page, which meant every other page's
empty state named a page instead of offering the field. Moving it to the navbar
made one thing load-bearing: pages no longer learn about a scan by having their
controls written into from outside, they learn about it from
``WORKSPACE_VERSION``. A scan that fails must therefore *not* bump it.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from dash import no_update

from roigbiv.ui.components import workspace_bar as wb
from roigbiv.ui.tests._tree import find_by_id, ids, text


def _capture_callbacks():
    captured = {}

    class _App:
        def callback(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    wb.register_callbacks(_App())
    return captured


class _FakeState:
    def __init__(self):
        self.workspace = None
        self.selected = None

    def set_workspace(self, workspace):
        self.workspace = workspace

    def set_selected_tifs(self, values):
        self.selected = set(values or [])


# ── the summary label ──────────────────────────────────────────────────────


def test_the_label_asks_for_a_workspace_when_there_is_none():
    assert "scan one" in text(SimpleNamespace(children=wb.summary_label(None)))


def test_the_label_names_the_workspace_and_its_size():
    ws = SimpleNamespace(input_root=Path("/data/prism_3sess"),
                         tifs=(Path("/a.tif"), Path("/b.tif")))
    body = text(SimpleNamespace(children=wb.summary_label(ws)))
    assert "prism_3sess" in body and "2 TIFs" in body


def test_the_registry_indicator_moved_into_the_disclosure():
    """It describes the workspace, not the navigation."""
    assert "scan a workspace" in wb.registry_label(None)
    ws = SimpleNamespace(db_path=Path("/ws/output/registry.db"))
    assert "registry.db" in wb.registry_label(ws)


# ── scanning ───────────────────────────────────────────────────────────────


def test_a_successful_scan_bumps_the_version_and_closes_the_disclosure(monkeypatch):
    ws = SimpleNamespace(input_root=Path("/ws"), output_root=Path("/ws/output"),
                         db_path=Path("/ws/output/registry.db"), tifs=())
    state = _FakeState()
    monkeypatch.setattr(wb, "get_app_state", lambda: state)
    monkeypatch.setattr(wb, "resolve_workspace", lambda _p: ws)

    _result, _registry, _label, version, is_open = _capture_callbacks()["_on_scan"](
        1, "/ws", 3)

    assert version == 4, "pages refresh off this counter"
    assert is_open is False
    assert state.workspace is ws


def test_a_failed_scan_does_not_bump_the_version(monkeypatch):
    """Otherwise pages refresh against the workspace they already had, and the
    disclosure closes over an error nobody got to read."""
    monkeypatch.setattr(wb, "get_app_state", lambda: _FakeState())
    monkeypatch.setattr(wb, "resolve_workspace",
                        lambda _p: (_ for _ in ()).throw(
                            FileNotFoundError("no such directory")))

    result, _registry, _label, version, is_open = _capture_callbacks()["_on_scan"](
        1, "/nope", 3)

    assert version is no_update
    assert is_open is True
    assert "no such directory" in text(result)


def test_an_empty_path_is_refused_without_touching_the_version(monkeypatch):
    monkeypatch.setattr(wb, "get_app_state", lambda: _FakeState())

    result, _registry, _label, version, is_open = _capture_callbacks()["_on_scan"](
        1, "", 0)

    assert version is no_update and is_open is True
    assert "Enter a path" in text(result)


# ── the disclosure ─────────────────────────────────────────────────────────


def test_the_store_is_the_only_answer_to_open_or_closed():
    """A remount zeroes n_clicks; deriving state from it would invert a
    deliberate choice on every navigation."""
    cbs = _capture_callbacks()
    assert cbs["_on_toggle"](1, True) is False
    assert cbs["_on_toggle"](1, False) is True
    assert cbs["_apply_open"](True) is True
    assert cbs["_apply_open"](None) is False


def test_the_disclosure_opens_itself_when_there_is_no_workspace(monkeypatch):
    monkeypatch.setattr(wb, "current_workspace", lambda: None)
    assert wb.collapse().is_open is True


# ── the TIF checklist ──────────────────────────────────────────────────────


def test_workspace_summary_renders_checklist(monkeypatch):
    # Detected TIFs as a checklist, all selected by default, with a "Select all"
    # master that starts checked. Option values are str(tif) so the run can map a
    # selection back to Path objects.
    monkeypatch.setattr(wb, "validate_tif", lambda _t: (None, (10, 32, 32)))
    tifs = (Path("/ws/a_mc.tif"), Path("/ws/b_mc.tif"))
    summary = wb.workspace_summary(
        SimpleNamespace(tifs=tifs, input_root=Path("/ws"),
                        output_root=Path("/ws/output")))

    child = find_by_id(summary, wb.TIF_SELECT_ID)
    assert type(child).__name__ == "Checklist"
    assert [o["value"] for o in child.options] == [str(t) for t in tifs]
    assert child.value == [str(t) for t in tifs]          # all selected
    # Selection persists per workspace so it survives a reload — and the move
    # to the navbar kept the id, so the stored key still matches.
    assert child.persistence == "/ws" and child.persistence_type == "local"

    master = find_by_id(summary, wb.TIF_SELECT_ALL_ID)
    assert type(master).__name__ == "Checklist"
    assert master.value == ["all"]


def test_an_invalid_tif_is_marked_rather_than_dropped(monkeypatch):
    def _validate(tif):
        if tif.name.startswith("bad"):
            raise ValueError("only 2 dimensions")
        return None, (10, 32, 32)

    monkeypatch.setattr(wb, "validate_tif", _validate)
    options, values = wb.tif_options_and_values(
        SimpleNamespace(tifs=(Path("/ws/good.tif"), Path("/ws/bad.tif"))))

    assert len(options) == 2 and len(values) == 2
    assert "only 2 dimensions" in text(SimpleNamespace(
        children=[options[1]["label"]]))


def test_select_all_pure_logic():
    # sync_select_all_values is the loop-free decision core for the master ↔
    # child checklist sync. no_update means "leave this control untouched".
    allv = ["a", "b", "c"]
    M, C = wb.TIF_SELECT_ALL_ID, wb.TIF_SELECT_ID
    sync = wb.sync_select_all_values

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
    tifs = (Path("/ws/a_mc.tif"), Path("/ws/b_mc.tif"))
    ws = SimpleNamespace(tifs=tifs)
    assert wb.selected_run_paths(ws, None) == list(tifs)        # None → all
    assert wb.selected_run_paths(ws, {str(tifs[1])}) == [tifs[1]]
    assert wb.selected_run_paths(ws, set()) == []               # empty → none


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


def test_the_stores_exist_before_any_page_mounts(monkeypatch):
    """The selection-sync sink fires on the checklist's *restored* value, so it
    has to be in the tree from the first render."""
    monkeypatch.setattr(wb, "current_workspace", lambda: None)
    present = ids(wb.stores())
    assert {wb.WORKSPACE_VERSION, wb.OPEN_STORE_ID, wb.TIF_SINK_ID} <= present


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
