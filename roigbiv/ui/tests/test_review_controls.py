"""Static-layout guards for the Review page UX cleanup.

Covers the non-clientside pieces: F-corrected default signal, the FOV/session
consolidation (active session is now a radio, the redundant dropdown is gone),
the Overlay toggle relocation out of the sidebar into the canvas toolbar, the
removal of the workspace-summary fields, and tooltip presence on view + train
controls. The iframe/postMessage/refresh behaviour is browser-only and not
unit-testable here.
"""
import pytest

from roigbiv.ui.pages import review
from roigbiv.ui.pages.review import (
    KIND_OPTIONS,
    _canvas_toolbar,
    _finetune_card,
    _selector_card,
    _view_controls_card,
)


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


def _ids(root):
    return {getattr(c, "id", None) for c in _walk(root)}


def _find_by_id(root, target_id):
    for c in _walk(root):
        if getattr(c, "id", None) == target_id:
            return c
    raise AssertionError(f"id={target_id!r} not found")


def _type(root, target_id):
    return type(_find_by_id(root, target_id)).__name__


def test_signal_default_is_f_corrected():
    kind = _find_by_id(_view_controls_card(), "roigbiv-review-kind")
    assert kind.value == "f", "F corrected must be the default signal"
    # dF/F stays available as an option.
    vals = {o["value"] for o in kind.options}
    assert vals == {"f", "dff"}


def test_kind_options_list_f_first():
    assert KIND_OPTIONS[0][0] == "f"


def test_active_session_is_radio_not_dropdown():
    # The redundant third dropdown is gone; active session is a radio that
    # mirrors the checked sessions.
    assert _type(_selector_card(), "roigbiv-review-active-session") == "RadioItems"
    # The multi-select checklist that feeds the cross-session overlay remains.
    assert _type(_selector_card(), "roigbiv-review-session-check") == "Checklist"
    assert _type(_selector_card(), "roigbiv-review-fov-select") == "Select"


def test_overlay_toggle_moved_to_canvas_toolbar():
    assert "roigbiv-review-overlay" not in _ids(_view_controls_card())
    assert "roigbiv-review-overlay" in _ids(_canvas_toolbar())


def test_workspace_summary_card_removed_from_review():
    # The Input/Output/Registry/TIFs card is no longer imported or used here.
    assert not hasattr(review, "workspace_summary_card")


def test_view_and_train_controls_have_tooltips():
    view_ids = _ids(_view_controls_card())
    for pid in ("roigbiv-review-kind", "roigbiv-review-color"):
        assert f"{pid}-help-icon" in view_ids, f"{pid} missing tooltip icon"
    assert "roigbiv-review-overlay-help-icon" in _ids(_canvas_toolbar())
    train_ids = _ids(_finetune_card())
    for pid in ("roigbiv-trainer-epochs", "roigbiv-trainer-lr"):
        assert f"{pid}-help-icon" in train_ids, f"{pid} missing tooltip icon"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
