"""The four routes, in operation order, and the paths that used to exist.

Each page renders through the real callback machinery rather than by calling
``layout()`` directly — the split moved state between modules, and a page that
imports cleanly can still fail on render.
"""
from __future__ import annotations

import warnings

import pytest

from roigbiv.ui.app import PAGES, build_app, resolve_route


@pytest.fixture(scope="module")
def client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = build_app()
    return app.server.test_client()


def _render(client, pathname: str):
    return client.post("/_dash-update-component", json={
        "output": "roigbiv-page-content.children",
        "outputs": {"id": "roigbiv-page-content", "property": "children"},
        "inputs": [{"id": "roigbiv-url", "property": "pathname",
                    "value": pathname}],
        "changedPropIds": ["roigbiv-url.pathname"],
        "state": [],
    })


def test_the_pages_are_in_operation_order():
    assert [label for _, label, _ in PAGES] == [
        "Motion correction", "Centroids", "Tracking", "Boundaries"]


@pytest.mark.parametrize("path", [p for p, _, _ in PAGES])
def test_every_page_renders(client, path):
    resp = _render(client, path)
    assert resp.status_code == 200, resp.get_data(as_text=True)[:500]


def test_the_root_serves_the_first_operation():
    page_path, redirect = resolve_route("/")
    assert page_path == "/motion-correction"
    assert redirect == "/motion-correction"


@pytest.mark.parametrize("old,new", [
    ("/pipeline", "/motion-correction"),
    ("/process", "/motion-correction"),
    ("/registry", "/motion-correction"),
    ("/track", "/tracking"),
    ("/cells", "/tracking"),
    ("/viewer", "/tracking"),
])
def test_retired_paths_redirect(old, new):
    """The address bar is rewritten rather than the new page served under the
    old URL: the nav's active="exact" matching keys on the path, and a stale one
    would leave every nav item unhighlighted."""
    page_path, redirect = resolve_route(old)
    assert page_path == new
    assert redirect == new


def test_a_live_page_does_not_redirect():
    assert resolve_route("/boundaries") == ("/boundaries", None)
    assert resolve_route("/boundaries/") == ("/boundaries", None)


def test_an_unknown_path_resolves_to_nothing():
    assert resolve_route("/nope") == (None, None)


def test_an_unknown_path_renders_a_notice_rather_than_a_blank(client):
    resp = _render(client, "/nope")
    assert resp.status_code == 200
    assert "Unknown page" in resp.get_data(as_text=True)


# ── the hazard the split introduced ────────────────────────────────────────


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _mounted_ids(component) -> set:
    seen, stack = set(), [component]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            seen.add(cid)
        children = getattr(node, "children", None)
        if children is not None:
            stack.append(children)
    return seen


def test_no_callback_writes_to_a_component_on_another_page():
    """The failure mode the page split makes possible.

    A callback fires when its *inputs* are on screen. Shared inputs — the run
    tick, the workspace-version store — are on every page, so a callback that
    reads one and writes a page-local component would fire everywhere and try
    to update something that is not there. ``suppress_callback_exceptions``
    hides it until the callback happens to fire, which is why this is checked
    rather than reasoned about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = build_app()

    from roigbiv.ui.components import run_panel, workspace_bar

    # Chrome outside the routed container — present no matter the page.
    chrome = {
        run_panel.TICK_ID, workspace_bar.WORKSPACE_VERSION,
        workspace_bar.OPEN_STORE_ID, workspace_bar.TIF_SINK_ID,
        workspace_bar.TOGGLE_ID, workspace_bar.COLLAPSE_ID,
        workspace_bar.PATH_ID, workspace_bar.SCAN_ID,
        workspace_bar.RESULT_ID, workspace_bar.REGISTRY_ID,
        workspace_bar.TIF_SELECT_ID, workspace_bar.TIF_SELECT_ALL_ID,
        "roigbiv-url", "roigbiv-redirect", "roigbiv-theme",
        "roigbiv-theme-toggle", "roigbiv-theme-toggle-icon",
        "roigbiv-page-content",
    }

    with app.server.test_request_context("/"):
        per_page = {path: _mounted_ids(page.layout()) | chrome
                    for path, _, page in PAGES}

    offenders = []
    for cb in app.callback_map.values():
        outs = {o.component_id for o in _as_list(cb["output"])
                if isinstance(getattr(o, "component_id", None), str)}
        ins = {i.component_id for i in _as_list(cb["inputs"])
               if isinstance(getattr(i, "component_id", None), str)}
        if not outs or not ins:
            continue
        for path, available in per_page.items():
            if ins <= available and not outs <= available:
                offenders.append(
                    f"on {path}: inputs {sorted(ins)} write missing "
                    f"{sorted(outs - available)}")

    assert not offenders, "\n".join(offenders)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
