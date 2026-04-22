"""Deprecation shim for the legacy Streamlit UI.

The ROIGBIV frontend has moved to a Dash + Plotly app. Launch it with::

    roigbiv-ui                    # console script installed by pyproject.toml
    python -m roigbiv.ui          # equivalent module invocation

``streamlit run app.py`` still works thanks to this file, but displays a
redirect page rather than the old four-tab UI. The old implementation lives
in git history; cherry-pick it if you have an in-flight workflow that relies
on it.
"""
from __future__ import annotations

_REDIRECT_LINES = (
    "ROIGBIV UI has moved",
    "",
    "The Streamlit interface has been replaced by a Dash + Plotly app with:",
    "  * workspace-rooted processing (output and registry live in your data dir)",
    "  * multi-mode ROI viewer (outline / fill / by-stage / by-feature)",
    "  * cross-session coordinated colors (same cell = same color across days)",
    "  * native HITL correction tools (polygon / freehand / eraser)",
    "",
    "Launch the new UI with:",
    "    roigbiv-ui",
    "  or equivalently",
    "    python -m roigbiv.ui",
    "",
    "Install dependencies if needed:",
    "    pip install 'roigbiv[ui]'",
)


def _run_streamlit_shim() -> None:
    import streamlit as st

    st.set_page_config(page_title="ROIGBIV — moved", layout="centered")
    st.title("ROIGBIV UI has moved")
    st.info(
        "The Streamlit interface has been superseded by a Dash + Plotly app. "
        "Launch it with `roigbiv-ui` (or `python -m roigbiv.ui`)."
    )
    st.markdown(
        "- Workspace-rooted processing — output and registry live inside your data directory.\n"
        "- Multi-mode ROI viewer (outline / fill / by-stage / by-feature).\n"
        "- Cross-session coordinated colors — the same cell is the same color across days.\n"
        "- Native HITL correction tools (polygon / freehand / eraser)."
    )
    st.subheader("Launch the new UI")
    st.code("roigbiv-ui\n# or\npython -m roigbiv.ui", language="bash")
    st.subheader("Install dependencies if needed")
    st.code("pip install 'roigbiv[ui]'", language="bash")


try:
    _run_streamlit_shim()
except Exception:  # noqa: BLE001
    # Streamlit not installed or running under plain python — print + exit.
    import sys

    for line in _REDIRECT_LINES:
        print(line)
    sys.exit(0)
