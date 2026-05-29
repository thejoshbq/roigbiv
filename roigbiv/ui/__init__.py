"""ROIGBIV Dash + Plotly user interface.

Primary frontend for the pipeline. Supersedes the legacy Streamlit ``app.py``.
Launch via ``roigbiv-ui`` (console script) or ``python -m roigbiv.ui``.

``build_app`` is defined in :mod:`roigbiv.ui.app` and imports Dash lazily —
don't eagerly import it here so :mod:`roigbiv.ui.services.*` stays importable
in environments without Dash installed (tests, pipeline-only installs).
"""
