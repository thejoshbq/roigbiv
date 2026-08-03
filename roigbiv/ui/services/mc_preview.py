"""Reading side of the live motion-correction preview sidecar.

The producer is :mod:`roigbiv.pipeline.mc_preview`, which writes
``{output_dir}/mc_preview/`` while a FOV is being registered. These helpers turn
that directory back into state dicts. Shared by the Flask routes the browser
polls (:mod:`roigbiv.ui.routes.mc_preview`) and by the Pipeline page's
server-side tick, which reads the same files directly rather than over HTTP.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from roigbiv.pipeline.mc_preview import preview_dir

#: A sidecar untouched for this long is presumed dead — the run was killed
#: rather than finishing, so the UI should stop presenting it as live.
#: Generous enough to survive a slow batch or a serialized GPU section.
STALE_AFTER_S = 20.0


def fov_preview_dir(workspace, stem: str) -> Path:
    """Sidecar dir for ``stem`` under ``workspace``'s output root.

    Raises :class:`ValueError` unless ``stem`` names a direct child of that
    root — the routes hand this untrusted query-string input.
    """
    if not stem or "/" in stem or "\\" in stem or stem in (".", ".."):
        raise ValueError("invalid stem")
    if workspace is None:
        raise ValueError("no workspace selected")
    root = Path(workspace.output_root).resolve()
    out_dir = (root / stem).resolve()
    if out_dir == root or root not in out_dir.parents:
        raise ValueError("stem escapes the workspace output root")
    return preview_dir(out_dir)


def read_state(pdir: Path) -> Optional[dict]:
    """Parse ``pdir/state.json``, adding a ``stale`` flag. None if unreadable.

    A torn read is not possible — the producer writes the file to a temporary
    name and ``os.replace``s it — but a *missing* one is normal (the FOV has
    not started yet), hence the quiet None.
    """
    try:
        state = json.loads((Path(pdir) / "state.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    updated = float(state.get("updated_at") or 0.0)
    state["stale"] = (time.time() - updated) > STALE_AFTER_S
    return state


def list_states(workspace) -> list[dict]:
    """Every FOV in the workspace with a preview sidecar, newest first.

    Globs per FOV rather than reading a shared "current FOV" pointer, so batch
    mode's concurrent workers both surface with nothing to race over.
    """
    if workspace is None:
        return []
    root = Path(workspace.output_root)
    states = []
    for state_path in sorted(root.glob("*/mc_preview/state.json")):
        state = read_state(state_path.parent)
        if state is not None:
            state.setdefault("stem", state_path.parent.parent.name)
            states.append(state)
    states.sort(key=lambda s: s.get("updated_at") or 0.0, reverse=True)
    return states


def latest_state(workspace) -> Optional[dict]:
    """The most recently updated FOV preview, live or not."""
    states = list_states(workspace)
    return states[0] if states else None
