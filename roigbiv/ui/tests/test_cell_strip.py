"""Per-cell crop strip (:mod:`roigbiv.ui.components.cell_strip`).

The strip is the page's evidence, not its decoration: it has to show a panel
per session at one fixed scale, and it has to keep showing a panel for the
session where the cell went missing. These check both, plus the geometry
edge cases that would otherwise surface as a blank card at runtime.
"""
from __future__ import annotations

import numpy as np
import pytest

from roigbiv.ui.components.cell_strip import cell_strip
from roigbiv.ui.services.loaders import ROIRender
from roigbiv.ui.services.tracked_cells import TrackedCell, TrackedFOV, TrackedSession


def _roi(label_id, cy, cx, status, *, radius=6):
    ys = [cy - radius, cy - radius, cy + radius, cy + radius]
    xs = [cx - radius, cx + radius, cx + radius, cx - radius]
    return ROIRender(
        label_id=label_id, source_stage=1, gate_outcome="accept",
        activity_type=None, area=int(np.pi * radius ** 2),
        centroid_yx=(float(cy), float(cx)), contours=[(ys, xs)],
        global_cell_id="cell-A", match_status=status,
    )


def _fov(positions, *, shape=(80, 80)):
    """A FOV whose single cell sits at *positions* (None = not detected)."""
    sessions, present, labels, centroids = [], [], [], []
    for i, pos in enumerate(positions):
        if pos is None:
            last = next(p for p in reversed(positions[:i] or positions) if p)
            rois = [_roi(-1, *last, "lost")]
            present.append(False)
            labels.append(None)
            centroids.append(None)
        else:
            rois = [_roi(1, *pos, "new" if not any(present) else "matched")]
            present.append(True)
            labels.append(1)
            centroids.append((float(pos[0]), float(pos[1])))
        sessions.append(TrackedSession(
            session_id=f"s{i}", stem=f"stem-{i}", session_date=None,
            sequence_index=i, output_dir=None,
            mean_M=np.zeros(shape, dtype=np.float32), rois=rois,
            n_matched=0, n_new=0, n_missing=0,
        ))

    cell = TrackedCell(global_cell_id="cell-A", index=1, present=present,
                       local_label_ids=labels, centroids=centroids,
                       anomalies=[])
    return TrackedFOV(fov_id="fov", animal_id="a", region="r",
                      sessions=sessions, cells=[cell]), cell


def _graphs(component):
    out = []

    def walk(node):
        if getattr(node, "figure", None) is not None:
            out.append(node)
        children = getattr(node, "children", None)
        if children is None:
            return
        for child in (children if isinstance(children, (list, tuple)) else [children]):
            walk(child)

    walk(component)
    return out


def _text(component) -> str:
    out = []

    def walk(node):
        if isinstance(node, str):
            out.append(node)
            return
        children = getattr(node, "children", None)
        if children is None:
            return
        for child in (children if isinstance(children, (list, tuple)) else [children]):
            walk(child)

    walk(component)
    return " ".join(out)


def test_one_crop_per_session():
    fov, cell = _fov([(40, 40), (42, 41), (41, 43)])
    assert len(_graphs(cell_strip(fov, cell))) == 3


def test_a_session_that_never_saw_the_cell_still_gets_a_panel():
    """The empty box *is* the dropout — dropping the panel would hide it."""
    fov, cell = _fov([(40, 40), (42, 41), None])
    strip = cell_strip(fov, cell)
    assert len(_graphs(strip)) == 3
    assert "not detected" in _text(strip)


def test_every_session_is_cropped_at_the_same_scale():
    """Different box sizes would make the comparison meaningless."""
    fov, cell = _fov([(40, 40), (42, 41), (41, 43)])
    shapes = {np.asarray(g.figure.data[0].z).shape for g in _graphs(cell_strip(fov, cell))}
    assert len(shapes) == 1, f"crops differ in size: {shapes}"


def test_the_crop_follows_the_cell_between_sessions():
    fov, cell = _fov([(20, 20), (60, 60)])
    first, second = (np.asarray(g.figure.data[0].z)
                     for g in _graphs(cell_strip(fov, cell)))
    # Same size, but taken from different parts of the frame — the outline
    # lands in the middle of each, which is the whole point.
    assert first.shape == second.shape
    for graph in _graphs(cell_strip(fov, cell)):
        outline = next(t for t in graph.figure.data if t.type == "scatter")
        assert 0 <= min(outline.x) and max(outline.x) <= first.shape[1]


def test_a_cell_at_the_frame_edge_clips_instead_of_raising():
    fov, cell = _fov([(2, 2), (78, 78)])
    graphs = _graphs(cell_strip(fov, cell))
    assert len(graphs) == 2
    assert all(np.asarray(g.figure.data[0].z).size > 0 for g in graphs)


def test_a_missing_mean_image_renders_an_empty_crop_not_a_crash():
    fov, cell = _fov([(40, 40), (42, 41)])
    fov.sessions[1].mean_M = None
    assert len(_graphs(cell_strip(fov, cell))) == 2


def test_no_selection_asks_for_one_instead_of_rendering_nothing():
    fov, _cell = _fov([(40, 40)])
    strip = cell_strip(fov, None)
    assert _graphs(strip) == []
    assert "Click a cell" in _text(strip)


def test_session_labels_are_carried_on_each_crop():
    fov, cell = _fov([(40, 40), (42, 41)])
    text = _text(cell_strip(fov, cell))
    assert "stem-0" in text and "stem-1" in text


def test_a_cell_that_arrives_late_borrows_the_position_it_will_have():
    """Nothing precedes it, so the earlier crops look where it turns up."""
    fov, cell = _fov([None, (55, 55)])
    graphs = _graphs(cell_strip(fov, cell))
    assert len(graphs) == 2
    assert all(np.asarray(g.figure.data[0].z).size > 0 for g in graphs)


@pytest.mark.parametrize("radius", [4, 20, 30])
def test_crop_box_scales_with_the_footprint(radius):
    fov, cell = _fov([(40, 40)])
    fov.sessions[0].rois = [_roi(1, 40, 40, "new", radius=radius)]
    crop = np.asarray(_graphs(cell_strip(fov, cell))[0].figure.data[0].z)
    assert crop.shape[0] >= 2 * radius
