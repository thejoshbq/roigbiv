"""
ROI G. Biv -- Interactive ROI curation interface.

Napari-based editor for reviewing and correcting merged ROI masks.
Supports: select, delete, draw, split, merge, undo/redo, and save.

Launch via Streamlit button or CLI:
  python -m roigbiv.curator --stem STEM --merged-dir PATH --projections-dir PATH
"""

import argparse
import copy
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label as connected_components
from skimage.draw import line as draw_line

log = logging.getLogger("curator")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CurationAction:
    """A single reversible curation operation."""
    action_type: str          # "delete", "merge", "split", "draw"
    description: str          # Human-readable, e.g. "Deleted ROI 7"
    mask_snapshot: np.ndarray  # Full mask *before* this action
    records_snapshot: list[dict]


class CurationHistory:
    """Undo/redo stack for high-level curation operations."""

    def __init__(self) -> None:
        self._undo: list[CurationAction] = []
        self._redo: list[CurationAction] = []

    def push(self, action: CurationAction) -> None:
        self._undo.append(action)
        self._redo.clear()

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def undo(self, current_mask: np.ndarray,
             current_records: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
        if not self._undo:
            return None
        action = self._undo.pop()
        # Push current state onto redo so we can restore it
        self._redo.append(CurationAction(
            action_type=action.action_type,
            description=f"Redo: {action.description}",
            mask_snapshot=current_mask.copy(),
            records_snapshot=copy.deepcopy(current_records),
        ))
        return action.mask_snapshot, action.records_snapshot

    def redo(self, current_mask: np.ndarray,
             current_records: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
        if not self._redo:
            return None
        action = self._redo.pop()
        self._undo.append(CurationAction(
            action_type=action.action_type,
            description=action.description.removeprefix("Redo: "),
            mask_snapshot=current_mask.copy(),
            records_snapshot=copy.deepcopy(current_records),
        ))
        return action.mask_snapshot, action.records_snapshot

    @property
    def log(self) -> list[str]:
        return [a.description for a in self._undo]


# ---------------------------------------------------------------------------
# Pure data operations
# ---------------------------------------------------------------------------

def delete_rois(
    mask: np.ndarray,
    records: list[dict],
    roi_ids: set[int],
) -> tuple[np.ndarray, list[dict]]:
    """Remove ROIs from the mask and records."""
    new_mask = mask.copy()
    for rid in roi_ids:
        new_mask[new_mask == rid] = 0
    new_records = [r for r in records if r["roi_id"] not in roi_ids]
    return new_mask, new_records


def merge_rois(
    mask: np.ndarray,
    records: list[dict],
    roi_ids: set[int],
    next_id: int,
) -> tuple[np.ndarray, list[dict], int]:
    """Merge multiple ROIs into a single ROI with a new ID."""
    new_mask = mask.copy()
    for rid in roi_ids:
        new_mask[new_mask == rid] = next_id

    # Build merged record
    originals = [r for r in records if r["roi_id"] in roi_ids]
    branches = set()
    for r in originals:
        for ch in r.get("source_branches", ""):
            if ch in "ABC":
                branches.add(ch)
    source_branches = "".join(sorted(branches)) if branches else "CURATED"

    ys, xs = np.where(new_mask == next_id)
    new_record = {
        "roi_id": next_id,
        "tier": "CURATED",
        "source_branches": source_branches,
        "iou_ab": 0.0,
        "iou_ac": 0.0,
        "iou_bc": 0.0,
        "a_label": -1,
        "b_label": -1,
        "c_label": -1,
        "centroid_y": int(np.mean(ys)) if len(ys) else -1,
        "centroid_x": int(np.mean(xs)) if len(xs) else -1,
        "area_px": int(len(ys)),
        "s2p_iscell_prob": -1.0,
        "review_flag": False,
    }

    new_records = [r for r in records if r["roi_id"] not in roi_ids]
    new_records.append(new_record)
    return new_mask, new_records, next_id + 1


def split_roi(
    mask: np.ndarray,
    records: list[dict],
    roi_id: int,
    separator_coords: list[np.ndarray],
    next_id: int,
) -> tuple[np.ndarray, list[dict], int]:
    """Split a single ROI along drawn separator line(s).

    Parameters
    ----------
    separator_coords : list of Nx2 arrays
        Each array is a sequence of (row, col) points forming a path.
    """
    roi_binary = mask == roi_id
    if not roi_binary.any():
        return mask, records, next_id

    # Rasterize separator lines as a barrier
    barrier = np.zeros(mask.shape, dtype=bool)
    for path in separator_coords:
        coords = np.asarray(path, dtype=int)
        for i in range(len(coords) - 1):
            rr, cc = draw_line(
                coords[i, 0], coords[i, 1],
                coords[i + 1, 0], coords[i + 1, 1],
            )
            valid = (
                (rr >= 0) & (rr < mask.shape[0])
                & (cc >= 0) & (cc < mask.shape[1])
            )
            barrier[rr[valid], cc[valid]] = True

    # Dilate barrier by 1px for reliability
    from scipy.ndimage import binary_dilation
    barrier = binary_dilation(barrier, iterations=1)

    # Cut the ROI
    roi_cut = roi_binary & ~barrier
    labeled, n_components = connected_components(roi_cut)

    if n_components < 2:
        return mask, records, next_id  # Separator didn't bisect

    # Find original record for inheritance
    original_rec = None
    for r in records:
        if r["roi_id"] == roi_id:
            original_rec = r
            break

    parent_branches = original_rec.get("source_branches", "") if original_rec else ""

    new_mask = mask.copy()
    new_mask[roi_binary] = 0  # Clear original
    new_records = [r for r in records if r["roi_id"] != roi_id]

    current_id = next_id
    for comp in range(1, n_components + 1):
        comp_pixels = labeled == comp
        new_mask[comp_pixels] = current_id

        ys, xs = np.where(comp_pixels)
        new_records.append({
            "roi_id": current_id,
            "tier": "CURATED",
            "source_branches": parent_branches,
            "iou_ab": 0.0,
            "iou_ac": 0.0,
            "iou_bc": 0.0,
            "a_label": -1,
            "b_label": -1,
            "c_label": -1,
            "centroid_y": int(np.mean(ys)),
            "centroid_x": int(np.mean(xs)),
            "area_px": int(len(ys)),
            "s2p_iscell_prob": -1.0,
            "review_flag": False,
        })
        current_id += 1

    return new_mask, new_records, current_id


def commit_drawn_roi(
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    records: list[dict],
    drawing_label: int,
    stem: str,
) -> list[dict]:
    """Create a record for a newly painted ROI.

    Compares old_mask to new_mask to find pixels painted with drawing_label.
    """
    drawn_pixels = new_mask == drawing_label
    if not drawn_pixels.any():
        return records

    ys, xs = np.where(drawn_pixels)
    new_record = {
        "roi_id": drawing_label,
        "tier": "MANUAL",
        "source_branches": "MANUAL",
        "iou_ab": 0.0,
        "iou_ac": 0.0,
        "iou_bc": 0.0,
        "a_label": -1,
        "b_label": -1,
        "c_label": -1,
        "centroid_y": int(np.mean(ys)),
        "centroid_x": int(np.mean(xs)),
        "area_px": int(len(ys)),
        "s2p_iscell_prob": -1.0,
        "review_flag": False,
        "fov": stem,
    }

    new_records = list(records)
    new_records.append(new_record)
    return new_records


def regenerate_records(
    mask: np.ndarray,
    original_records: list[dict],
    stem: str,
) -> list[dict]:
    """Rebuild records from the current mask state.

    Preserves metadata for unmodified ROIs, creates MANUAL records for
    unknown IDs, and drops records for deleted ROIs.
    """
    original_by_id = {r["roi_id"]: r for r in original_records}
    current_ids = set(np.unique(mask[mask > 0]).astype(int))

    new_records = []
    for roi_id in sorted(current_ids):
        ys, xs = np.where(mask == roi_id)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        area = int(len(ys))

        if roi_id in original_by_id:
            rec = copy.deepcopy(original_by_id[roi_id])
            rec["centroid_y"] = cy
            rec["centroid_x"] = cx
            rec["area_px"] = area
        else:
            rec = {
                "roi_id": roi_id,
                "tier": "MANUAL",
                "source_branches": "MANUAL",
                "iou_ab": 0.0,
                "iou_ac": 0.0,
                "iou_bc": 0.0,
                "a_label": -1,
                "b_label": -1,
                "c_label": -1,
                "centroid_y": cy,
                "centroid_x": cx,
                "area_px": area,
                "s2p_iscell_prob": -1.0,
                "review_flag": False,
            }
        rec["fov"] = stem
        new_records.append(rec)

    return new_records


def save_curated(
    mask: np.ndarray,
    records: list[dict],
    stem: str,
    merged_dir: str | Path,
    action_log: list[str],
    backup: bool = True,
) -> None:
    """Write curated mask, records CSV, and curation log to disk."""
    merged_dir = Path(merged_dir)
    mask_path = merged_dir / f"{stem}_merged_masks.tif"
    csv_path = merged_dir / f"{stem}_merge_records.csv"
    log_path = merged_dir / f"{stem}_curation_log.json"

    # Backup originals
    if backup:
        for p in (mask_path, csv_path):
            if p.exists():
                bak = p.with_suffix(f".bak{p.suffix}")
                shutil.copy2(p, bak)

    # Write mask
    tifffile.imwrite(str(mask_path), mask.astype(np.uint16))

    # Write records
    final_records = regenerate_records(mask, records, stem)
    pd.DataFrame(final_records).to_csv(str(csv_path), index=False)

    # Write curation log
    log_entry = {
        "stem": stem,
        "curated_at": datetime.now(timezone.utc).isoformat(),
        "n_rois": int(len(final_records)),
        "actions": action_log,
    }
    log_path.write_text(json.dumps(log_entry, indent=2))

    log.info("Saved curated mask (%d ROIs) to %s", len(final_records), mask_path)


# ---------------------------------------------------------------------------
# Napari GUI
# ---------------------------------------------------------------------------

def _find_roi_at_cursor(mask: np.ndarray, position: tuple) -> int | None:
    """Return the ROI label at (row, col) or None if background."""
    r, c = int(round(position[-2])), int(round(position[-1]))
    if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
        val = int(mask[r, c])
        return val if val > 0 else None
    return None


class CuratorState:
    """Mutable state for a curation session."""

    def __init__(
        self,
        mask: np.ndarray,
        records: list[dict],
        stem: str,
        merged_dir: Path,
        projections_dir: Path,
    ) -> None:
        self.mask = mask
        self.records = records
        self.stem = stem
        self.merged_dir = merged_dir
        self.projections_dir = projections_dir
        self.history = CurationHistory()
        self.next_roi_id: int = int(mask.max()) + 1 if mask.any() else 1
        self.selected_ids: list[int] = []
        self.split_target: int | None = None
        self._drawing = False

    @property
    def drawing(self) -> bool:
        return self._drawing


def _build_curator_widget(viewer, labels_layer, state: CuratorState):
    """Build and dock the curation control panel."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)

    # ── ROI Info ──────────────────────────────────────────────────────
    info_group = QGroupBox("ROI Info")
    info_layout = QVBoxLayout()
    info_label = QLabel("Click an ROI to inspect")
    info_layout.addWidget(info_label)
    info_group.setLayout(info_layout)
    layout.addWidget(info_group)

    # ── Selection ─────────────────────────────────────────────────────
    sel_group = QGroupBox("Selection")
    sel_layout = QVBoxLayout()
    sel_list = QListWidget()
    sel_list.setMaximumHeight(120)
    sel_layout.addWidget(sel_list)

    sel_btn_row = QHBoxLayout()
    btn_add_sel = QPushButton("Add Hovered")
    btn_clear_sel = QPushButton("Clear")
    sel_btn_row.addWidget(btn_add_sel)
    sel_btn_row.addWidget(btn_clear_sel)
    sel_layout.addLayout(sel_btn_row)
    sel_group.setLayout(sel_layout)
    layout.addWidget(sel_group)

    # ── Actions ───────────────────────────────────────────────────────
    act_group = QGroupBox("Actions")
    act_layout = QVBoxLayout()

    btn_delete = QPushButton("Delete Selected")
    btn_merge = QPushButton("Merge Selected")
    btn_split = QPushButton("Split Mode")
    btn_split_confirm = QPushButton("Confirm Split")
    btn_split_confirm.setEnabled(False)
    btn_split_cancel = QPushButton("Cancel Split")
    btn_split_cancel.setEnabled(False)
    btn_draw = QPushButton("Draw New ROI")
    btn_draw_commit = QPushButton("Commit Drawing")
    btn_draw_commit.setEnabled(False)

    act_layout.addWidget(btn_delete)
    act_layout.addWidget(btn_merge)

    split_row = QHBoxLayout()
    split_row.addWidget(btn_split)
    split_row.addWidget(btn_split_confirm)
    split_row.addWidget(btn_split_cancel)
    act_layout.addLayout(split_row)

    draw_row = QHBoxLayout()
    draw_row.addWidget(btn_draw)
    draw_row.addWidget(btn_draw_commit)
    act_layout.addLayout(draw_row)

    act_group.setLayout(act_layout)
    layout.addWidget(act_group)

    # ── Undo / Redo ───────────────────────────────────────────────────
    hist_row = QHBoxLayout()
    btn_undo = QPushButton("Undo")
    btn_redo = QPushButton("Redo")
    hist_row.addWidget(btn_undo)
    hist_row.addWidget(btn_redo)
    layout.addLayout(hist_row)

    # ── Save ──────────────────────────────────────────────────────────
    save_group = QGroupBox("Save")
    save_layout = QVBoxLayout()
    btn_save = QPushButton("Save Curated Masks")
    btn_save.setStyleSheet("font-weight: bold;")
    status_label = QLabel("")
    save_layout.addWidget(btn_save)
    save_layout.addWidget(status_label)
    save_group.setLayout(save_layout)
    layout.addWidget(save_group)

    layout.addStretch()

    # ── Helper: refresh the labels layer from state ───────────────────
    def _refresh_layer():
        labels_layer.data = state.mask
        labels_layer.refresh()

    def _update_info(roi_id: int | None):
        if roi_id is None:
            info_label.setText("Click an ROI to inspect")
            return
        rec = next((r for r in state.records if r["roi_id"] == roi_id), None)
        if rec:
            info_label.setText(
                f"ROI {roi_id}  |  tier: {rec.get('tier', '?')}  |  "
                f"area: {rec.get('area_px', '?')} px  |  "
                f"branches: {rec.get('source_branches', '?')}"
            )
        else:
            ys, xs = np.where(state.mask == roi_id)
            info_label.setText(
                f"ROI {roi_id}  |  tier: (new)  |  area: {len(ys)} px"
            )

    def _refresh_sel_list():
        sel_list.clear()
        for rid in state.selected_ids:
            sel_list.addItem(f"ROI {rid}")

    # ── Mouse callback: show info on hover ────────────────────────────
    @labels_layer.mouse_move_callbacks.append
    def _on_mouse_move(layer, event):
        roi_id = _find_roi_at_cursor(state.mask, event.position)
        _update_info(roi_id)

    # ── Add to selection ──────────────────────────────────────────────
    def _add_to_selection():
        # Use the label currently under cursor via the labels layer
        pos = viewer.cursor.position
        roi_id = _find_roi_at_cursor(state.mask, pos)
        if roi_id and roi_id not in state.selected_ids:
            state.selected_ids.append(roi_id)
            _refresh_sel_list()

    btn_add_sel.clicked.connect(_add_to_selection)

    def _clear_selection():
        state.selected_ids.clear()
        _refresh_sel_list()
        state.split_target = None

    btn_clear_sel.clicked.connect(_clear_selection)

    # ── Delete ────────────────────────────────────────────────────────
    def _do_delete():
        if not state.selected_ids:
            status_label.setText("No ROIs selected.")
            return
        ids = set(state.selected_ids)
        # Snapshot for undo
        state.history.push(CurationAction(
            action_type="delete",
            description=f"Deleted ROIs {sorted(ids)}",
            mask_snapshot=state.mask.copy(),
            records_snapshot=copy.deepcopy(state.records),
        ))
        state.mask, state.records = delete_rois(state.mask, state.records, ids)
        state.selected_ids.clear()
        _refresh_sel_list()
        _refresh_layer()
        status_label.setText(f"Deleted {len(ids)} ROI(s).")

    btn_delete.clicked.connect(_do_delete)

    # ── Merge ─────────────────────────────────────────────────────────
    def _do_merge():
        if len(state.selected_ids) < 2:
            status_label.setText("Select at least 2 ROIs to merge.")
            return
        ids = set(state.selected_ids)
        state.history.push(CurationAction(
            action_type="merge",
            description=f"Merged ROIs {sorted(ids)} -> {state.next_roi_id}",
            mask_snapshot=state.mask.copy(),
            records_snapshot=copy.deepcopy(state.records),
        ))
        state.mask, state.records, state.next_roi_id = merge_rois(
            state.mask, state.records, ids, state.next_roi_id,
        )
        state.selected_ids.clear()
        _refresh_sel_list()
        _refresh_layer()
        status_label.setText(f"Merged into ROI {state.next_roi_id - 1}.")

    btn_merge.clicked.connect(_do_merge)

    # ── Split ─────────────────────────────────────────────────────────
    _split_shapes_layer = [None]  # mutable container for closure

    def _enter_split_mode():
        if len(state.selected_ids) != 1:
            status_label.setText("Select exactly 1 ROI to split.")
            return
        state.split_target = state.selected_ids[0]
        # Add shapes layer for drawing the separator
        shapes = viewer.add_shapes(
            name="split_separator",
            shape_type="path",
            edge_color="red",
            edge_width=2,
        )
        shapes.mode = "add_path"
        _split_shapes_layer[0] = shapes
        btn_split.setEnabled(False)
        btn_split_confirm.setEnabled(True)
        btn_split_cancel.setEnabled(True)
        status_label.setText(
            f"Draw line(s) across ROI {state.split_target}, then Confirm."
        )

    btn_split.clicked.connect(_enter_split_mode)

    def _exit_split_mode():
        if _split_shapes_layer[0] is not None:
            try:
                viewer.layers.remove(_split_shapes_layer[0])
            except ValueError:
                pass
            _split_shapes_layer[0] = None
        state.split_target = None
        btn_split.setEnabled(True)
        btn_split_confirm.setEnabled(False)
        btn_split_cancel.setEnabled(False)

    def _confirm_split():
        shapes = _split_shapes_layer[0]
        if shapes is None or state.split_target is None:
            _exit_split_mode()
            return

        separator_coords = [np.asarray(d) for d in shapes.data]
        if not separator_coords:
            status_label.setText("No separator drawn.")
            _exit_split_mode()
            return

        target = state.split_target
        state.history.push(CurationAction(
            action_type="split",
            description=f"Split ROI {target}",
            mask_snapshot=state.mask.copy(),
            records_snapshot=copy.deepcopy(state.records),
        ))

        old_next = state.next_roi_id
        state.mask, state.records, state.next_roi_id = split_roi(
            state.mask, state.records, target, separator_coords,
            state.next_roi_id,
        )

        if state.next_roi_id == old_next:
            # Split didn't produce multiple components
            state.history._undo.pop()  # Remove the no-op snapshot
            status_label.setText("Split failed: separator didn't bisect the ROI.")
        else:
            n_parts = state.next_roi_id - old_next
            status_label.setText(
                f"Split ROI {target} into {n_parts} parts "
                f"(IDs {old_next}-{state.next_roi_id - 1})."
            )

        state.selected_ids.clear()
        _refresh_sel_list()
        _exit_split_mode()
        _refresh_layer()

    btn_split_confirm.clicked.connect(_confirm_split)
    btn_split_cancel.clicked.connect(lambda: (
        _exit_split_mode(), status_label.setText("Split cancelled.")
    ))

    # ── Draw ──────────────────────────────────────────────────────────
    _draw_snapshot = [None]  # mask snapshot before drawing began

    def _enter_draw_mode():
        _draw_snapshot[0] = state.mask.copy()
        labels_layer.selected_label = state.next_roi_id
        labels_layer.mode = "paint"
        state._drawing = True
        btn_draw.setEnabled(False)
        btn_draw_commit.setEnabled(True)
        status_label.setText(
            f"Paint new ROI (label {state.next_roi_id}). "
            "Use brush or press P for polygon. Click Commit when done."
        )

    btn_draw.clicked.connect(_enter_draw_mode)

    def _commit_drawing():
        if _draw_snapshot[0] is None:
            return

        current_data = labels_layer.data
        drawing_label = state.next_roi_id

        # Check if anything was actually drawn
        if not (current_data == drawing_label).any():
            status_label.setText("Nothing drawn. Exiting draw mode.")
            labels_layer.data = _draw_snapshot[0]
            state.mask = _draw_snapshot[0]
        else:
            state.history.push(CurationAction(
                action_type="draw",
                description=f"Drew new ROI {drawing_label}",
                mask_snapshot=_draw_snapshot[0],
                records_snapshot=copy.deepcopy(state.records),
            ))
            state.mask = current_data.copy()
            state.records = commit_drawn_roi(
                _draw_snapshot[0], state.mask, state.records,
                drawing_label, state.stem,
            )
            state.next_roi_id += 1
            status_label.setText(f"Committed ROI {drawing_label}.")

        _draw_snapshot[0] = None
        state._drawing = False
        labels_layer.mode = "pan_zoom"
        btn_draw.setEnabled(True)
        btn_draw_commit.setEnabled(False)
        _refresh_layer()

    btn_draw_commit.clicked.connect(_commit_drawing)

    # ── Undo / Redo ───────────────────────────────────────────────────
    def _do_undo():
        result = state.history.undo(state.mask, state.records)
        if result is None:
            status_label.setText("Nothing to undo.")
            return
        state.mask, state.records = result
        _refresh_layer()
        status_label.setText("Undone.")

    def _do_redo():
        result = state.history.redo(state.mask, state.records)
        if result is None:
            status_label.setText("Nothing to redo.")
            return
        state.mask, state.records = result
        _refresh_layer()
        status_label.setText("Redone.")

    btn_undo.clicked.connect(_do_undo)
    btn_redo.clicked.connect(_do_redo)

    # Keybindings
    @viewer.bind_key("Control-Z")
    def _key_undo(viewer):
        if not state.drawing:
            _do_undo()

    @viewer.bind_key("Control-Shift-Z")
    def _key_redo(viewer):
        if not state.drawing:
            _do_redo()

    # ── Save ──────────────────────────────────────────────────────────
    def _do_save():
        try:
            save_curated(
                mask=state.mask,
                records=state.records,
                stem=state.stem,
                merged_dir=state.merged_dir,
                action_log=state.history.log,
                backup=True,
            )
            n = len(np.unique(state.mask[state.mask > 0]))
            status_label.setText(
                f"Saved {n} ROIs. Re-run Stages 6-7 to update traces."
            )
        except Exception as e:
            status_label.setText(f"Save failed: {e}")
            log.exception("Save failed")

    btn_save.clicked.connect(_do_save)

    # Dock the widget
    viewer.window.add_dock_widget(widget, name="ROI Curator", area="right")
    return widget


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _resolve_mean_path(projections_dir: Path, stem: str) -> Path:
    """Return the mean-projection path for *stem*, or raise FileNotFoundError."""
    for name in (f"{stem}_mean.tif", f"{stem}_mc_mean.tif"):
        p = projections_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No mean projection found for '{stem}' in {projections_dir}.\n"
        f"Expected: {stem}_mean.tif or {stem}_mc_mean.tif"
    )


def open_curator(
    stem: str,
    merged_dir: str,
    projections_dir: str,
) -> None:
    """Open the interactive ROI curation interface for a single FOV.

    Parameters
    ----------
    stem : FOV identifier (e.g. "mouse1_fov3")
    merged_dir : directory containing {stem}_merged_masks.tif and
                 {stem}_merge_records.csv
    projections_dir : directory containing {stem}_mean.tif
    """
    import napari

    merged_dir = Path(merged_dir)
    projections_dir = Path(projections_dir)

    # Load data
    mask_path = merged_dir / f"{stem}_merged_masks.tif"
    csv_path = merged_dir / f"{stem}_merge_records.csv"
    mean_path = _resolve_mean_path(projections_dir, stem)

    if not mask_path.exists():
        raise FileNotFoundError(f"Merged mask not found: {mask_path}")

    mask = tifffile.imread(str(mask_path)).astype(np.uint16)
    mean_img = tifffile.imread(str(mean_path)).astype(np.float32)

    records: list[dict] = []
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
        records = df.to_dict("records")

    # Build viewer
    viewer = napari.Viewer(title=f"ROI Curator -- {stem}")
    viewer.add_image(mean_img, name="mean", colormap="gray")

    labels_layer = viewer.add_labels(mask, name="ROIs", opacity=0.5)

    # Initialize state and build widget
    state = CuratorState(
        mask=mask,
        records=records,
        stem=stem,
        merged_dir=merged_dir,
        projections_dir=projections_dir,
    )

    _build_curator_widget(viewer, labels_layer, state)

    napari.run()


def main():
    ap = argparse.ArgumentParser(
        description="ROI G. Biv -- Interactive ROI curation interface",
    )
    ap.add_argument("--stem", required=True, help="FOV stem name")
    ap.add_argument("--merged-dir", required=True,
                    help="Directory with merged masks + records CSV")
    ap.add_argument("--projections-dir", required=True,
                    help="Directory with mean projections")
    args = ap.parse_args()

    open_curator(args.stem, args.merged_dir, args.projections_dir)


if __name__ == "__main__":
    main()
