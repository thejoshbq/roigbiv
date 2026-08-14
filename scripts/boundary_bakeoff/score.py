"""Score one arm's label image against hand-drawn ground truth.

Matching is ``roigbiv.eval.match.iou_match`` unchanged — greedy one-to-one,
IoU >= threshold. The only thing added here is the *seeded subset*: the GT cells
that a confirmed centroid actually sits inside. Seeding cannot help a cell
nobody marked, so a whole-FOV mean IoU understates the change by however many
cells were never confirmed. Both numbers are reported; neither alone is honest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from roigbiv.eval.match import iou_match
from roigbiv.pipeline.seeded_masks import ORIGIN_FLOW


@dataclass
class ArmScore:
    arm: str
    stem: str
    n_gt: int
    n_pred: int
    n_tp: int
    n_fp: int
    n_fn: int
    mean_iou: float                       # over matched pairs, whole FOV
    mean_iou_seeded: Optional[float]      # over matched pairs on seeded GT cells
    n_gt_seeded: int
    min_iou: float
    n_disk_fallback: int = 0
    n_orphan_basin_px: int = 0
    # Restricted to predictions whose boundary actually came from the flow
    # field. The headline mean_iou averages those together with the disks the
    # arm fell back to, so on a FOV where the detector fires on a third of the
    # cells it mostly reports the disk arm back to you.
    mean_iou_flow: Optional[float] = None
    n_flow: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def precision(self) -> Optional[float]:
        denom = self.n_tp + self.n_fp
        return self.n_tp / denom if denom else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.n_tp + self.n_fn
        return self.n_tp / denom if denom else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=self.precision, recall=self.recall, f1=self.f1)
        return d


def seeded_gt_labels(
    gt_labels: np.ndarray, seeds: dict[int, tuple[float, float]],
) -> set[int]:
    """GT labels containing at least one confirmed centroid.

    Containment, not nearest-neighbour: a centroid inside a hand-drawn ROI is
    unambiguous evidence a human marked that cell, which is what this subset is
    for. A distance rule would need a radius, and the radius would become a
    second knob quietly shaping the headline number.
    """
    gt = np.asarray(gt_labels)
    H, W = gt.shape[:2]
    hit: set[int] = set()
    for cy, cx in seeds.values():
        y, x = int(round(cy)), int(round(cx))
        if 0 <= y < H and 0 <= x < W and gt[y, x]:
            hit.add(int(gt[y, x]))
    return hit


def score_arm(
    arm, gt_labels: np.ndarray, stem: str,
    seeds: dict[int, tuple[float, float]], *, min_iou: float = 0.3,
) -> ArmScore:
    result = iou_match(gt_labels, arm.labels, min_iou=min_iou)
    ious = [iou for _g, _p, iou in result.tp]

    seeded_gt = seeded_gt_labels(gt_labels, seeds)
    seeded_ious = [iou for g, _p, iou in result.tp if g in seeded_gt]

    origins = getattr(arm, "origins", None) or {}
    flow_ious = [iou for _g, p, iou in result.tp
                 if origins.get(int(p)) == ORIGIN_FLOW]

    notes: list[str] = []
    if origins and arm.n_disk_fallback:
        notes.append(
            f"{arm.n_disk_fallback} of {len(origins)} seed(s) captured no flow "
            f"basin and fell back to a disk; mean_iou_flow is the only number "
            f"here that describes a seeded boundary")
    if seeded_gt and len(seeded_gt) < len(np.unique(gt_labels)) - 1:
        notes.append(
            f"{len(seeded_gt)} of {len(np.unique(gt_labels)) - 1} GT cells carry "
            f"a confirmed centroid; seeding cannot affect the rest")

    return ArmScore(
        arm=arm.name, stem=stem,
        n_gt=int(len(np.unique(gt_labels)) - 1),
        n_pred=int(len(np.unique(arm.labels)) - 1),
        n_tp=result.n_tp, n_fp=result.n_fp, n_fn=result.n_fn,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        mean_iou_seeded=float(np.mean(seeded_ious)) if seeded_ious else None,
        n_gt_seeded=len(seeded_gt),
        min_iou=float(min_iou),
        n_disk_fallback=arm.n_disk_fallback,
        n_orphan_basin_px=arm.n_orphan_basin_px,
        mean_iou_flow=float(np.mean(flow_ious)) if flow_ious else None,
        n_flow=len(flow_ious),
        notes=notes,
    )
