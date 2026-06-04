"""Stratified detection metrics for eval harness.

Recall is the headline metric per stratum. F1/precision are also reported but
never collapsed into a single aggregate across strata.

Activity types follow ROI.activity_type from roigbiv/pipeline/types.py:
phasic | sparse | tonic | silent | ambiguous.

FN ROIs have no assigned activity type (unknown). Tonic and silent recall
values are lower bounds — manual GT under-represents these types (Blindspot 13).
"""
from __future__ import annotations

from .match import MatchResult

ACTIVITY_TYPES = ("phasic", "sparse", "tonic", "silent", "ambiguous")
LOWER_BOUND_TYPES = frozenset({"tonic", "silent"})


def stratified_metrics(
    match_result: MatchResult,
    pred_metadata: dict[int, dict],
) -> dict:
    """Compute per-stratum and overall detection metrics.

    Parameters
    ----------
    match_result : MatchResult from iou_match(gt_labels, pred_labels).
    pred_metadata : dict mapping pred_label_id → roi_metadata entry dict.
        Expected key: "activity_type" (str). Missing entries → "unknown".

    Returns
    -------
    dict with keys:
      "overall": {recall, precision, f1, tp, fp, fn}
      "by_type": {type_name: {recall, precision, f1, tp, fp, fn,
                              lower_bound: bool}}
      "stage_cascade": list or None (populated by harness if pipeline_log present)
      "warnings": list[str]
    """
    warnings: list[str] = []

    def _get_type(pred_label: int) -> str:
        meta = pred_metadata.get(pred_label)
        if meta is None:
            return "unknown"
        return meta.get("activity_type") or "unknown"

    # Map gt_label → pred_label for TP lookup
    gt_to_pred = {gt: pred for gt, pred, _ in match_result.tp}
    pred_to_gt = {pred: gt for gt, pred, _ in match_result.tp}

    # Count TP by activity_type (from pred metadata of matched ROI)
    tp_by_type: dict[str, int] = {}
    for _, pred_label, _ in match_result.tp:
        atype = _get_type(pred_label)
        tp_by_type[atype] = tp_by_type.get(atype, 0) + 1

    # FP by activity_type (from pred metadata of unmatched pred ROI)
    fp_by_type: dict[str, int] = {}
    for pred_label in match_result.fp:
        atype = _get_type(pred_label)
        fp_by_type[atype] = fp_by_type.get(atype, 0) + 1

    # FN by activity_type — unknown because GT has no type labels
    fn_count = match_result.n_fn
    fn_by_type: dict[str, int] = {"unknown": fn_count}

    all_types = set(ACTIVITY_TYPES) | set(tp_by_type) | set(fp_by_type) | {"unknown"}

    by_type: dict[str, dict] = {}
    for atype in sorted(all_types):
        tp_n = tp_by_type.get(atype, 0)
        fp_n = fp_by_type.get(atype, 0)
        # FN for this type: we can only know about FN for "unknown" (all unmatched GT)
        # For named types, we can't split FN by type since GT has no labels
        fn_n = fn_by_type.get(atype, 0)

        recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else float("nan")
        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else float("nan")
        f1 = _f1(recall, precision)
        is_lb = atype in LOWER_BOUND_TYPES or atype == "unknown"

        by_type[atype] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "tp": tp_n,
            "fp": fp_n,
            "fn": fn_n,
            "lower_bound": is_lb,
        }
        if is_lb and tp_n > 0:
            warnings.append(
                f"Stratum '{atype}': recall={recall:.3f} is a lower bound "
                f"(FN ROIs have no assigned type — Blindspot 13)."
            )

    # Overall
    tp_total = match_result.n_tp
    fp_total = match_result.n_fp
    fn_total = match_result.n_fn
    rec_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else float("nan")
    prec_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else float("nan")

    return {
        "overall": {
            "recall": rec_total,
            "precision": prec_total,
            "f1": _f1(rec_total, prec_total),
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
        },
        "by_type": by_type,
        "warnings": warnings,
    }


def _f1(recall: float, precision: float) -> float:
    if recall != recall or precision != precision:  # nan check
        return float("nan")
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)
