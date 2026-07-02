"""Tests for roigbiv.benchmark.matcher — optimal IoU-based ROI matching."""
import json

import numpy as np
import pytest

from roigbiv.benchmark.matcher import (
    MatchResult,
    detection_metrics,
    match,
    match_at_thresholds,
    save_match_result,
    save_match_results,
)


def _disc(H, W, cy, cx, r, label):
    """Draw a filled disc with given label into a (H, W) uint16 array."""
    img = np.zeros((H, W), dtype=np.uint16)
    y, x = np.ogrid[:H, :W]
    img[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = label
    return img


def test_perfect_match():
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = _disc(64, 64, 32, 32, 8, 1)
    result = match(gt, pred)
    assert result.n_tp == 1
    assert result.n_fp == 0
    assert result.n_fn == 0
    assert result.tp[0][2] == pytest.approx(1.0)


def test_no_overlap():
    gt = _disc(64, 64, 10, 10, 6, 1)
    pred = _disc(64, 64, 50, 50, 6, 1)
    result = match(gt, pred)
    assert result.n_tp == 0
    assert result.n_fp == 1
    assert result.n_fn == 1


def test_partial_overlap_above_threshold():
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = _disc(64, 64, 36, 32, 8, 1)  # shifted 4px down (cy 32 -> 36)
    result = match(gt, pred, min_iou=0.3)
    assert result.n_tp == 1
    assert result.tp[0][2] >= 0.3


def test_partial_overlap_below_threshold():
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = _disc(64, 64, 48, 32, 8, 1)  # shifted 16px down (cy 32 -> 48)
    result = match(gt, pred, min_iou=0.3)
    assert result.n_tp == 0
    assert result.n_fp == 1
    assert result.n_fn == 1


def test_duplicate_predictions():
    """ONE gt disc, TWO pred discs both overlapping gt above threshold.

    The Hungarian algorithm should assign the highest-IoU pred to the gt,
    leaving the other pred unmatched as a false positive.
    """
    # Single GT disc at (32, 32) with radius 8.
    gt = _disc(64, 64, 32, 32, 8, 1)

    # Pred label 1: perfect overlap with GT (IoU ~1.0).
    # Pred label 2: drawn second at (32, 22) r=6 — its circle geometrically
    # overlaps pred label 1's circle (center distance 10 < r1+r2=14), but
    # being drawn after label 1 it overwrites the shared pixels, so the two
    # pred labels end up as disjoint regions while pred label 2 still
    # partially overlaps GT (IoU ~0.09, well under the 0.3 threshold, so it
    # only ever competes for the single available GT slot as a weaker option).
    pred = np.zeros((64, 64), dtype=np.uint16)
    y, x = np.ogrid[:64, :64]
    pred[(y - 32) ** 2 + (x - 32) ** 2 <= 8 ** 2] = 1
    pred[(y - 32) ** 2 + (x - 22) ** 2 <= 6 ** 2] = 2

    result = match(gt, pred, min_iou=0.3)

    # Exactly one TP (Hungarian matches the highest-IoU pred to gt).
    assert result.n_tp == 1, f"Expected n_tp=1, got {result.n_tp}"
    # Exactly one FP (the other pred, which has no gt to match).
    assert result.n_fp == 1, f"Expected n_fp=1, got {result.n_fp}"
    # Zero FN (the single gt was matched).
    assert result.n_fn == 0, f"Expected n_fn=0, got {result.n_fn}"


def test_merged_prediction():
    """TWO separate gt discs, ONE pred region — only one gt can be matched.

    With one predicted region and two GT objects, the Hungarian solver can
    only ever produce a single (gt, pred) pairing (min(n_gt, n_pred) == 1) —
    so as long as that one pairing clears the threshold, the outcome is
    deterministically one TP plus one FN for the un-paired GT object,
    regardless of whether the pred region's footprint literally touches the
    second GT disc. Pred is built as an exact copy of GT1 (IoU == 1.0) to
    keep the assertion unambiguous rather than resting on a boundary IoU
    value that's fragile to compute by hand for an irregular "spanning" shape.
    """
    # Two GT discs, far enough apart to not overlap each other.
    gt1 = _disc(64, 64, 16, 32, 8, 1)
    gt2 = _disc(64, 64, 48, 32, 8, 2)
    gt = gt1 + gt2

    # Pred: a single region that exactly matches GT1 (IoU == 1.0 with GT1,
    # no overlap at all with GT2). Only one pred label exists, so it can
    # only ever be assigned to one of the two GT objects.
    pred = _disc(64, 64, 16, 32, 8, 1)

    result = match(gt, pred, min_iou=0.3)

    assert result.n_tp == 1, f"Expected n_tp=1, got {result.n_tp}"
    assert result.n_fp == 0, f"Expected n_fp=0, got {result.n_fp}"
    assert result.n_fn == 1, f"Expected n_fn=1 (the missed GT2), got {result.n_fn}"
    # The matched pair must be GT1 (label 1), and GT2 (label 2) is the miss.
    assert result.tp[0][0] == 1
    assert result.fn == [2]


def test_empty_gt():
    gt = np.zeros((64, 64), dtype=np.uint16)
    pred = _disc(64, 64, 32, 32, 8, 1)
    result = match(gt, pred)
    assert result.n_tp == 0
    assert result.n_fp == 1
    assert result.n_fn == 0

    # match_at_thresholds must not raise on empty GT.
    results = match_at_thresholds(gt, pred, thresholds=(0.3, 0.5))
    assert all(r.n_tp == 0 for r in results.values())


def test_empty_pred():
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = np.zeros((64, 64), dtype=np.uint16)
    result = match(gt, pred)
    assert result.n_tp == 0
    assert result.n_fp == 0
    assert result.n_fn == 1

    # match_at_thresholds must not raise on empty pred.
    results = match_at_thresholds(gt, pred, thresholds=(0.3, 0.5))
    assert all(r.n_tp == 0 for r in results.values())


def test_both_empty():
    gt = np.zeros((64, 64), dtype=np.uint16)
    pred = np.zeros((64, 64), dtype=np.uint16)
    result = match(gt, pred)
    assert result.n_tp == 0
    assert result.n_fp == 0
    assert result.n_fn == 0

    results = match_at_thresholds(gt, pred, thresholds=(0.3, 0.5))
    assert all(r.n_tp == 0 for r in results.values())


def test_match_at_thresholds_reuses_assignment():
    """One shared Hungarian assignment must split correctly at each threshold."""
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = _disc(64, 64, 37, 32, 8, 1)  # shifted 5px, IoU ~0.35-0.4

    thresholds = (0.3, 0.5, 0.7)
    results = match_at_thresholds(gt, pred, thresholds=thresholds)

    result_0_3 = results[0.3]
    assert result_0_3.n_tp == 1, "IoU ~0.35-0.4 should be tp at threshold 0.3"
    assert result_0_3.n_fn == 0
    assert result_0_3.n_fp == 0

    for threshold in (0.5, 0.7):
        result = results[threshold]
        assert result.n_tp == 0, f"IoU ~0.35-0.4 should not be tp at threshold {threshold}"
        assert result.n_fn == 1, f"Expected n_fn=1 at threshold {threshold}, got {result.n_fn}"
        assert result.n_fp == 1, f"Expected n_fp=1 at threshold {threshold}, got {result.n_fp}"


def test_to_dict_from_dict_roundtrip():
    """tp tuples must survive a to_dict()/from_dict() round trip through JSON-shaped lists."""
    original = MatchResult(
        tp=[(1, 10, 0.95), (2, 20, 0.87)],
        fp=[15, 25],
        fn=[3, 4],
        min_iou=0.5,
    )

    payload = original.to_dict()
    assert isinstance(payload["tp"], list)
    # asdict() preserves tuples as tuples (they're not JSON-serialized yet).
    assert isinstance(payload["tp"][0], tuple)

    # Simulate a JSON round trip: tuples become lists.
    payload_json = {
        "tp": [list(t) for t in payload["tp"]],
        "fp": payload["fp"],
        "fn": payload["fn"],
        "min_iou": payload["min_iou"],
    }

    reconstructed = MatchResult.from_dict(payload_json)

    assert reconstructed.n_tp == original.n_tp
    assert reconstructed.n_fp == original.n_fp
    assert reconstructed.n_fn == original.n_fn
    assert reconstructed.min_iou == original.min_iou
    assert all(isinstance(t, tuple) for t in reconstructed.tp)
    assert reconstructed.tp == original.tp
    assert reconstructed.fp == original.fp
    assert reconstructed.fn == original.fn


def test_detection_metrics_basic():
    result = MatchResult(
        tp=[(1, 10, 0.90), (2, 20, 0.80)],
        fp=[15],
        fn=[3],
        min_iou=0.3,
    )

    metrics = detection_metrics(result)

    # precision = n_tp / (n_tp + n_fp) = 2 / (2 + 1) = 2/3
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    # recall = n_tp / (n_tp + n_fn) = 2 / (2 + 1) = 2/3
    assert metrics.recall == pytest.approx(2.0 / 3.0)
    # f1 = 2 * P * R / (P + R) = 2/3
    assert metrics.f1 == pytest.approx(2.0 / 3.0)
    # mean_iou = (0.90 + 0.80) / 2 = 0.85
    assert metrics.mean_iou == pytest.approx(0.85)
    assert metrics.median_iou == pytest.approx(0.85)
    assert metrics.false_positive_count == 1
    assert metrics.false_negative_count == 1


def test_detection_metrics_zero_denominators():
    result = MatchResult(tp=[], fp=[], fn=[], min_iou=0.3)

    metrics = detection_metrics(result)

    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None
    assert metrics.mean_iou is None
    assert metrics.median_iou is None
    # Counts are always set, never None.
    assert metrics.false_positive_count == 0
    assert metrics.false_negative_count == 0


def test_detection_metrics_only_fp():
    result = MatchResult(tp=[], fp=[1, 2], fn=[], min_iou=0.3)

    metrics = detection_metrics(result)

    assert metrics.precision == 0.0
    assert metrics.recall is None  # 0 / 0 undefined
    assert metrics.f1 is None
    assert metrics.mean_iou is None
    assert metrics.false_positive_count == 2
    assert metrics.false_negative_count == 0


def test_detection_metrics_only_fn():
    result = MatchResult(tp=[], fp=[], fn=[1, 2], min_iou=0.3)

    metrics = detection_metrics(result)

    assert metrics.precision is None  # 0 / 0 undefined
    assert metrics.recall == 0.0
    assert metrics.f1 is None
    assert metrics.mean_iou is None
    assert metrics.false_positive_count == 0
    assert metrics.false_negative_count == 2


def test_save_match_result_and_reload(tmp_path):
    result = MatchResult(
        tp=[(1, 10, 0.95), (2, 20, 0.87)],
        fp=[15],
        fn=[3],
        min_iou=0.5,
    )

    # Save to a nested path to verify parent dir creation.
    nested_path = tmp_path / "sub" / "dir" / "match.json"
    returned_path = save_match_result(result, nested_path)

    assert returned_path == nested_path
    assert nested_path.exists()
    assert nested_path.parent.exists()

    loaded_dict = json.loads(nested_path.read_text())
    loaded_result = MatchResult.from_dict(loaded_dict)

    assert loaded_result.n_tp == result.n_tp
    assert loaded_result.n_fp == result.n_fp
    assert loaded_result.n_fn == result.n_fn
    assert loaded_result.min_iou == result.min_iou
    assert loaded_result.tp == result.tp
    assert loaded_result.fp == result.fp
    assert loaded_result.fn == result.fn


def test_save_match_results_multi_threshold(tmp_path):
    result_a = MatchResult(tp=[(1, 10, 0.95)], fp=[], fn=[], min_iou=0.3)
    result_b = MatchResult(tp=[], fp=[10], fn=[1], min_iou=0.5)

    results = {0.3: result_a, 0.5: result_b}

    nested_path = tmp_path / "multi" / "results.json"
    returned_path = save_match_results(results, nested_path)

    assert returned_path == nested_path
    assert nested_path.exists()

    loaded_data = json.loads(nested_path.read_text())

    assert "thresholds" in loaded_data
    thresholds_dict = loaded_data["thresholds"]

    assert "0.3" in thresholds_dict
    assert "0.5" in thresholds_dict

    loaded_a = MatchResult.from_dict(thresholds_dict["0.3"])
    loaded_b = MatchResult.from_dict(thresholds_dict["0.5"])

    assert loaded_a.n_tp == result_a.n_tp
    assert loaded_a.min_iou == result_a.min_iou
    assert loaded_b.n_tp == result_b.n_tp
    assert loaded_b.min_iou == result_b.min_iou
