"""Tests for roigbiv.eval.match."""
import numpy as np
import pytest
from roigbiv.eval.match import iou_match, MatchResult


def _disc(H, W, cy, cx, r, label):
    """Draw a filled disc with given label into a (H, W) uint16 array."""
    img = np.zeros((H, W), dtype=np.uint16)
    y, x = np.ogrid[:H, :W]
    img[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = label
    return img


def test_perfect_match():
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = _disc(64, 64, 32, 32, 8, 1)
    result = iou_match(gt, pred)
    assert result.n_tp == 1
    assert result.n_fp == 0
    assert result.n_fn == 0
    assert result.tp[0][2] == pytest.approx(1.0)


def test_no_overlap():
    gt = _disc(64, 64, 10, 10, 6, 1)
    pred = _disc(64, 64, 50, 50, 6, 1)
    result = iou_match(gt, pred)
    assert result.n_tp == 0
    assert result.n_fp == 1
    assert result.n_fn == 1


def test_partial_overlap_above_threshold():
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = _disc(64, 64, 36, 32, 8, 1)  # shifted 4px, ~50% overlap
    result = iou_match(gt, pred, min_iou=0.3)
    assert result.n_tp == 1


def test_partial_overlap_below_threshold():
    gt = _disc(64, 64, 32, 32, 8, 1)
    pred = _disc(64, 64, 48, 32, 8, 1)  # shifted 16px, negligible overlap
    result = iou_match(gt, pred, min_iou=0.3)
    assert result.n_tp == 0
    assert result.n_fn == 1
    assert result.n_fp == 1


def test_greedy_one_to_one():
    # Two GT discs; one pred disc overlaps both — should match to the higher IoU one
    gt = np.zeros((64, 64), dtype=np.uint16)
    gt[10:20, 10:20] = 1
    gt[12:22, 12:22] = 2  # partially overlapping GT region (label 2 wins overlap area)
    pred = np.zeros((64, 64), dtype=np.uint16)
    pred[10:20, 10:20] = 1  # identical to GT label 1

    result = iou_match(gt, pred)
    assert result.n_tp == 1
    assert result.n_fn >= 0


def test_multiple_matches():
    H, W = 64, 64
    gt1 = _disc(H, W, 16, 16, 6, 1)
    gt2 = _disc(H, W, 16, 48, 6, 2)
    pred1 = _disc(H, W, 16, 16, 6, 10)
    pred2 = _disc(H, W, 16, 48, 6, 20)
    gt = gt1 + gt2
    pred = pred1 + pred2
    result = iou_match(gt, pred)
    assert result.n_tp == 2
    assert result.n_fp == 0
    assert result.n_fn == 0
