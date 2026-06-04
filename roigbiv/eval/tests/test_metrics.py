"""Tests for roigbiv.eval.metrics."""
import math
import numpy as np
from roigbiv.eval.match import MatchResult
from roigbiv.eval.metrics import stratified_metrics


def _metadata(*entries):
    """Build pred_metadata dict from (label_id, activity_type) pairs."""
    return {lid: {"label_id": lid, "activity_type": atype} for lid, atype in entries}


def test_perfect_phasic():
    match = MatchResult(tp=[(1, 10, 0.9)], fp=[], fn=[])
    meta = _metadata((10, "phasic"))
    result = stratified_metrics(match, meta)
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 0
    assert result["overall"]["fn"] == 0
    assert result["overall"]["recall"] == pytest.approx(1.0)
    assert result["by_type"]["phasic"]["tp"] == 1


def test_fn_goes_to_unknown():
    match = MatchResult(tp=[], fp=[], fn=[1, 2])
    result = stratified_metrics(match, {})
    assert result["by_type"]["unknown"]["fn"] == 2
    assert result["by_type"]["unknown"]["lower_bound"] is True


def test_fp_typed():
    match = MatchResult(tp=[], fp=[5, 6], fn=[])
    meta = _metadata((5, "phasic"), (6, "tonic"))
    result = stratified_metrics(match, meta)
    assert result["by_type"]["phasic"]["fp"] == 1
    assert result["by_type"]["tonic"]["fp"] == 1


def test_lower_bound_warning_tonic():
    match = MatchResult(tp=[(1, 10, 0.8)], fp=[], fn=[])
    meta = _metadata((10, "tonic"))
    result = stratified_metrics(match, meta)
    assert result["by_type"]["tonic"]["lower_bound"] is True
    assert any("lower bound" in w for w in result["warnings"])


def test_overall_recall():
    match = MatchResult(tp=[(1, 10, 0.9), (2, 20, 0.8)], fp=[30], fn=[3])
    meta = _metadata((10, "phasic"), (20, "sparse"), (30, "phasic"))
    result = stratified_metrics(match, meta)
    ov = result["overall"]
    assert ov["tp"] == 2
    assert ov["fp"] == 1
    assert ov["fn"] == 1
    assert ov["recall"] == pytest.approx(2 / 3)
    assert ov["precision"] == pytest.approx(2 / 3)


import pytest
