"""Smoke tests for :mod:`roigbiv.ui.services.colors`."""
from __future__ import annotations

from roigbiv.pipeline.corrections import USER_STAGE_SENTINEL
from roigbiv.ui.services.colors import (
    FEATURE_LABELS,
    FEATURE_PALETTE,
    STAGE_LABELS,
    STAGE_PALETTE,
    color_for_feature,
    color_for_gcid,
    color_for_stage,
)


def test_stage_palette_covers_pipeline_stages_and_user_sentinel() -> None:
    assert set(STAGE_PALETTE) >= {1, 2, 3, 4, USER_STAGE_SENTINEL}


def test_stage_labels_in_sync_with_palette_keys() -> None:
    assert set(STAGE_PALETTE) == set(STAGE_LABELS)


def test_feature_palette_covers_all_activity_types() -> None:
    expected = {"phasic", "sparse", "tonic", "silent", "ambiguous"}
    assert set(FEATURE_PALETTE) == expected
    assert set(FEATURE_LABELS) == expected


def test_color_for_stage_falls_back_on_unknown() -> None:
    # Unknown stages should still return a valid rgba string.
    result = color_for_stage(999)
    assert result.startswith("rgba(")


def test_color_for_feature_handles_none() -> None:
    result = color_for_feature(None)
    assert result.startswith("rgba(")


def test_color_for_gcid_is_deterministic() -> None:
    a = color_for_gcid("00000000-0000-0000-0000-000000000001")
    b = color_for_gcid("00000000-0000-0000-0000-000000000001")
    c = color_for_gcid("00000000-0000-0000-0000-000000000002")
    assert a == b
    assert a != c


def test_color_for_gcid_none_returns_untracked() -> None:
    result = color_for_gcid(None)
    assert result.startswith("rgba(")
