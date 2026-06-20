"""Tests for the acquisition/lens profile mechanism (Layer 0).

Pins the registry contract and — most importantly — the merge precedence
``defaults < profile < explicit user flags`` and the historical-CLI-default
backfill that keeps the GRIN path byte-identical after the argparse
``default=None`` migration.
"""
from __future__ import annotations

from argparse import Namespace

import pytest

from roigbiv.pipeline.profiles import (
    AUTO,
    PROFILES,
    STAGE1_CLI_DEFAULTS,
    get_profile,
    list_profiles,
    merged_overrides,
)
from roigbiv.pipeline.run import _build_explicit_stage1, _resolve_profile_name
from roigbiv.pipeline.types import PipelineConfig


# ── registry ────────────────────────────────────────────────────────────────

def test_grin_profile_is_empty_noop():
    assert get_profile("grin") == {}


def test_prism_profile_is_single_channel_cyto3():
    p = get_profile("prism")
    assert p["channels"] == (0, 0)          # the dominant Phase-A lever
    assert p["cellpose_model"] == "cyto3"   # generalist, not the GRIN-overfit model
    assert p["use_denoise"] is False
    assert p["diameter"] == 56
    assert "cpsam" not in p.values()        # CP4-only; never referenced


def test_generic_profile_is_conservative():
    p = get_profile("generic")
    assert p["channels"] == (0, 0)
    assert p.get("diameter_auto") is True
    # least-certain path must not be the most experimental one:
    assert "adaptive_gates" not in p
    assert "ensemble" not in p


def test_get_profile_auto_raises():
    with pytest.raises(ValueError):
        get_profile(AUTO)


def test_get_profile_unknown_raises():
    with pytest.raises(ValueError):
        get_profile("two-photon-deluxe")


def test_list_profiles_auto_first():
    lp = list_profiles()
    assert lp[0] == AUTO
    assert set(lp) == {AUTO, *PROFILES}


def test_get_profile_returns_copy():
    get_profile("prism")["diameter"] = 999
    assert get_profile("prism")["diameter"] == 56   # registry not mutated


# ── merge precedence ─────────────────────────────────────────────────────────

def test_merged_overrides_precedence_and_label():
    base = {"fs": 7.5, "tau": 1.0}
    out = merged_overrides(
        "prism", base,
        [{"min_area": 999},                 # explicit user flag beats profile's 900
         {"diameter": 30}],                 # explicit beats profile's 56
    )
    assert out["fs"] == 7.5                  # base preserved
    assert out["min_area"] == 999           # explicit > profile
    assert out["diameter"] == 30            # explicit > profile
    assert out["channels"] == (0, 0)        # profile > base default
    assert out["profile"] == "prism"        # label recorded


def test_merged_overrides_later_dict_wins():
    out = merged_overrides("grin", {}, [{"diameter": 10}, {"diameter": 20}])
    assert out["diameter"] == 20


# ── _build_explicit_stage1 backfill ──────────────────────────────────────────

def _args(**kw) -> Namespace:
    base = dict(model=None, diameter=None, diameter_auto=None,
                cellprob_threshold=None, flow_threshold=None, channels=None)
    base.update(kw)
    return Namespace(**base)


def test_explicit_stage1_backfills_historical_defaults_for_grin():
    """All flags omitted + empty (grin) profile → historical CLI defaults, NOT
    the (different) dataclass defaults. Critically flow_threshold == 0.6."""
    out = _build_explicit_stage1(_args(), get_profile("grin"))
    assert out["flow_threshold"] == 0.6 == STAGE1_CLI_DEFAULTS["flow_threshold"]
    assert out["diameter"] == 12
    assert out["cellprob_threshold"] == -2.0
    assert out["channels"] == (1, 2)
    assert out["diameter_auto"] is False
    assert out["cellpose_model"].endswith("current_model")   # pinned default string


def test_explicit_stage1_does_not_backfill_keys_the_profile_sets():
    """PRISM sets channels/flow/cellprob/diameter/model — backfill must not clobber."""
    out = _build_explicit_stage1(_args(), get_profile("prism"))
    assert "channels" not in out          # profile owns it (→ (0,0))
    assert "flow_threshold" not in out    # profile owns it (→ 0.4)
    assert "cellprob_threshold" not in out
    assert "diameter" not in out
    assert "cellpose_model" not in out


def test_explicit_stage1_flag_wins_over_profile():
    out = _build_explicit_stage1(_args(diameter=30, flow_threshold=0.9),
                                 get_profile("prism"))
    assert out["diameter"] == 30
    assert out["flow_threshold"] == 0.9


# ── end-to-end cfg construction (regression guards) ──────────────────────────

def _cfg_for(profile_name, **arg_overrides) -> PipelineConfig:
    args = _args(**arg_overrides)
    bundle = get_profile(profile_name)
    explicit = _build_explicit_stage1(args, bundle)
    return PipelineConfig(**merged_overrides(profile_name, {"fs": 7.5}, [explicit]))


def test_grin_cfg_byte_identical_to_prior_cli_defaults():
    cfg = _cfg_for("grin")
    assert cfg.profile == "grin"
    assert cfg.channels == (1, 2)
    assert cfg.flow_threshold == 0.6      # historical CLI default preserved
    assert cfg.diameter == 12
    assert cfg.cellprob_threshold == -2.0
    assert cfg.use_denoise is True
    assert cfg.cellpose_model.endswith("current_model")


def test_prism_cfg_applies_single_channel_generalist():
    cfg = _cfg_for("prism")
    assert cfg.profile == "prism"
    assert cfg.channels == (0, 0)
    assert cfg.cellpose_model == "cyto3"
    assert cfg.use_denoise is False
    assert cfg.diameter == 56
    assert cfg.flow_threshold == 0.4
    assert cfg.min_area == 900 and cfg.max_area == 9000


def test_explicit_flag_overrides_prism_profile_in_cfg():
    cfg = _cfg_for("prism", diameter=30, model="cyto2")
    assert cfg.diameter == 30
    assert cfg.cellpose_model == "cyto2"
    assert cfg.channels == (0, 0)         # untouched profile value remains


def test_resolve_profile_name_auto_maps_to_grin_for_now(tmp_path):
    assert _resolve_profile_name(AUTO, tmp_path) == "grin"
    assert _resolve_profile_name("prism", tmp_path) == "prism"


def test_profile_serializes_in_summary_for_log():
    cfg = _cfg_for("prism")
    assert cfg.summary_for_log()["profile"] == "prism"
    assert cfg.summary_for_log()["channels"] == [0, 0]   # tuple → list
