"""
Env-driven registry config (:mod:`roigbiv.registry.config`).

Covers the three matcher settings that were changed after measuring the
three-session prism FOV, where the shipped defaults produced no cross-session
match at all: the ``d_cutoff`` escape hatch, and the accept threshold.
"""
from __future__ import annotations

import pytest

from roigbiv.registry.config import RegistryConfig, build_adapter_config


@pytest.fixture
def clean_env(monkeypatch):
    """A pristine env — these settings are all read from os.environ."""
    for name in ("ROIGBIV_ROICAT_D_CUTOFF", "ROIGBIV_FOV_ACCEPT_THRESHOLD",
                 "ROIGBIV_FOV_REVIEW_THRESHOLD", "ROIGBIV_REGISTRY_DSN",
                 "ROIGBIV_BLOB_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ── d_cutoff ───────────────────────────────────────────────────────────────


def test_d_cutoff_is_unset_by_default(clean_env):
    """Unset means "let ROICaT infer" — correct on real-sized FOVs."""
    assert RegistryConfig.from_env().roicat_d_cutoff is None
    assert build_adapter_config(RegistryConfig.from_env()).d_cutoff is None


def test_d_cutoff_env_var_reaches_the_adapter(clean_env):
    """The knob exists precisely because ROICaT's inference returns None on
    small inputs and is then dereferenced unconditionally."""
    clean_env.setenv("ROIGBIV_ROICAT_D_CUTOFF", "0.7")

    cfg = RegistryConfig.from_env()
    assert cfg.roicat_d_cutoff == 0.7
    assert build_adapter_config(cfg).d_cutoff == 0.7


def test_blank_d_cutoff_is_treated_as_unset(clean_env):
    """An exported-but-empty var must not become 0.0, which would prune
    every edge and silently cluster nothing."""
    clean_env.setenv("ROIGBIV_ROICAT_D_CUTOFF", "  ")
    assert RegistryConfig.from_env().roicat_d_cutoff is None


def test_a_malformed_d_cutoff_fails_loudly(clean_env):
    clean_env.setenv("ROIGBIV_ROICAT_D_CUTOFF", "not-a-number")
    with pytest.raises(ValueError):
        RegistryConfig.from_env()


# ── thresholds ─────────────────────────────────────────────────────────────


def test_accept_threshold_defaults_to_the_measured_value(clean_env):
    """0.8, not the 0.9 carried over from v3 bring-up.

    The prism FOV matched correctly at 0.826 and 0.872; 0.9 sent both to
    review, and a review writes no session row, so the next session had
    nothing to match against either.
    """
    assert RegistryConfig.from_env().fov_accept_threshold == 0.8


def test_accept_threshold_is_still_overridable(clean_env):
    """Dense FOVs have more ROIs behind the posterior and can afford 0.9."""
    clean_env.setenv("ROIGBIV_FOV_ACCEPT_THRESHOLD", "0.95")
    assert RegistryConfig.from_env().fov_accept_threshold == 0.95


def test_review_threshold_is_unchanged(clean_env):
    assert RegistryConfig.from_env().fov_review_threshold == 0.5


def test_accept_stays_above_review_at_the_defaults(clean_env):
    cfg = RegistryConfig.from_env()
    assert cfg.fov_accept_threshold > cfg.fov_review_threshold
