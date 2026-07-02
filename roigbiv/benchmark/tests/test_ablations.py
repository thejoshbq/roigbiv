"""Tests for roigbiv.benchmark.ablations — the ablation preset registry
(issue #33)."""
from __future__ import annotations

import pytest

from roigbiv.benchmark.ablations import ABLATIONS, ALL, get_ablation, list_ablations
from roigbiv.pipeline.types import PipelineConfig

_EXPECTED_NAMES = {
    "raw_only", "denoised_only", "dual_branch", "cellpose3_only",
    "cellpose_sam_only", "stage3_off", "stage4_off",
    "residual_refinement_off", "joint_deconfliction_off",
}


# ── registry ────────────────────────────────────────────────────────────────

def test_all_nine_ablations_registered():
    assert set(ABLATIONS) == _EXPECTED_NAMES


def test_get_ablation_returns_copy():
    get_ablation("raw_only")["use_denoise"] = "mutated"
    assert get_ablation("raw_only")["use_denoise"] is False   # registry not mutated


def test_get_ablation_all_sentinel_raises():
    with pytest.raises(ValueError):
        get_ablation(ALL)


def test_get_ablation_unknown_raises():
    with pytest.raises(ValueError):
        get_ablation("not-a-real-ablation")


def test_list_ablations_all_first():
    names = list_ablations()
    assert names[0] == ALL
    assert set(names) == {ALL, *ABLATIONS}


# ── per-preset content ───────────────────────────────────────────────────────

def test_joint_deconfliction_off_is_empty_noop():
    assert get_ablation("joint_deconfliction_off") == {}


def test_raw_only_disables_all_denoise_mechanisms():
    p = get_ablation("raw_only")
    assert p["enable_denoised_branch"] is False
    assert p["denoiser_backend"] == "none"
    assert p["deepcad_denoise"] is False
    assert p["use_denoise"] is False
    assert p["use_pmd_denoise"] is False


@pytest.mark.parametrize("name", ["denoised_only", "dual_branch"])
def test_denoised_presets_bridge_both_denoiser_surfaces(name):
    p = get_ablation(name)
    # deepcad_denoise is the surface that actually executes (issue #37).
    assert p["deepcad_denoise"] is True
    # enable_denoised_branch/denoiser_backend (issue #34) are set too, for
    # forward-compat once the two surfaces reconcile.
    assert p["enable_denoised_branch"] is True
    assert p["denoiser_backend"] == "deepcad_rt"
    # Regression guard: denoiser_backend != "none" unconditionally requires
    # denoiser_model_path is not None (foundation.py's guard fires regardless
    # of enable_denoised_branch) — a None here means every FOV run under this
    # ablation would raise ValueError before any pipeline stage executes.
    assert p["denoiser_model_path"] is not None


def test_cellpose3_only_matches_dataclass_default():
    assert get_ablation("cellpose3_only")["stage1_backend"] == "cellpose3"


def test_cellpose_sam_only_selects_sidecar_backend():
    assert get_ablation("cellpose_sam_only")["stage1_backend"] == "cpsam_sidecar"


def test_stage3_off_disables_stage_3_only():
    p = get_ablation("stage3_off")
    assert p["enable_stage_3"] is False
    assert "enable_stage_4" not in p


def test_stage4_off_disables_stage_4_only():
    p = get_ablation("stage4_off")
    assert p["enable_stage_4"] is False
    assert "enable_stage_3" not in p


def test_residual_refinement_off_sets_candidate_union_mode():
    assert get_ablation("residual_refinement_off")["pipeline_mode"] == "candidate_union"


# ── PipelineConfig construction (shallow integration guard) ─────────────────

@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_ablation_overrides_construct_valid_pipelineconfig(name):
    cfg = PipelineConfig(**{**{"fs": 7.5}, **get_ablation(name)})
    assert isinstance(cfg, PipelineConfig)


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_ablation_respects_denoiser_model_path_invariant(name):
    """Pins the exact guard in roigbiv/pipeline/foundation.py: whenever
    denoiser_backend != "none", denoiser_model_path must be set. Checked
    directly against the registry (not via a full run_foundation call) so
    this stays a fast, isolated unit test."""
    p = get_ablation(name)
    backend = p.get("denoiser_backend", "none")
    if backend != "none":
        assert p.get("denoiser_model_path") is not None
