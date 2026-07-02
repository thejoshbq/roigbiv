"""Ablation registry — parameter-override bundles for measuring the marginal
contribution of each pipeline component (Milestone A roadmap item A9,
issue #33).

Same shape as roigbiv.pipeline.profiles: a flat dict-of-dicts keyed by
PipelineConfig field names, splatted via merged_overrides()'s explicit_dicts
seam so ablation overrides win over the lens/optics profile bundle. No new
field is added to PipelineConfig itself — which ablation produced a given
FovRunResult is recorded on FovRunResult.ablation instead.

Per issue #33's own out-of-scope note: "Ablation targets that don't exist yet
are wired as presets; they activate as their features land." Two presets
(residual_refinement_off, joint_deconfliction_off) are INERT TODAY — see the
per-preset comments — by design, not by omission.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["ABLATIONS", "ALL", "get_ablation", "list_ablations"]

ALL = "all"   # CLI/registry sentinel: expand to every registered ablation name

# denoiser_backend != "none" unconditionally requires denoiser_model_path to be
# set (roigbiv/pipeline/foundation.py's validation guard fires regardless of
# enable_denoised_branch). Nothing actually consumes this path yet — the real
# execution happens via deepcad_denoise below — so this exists purely to
# satisfy that guard without tripping a ValueError on every FOV. Replace with
# a real checkpoint path once a real backend-dispatch stage lands.
_DENOISER_MODEL_PATH_PLACEHOLDER = Path("<unset:ablation-placeholder>")

# Flat override bundles keyed by PipelineConfig field names.
ABLATIONS: dict[str, dict] = {
    "raw_only": {
        # Explicit control arm: no denoised branch, no Cellpose3-internal
        # restoration, no PMD spatiotemporal denoise. Deliberately more
        # aggressive than "just leave defaults" (use_denoise defaults True) so
        # this preset is a genuinely unprocessed baseline.
        "enable_denoised_branch": False,
        "denoiser_backend": "none",
        "deepcad_denoise": False,
        "use_denoise": False,
        "use_pmd_denoise": False,
    },
    "denoised_only": {
        # BEST-EFFORT BRIDGE across the two disconnected denoiser surfaces:
        # issue #34's enable_denoised_branch/denoiser_backend are validated
        # but inert (no stage consumes them yet); issue #37's deepcad_denoise
        # is the only one that actually executes (produces {stem}_deepcad.tif
        # via foundation.py). Both are set so this preset is correct today
        # (via deepcad_denoise) and stays correct once #34/#37 reconcile.
        #
        # INERT-FOR-DETECTION until raw-vs-denoised branch routing lands (see
        # FOVData.denoised_path's docstring: "no stage consumes this yet") —
        # detection still runs on raw/mean_M regardless of this preset today.
        "enable_denoised_branch": True,
        "denoiser_backend": "deepcad_rt",
        "denoiser_model_path": _DENOISER_MODEL_PATH_PLACEHOLDER,
        "deepcad_denoise": True,
    },
    "dual_branch": {
        # Same bridge as denoised_only. Today the two presets are functionally
        # identical (both just produce the denoised artifact without changing
        # what detection consumes) — kept as separate registry entries per
        # issue #33's explicit preset list; they diverge once raw+denoised
        # branch routing lands (denoised_only routes Stage 1 to the denoised
        # movie only; dual_branch runs both and reconciles).
        "enable_denoised_branch": True,
        "denoiser_backend": "deepcad_rt",
        "denoiser_model_path": _DENOISER_MODEL_PATH_PLACEHOLDER,
        "deepcad_denoise": True,
    },
    "cellpose3_only": {
        # Matches the PipelineConfig default; explicit for the ablation
        # matrix / documentation, and so a future default flip doesn't
        # silently redefine what this preset means.
        "stage1_backend": "cellpose3",
    },
    "cellpose_sam_only": {
        # Real, wired sidecar branch (roigbiv/pipeline/stage1.py). Requires
        # the cp-sam conda env; fails gracefully per-FOV (recorded as
        # FovRunResult.error) if that env isn't configured on the
        # benchmarking machine.
        "stage1_backend": "cpsam_sidecar",
    },
    "stage3_off": {
        "enable_stage_3": False,
    },
    "stage4_off": {
        "enable_stage_4": False,
    },
    "residual_refinement_off": {
        # pipeline_mode is CURRENTLY INERT PLUMBING (types.py: "no stage reads
        # this field yet"). This sets the forward-looking value (candidate_
        # union, i.e. NOT candidate_union_with_residual_refinement) so the
        # moment a stage starts branching on cfg.pipeline_mode, this preset
        # activates correctly with zero further changes to this file.
        "pipeline_mode": "candidate_union",
    },
    "joint_deconfliction_off": {
        # NO CONFIG FIELD EXISTS for joint deconfliction anywhere in
        # PipelineConfig — ADR-0001 names it as a future pipeline stage.
        # Wired as a documented no-op per issue #33's own out-of-scope note
        # ("targets that don't exist yet are wired as presets; they activate
        # as their features land").
    },
}


def get_ablation(name: str) -> dict:
    """Return a fresh copy of the override bundle for *name*.

    Raises
    ------
    ValueError
        If *name* is ``all`` (must be expanded to concrete names upstream —
        resolve in the CLI/runner, mirrors profiles.get_profile's ``auto``
        handling) or unknown.
    """
    if name == ALL:
        raise ValueError(
            f"{ALL!r} must be expanded to concrete ablation names before "
            "get_ablation() (resolve in the CLI/runner)."
        )
    if name not in ABLATIONS:
        raise ValueError(
            f"unknown ablation {name!r}; choose one of {sorted(ABLATIONS)} "
            f"or {ALL!r}."
        )
    return dict(ABLATIONS[name])


def list_ablations() -> list[str]:
    """Ablation choices for CLI ``--ablation`` (``all`` first)."""
    return [ALL, *sorted(ABLATIONS)]
