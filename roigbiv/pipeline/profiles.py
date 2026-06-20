"""Acquisition/lens profiles — parameter bundles for lens-agnostic detection.

The pipeline's defaults (``PipelineConfig``) are tuned for 512² **GRIN** imaging
(bright, round, ~12 px somata). Dim, diffuse **PRISM** FOVs (~56 px) need a
different Stage-1 configuration. Rather than make a user hand-set ~8 CLI flags,
a *profile* bundles the corrections behind one selector.

This is a **Python dict registry** (not a YAML loader): flat dicts keyed by
``PipelineConfig`` field names, so they splat directly and cannot drift from the
dataclass. The merge precedence enforced by the callers is::

    PipelineConfig defaults  <  profile bundle  <  explicit user flags

Empirical grounding (Phase A, on Logan's PRISM ``post-007``; see the gated-workflow
plan): the dominant PRISM fix is **single-channel** Stage-1 input. Feeding
``vcorr_S`` as Cellpose's nucleus channel (``channels=(1,2)``) suppresses
segmentation on PRISM's diffuse correlation map — switching to ``channels=(0,0)``
takes cyto3 from 0 → 13 detections on ``mean_M``; dropping ``denoise`` and the
generalist ``cyto3`` model add the rest (→16). The GRIN-fine-tuned deployed model
is the *worst* on PRISM, so PRISM uses the ``cyto3`` generalist.

``cpsam`` (Cellpose-SAM) is deliberately **not** referenced here: it is a
Cellpose 4.x model and cannot load under this repo's ``cellpose<4.0.0`` pin. A
CP4 generalist is a separate, deferred sidecar track.

Gate/area constants (``min_area``/``max_area``) come from
``scripts/measure_prism_scale.py`` (Logan FOVs: median diameter≈56, area≈2480,
p95≈3350). Constants still marked PENDING in the plan's provenance table
(``min_solidity``, ``max_eccentricity``, ``tile_norm_blocksize``,
``flow_threshold``, ``cellprob_threshold``) are included so the profile is usable
end-to-end, but should be confirmed by A/B against PRISM ground truth before they
are treated as load-bearing.
"""
from __future__ import annotations

__all__ = [
    "PROFILES", "AUTO", "STAGE1_CLI_DEFAULTS",
    "get_profile", "list_profiles", "is_profile", "merged_overrides",
]

AUTO = "auto"

# Historical CLI defaults for the Stage-1 args that are made ``default=None`` in
# argparse so a profile can fill them. When a flag is omitted AND the resolved
# profile does not set the key, these values are backfilled so an omitted flag
# keeps its exact prior CLI behavior. NOTE: ``flow_threshold`` is 0.6 here — the
# historical CLI default — which deliberately differs from the PipelineConfig
# dataclass default (0.4); backfilling preserves GRIN CLI byte-identity.
STAGE1_CLI_DEFAULTS: dict = {
    "diameter": 12,
    "diameter_auto": False,
    "cellprob_threshold": -2.0,
    "flow_threshold": 0.6,
    "channels": (1, 2),
}

# Flat override bundles keyed by PipelineConfig field names. ``grin`` is empty =
# the dataclass defaults (the working 512² baseline), so selecting it is a no-op.
PROFILES: dict[str, dict] = {
    "grin": {},
    "prism": {
        # ── Group 1: Stage-1 input + model + diameter (Phase-A grounded) ──
        "channels": (0, 0),            # single-channel on mean_M (dominant lever: 0→13)
        "cellpose_model": "cyto3",     # generalist beats the GRIN-overfit deployed model
        "use_denoise": False,          # denoise_cyto3 suppresses dim PRISM signal
        "diameter": 56,                # measure_prism_scale.py median
        # ── Group 2: Gate-1 area bounds (widened by the Stage-1 recall OFAT) ──
        # Bounds widened from the measure_prism_scale.py 1500/5000 after a
        # drift-guarded one-factor-at-a-time matrix on VI15_D2_FOV2 (pre-005):
        # min_area 1500→900 recovers small dim somata; max_area 5000→9000
        # recovers large single-soma masks the 5000 ceiling clipped. Net 11→17
        # accepts, 0 rejects, on that FOV. CAVEAT: max_area=9000 also admits
        # genuine 2-soma merges — KEEP the peak-count check (skimage
        # peak_local_max, ~28 px sep) downstream so ≥2-peak masks flag rather
        # than accept (the label-11 merge case). See scripts/stage1_matrix/.
        "min_area": 900,
        "max_area": 9000,
        # ── Group 3: relax for cyto3's ugly-but-valid masks (recall OFAT) ──
        "min_solidity": 0.40,
        "max_eccentricity": 0.97,    # 0.95→0.97 recovers the 1 elongated reject
        "tile_norm_blocksize": 256,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        # ── Foundation (upstream of all gates) — PENDING A/B (A1 ran at 32) ──
        "mc_strip_height": 48,
    },
    # Conservative "unknown optics" fallback: single-channel generalist with a
    # per-FOV diameter estimate. Deliberately NO adaptive_gates / NO ensemble —
    # the least-certain path must not also be the most experimental one.
    "generic": {
        "channels": (0, 0),
        "cellpose_model": "cyto3",
        "use_denoise": False,
        "diameter_auto": True,
    },
}


def is_profile(name: str) -> bool:
    """True if *name* is a concrete profile (not ``auto``, not unknown)."""
    return name in PROFILES


def get_profile(name: str) -> dict:
    """Return a fresh copy of the override bundle for *name*.

    Raises
    ------
    ValueError
        If *name* is ``auto`` (must be resolved to a concrete profile upstream,
        where explicit-vs-default flags are still distinguishable) or unknown.
    """
    if name == AUTO:
        raise ValueError(
            "'auto' must be resolved to a concrete profile before get_profile() "
            "(resolve in the CLI/UI where explicit flags are distinguishable)."
        )
    if name not in PROFILES:
        raise ValueError(
            f"unknown profile {name!r}; choose one of {sorted(PROFILES)} or 'auto'."
        )
    return dict(PROFILES[name])


def list_profiles() -> list[str]:
    """Profile choices for CLI ``--profile`` / the UI dropdown (``auto`` first)."""
    return [AUTO, *sorted(PROFILES)]


def merged_overrides(profile_name: str, base: dict,
                     explicit_dicts: "list[dict]") -> dict:
    """Merge config overrides with precedence ``base < profile < explicit``.

    Returns a single flat dict for ``PipelineConfig(**out)``. Merging into one
    dict (last-wins) is required because Python forbids the same key appearing in
    two ``**`` splats, and the profile bundle overlaps the explicit override dicts
    (e.g. ``min_area``).

    Parameters
    ----------
    profile_name : concrete profile name (NOT ``auto`` — resolve upstream).
    base : always-present, non-profile-able kwargs (fs, tau, output_dir, …).
    explicit_dicts : ordered list of None-filtered user-override dicts; later
        dicts win over earlier ones, and all win over the profile bundle.
    """
    out = dict(base)
    out.update(get_profile(profile_name))      # raises on auto/unknown
    for d in explicit_dicts:
        out.update(d)
    out["profile"] = profile_name
    return out
