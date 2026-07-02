# Limitations — what ROI G. Biv does not claim

This page states, in plain terms, what ROI G. Biv (roigbiv) does **not** claim to
do or guarantee. It is a scope document, not a defect list: each item below is
either an intentional design boundary or a claim that requires more validation
data before it can be strengthened. See
[`docs/design/OVERVIEW.md`](design/OVERVIEW.md) for the full architecture and
per-stage behavior.

## Claim 1: Stage 4 detects tonic *candidates*, not proven tonic neurons

Stage 4 (`roigbiv/pipeline/stage4.py:4-14`) flags cells whose activity pattern
looks tonic — sustained, low-variance fluorescence with no discrete transients —
using a bandpass + inner/outer correlation-contrast signal. This is a
morphological/statistical proxy for tonic firing, not a validated
electrophysiological or ground-truth label.

Accordingly, Stage 4 has no accept tier by default: passing candidates receive
`gate_outcome="flag"` / `confidence="requires_review"`, and failures are
rejected. Every Stage-4 detection routes to human review unless the pipeline
operator explicitly opts into the (off-by-default) tonic accept tier
(`roigbiv/pipeline/types.py`, `tonic_accept_tier: bool = False`) — and even
then, Stage-4-sourced candidates are never auto-promoted; only anatomically
detected (Stage 1/2) ROIs are eligible
(`docs/design/OVERVIEW.md`, §8, "Stage 4 has no accept tier").

## Claim 2: denoised-only detections require raw-movie validation

roigbiv has two independent denoised-branch paths — Cellpose3's built-in
`denoise_cyto3` (`use_denoise`) and an optional out-of-process DeepCAD-RT
sidecar (`deepcad_denoise`), both in `roigbiv/pipeline/types.py`. (A separate
PMD spatiotemporal denoiser also exists behind `use_pmd_denoise`, but it
replaces the residual in place rather than offering a raw-vs-denoised branch
choice, so it isn't a third instance of this specific caveat.) Neither
denoised-branch path is cross-validated against the raw (non-denoised) movie
by the pipeline itself:
the DeepCAD-RT module "only produces `{stem}_deepcad.tif` + provenance... it
does not decide which stages consume the denoised branch vs. the raw branch —
that routing is a separate concern" (`roigbiv/pipeline/deepcad.py:21-26`).

A detection made only on a denoised branch, with no corresponding signal on the
raw movie, should be treated as provisional. roigbiv does not currently
guarantee or automatically check that denoised-only detections reproduce on
raw data.

## Claim 3: not claiming general segmentation SOTA on all calcium imaging

roigbiv is scoped to **two-photon calcium imaging** with the default indicator
GCaMP6s (`README.md`, `docs/design/OVERVIEW.md` §1: "detects regions of
interest (ROIs)... in two-photon calcium imaging movies of the mouse brain.
The default indicator is GCaMP6s"). It is not benchmarked against, and makes
no performance claims for, one-photon/miniscope imaging, other indicators,
other species, or general-purpose cell-segmentation datasets outside two-photon
calcium imaging. Any apparent advantage over other segmentation tools is
specific to the sequential subtractive setting it was built and validated in.

## Claim 4: L+S+T decomposition is intentionally not implemented

roigbiv's Foundation stage splits each movie into a low-rank background (L)
and sparse residual (S) via truncated SVD
(`docs/design/OVERVIEW.md` §3.2, "Truncated-SVD low-rank / sparse (L+S)
background split"). There is no third temporal (T) term in the wired
pipeline. A separate, more rigorous robust-PCA L+S implementation exists in
`roigbiv/pipeline/rpca.py` but is gated behind a config field
(`background_method`) that does not exist on `PipelineConfig` and has no
runtime call site — it is "aspirational/disabled until a config field and call
site are added" (`docs/design/OVERVIEW.md` §11, "RPCA robust background —
implemented but NOT wired"). An L+S+T extension has been considered and
explicitly rejected for now — not separably identifiable without strong
priors — per `docs/adr/0001-non-destructive-candidate-union.md` (Decision 4,
"Do not build yet"); it is not on a committed roadmap.

## Claim 5: cross-session probabilities are calibrated only within validated data regimes

The cross-session FOV-matching registry (`roigbiv/registry/orchestrator.py`,
`register_or_match`) scores candidate matches with a logistic posterior over
FOV-level features (shared-ROI fraction, alignment quality, cluster cohesion).
Until a labeled cross-session pair set exists, the model's coefficients are
hand-tuned priors, not fit from data — "the coefficients are hand priors...
[this is] the normal state of the system until labeled cross-session pairs are
collected" (`roigbiv/registry/calibration.py`, `DEFAULT_FOV_COEFS` and
`CalibrationModel`). The accept/review thresholds
(`ROIGBIV_FOV_ACCEPT_THRESHOLD`, `ROIGBIV_FOV_REVIEW_THRESHOLD`) are
configurable, but the underlying posterior is only as accurate as the data
regime it has been validated against. Match probabilities from FOVs or imaging
conditions outside that regime should be treated as provisional, not
calibrated.
