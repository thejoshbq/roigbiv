# Phase M — Cellpose 4.x Migration: Discovery (read-only)

**Branch:** `chore/0-phasem-cellpose4x` (off baseline `fix/0-virtual-residual` @ `0138689`).
**Engagement:** recall-refinement, Phase M (prerequisite migration that gates Phases 3/4).
**Status:** discovery only — **no env mutation, no checkpoint change**. This phase carries
`migration` + `data_integrity` risk and touches the protected `models/deployed/` checkpoint, so
per `discovery_before_code` the findings below are surfaced *before* any implementation.

## TL;DR — the Phase-0-gate decision is contradicted by the environment

At the Phase-0 gate the resolved decision was **"UPGRADE cellpose to ≥4.x (full upgrade, not a
sidecar)."** Discovery shows that decision is **unworkable as stated** and would damage the
load-bearing foundation. The decisive facts:

1. **The deployed CP3 checkpoint CANNOT load under cellpose 4.x.** Hard error, verified:
   `ValueError: This model does not appear to be a CP4 model. CP3 models are not compatible with CP4.`
   (loading `models/deployed/current_model` under cellpose 4.2.1.1).
2. **Phase 3's entire purpose is an A/B of `cpsam` vs the deployed CP3 checkpoint.** If we do a full
   in-place upgrade, CP3 can no longer run *at all* → **the Phase 3 A/B becomes impossible in one
   env.** Running that comparison *requires* cellpose 3.x (CP3) and 4.x (cpsam) co-present. The
   sidecar is therefore **not a preference — it is mandatory for the comparison the engagement asks
   for.**
3. **A full in-place upgrade threatens the foundation.** The canonical `roigbiv` env is
   `numpy 1.26.4 / suite2p 0.14.5`; cellpose 4.x requires **numpy 2.x** (`cp-sam` env carries
   `numpy 2.4.6`). Forcing numpy 2.x into `roigbiv` risks suite2p 0.14.5 (Stage 2 + Foundation
   registration) and the custom `torch 2.12.0+cu130` sm_120 build — i.e. the parts of the pipeline
   the engagement explicitly declares CORRECT and out of scope.
4. **A working cellpose 4.x sidecar already exists** (`cp-sam` conda env, cellpose 4.2.1.1) and is
   **GPU-capable on the RTX 5080** (verified: `torch.cuda.get_device_capability == (12, 0)`, GPU
   matmul + a real `cpsam` GPU `eval` both succeed). `profiles.py:22-24` already anticipates this
   "separate, deferred sidecar track."

**Conclusion to ratify at the gate:** keep `roigbiv` on cellpose 3.x (CP3 + suite2p + GPU build
intact) and run `cpsam` out-of-process via the existing `cp-sam` sidecar env. This *reverses* the
Phase-0 "full upgrade" decision — which is exactly the kind of directive/reality contradiction
`discovery_before_code` says to STOP and report rather than code around.

## Environment reality

| | `roigbiv` (canonical) | `cp-sam` (sidecar, exists) |
|---|---|---|
| cellpose | **3.1.1.2** | **4.2.1.1** |
| torch | 2.12.0+cu130 (sm_120 build) | 2.12.0 — **GPU works on 5080** (cap 12,0 verified) |
| numpy | **1.26.4** | **2.4.6** |
| suite2p | 0.14.5 | — (not needed; Stage 2 stays in roigbiv) |
| deployed CP3 checkpoint | loads ✅ (production) | **load FAILS ❌** (CP3≠CP4) |
| stock `cpsam` | unusable (silent fallback under 3.x) | loads + GPU eval ✅ |

## Cellpose 3.x → 4.x API breakage (verified in `cp-sam` env)

Relevant to `roigbiv/pipeline/stage1.py`:

- **`cellpose.denoise.DenoiseModel` is GONE in 4.x** (`hasattr(cellpose.denoise,'DenoiseModel')`
  → `False`). `denoise_mean_S` / `denoise_cyto3` (`stage1.py:110-142`, invoked at `:236-238` when
  `cfg.use_denoise`) has **no 4.x equivalent**. cpsam is noise-robust by design, so under a cpsam
  path the denoise step is simply dropped; the CP3 path still needs it (another reason CP3 stays on
  3.x).
- **`CellposeModel.eval` still accepts `channels=` / `channel_axis=` / `normalize=` / `diameter=` /
  `cellprob_threshold=` / `flow_threshold=`** (kwargs present in 4.x signature), but cpsam is
  **channel-invariant** — the `channels=(1,2)` cyto/nucleus role convention (`stage1.py:248,283`) is
  tolerated but semantically inert. cpsam takes up to 3 channels with no role assignment (this is
  what *enables* Phase 4's 3-channel enrichment, and what makes the CP3 `channels=(1,2)` convention
  non-portable).
- **`CellposeModel.__init__`** still takes `pretrained_model=`, `model_type=`, `gpu=`, `nchan=` —
  so the resolver/load shape (`stage1.py:230-231`) is signature-compatible; only the *checkpoint
  format* is not.
- `flows` tuple still length-3 (cellprob extraction at `stage1.py:295-298` is structurally OK).

## Implication for the engagement sequence

- **Phase 3** ("cpsam vs deployed CP3") is only meaningful as a **two-env A/B**: CP3 in-process
  (roigbiv/3.x), cpsam out-of-process (cp-sam/4.x). A full upgrade would delete one arm of the
  comparison.
- **Phase 4** (3-channel enrichment) depends on cpsam's channel-invariance and so lives on the
  sidecar path; it cannot be done on CP3 (`channels=(1,2)` is fixed 2-channel).
- The deployed CP3 checkpoint is **data of record** in a protected zone. Nothing here retrains or
  overwrites it; the sidecar approach keeps it loadable and unchanged.

## Options at the gate (your call — no code beyond this doc until you choose)

- **(M1) Sidecar integration (recommended).** Keep `roigbiv` on cellpose 3.x. Add an OFF-by-default
  `stage1_backend ∈ {cellpose3, cpsam_sidecar}` config field. When `cpsam_sidecar` is selected,
  Stage 1 hands `mean_M`/`vcorr_S` to the `cp-sam` env out-of-process (subprocess CLI or a thin
  file/np handoff) and reads back the label image — Stage 1's *inputs and outputs are unchanged*, so
  Gate 1 / subtraction / provenance / the residual engine are untouched. Enables the Phase 3 A/B
  cleanly. Zero risk to suite2p/Foundation/the GPU build/the CP3 checkpoint.
- **(M2) Full in-place upgrade of `roigbiv` to cellpose 4.x.** Matches the literal Phase-0 wording,
  but: strands the deployed CP3 checkpoint (unloadable), forces numpy 2.x (suite2p/Foundation/GPU
  build at risk), deletes the denoise step, and **makes the Phase 3 A/B impossible**. High blast
  radius into explicitly-out-of-scope subsystems. Not recommended.
- **(M3) Fresh dedicated env** (clone `roigbiv` → bump only cellpose+numpy). Same checkpoint/Phase-3
  problems as M2, plus a second full GPU/torch+suite2p stack to maintain. The existing `cp-sam`
  env already covers the cpsam-only need, so this buys nothing over M1.

## Recommendation

Adopt **M1 (sidecar)**. It is the only option that (a) preserves the protected CP3 checkpoint and
the out-of-scope foundation byte-for-byte, and (b) actually permits the Phase 3 cpsam-vs-CP3 A/B the
engagement requires. It also realizes the "deferred sidecar track" already documented in
`profiles.py:22-24`. The `cpsam` default stays OFF (`no_default_flip`); a default flip would only be
considered after the Phase 3 A/B + explicit approval.

## What this discovery did NOT do

No conda env was modified. No checkpoint was loaded into, retrained, or overwritten. All probes were
read-only (version queries, a signature inspection, a load-attempt that failed cleanly, and tiny
synthetic-image evals). The deployed CP3 checkpoint is untouched.
