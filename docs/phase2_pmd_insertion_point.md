# Phase 2 — PMD Denoising Insertion-Point Analysis

**Branch:** `feat/0-phase2-pmd` (cut from `fix/0-virtual-residual`, the engagement's resolved baseline).
**Status:** Discovery sub-step (Phase 2 directive item 1). **No behavior changed.** This document is
the "document it before coding" deliverable; implementation (item 2) and the A/B (item 3) follow only
after the gate decisions below are resolved.

**Goal:** add an OPTIONAL penalized-matrix-decomposition (PMD) spatiotemporal denoiser that feeds the
*detection* of Stages 3 and 4 only, behind an OFF-by-default flag, **without** disturbing the L+S
decomposition, Stage 2's Suite2p reuse, or the ResidualView/SourceLayer on-demand reconstruction
contract (preserved byte-for-byte).

---

## 1. Where Stages 3 and 4 read their residual

Both stages receive the single live `fov.residual_view` object (a `ResidualView`) and read it through
its three primitives (`read_chunk`, `read_rows`, `read_pixels`; `residual.py:216-273`). There are two
*distinct* kinds of read inside each stage — **detection scan** and **trace extraction** — and the
distinction is load-bearing for the design (§8, decision D1).

| Stage | Detection-scan read (the SNR-limited step PMD targets) | Trace-extraction read |
|---|---|---|
| 3 | `residual_view.read_rows(y0, y1)` streamed into the per-pixel MAD/template GPU kernel — `stage3.py:357-368` (MAD σ at `stage3.py:132-135`) | `extract_traces_from_residual(residual_view, masks, …)` — `stage3.py:496-498` |
| 4 | `detrend_to_memmap(residual_view, …)` → `view.read_rows(y0,y1)` per band — `stage4.py:473-475`, `stage4.py:80-88`; then bandpass on the detrended memmap | `extract_traces_from_residual(residual_view, masks, …)` — `stage4.py:519-522` |

Call sites in `run.py`: Stage 3 at `run.py:784-787`, Stage 4 at `run.py:917-918`.

**Residual identity differs between the two stages.** Stage 3 reads S₂ (deepest subtraction so far).
Between the stages, `run_source_subtraction(fov.residual_view, …)` returns `view_s3` and
`fov.residual_view = view_s3` (`run.py:870-877`), so Stage 4 reads S₃ = S₂ − (Stage-3 sources).

## 2. The single cleanest insertion point

**One point, in `run.py`, immediately before the Stage 3 block (`run.py:775`)**: substitute
`fov.residual_view` with a PMD-denoised, dense-backed `ResidualView`. Stage 4 inherits the denoised
base **automatically and through the existing machinery**, because:

- Stage-3 source subtraction advances the view via `view.with_source(...)`
  (`subtraction.py:344`, called from `run_source_subtraction`), and
- `with_source` constructs the new `ResidualView` with `dense=self._dense` (`residual.py:194-203`) —
  it **carries the dense base forward**. So `view_s3` = (PMD dense base) − (Stage-3 source layers),
  exactly the S₃ Stage 4 should see, with no change to `with_source` or the subtraction code.

No other insertion point has this property: a per-read wrapper object would be silently dropped at the
`with_source` rebuild (it hardcodes `ResidualView(...)`), so Stage 4 would lose the denoise.

## 3. Mechanism: materialize-once dense base — NOT a lazy per-read wrapper

The naive "wrap the view and denoise each `read_*` call" approach is **rejected**:

- PMD is a **global, patch-wise spatiotemporal decomposition** (low-rank + penalty over each spatial
  patch across the full time axis). Applying it independently per read request yields mutually
  inconsistent results depending on the access pattern, and re-fits the decomposition on every read.
- `read_chunk(t0,t1)` returns only a **temporal slab** (`cs, Ly, Lx`) — full-T-over-patch PMD is
  impossible there.

**Correct mechanism** (mirrors how Stage 4 already materializes `S3_detrended.dat` / filtered memmaps
at `stage4.py:80-129`):

1. Stream the current residual out via `read_rows(y0, y1)` (each band is `(T, h, Lx)` — full T, the
   shape PMD patches want), fit PMD per spatial patch (patch-parallel, bounded RAM), and write the
   denoised result to a float32 memmap `S_pmd.dat` (≈ same on-disk size as Stage 4's existing
   intermediate memmaps).
2. Wrap that memmap as the dense base of a fresh `ResidualView`: `ResidualView(shape, dense=<memmap>)`.
   `__init__` assigns `self._dense = dense` **without copying** (`residual.py:112`); `read_chunk`
   slices then copies (`np.array(self._dense[t0:t1])`, `residual.py:220`) → bounded RAM preserved.

This exercises the **already-tested `_dense` read path** (the `test_residual_view.py` oracle covers
it), so the reconstruction arithmetic is byte-for-byte unchanged. **The ResidualView/SourceLayer
engine source is not modified.** (At most an *additive*, optional memmap-friendly constructor helper;
the existing `from_dense` does a full `np.asarray` copy `residual.py:171` and must not be used for the
large memmap — pass the memmap straight to the constructor instead.)

## 4. Separation from L+S, Stage 1, and Stage 2 (verified)

- **L+S** is computed once in Foundation from `data.bin` (raw int16 movie) and stored as SVD factors
  (`US_k`, `V_k_full`); the residual is virtual (`M − L − Σ sources`). Denoising the materialized
  residual never touches the factors → L+S unchanged.
- **Stage 1** segments on `mean_M`/`vcorr_S` summary images (`run.py:502-504`), not the residual view.
- **Stage 2** detection uses Suite2p outputs produced in Foundation from `data.bin`; its trace
  extraction (`stage2.py:231-236`) runs **before** Stage 3, on the pre-PMD view. Inserting PMD at
  `run.py:775` is strictly downstream of all Stage-2 reads → Stage 2 unaffected.

## 5. Streaming / memory contract the implementation must honor

- float32 throughout; read bands via `read_rows` (`(T, h, Lx)`), write `S_pmd.dat` as a float32 memmap.
- Patch-parallel, chunked, bounded peak RAM; GPU with CPU fallback (mirror `force_cpu` / device
  selection at `stage3.py:307-310`).
- Note: `compute_std_map` accumulates in **float64** (`subtraction.py:620`) — that invariant lives in
  the consumer and is unaffected by feeding it a dense-backed view.

## 6. Config plumbing (OFF by default)

Mirror the existing optional-denoise precedent (`use_denoise` → `denoise_mean_S`, `stage1.py:236-245`,
default `True` at `types.py`): add to the `PipelineConfig` dataclass (`types.py`) a single flag
`use_pmd_denoise: bool = False` (plus any PMD hyperparameters as fields with defaults). Read it once at
the `run.py:775` insertion point. **No default flip** in this phase.

## 7. Reconstruction-contract preservation statement

The on-demand contract (`S = M − L − Σ sources`, `residual.py:80-92`, oracle-verified in
`test_residual_view.py`) is preserved byte-for-byte: the PMD path swaps the *base* the view
reconstructs from (SVD-factor base → dense memmap base) using existing constructor + read code, and
leaves the source-subtraction chain (`with_source`) and all read math untouched. A regression run of
`test_residual_view.py` plus a new test asserting "PMD-off view == current SVD-factor view" will gate
the change.

---

## 8. Open decisions for the gate (need a human steer before coding)

**D1 — Detection-only vs detection+trace-extraction (blast radius / one-variable purity).**
The §2 single-point swap denoises *everything* read from the view downstream — including
`extract_traces_from_residual` (`stage3.py:496`, `stage4.py:519`) and `compute_std_map`
(`run.py:866`). That means final traces, the subtraction profile, and hence QC features / classifier
inputs (`classify.py`, `qc_features.py`) would all see denoised data — arguably more than "one
variable."
- **Option D1-a (recommended): detection-input-only.** Add an optional `detect_view=None` parameter to
  `run_stage3`/`run_stage4` (defaults to `residual_view` → byte-identical when PMD off). Use
  `detect_view` for the detection scan (`stage3.py:357`, `stage4.py:473`) and keep `residual_view`
  (non-denoised) for `extract_traces_from_residual`. Cleanest one-variable change: detection recall is
  the measured outcome; trace/QC/classifier semantics stay fixed. Cost: PMD must be fit for the
  detection input of each stage — and Stage 3 (S₂) vs Stage 4 (S₃) differ, so either two fits, or fit
  once on S₂ and accept Stage-4 detection on (PMD-S₂ − Stage-3 sources) via the dense-base propagation
  of §2 (one fit, recommended sub-choice).
- **Option D1-b: full swap (§2 as written).** Simplest; single fit; propagates for free. But couples
  trace extraction, std, and classifier inputs to PMD — a wider change to characterize in the A/B.

**D2 — Subtraction-profile coupling.** Under D1-b, `compute_std_map(fov.residual_view,…)` (`run.py:866`)
computes the Stage-3 subtraction std on denoised data. Internally consistent, but a behavioral change
to flag. Under D1-a it is avoided (std stays on the original residual).

**D3 — PMD library dependency (prerequisite; currently MISSING).** No PMD-lineage package
(`funimag`/`trefide`/`masknmf`/`localmd`) is installed in any env (`roigbiv`, `caiman`, …) and there is
no PMD reference in the repo. Implementation cannot start until this is chosen:
- **`localmd` / maskNMF-toolbox (recommended):** modern Paninski-lab PMD, torch/GPU-native,
  pip-installable — best fit for the GPU + streaming model.
- **`funimag` + `trefide`:** the original lineage, but `trefide` needs a fragile C++/Cython build;
  high install risk.
- **Minimal in-repo PMD** (patch-wise truncated SVD + penalty): no external dep, full control, but
  more code and a reproducibility burden.

---

## 9. Verification plan (for the implementation phase, not run here)

- Unit: extend `test_residual_view.py` with a "PMD-off ≡ SVD-factor view" equivalence test (proves the
  OFF default is a true no-op) and a dense-base propagation test through `with_source`.
- A/B (engagement harness, already stratified): `python -m roigbiv.eval.harness --batch
  experiments/harness/heldout_fovs.txt …`, PMD on vs off, stratified recall for Stage 3 (sparse) and
  Stage 4 (tonic) targets, plus compute-time delta and peak GPU/host memory, with an explicit OOM
  check on the smallest-VRAM target.
