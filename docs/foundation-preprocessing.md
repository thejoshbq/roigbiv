# Foundation Preprocessing: PrairieView → motion-corrected stack

**Scope.** This document covers the *foundational* steps that turn raw PrairieView two-photon output into the input ROI G. Biv consumes: **stacking** single-frame TIFFs into one container, **motion correction**, and the **SVD / L+S background + summary images** that the detection stages read. It audits the legacy notebook pipeline (`docs/legacy/legacy_mc/`) step by step against the current in-repo implementation, and for each step gives an optimization proposition with a quality gate that must not regress relative to legacy.

Reference dataset throughout: `data/logan_cousa_trial/` (Logan Cousa's DS-Prism-3 FOV2). When behavior is in question, `docs/roi-pipeline-specification.md` is authoritative; this doc defers to it.

**TL;DR.** Two of the three legacy stages are already obsolete or replaced by faster, bit-exact in-repo code (`io.py`); the housekeeping/rename stage is dead work, and HDF5 stacking is superseded by a single-pass BigTIFF streamer. The one genuinely open question is **motion correction**: the proven backend (`phasecorr` / Suite2p) matches legacy quality but is slow (~370 s foundation on a 2.3k-frame FOV); the ~10× faster GPU backend (`rowwise-pcc`) currently regresses quality and so cannot be adopted as-is. Three concrete directions to close that gap are laid out in [§5](#5-motion-correction--the-open-decision), with a shared validation recipe.

---

## 1. The dataset

`data/logan_cousa_trial/` holds three acquisition trials from one prism-lens FOV. Each trial is a directory of **one `.ome.tif` file per frame** — PrairieView/Bruker writes the stream frame-by-frame, not as a stack.

| Trial dir | Frames | Size | Notes |
| --- | --- | --- | --- |
| `..._pre-005/` | 2,271 | 4.5 GB | pre-treatment |
| `..._post-007/` | 2,144 | 4.2 GB | post-treatment |
| `..._beh-006/` | 27,253 | 54 GB | behavior session (long) |

Per-frame characteristics (read from the data):

- **1024 × 1024, uint16, single-page** per file (~2.1 MB/frame). This is the **prism** scale — 4× the linear resolution of the 512² GRIN FOVs the base config was tuned for. See `configs/pipeline.prism.yaml` and [§6](#6-frame-rate-and-prism-scale-notes).
- Filenames carry a **cycle/channel/index** scheme: `..._Cycle00001_Ch2_000001.ome.tif`. The 6-digit suffix is the frame order; `Ch2` is the imaging channel.
- **Frame `000001` is the OME-XML master** (it embeds the multi-file OME metadata; ~2.9 MB vs ~2.1 MB for the rest). Every other frame is `BinaryOnly` and points back to it.
- Each trial directory also contains PrairieView **sidecars**: a `*.xml` (~1.1 MB — per-frame timing, laser/PMT/scan settings, the true frame rate) and a `*.env` (~70 KB — acquisition environment), plus a `References/` subdir with 16-bit and 8-bit reference TIFFs. None of these are imaging frames.

---

## 2. The pipeline at a glance: legacy → current

| Step | Legacy (notebooks) | Current roigbiv | Status |
| --- | --- | --- | --- |
| 0. Housekeeping + `.ome.tif`→`.tif` rename | NB1 / NB3 cells 0–3 | not needed — `io.py` ignores sidecars by pattern, reads `.ome.tif` in place | **Obsoleted** |
| 1. Stack frames → one container | NB3 `write_hdf5_from_tif` → `.h5` | `io.py::assemble_prairie_stack` → BigTIFF | **Replaced, faster + bit-exact** |
| 2. Motion correction | NB2 SIMA `HiddenMarkov2D` | `foundation.py::run_motion_correction` (backend-selected) | **Built; speed is the open question** |
| 3. SVD / L+S background + summaries | (none — roigbiv-specific) | `foundation.py::compute_background_separation` | **In-repo; not a legacy regression target** |
| (4× temporal averaging) | NB2 cell-11, post-extraction | acquisition-time concern; `PipelineConfig.frame_averaging` tracks it | **Frame-rate note, [§6](#6-frame-rate-and-prism-scale-notes)** |

Notebooks referenced:
- **NB1** `2P_preprocessing_v1.ipynb` — folder housekeeping.
- **NB2** `PFC Data Pipeline_EMD.ipynb` — SIMA motion correction + ROI extraction (Python 2).
- **NB3** `write_hdf5_from_tif3 - Py2_Mike_allinone.ipynb` — single-frame TIFFs → HDF5.

---

## 3. Step 0 — Folder housekeeping & rename

### What legacy does
NB1 (and the duplicated cells 0–3 of NB3) walk each session directory and:
1. create an `extras/` mirror tree;
2. `shutil.move` the `References/` folder, `*.env`, and `*.xml` sidecars into `extras/`;
3. `os.rename` every `*.ome.tif` → `*.tif` (strip the `.ome` infix).

No pixel data is touched — it is pure filesystem bookkeeping so a downstream glob like `*.tif` finds only frames.

### Cost
A full directory-move pass over every session (cheap in bytes, but it mutates the raw acquisition tree in place — destructive to provenance, and a failure mid-move leaves a half-renamed directory).

### Optimization proposition — **drop this step entirely** *(recommended)*
The current loader makes it unnecessary. `io.py::discover_tifs` (`roigbiv/io.py:150`) detects PrairieView sessions by matching the frame pattern directly:

```python
_PRAIRIE_FRAME_PATTERN = re.compile(r'_Cycle\d+_Ch\d+_(\d+)\.ome\.tif$')   # io.py:31
```

`_detect_prairie_sessions` (`io.py:34`) samples up to 5 files per immediate subdir and flags a session when ≥2 match. The `.xml`/`.env` sidecars and `References/` never match the frame regex, so they are ignored without being moved — the raw tree stays pristine. The `.ome` infix is part of the pattern, so **no rename is needed**.

- **Speedup:** eliminates an entire filesystem-move pass and its failure modes.
- **Quality gate:** none — no pixel I/O, and *not* mutating the raw tree is strictly safer than legacy.

---

## 4. Step 1 — Stacking single frames into one container

### What legacy does
NB3 `write_into_hdf5` builds one HDF5 file (`/imaging`, shape `(T, Y, X)`, uint16, chunks `(1, Y, X)`, no compression) from the renamed `*.tif` files, sorted by the trailing integer in the filename. Concretely it makes **multiple full passes over the data**:
1. open the first file to learn frame shape/dtype;
2. a counting pass — `calculate_num_frames` opens every file and `PIL.seek`s to EOF to total the frames;
3. a write pass — a pure-Python nested loop `for file: for frame:` doing `PIL.seek` + `np.array` + one HDF5 write per frame;
4. a **full read-back verification pass** — every frame is read back from HDF5 and compared pixel-for-pixel to the source TIFF (`np.all(... == ...)`), doubling total I/O.

dtype flows straight through from PIL (uint16, no cast); no averaging happens here.

### Cost
≈ 4–5× the dataset in disk I/O (count + write + verify, plus per-file re-counts), all single-threaded in Python, one frame in RAM at a time. For `beh-006` (54 GB) that is hundreds of GB of I/O before motion correction even starts. NB3's `main()` also has a latent bug — it only processes the *last* directory `os.walk` yields, not all of them.

### Optimization proposition — **adopt `assemble_prairie_stack` (BigTIFF, single-pass)** *(recommended; already the default)*
`io.py::assemble_prairie_stack` (`roigbiv/io.py:51-143`) is what `discover_tifs` already calls for PrairieView sessions. It:

- **auto-detects the channel** by frequency of the `_ChN_` token when `channel=None`;
- **sorts by the 6-digit frame index** via the same regex group (`io.py:85-86`) — robust to lexical sort pitfalls;
- **skips zero-byte placeholder files** (Bruker writes empties mid-acquisition);
- **streams frames in one pass** into `tifffile.TiffWriter(..., bigtiff=True, is_ome=False)` with `contiguous=True` — no >4 GB ceiling (the failure that broke NB2's TIFF export), no OME cross-series re-read, one frame in RAM at a time;
- writes to a `*.tmp.tif` sidecar and **atomically renames** on success → `{root}/_stacks/{session}.tif`, so a crash never leaves a half-written stack that looks complete.

`validate_tif` (`io.py:278`) then enforces a genuine 3-D multi-page stack (≥2 pages) before anything downstream runs, and the `_stacks/` path is excluded from re-discovery so outputs are never re-ingested as inputs.

- **Speedup vs legacy:** one streaming pass instead of count + write + verify (≈4–5×), plus no HDF5-vs-TIFF format detour. Output is a BigTIFF the rest of the pipeline reads directly.
- **Quality gate — bit-exactness.** The frame copy is lossless (uint16 in, uint16 out, no cast), so the stack is bit-identical to the source frames. Legacy proved this with a full read-back; the cheap standing check is **frame-count equality** (`len(stack) == number of source frames`) plus a **sampled** `np.array_equal` of, say, 20 random frames against their source `.ome.tif`. That catches ordering/truncation regressions at a tiny fraction of the legacy verify cost.

### Optional further speedup (document, don't adopt blindly)
The write is contiguous and serial, but the **reads are I/O-bound**. A bounded `ThreadPoolExecutor` that prefetches the next N frames while the writer appends would overlap read latency with write, with no change to output bytes (writer stays single-threaded and in-order). Treat this as a *measure-if-stacking-dominates* follow-up — only worth it if profiling shows stacking (not motion correction) is the bottleneck, which on this dataset it is not.

---

## 5. Step 2 — Motion correction — the open decision

This is the one foundational step where the speed/quality trade-off is unresolved, so it is presented as three directions with a shared quality gate rather than a single recommendation.

### What legacy does
NB2 runs `sima.motion.HiddenMarkov2D(granularity='row', max_displacement=[50, 50])` on the HDF5 — an HMM that estimates a displacement **per image row** (in-plane non-rigid at row granularity), capped at ±50 px. It is slow enough that the notebook's own attempt to export the corrected stack to TIFF16 *failed* (SIMA's writer overflows the 32-bit TIFF offset past ~4 GB). SIMA builds its reference internally.

### What roigbiv does today
`foundation.py::run_motion_correction` (`roigbiv/pipeline/foundation.py:40-145`) dispatches on `cfg.motion_correction_backend`:

- **`phasecorr` (default)** — hands the stack to **Suite2p** for rigid + non-rigid registration (`cfg.nonrigid=True`), then exports `{stem}_mc.tif` best-effort. This is the proven path. `PipelineConfig.motion_correction_backend` defaults to `"phasecorr"` (`types.py`), and `test_registration.py::test_default_backend_is_phasecorr` regression-guards that default so the fast backend can't silently take over.
- **`rowwise-pcc` (opt-in)** — `roigbiv/pipeline/registration.py`, a pure-PyTorch GPU backend: build an iterative high-correlation template, rigid pre-align each frame by FFT phase correlation (`torch.fft.rfft2`) with 3-point parabolic subpixel refinement, then a row-wise non-rigid pass (split each frame into `strip_height`=8 px strips, phase-correlate each strip, smooth into a per-row displacement field, resample with `grid_sample`). VRAM-budgeted to ~4 GB/batch; falls back to CPU.

`do_registration` defaults to `False` because the lab's `*_mc.tif` inputs arrive pre-corrected; for the *raw* `logan_cousa_trial` frames it must be `True` so registration actually runs.

### Measured speed and quality
From the in-repo bench harness (`scripts/bench_motion_correction.py`), metrics computed on each backend's **temporal-mean image**, all **z-normalized** (scale-invariant). `lap_var`/`lap_var_smooth` = cell-edge sharpness; `grad_anisotropy_xy` ≈ 1.0 is healthy, <1.0 signals horizontal jitter/warp blur; `banding_score` higher = more horizontal banding.

**`mc_bench_beh006` — 1,500 frames:**

| Backend | lap_var_smooth | lap_var | banding ↓ | aniso_xy →1 | seconds |
| --- | --- | --- | --- | --- | --- |
| raw (no correction) | 0.004027 | 0.335 | 9.96e-5 | 0.793 | 3.6 |
| **rowwise-pcc** | 0.003408 | 0.155 | 8.80e-5 | **0.673** | **12.9** |
| **phasecorr** | 0.003967 | 0.105 | **5.93e-5** | 0.721 | 174.0 |
| legacy_ref (PNG)\* | 0.002953 | 0.047 | 1.30e-4 | 0.752 | — |

**`mc_bench_beh006_hi` — 1,500 frames at the high-motion window (start 13,500, picked by `scan_motion.py`):**

| Backend | lap_var_smooth | lap_var | banding ↓ | aniso_xy →1 | seconds |
| --- | --- | --- | --- | --- | --- |
| raw | 0.004549 | 0.373 | 1.16e-4 | 0.791 | 3.1 |
| **rowwise-pcc** | 0.003882 | 0.178 | 9.75e-5 | **0.672** | **12.4** |
| **phasecorr** | 0.004489 | 0.118 | **6.99e-5** | 0.713 | 177.2 |
| legacy_ref (PNG)\* | 0.002953 | 0.047 | 1.30e-4 | 0.752 | — |

**`mc_bench_pre005` — 1,200 frames** (lap_var_smooth not recorded in this earlier run): rowwise-pcc 10.7 s, phasecorr **128.6 s**; rowwise aniso 0.671 vs phasecorr 0.704; banding 8.49e-5 vs 8.68e-5.

End-to-end, on the full **2,271-frame `pre-005`** FOV with `phasecorr`, the foundation stage (motion correction + SVD background) took **370 s** (`experiments/runs/mc_validate_pre005/pipeline_log.json`, `timings_s.foundation_s`), vs 3.6 s Stage 1 and 24.4 s Stage 2 — i.e. **motion correction dominates the entire pipeline wall-clock.** Extrapolated to the 27k-frame `beh-006`, `phasecorr` is on the order of tens of minutes.

> \* **Reading `legacy_ref`.** It is the legacy SIMA output rendered to an **8-bit percentile-stretched PNG**, not a raw-scale measurement. The metrics are z-normalized so they are *directionally* comparable, but 8-bit quantization deflates its absolute Laplacian — do **not** treat its `lap_var` as a hard numeric floor. The apples-to-apples quality anchor is **`phasecorr`** (identical 16-bit harness, proven backend). What `legacy_ref` *is* good for: confirming `phasecorr`'s **banding is already lower** than legacy (5.9–7.0e-5 vs 1.30e-4) — i.e. the proven backend is at least as clean as SIMA on the hallmark artifact.

### What the numbers say
- **`phasecorr` meets the bar but is the bottleneck.** It holds cell-edge sharpness essentially equal to raw on `lap_var_smooth` (0.00397 vs 0.00403; 0.00449 vs 0.00455) and has the **lowest banding of any backend, legacy included**. Cost: 128–177 s per ~1.5k frames; 370 s foundation on 2.3k frames.
- **`rowwise-pcc` is ~10–14× faster but regresses quality.** Across every window it has the **lowest `lap_var_smooth`** and the **lowest `grad_anisotropy_xy`** (0.67 vs phasecorr's 0.70–0.72 and raw's 0.79). Since anisotropy and banding are scale-invariant and computed identically, this is a real signal — its per-row warps smear horizontal edges on dim prism data. **As-is it violates the no-regression constraint** and cannot be the recommendation.

### The three directions

| # | Direction | Mechanism | Expected speed | Quality risk | What it takes to adopt |
| --- | --- | --- | --- | --- | --- |
| **A** | **Keep `phasecorr`, tune Suite2p for speed** | Sweep registration params — `batch_size`, non-rigid block size, two-step registration, `maxregshift`/`smooth_sigma`; confirm whether the ceiling is CPU-bound | Modest — proven backend, 128–177 s baseline | **Lowest** — already meets/beats legacy | A param sweep through the bench harness; no new code path |
| **B** | **Fix `rowwise-pcc` to the legacy bar** | Pursue the experimental `rowwise-pcc-fixed` knobs already stubbed in the bench (`prefilter=True`, `strip_height=16`); suppress the noise-driven per-row warps that cost anisotropy/banding | **Highest (~10×)** *if* it lands | Currently fails the gate; unproven | Bench `rowwise-pcc-fixed` until its `grad_anisotropy_xy` and `banding_score` reach `phasecorr` parity and `lap_var_smooth` stops regressing |
| **C** | **Hybrid: GPU-rigid everywhere + selective non-rigid** | Use the fast GPU rigid phase-correlation stage on all frames (rigid alignment doesn't band), and apply the non-rigid pass **only** on high-motion windows flagged by `scan_motion.py` | High — rigid is cheap on GPU; non-rigid runs on a fraction of frames | Medium — hinges on prism motion being mostly rigid drift; needs validation on quiescent vs high-motion windows | Confirm rigid-only meets the gate on low-motion windows; define a motion-magnitude trigger for the non-rigid pass |

All three are gated **identically** (next section), so they are directly comparable once benched. Direction **A** is the safe floor; **B** is the high-payoff gamble; **C** is the principled middle path that exploits the existing `scan_motion.py` scout. None is adopted here — the bench harness exists to let the choice be made on data.

---

## 6. Frame rate and prism-scale notes

- **Frame rate.** `PipelineConfig.fs` is the *acquisition* rate; pass `--fs 7.5` for these stacks if they are 4×-averaged, or the true per-frame rate read from the trial's `*.xml`. Wrong `fs` miscalibrates Stage 3 GCaMP templates, Stage 4 bandpass, and deconvolution τ. The legacy 4× temporal averaging (NB2 cell-11, `np.nanmean` over groups of 4) happened *post-extraction* on traces, not on the movie — `PipelineConfig.frame_averaging` records the factor for provenance; it is not a foundation step to re-implement.
- **Prism scale.** These are 1024² prism FOVs; cell somata are ~56 px across vs ~12 px on 512² GRIN data (measured by `scripts/measure_prism_scale.py`: median diameter ≈56 px, median area ≈2480 px², p95 ≈3350 px²). Use `configs/pipeline.prism.yaml` (Cellpose diameter 56, Gate-1 area 1500–5000 px², `--template-threshold 9.0` to keep Stage 3 from saturating on prism background). This is downstream of foundation but governs whether the corrected stack yields sensible detections.

## 7. Step 3 — SVD / L+S background + summary images (for completeness)

Not a legacy step and not a target of this optimization pass (no legacy baseline to regress against), but it shares the foundation stage's wall-clock so it is documented. `foundation.py::compute_background_separation` (`foundation.py:265`) memmaps the registered `data.bin` (int16, zero-copy), bins to ≤`svd_bin_frames` (5000), runs `torch.svd_lowrank(q=n_svd=200)` on GPU (seeded for determinism, CPU fallback on OOM), and forms the background `L` from `k_background`=30 components in closed form. (An opt-in `background_method="rpca"` swaps this truncated SVD for a robust low-rank+sparse decomposition — `roigbiv/pipeline/rpca.py` — that keeps bright sparse sources out of `L`; it factors the robust `L` back into the same `U/S/V` form so everything below is unchanged. RPCA holds ~5 live copies of the binned matrix, so its `rpca_bin_frames` target is auto-sized against *free* GPU memory — `rpca.estimate_rpca_bin_frames` — and the foundation retries coarser, then on CPU, before giving up; this keeps large FOVs, e.g. 1024×1024, on the GPU. The svd/rpca choice and the RPCA knobs are exposed in the Dash UI as well as the CLI.) The residual `S = M − L` is **never materialized to disk** — a lazy `ResidualView` reconstructs it chunk-by-chunk on demand, and summary accumulation is capped at 128-frame chunks (`foundation.py:698-702`) to bound peak RAM (~500 MB). These are existing, deliberate perf choices; leave them as-is unless profiling says otherwise.

---

## 8. Validation recipe

Any proposed change must be shown to hold the bar *before* adoption. The tooling already exists.

1. **Stacking (Step 1).** After `assemble_prairie_stack`, assert `frame_count(stack) == #source .ome.tif frames`, then sample ~20 random frames and `np.array_equal` each against its source `.ome.tif`. Bit-exact ⇒ no possible quality regression.
2. **Motion correction (Step 2).**
   - `python scripts/scan_motion.py <stack>` → pick the top high-motion window (most discriminative for non-rigid backends).
   - `python scripts/bench_motion_correction.py` over `{raw, candidate, phasecorr, legacy_ref}` on that window (and one quiescent window).
   - **Gate:** the candidate must reach **`phasecorr` parity** on `grad_anisotropy_xy` and `banding_score` (the scale-invariant discriminators) and must **not regress `lap_var_smooth`** below `phasecorr`. Confirm visually against the `mean_*.png` / `montage.png` the bench writes. `legacy_ref` is the directional sanity anchor, not a numeric floor (8-bit PNG, see §5).
3. **End-to-end.** `roigbiv-pipeline --input data/logan_cousa_trial/<trial>/ --fs 7.5` (add `--config configs/pipeline.prism.yaml` knobs for prism scale). Check `summary/` images look right and compare `pipeline_log.json` `timings_s.foundation_s` against the 370 s `pre-005` baseline to quantify the speedup.

---

## 9. Summary

| Step | Recommendation | Expected speedup | Quality gate |
| --- | --- | --- | --- |
| 0. Housekeeping/rename | **Drop** — loader ignores sidecars; don't mutate the raw tree | Removes a full move pass | None (no pixel I/O) |
| 1. Stacking | **Adopt `assemble_prairie_stack`** (already default) — single-pass BigTIFF | ≈4–5× less I/O vs HDF5 count+write+verify | Bit-exact (frame-count + sampled `array_equal`) |
| 2. Motion correction | **Open** — bench Directions A/B/C; do **not** ship `rowwise-pcc` as-is | A: modest · B/C: up to ~10× | Parity with `phasecorr` on `aniso_xy` + `banding`; no `lap_var_smooth` regression |
| 3. SVD/summaries | Keep as-is (no legacy baseline) | n/a | n/a |

**Bottom line.** Stacking and housekeeping are already faster and safer in-repo than legacy and meet the no-regression bar trivially (bit-exact / no-op). The remaining speed win — and the only real risk to quality — lives entirely in **motion correction**, where `phasecorr` is the trustworthy-but-slow anchor and the path to a 10× speedup runs through proving one of the three directions to `phasecorr` parity on the bench harness that already exists.

---

### Appendix — legacy parameter reference (provenance)

| Parameter | Legacy value | Source |
| --- | --- | --- |
| MC algorithm | `sima.motion.HiddenMarkov2D` | NB2 |
| MC granularity | `'row'` (non-rigid, per-row) | NB2 |
| MC max displacement | `[50, 50]` px (y, x) | NB2 |
| MC reference | SIMA-internal | NB2 |
| Stack container | HDF5 `/imaging`, `(T,Y,X)`, uint16 | NB3 |
| HDF5 chunks / compression | `(1, Y, X)` / none | NB3 |
| Frame sort key | trailing integer (`\d+$`) | NB3 |
| dtype handling | pass-through uint16, no cast | NB3 |
| Temporal averaging | 4× post-extraction (`np.nanmean`) | NB2 cell-11 |
