# Phase 0 — Discovery

Branch: `feat/robust-subtraction-and-baselines`

---

## Hardware

| Resource | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 5080 |
| VRAM | 15,840 MB (16,303 MiB reported by nvidia-smi) |
| Driver | 580.159.03 |
| System RAM | 123 GiB total, ~106 GiB available |
| CPU cores | 32 |
| PyTorch CUDA version | 13.0 (`torch 2.12.0+cu130`) |
| CUDA compute capability | sm_120 |

**RESOLVED (2026-06-16):** The GPU is fully usable. `roigbiv` now ships
`torch 2.12.0+cu130`; `torch.cuda.get_arch_list()` includes **`sm_120`**,
`cuda_compute_capable()` returns **`True`**, and real ops run on `cuda:0`
(`svd_lowrank` 4000×4000 q=30 in ~0.06 s). The earlier "sm_50–sm_90 build, install
nightly" conclusion was a **stale** snapshot of an older cu126 wheel.

The real cause of the earlier CPU-only runs was **VRAM contention, not the build**:
the local-Qwen MCP server keeps a large model (e.g. `qwen3-coder:30b` ≈ 18 GB) resident
on the 16 GB card for minutes after each call, leaving the pipeline ~0.7 GB free → it
OOM'd and fell back to CPU. `cuda_compute_capable()` caught that transient OOM and
`stage1.py` misreported it as an sm/CC mismatch, which made the GPU look permanently
broken. Mitigations now in place: a per-run VRAM preflight (`pipeline/gpuguard.py`,
default on; `--no-free-gpu` to disable) that unloads ollama's model before GPU stages,
an OOM-vs-sm_mismatch-aware probe (`pipeline/device.py::cuda_unavailable_reason`), and
`OLLAMA_KEEP_ALIVE=0` on the ollama daemon so models don't squat. Chunk size
(`subtract_chunk_frames=2000`) is sized for the full 123 GiB RAM; no tiling required.

---

## Solver Insertion Point

**Primary solver:** `roigbiv/pipeline/subtraction.py:129-177`
Function: `solve_traces_from_chunks(design, T, chunk_iter, cfg) → (N, T) float32`

- `design`: (P, N) float32 — design matrix over union pixels
- `T`: int — total frames
- `chunk_iter`: iterable yielding `(t0, t1, S_chunk)` where S_chunk is (cs, P) float32
- `cfg`: `PipelineConfig` — uses `subtract_ridge_lambda_scale`, `force_cpu`
- Returns: (N, T) float32 traces; no non-negativity constraint enforced at this level

**Dispatch site:** `estimate_traces_simultaneous()`, line 217:
```python
traces = solve_traces_from_chunks(design, T, _iter(), cfg)
```
This is the single line to replace with a conditional dispatch.

**Three call sites in `run.py`** (lines 384, 501, 643) all pass `cfg` through
`run_source_subtraction()` → `estimate_traces_simultaneous()`. No other changes needed
there.

**NNLS fallback** (`_nnls_refine_flagged()`, lines 516–560): separate from the primary
solve; triggered post-validation for anticorrelation-flagged ROIs. Unchanged by this work.

---

## Config Threading Path

- **Dataclass**: `roigbiv/pipeline/types.py:218-223` — subtraction section of
  `PipelineConfig`. New fields go here.
- **No YAML loader at runtime**: `configs/pipeline.yaml` is documentation-only.
  All config flows through CLI → argparse → `PipelineConfig(**kwargs)` in
  `run.py:_run_single()` (line ~1114).
- **CLI args**: add to the argparse group in `run.py:main()` near existing
  `--subtract-*` flags; thread to `PipelineConfig` in `_run_single()`.

New fields to add after line 223:
```python
subtract_solver: str = "ridge"           # "ridge" | "robust"
subtract_robust_kappa: float = 0.5       # one-sided Huber threshold (sigma units)
subtract_robust_max_iter: int = 5        # IRLS iteration cap
```

---

## Held-Out FOV Set

Val split: seed=42, val_frac=0.15 from `scripts/train.py:load_dataset()`.
Total annotated FOVs: 92. Val count: 13 (last 13 of shuffled stems).

All 13 have `_mc.tif` movies locally under `data/JOSH/ROIGBIV-DATA/`.
Canonical paths listed in `experiments/harness/heldout_fovs.txt`.

GT masks: `data/JOSH/ROIGBIV-DATA/cellpose_ready/masks/{stem}_mc_masks.tif`
Format: (H, W) uint16 label image; 0 = background, 1..N = ROI labels.

---

## Metrics Module Status

| Component | Status | Location |
|-----------|--------|----------|
| IoU matching | EXISTS | `scripts/diagnostic_compare.py:41-89` — `iou_match()` |
| Stratified recall/precision/F1 | MISSING | must build `roigbiv/eval/metrics.py` |
| Section 5.2 diagnostics wrapper | PARTIAL | JSON output exists at `subtraction_report_residual_S{N}.json`; no Python loader |
| Harness entry point | MISSING | must build `roigbiv/eval/harness.py` |

---

## Ground-Truth Format

- File: `data/JOSH/ROIGBIV-DATA/cellpose_ready/masks/{stem}_mc_masks.tif`
- Type: (H, W) uint16 label image, tifffile-readable
- Coverage: spatial ROI masks only — no per-ROI activity-type labels

---

## Activity-Label Derivation Strategy

`ROI.activity_type` is written to `roi_metadata.json` after the pipeline classify step.
GT masks carry no type labels. Strategy:

1. Run the NNLS pipeline on each held-out FOV → produces `roi_metadata.json` with
   `activity_type` per detected ROI.
2. Match GT ROIs to detected ROIs by IoU ≥ 0.3.
3. For matched ROIs (TP): activity_type taken from the pipeline's `roi_metadata.json`.
4. For unmatched GT ROIs (FN): activity_type = `"unknown"`. Per-stratum recall for
   tonic and silent is a **lower bound** (Blindspot 13 — manual GT under-represents
   these types). All tonic/silent recall values must be labelled as such in reports.
5. FP ROIs (predicted, no GT match): contribute to per-stratum precision denominator
   via their pipeline-assigned activity_type.

This means recall stratification is relative to what the NNLS run assigned — not an
absolute ground truth for activity type. This limitation is acknowledged and documented
in the comparison report.

---

## No Blockers

All insertion points located, all val FOV movies present, no missing components prevent
starting Phase 1.
