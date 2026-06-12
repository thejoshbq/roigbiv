# Auto-tuning / self-correction — feasibility & design

**Status:** design / feasibility assessment. No code shipped yet.
**Scope of this doc:** whether and how to let the pipeline tune its own parameters
instead of running everything at fixed defaults. Cross-reference the behavioral
source of truth in [`docs/roi-pipeline-specification.md`](roi-pipeline-specification.md)
for foundation / gate / residual terminology.

---

## 1. Problem & scope

The pipeline runs with fixed parameters end to end — motion correction, the four
detection stages, and every gate read their values from `PipelineConfig`
defaults (`roigbiv/pipeline/types.py:159-392`; ~100 tunable fields). Those values
are good general defaults, but per-FOV the *optimal* values differ: a dim Prism
stack wants different motion-correction smoothing than a crisp GRIN stack; a
denser FOV wants different Cellpose thresholds. The idea is **self-correction** —
let the pipeline either auto-adjust or sweep a set of parameter variations and
converge on the best output.

The central constraint: **auto-tuning is only as trustworthy as its objective
function.** "Best output" has to be a number the machine can compute, and that
number has to actually track quality rather than a gameable proxy. The signals
the pipeline can compute split into two classes with very different
trustworthiness (§2), and that split forces the design decision:

> **Hybrid by subsystem.** Closed-loop auto-select *only* where a reference-free,
> non-circular quality metric exists (motion correction / foundation). Everywhere
> a metric would be circular or gameable (detection / gates), use open-loop:
> sweep, rank, and surface variants to the human reviewer — never auto-pick.

**First target: motion correction.** It is the subsystem where the metric is
genuinely sound, the payoff is high (MC quality propagates into every downstream
stage), and — critically — the measurement machinery already exists and has been
validated against a legacy quality bar (§3).

---

## 2. Efficacy assessment

Two classes of objective signal the pipeline already computes:

### 2a. Reference-free image-quality metrics — *trustworthy*

For motion correction, quality is measurable from the registered temporal-mean
image alone, with no labels and no ground-truth ROI set. These metrics live today
in `scripts/bench_motion_correction.py` (`compute_metrics`, line 119):

| Metric | Meaning | Direction |
| --- | --- | --- |
| `lap_var_smooth` | variance of the Laplacian after a light Gaussian blur — cell-edge sharpness with shot/scan noise suppressed | **higher = sharper** (headline) |
| `banding_score` | residual variance of the high-passed per-row mean profile — isolates horizontal per-row jitter (the `rowwise-pcc` failure mode) | **lower = less banding** |
| `grad_anisotropy_xy` | horizontal/vertical gradient-energy ratio | **~1.0 = isotropic** (x-jitter pulls it below 1) |
| `grad_energy`, `tenengrad`, `contrast_rms` | auxiliary sharpness / dynamic-range measures | higher = sharper |

These are **not circular**: a sharper, less-banded temporal mean is unambiguously
a better-registered movie, independent of any detection decision made later. And
they are **already validated** — the bench/sweep harnesses scored candidate
backends against the legacy SIMA-corrected mean as the bar
(`scripts/sweep_suite2p_reg.py`, "Acceptance = `lap_var_smooth` >= that bar"). The
motion traces themselves (`motion_x`, `motion_y` from `run_motion_correction`)
add magnitude/smoothness as secondary signals.

**Conclusion: closed-loop auto-select is sound for motion correction.** The
machine can pick the winner and the choice is defensible.

### 2b. Detection / gate outcomes — *not trustworthy for auto-select*

For the detection stages, the available signals are ROI counts, gate
accept/flag/reject tallies, per-ROI `confidence`, `snr`, and `n_stages_detected`.
There is **no ground-truth ROI set** for a real FOV. That makes every aggregate
objective gameable:

- "Maximize accepted ROIs" is trivially won by loosening `cellprob_threshold`,
  `flow_threshold`, or the gate cutoffs — more detections, not better ones.
- "Maximize mean confidence / SNR" biases toward only the brightest cells and
  silently drops the dim, rare neurons the sequential subtractive design exists
  to find.
- Any of these optimizes a **proxy that drifts from biological truth**, and the
  drift is invisible because there is nothing to check it against. Worse, an
  auto-tuned threshold makes runs non-reproducible and bakes a confirmation bias
  into the data of record.

**Conclusion: detection/gate parameters must not be auto-selected.** They can
still be *swept* — but the output is a ranked comparison for a human to choose
from (§5), not a machine decision.

### Why the split is correct, not a compromise

The hybrid line falls exactly where trustworthy metrics exist. We auto-tune MC
because there is a real, reference-free yardstick; we refuse to auto-tune
detection because there is not. The boundary is drawn by the data, not by
caution.

---

## 3. Mechanism inventory — what we reuse

The orchestration is the only thing missing. Every primitive already exists:

- **Foundation caching + resume.** Foundation (motion correction + truncated SVD
  + L+S background) is cached and resumable (`roigbiv/pipeline/resume.py`).
  `_FINGERPRINT_EXCLUDE` (`resume.py:128`) deliberately drops `enable_stage_2/3/4`
  and `foundation_only` from the config fingerprint, so a downstream variant
  resumes from foundation in 1–2 min instead of re-running the ~20 min
  foundation. Per-stage *parameter* knobs (e.g. `cellprob_threshold`) **are**
  fingerprinted and correctly invalidate — so a detection sweep re-runs only the
  changed stages. This is what makes detection sweeps cheap (§5).
- **Config-agnostic batch pool.** `roigbiv/pipeline/batch.py` takes
  `list[tuple[Path, PipelineConfig]]` and a shared GPU lock. It is already
  agnostic to whether the FOV or the *config* varies across jobs — so it can
  drive a parameter sweep with no change to the pool itself (GPU-capped at 2
  concurrent workers).
- **Validated sweep harnesses.** `scripts/bench_motion_correction.py` and
  `scripts/sweep_suite2p_reg.py` already run a stack subset through multiple
  backends/param-sets, build the registered temporal mean, score it, and emit a
  ranked table + montage. The `experiments/runs/mc_*` directories are their
  output. These are loose-rigor `scripts/`, driven by hand — not a pipeline mode.
- **Subset / mean helpers.** `make_subset` and `temporal_mean_tif`
  (`bench_motion_correction.py:135,151`) already do page-selective frame
  subsetting and chunked temporal-mean reconstruction.

The gap is a single integration: promote the metrics into the package and wrap
the existing select-best logic as a real foundation step.

---

## 4. Recommended MC-first architecture (closed-loop)

### 4.0 Prerequisite refactor — promote the metrics into the package

The metric functions currently live in a loose-rigor script. Move them into a
real module, e.g. **`roigbiv/pipeline/mc_metrics.py`**:
`compute_metrics`, `lap_var_smooth`, `lap_var`, `grad_energy`, `tenengrad`,
`grad_anisotropy_xy`, `banding_score`, `contrast_rms`, plus `temporal_mean_tif`
and `make_subset`. Have `scripts/bench_motion_correction.py` and
`scripts/sweep_suite2p_reg.py` import from the new module so the scripts keep
working and the duplication is removed. This is the one piece of code marked as a
hard prerequisite; everything else in §4 builds on it.

### 4.1 New config + flag

- `PipelineConfig.mc_autotune: bool = False` (`types.py`, in the MC block at
  `types.py:210-230`).
- `--mc-autotune` CLI flag (`roigbiv/pipeline/run.py` parser).
- **Active only when registration runs.** Pre-corrected `*_mc.tif` inputs have
  `do_registration = False`, so there is nothing to tune — auto-tune must be a
  no-op in that case and log that it was skipped. Call this out so it isn't
  silently a no-op on the lab's averaged `_mc` stacks.

### 4.2 Hook point

The dispatch already lives in one function: `run_motion_correction`
(`roigbiv/pipeline/foundation.py:40`) selects the backend on
`cfg.motion_correction_backend`. Auto-tune wraps the *selection* step:

1. Take a representative frame subset of the input (reuse `make_subset`; default
   ~1200 frames, configurable window).
2. For each candidate `(backend, param-overrides)` in the grid, run the backend
   on the subset and build the registered temporal mean (reuse the
   per-backend runners already in `bench_motion_correction.py`:
   `run_rowwise` → `run_rowwise_pcc_register`, `run_phasecorr` → `run_suite2p_fov`).
3. Score each mean with `compute_metrics`; apply the scoring policy (§4.4).
4. Pick the winner; set `cfg.motion_correction_backend` (and the winning
   `mc_*` overrides) accordingly.
5. Run **full** foundation once with the winning config — the subset was only for
   selection.

### 4.3 Candidate grid — start small

Bounded set, expanded later:

- `phasecorr` (default) × `smooth_sigma_time ∈ {default, higher}` — the knob the
  Suite2p sweep found most impactful for dim-frame shift-estimation SNR.
- `rowwise-pcc` × a small cross of `mc_smooth_sigma_rows` and `mc_strip_height`
  (the strip-regularization knobs that closed the synthetic gap).

**Exclude `legacy` (SIMA sidecar) from the default grid** — it is CPU-only and
tens-of-minutes-to-hours per FOV, far too expensive to run as a candidate. Leave
it available as an explicit opt-in only.

### 4.4 Scoring policy — guarded maximization

Maximize the headline sharpness **subject to guards**, so a candidate cannot win
by sharpening noise or trading isotropy for a banded edge:

```
eligible  ⟺  banding_score ≤ B_ceiling  AND  0.8 ≤ grad_anisotropy_xy ≤ 1.2
winner    =  argmax(lap_var_smooth) over eligible candidates
```

`B_ceiling` calibrated from the baseline (default-backend) banding on the same
subset (e.g. baseline × a small factor). If no candidate is eligible, fall back
to the configured default backend and log the fallback — never ship a banded
winner silently. The exact constants are tuning parameters of the auto-tuner
itself and should be exposed, not hard-coded.

### 4.5 Provenance & reproducibility

Record into `pipeline_log.json`: the winning backend + overrides, and the full
candidate scoreboard (every candidate's `lap_var_smooth`, `banding_score`,
`grad_anisotropy_xy`, seconds). This makes an auto-tuned run **reproducible and
auditable**, consistent with the per-ROI provenance the pipeline already keeps
(`source_stage`, `gate_outcome`, per-stage scores). A reader must be able to see
*why* a backend was chosen.

### 4.6 Cost

Subset scoring is a few candidates × ~1200 frames (a couple of minutes each,
serialized on the GPU lock), then **one** full foundation with the winner. The
overhead is small relative to a full ~20 min foundation, and tiny relative to
re-running foundation per parameter by hand.

---

## 5. Detection-stage open-loop design (deferred — documented for the record)

The second half of the hybrid, sketched so the boundary is explicit; **not built
in the first integration.**

- A config-grid sweep orchestrator reusing `batch.py` (the config-agnostic pool)
  + resume + the foundation cache. Foundation runs **once**; each detection
  variant resumes from it (`resume.py` fingerprint excludes stage toggles,
  includes parameter knobs, so only changed stages re-run).
- Output is a **ranked comparison plus per-variant overlays**, surfaced to the
  HITL reviewer in the Dash Review page — *decision support, not auto-select.*
  The human picks the variant; the pipeline never commits a detection-parameter
  choice on its own.
- Cost model: one ~20 min foundation + N × ~2 min downstream variants (Stage 1
  ~1–2 min, subtraction ~30 s, Stage 2 ~1–2 min; Stage 3 ~10 min is the
  expensive one and should be swept sparingly).

This preserves the repo's HITL-additive philosophy: corrections and choices are
human, recorded, and never silently mutate the data of record.

---

## 6. Risks & caveats

- **Subset representativeness.** Scoring on a frame window may not reflect
  whole-movie registration quality (e.g. motion that only appears late).
  Mitigate: a representative window, or score on multiple windows and aggregate.
- **Metric blind spots.** `lap_var` rewards sharpness and could, in principle,
  reward sharpened *noise*. Mitigated by the pre-smoothing in `lap_var_smooth`
  plus the banding/anisotropy guards — but the guards must stay, and their
  thresholds are themselves tunable and should be validated per imaging format.
- **Bounded candidate cost.** Keep the grid small; never let `legacy`/SIMA into
  the default candidate set.
- **`_mc` no-op.** Auto-tune does nothing for pre-corrected input — must be an
  explicit, logged skip, not a silent pass.
- **Determinism.** Auto-decisions must be reproducible; record the scoreboard
  (§4.5) and keep backend runs seeded/deterministic where the backends allow.

---

## 7. Recommended phasing

1. **Prerequisite refactor** — extract `roigbiv/pipeline/mc_metrics.py` from
   `scripts/bench_motion_correction.py`; repoint both scripts at it.
2. **MC closed-loop auto-tune** — `--mc-autotune` / `mc_autotune`, hooked into
   `run_motion_correction`, with the guarded scoring policy and provenance
   logging. First real integration.
3. **Detection open-loop sweep + HITL ranked review** — later; reuses
   batch/resume/foundation-cache; surfaces ranked variants to the reviewer.

---

## Reference map

| Concern | Location |
| --- | --- |
| Config surface (all knobs) | `roigbiv/pipeline/types.py:159-392`; MC block `:210-230` |
| MC backend dispatch (auto-tune hook point) | `roigbiv/pipeline/foundation.py:40` (`run_motion_correction`) |
| Row-wise PCC backend | `roigbiv/pipeline/registration.py` (`run_rowwise_pcc_register`) |
| Resume fingerprint / cache reuse | `roigbiv/pipeline/resume.py:128` (`_FINGERPRINT_EXCLUDE`) |
| Config-agnostic worker pool | `roigbiv/pipeline/batch.py` |
| Metrics to promote | `scripts/bench_motion_correction.py:119` (`compute_metrics`) |
| Existing sweep pattern + true-mean-from-bin | `scripts/sweep_suite2p_reg.py` |
| Behavioral source of truth | `docs/roi-pipeline-specification.md` |
