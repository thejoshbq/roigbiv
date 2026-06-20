#!/usr/bin/env python3
"""PRISM Stage-1 recall matrix — concurrent OFAT from a pinned baseline.

Faithful, drift-guarded one-factor-at-a-time perturbation of the verified
pre-005 prism-profile baseline (11 soma-scale accepts). See the engagement
directive for context.

Design notes
------------
* **Canonical baseline.** The baseline config is *read from the pre-005
  manifest's* ``cfg_snapshot`` and instantiated directly via
  ``PipelineConfig(**d)`` — never hand-assembled from profile + CLI flags
  (that assembly path is what produced Run A's silent ``tile_norm_blocksize``
  reversion). Every run = this pinned dict + exactly ONE documented delta +
  the scout operational overlay.
* **Drift guard.** After each variant's config is built, its resolved
  ``summary_for_log()`` snapshot is asserted equal to ``pin + delta + overlay``
  (ignoring per-run ``output_dir``). Any other differing field ⇒ FAILED-DRIFT,
  the run is excluded from interpretation.
* **One foundation, many gates.** Under ``channels=(0,0)`` the Stage-1 detector
  input is ``mean_M`` (SVD-independent), identical across every variant. So
  Foundation/motion-correction runs ONCE; each variant then calls the *exact
  two functions* run_pipeline calls — ``run_cellpose_detection`` (run.py:512)
  and ``evaluate_gate1`` (run.py:531) — in-process. This is bit-faithful to the
  full-pipeline Stage-1 path, removes per-run motion-correction noise as a
  confound, and keeps every candidate mask in memory for the metrics. (It also
  makes the directive's 2-worker VRAM concurrency moot: there is only ever one
  Cellpose context live, and the per-variant cellpose passes are seconds each.)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.types import PipelineConfig
from roigbiv.pipeline.foundation import run_foundation
from roigbiv.pipeline.stage1 import run_cellpose_detection
from roigbiv.pipeline.gate1 import evaluate_gate1

REPO = Path(__file__).resolve().parents[2]
PIN_MANIFEST = REPO / "output/prism_profile/052126_DS-Prism-3_VI15_D2_FOV2_pre-005/.roigbiv_manifest.json"
INPUT = REPO / "data/logan_cousa_trial/_stacks/052126_DS-Prism-3_VI15_D2_FOV2_pre-005.tif"
STEM = "052126_DS-Prism-3_VI15_D2_FOV2_pre-005"
OUT_ROOT = REPO / "output/stage1_matrix"

# Scout operational overlay applied to every matrix run (the pinned baseline was
# itself a full run; scout reproduces its Stage-1 result under channels=(0,0)).
OVERLAY = {"scout_mode": True, "no_viewer": True}

# Each run = BASELINE + exactly this delta. R0 = control (no delta).
DELTAS = {
    "R0": {},
    "R_gate": {"min_area": 900, "max_eccentricity": 0.97, "max_area": 9000},
    "R_cellprob_n1": {"cellprob_threshold": -1.0},
    "R_cellprob_n2": {"cellprob_threshold": -2.0},
    "R_flow": {"flow_threshold": 0.6},
    "R_denoise": {"use_denoise": True},
    # Phase 3 composition — R_gate bounds alone (only Phase-2-validated clean lever).
    "composition": {"min_area": 900, "max_eccentricity": 0.97, "max_area": 9000},
}
# Run-specific metric focus (for the report).
DETECTION_RUNS = {"R_cellprob_n1", "R_cellprob_n2", "R_flow", "R_denoise"}

IGNORE_DRIFT = {"output_dir"}          # operational; varies per run by design
TUPLE_FIELDS = {"channels"}
VALID_FIELDS = {f.name for f in fields(PipelineConfig)}
SOMA_RADIUS_PX = 28                    # ~one soma radius for peak separation
BIG_MASK_PX = 5000                     # masks above this get a peak-count check


# ── config assembly + drift ────────────────────────────────────────────────
def load_pin() -> dict:
    return json.loads(PIN_MANIFEST.read_text())["cfg_snapshot"]


def expected_snapshot(pin: dict, run_id: str, out_dir: Path) -> dict:
    exp = dict(pin)
    exp.update(OVERLAY)
    exp.update(DELTAS[run_id])
    exp["output_dir"] = str(out_dir)
    return exp


def build_cfg(pin: dict, run_id: str, out_dir: Path) -> PipelineConfig:
    exp = expected_snapshot(pin, run_id, out_dir)
    d = {k: v for k, v in exp.items() if k in VALID_FIELDS}
    for tf in TUPLE_FIELDS:
        if tf in d and isinstance(d[tf], list):
            d[tf] = tuple(d[tf])
    return PipelineConfig(**d)


def _norm(v):
    return list(v) if isinstance(v, (list, tuple)) else v


def drift_offenders(resolved: dict, pin: dict, run_id: str, out_dir: Path) -> dict:
    """Fields where the resolved snapshot deviates from pin+delta+overlay."""
    exp = expected_snapshot(pin, run_id, out_dir)
    out = {}
    for k in set(resolved) | set(exp):
        if k in IGNORE_DRIFT:
            continue
        rv, ev = _norm(resolved.get(k, "<absent>")), _norm(exp.get(k, "<absent>"))
        if rv != ev:
            out[k] = {"expected": ev, "resolved": rv}
    return out


# ── metrics ────────────────────────────────────────────────────────────────
def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def _matches_any(mask: np.ndarray, refs: list[np.ndarray], iou_thresh: float = 0.3) -> bool:
    """True if `mask` overlaps any reference candidate (IoU >= thresh)."""
    a = mask
    for r in refs:
        inter = np.logical_and(a, r).sum()
        if inter == 0:
            continue
        union = np.logical_or(a, r).sum()
        if union and inter / union >= iou_thresh:
            return True
    return False


def _peak_count(mask: np.ndarray, mean_M: np.ndarray) -> int:
    """Local-maxima count inside a mask (single soma vs residual merge)."""
    from skimage.feature import peak_local_max

    img = np.where(mask, mean_M, 0.0)
    peaks = peak_local_max(
        img, min_distance=SOMA_RADIUS_PX, labels=mask.astype(int),
        exclude_border=False,
    )
    return int(len(peaks))


def _intensity_over_bg(mask: np.ndarray, mean_M: np.ndarray, bg_median: float,
                       bg_p95: float) -> dict:
    mv = float(mean_M[mask].mean())
    denom = (bg_p95 - bg_median) or 1.0
    return {
        "mask_mean": round(mv, 2),
        "over_bg": round(mv - bg_median, 2),
        "norm_contrast": round((mv - bg_median) / denom, 3),
        "pct_rank": round(float((mean_M < mv).mean()) * 100, 1),
    }


# ── per-variant execution ──────────────────────────────────────────────────
def run_variant(run_id: str, fov, pin: dict) -> dict:
    out_dir = OUT_ROOT / run_id / STEM
    (out_dir / "stage1").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(pin, run_id, out_dir)
    resolved = cfg.summary_for_log()
    offenders = drift_offenders(resolved, pin, run_id, out_dir)
    drift_pass = not offenders

    # Exact run_pipeline Stage-1 calls (run.py:512 + run.py:531).
    candidates, probs, label_image, cellprob_map = run_cellpose_detection(
        fov.mean_M, fov.vcorr_S, cfg, max_S=fov.max_S,
    )
    rois = evaluate_gate1(
        candidates, probs, fov.mean_M, fov.vcorr_S, fov.dog_map, cfg,
        starting_label_id=1,
    )

    n_det = len(candidates)
    n_acc = sum(1 for r in rois if r.gate_outcome == "accept")
    n_flag = sum(1 for r in rois if r.gate_outcome == "flag")
    n_rej = sum(1 for r in rois if r.gate_outcome == "reject")

    # Persist outputs (mirror run.py:521-568) + manifest snapshot + drift.
    mask_img = np.zeros(fov.mean_M.shape, dtype=np.uint16)
    for r in rois:
        if r.gate_outcome in ("accept", "flag"):
            mask_img[r.mask] = r.label_id
    tifffile.imwrite(str(out_dir / "stage1" / "stage1_masks.tif"), mask_img)
    tifffile.imwrite(str(out_dir / "stage1" / "stage1_probs.tif"),
                     cellprob_map.astype(np.float32))
    for name, arr in [("mean_M", fov.mean_M), ("vcorr_S", fov.vcorr_S),
                      ("dog_map", fov.dog_map)]:
        tifffile.imwrite(str(out_dir / "summary" / f"{name}.tif"),
                         arr.astype(np.float32))

    (out_dir / "stage1" / "stage1_report.json").write_text(json.dumps({
        "detected": n_det, "accepted": n_acc, "flagged": n_flag, "rejected": n_rej,
        "rois": [r.to_serializable() for r in rois],
    }, indent=2))
    (out_dir / ".roigbiv_manifest.json").write_text(json.dumps({
        "input_tif": str(INPUT), "cfg_snapshot": resolved,
    }, indent=2))
    (out_dir / "drift.json").write_text(json.dumps({
        "run_id": run_id, "delta": DELTAS[run_id], "drift_pass": drift_pass,
        "offending_fields": offenders,
    }, indent=2))

    return {
        "run_id": run_id, "delta": DELTAS[run_id], "drift_pass": drift_pass,
        "offending_fields": offenders, "out_dir": str(out_dir),
        "detected": n_det, "accepted": n_acc, "flagged": n_flag, "rejected": n_rej,
        "candidates": candidates, "rois": rois,
    }


def analyse(results: dict, fov) -> dict:
    """Run-specific metrics, computed against R0 as the detection reference."""
    mean_M = fov.mean_M
    bg_median = float(np.percentile(mean_M, 50))
    bg_p95 = float(np.percentile(mean_M, 95))
    r0 = results["R0"]
    r0_cands = r0["candidates"]

    analysis = {"bg_median": round(bg_median, 2), "bg_p95": round(bg_p95, 2),
                "runs": {}}

    for run_id, res in results.items():
        entry = {"detected": res["detected"], "accepted": res["accepted"],
                 "flagged": res["flagged"], "rejected": res["rejected"],
                 "drift_pass": res["drift_pass"]}
        rois = res["rois"]
        cands = res["candidates"]

        # Reject-reason histogram.
        hist = {}
        for r in rois:
            if r.gate_outcome == "reject":
                for gr in r.gate_reasons:
                    key = gr.split(":")[0]
                    hist[key] = hist.get(key, 0) + 1
        entry["reject_reasons"] = hist

        if run_id in ("R_gate", "composition"):
            # Detection is identical to R0 (gates-only delta); compare
            # per-candidate gate flips reject->accept/flag = recovered ROIs.
            recovered = []
            for i, r in enumerate(rois):
                r0_outcome = r0["rois"][i].gate_outcome
                if r0_outcome == "reject" and r.gate_outcome in ("accept", "flag"):
                    info = {"label": r.label_id, "area": int(r.area),
                            "ecc": round(float(r.eccentricity), 3),
                            "outcome": r.gate_outcome,
                            "was": r0["rois"][i].gate_reasons}
                    if r.area > BIG_MASK_PX:
                        info["peak_count"] = _peak_count(cands[i], mean_M)
                    recovered.append(info)
            entry["recovered"] = recovered

        elif run_id in DETECTION_RUNS:
            # New detections = candidates not overlapping any R0 candidate.
            new = []
            for i, c in enumerate(cands):
                if not _matches_any(c, r0_cands):
                    iob = _intensity_over_bg(c, mean_M, bg_median, bg_p95)
                    iob.update({"label": rois[i].label_id, "area": int(rois[i].area),
                                "outcome": rois[i].gate_outcome})
                    new.append(iob)
            entry["delta_detected_vs_r0"] = res["detected"] - r0["detected"]
            entry["new_detections"] = new

        analysis["runs"][run_id] = entry
    return analysis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(DELTAS),
                    help="subset of run_ids (default: all)")
    args = ap.parse_args()

    pin = load_pin()
    print(f"[matrix] pinned baseline from {PIN_MANIFEST.name}: "
          f"max_ecc={pin['max_eccentricity']} min_sol={pin['min_solidity']} "
          f"tile_norm={pin['tile_norm_blocksize']} scout(pin)={pin['scout_mode']}")

    # Foundation ONCE (scout, baseline cfg) — shared mean_M for all variants.
    found_dir = OUT_ROOT / "_foundation" / STEM
    found_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = build_cfg(pin, "R0", found_dir)
    print(f"[matrix] running Foundation once (scout) -> {found_dir}")
    fov = run_foundation(INPUT, base_cfg, found_dir, gpu_lock=None)
    print(f"[matrix] foundation done: mean_M {fov.mean_M.shape}, "
          f"max_S={'present' if fov.max_S is not None else 'None (scout)'}")

    results = {}
    for run_id in DELTAS:                       # always run R0 (reference)
        if run_id not in args.runs and run_id != "R0":
            continue
        print(f"\n[matrix] === {run_id}  Δ={DELTAS[run_id]} ===")
        res = run_variant(run_id, fov, pin)
        flag = "OK" if res["drift_pass"] else f"FAILED-DRIFT {res['offending_fields']}"
        print(f"[matrix] {run_id}: detected={res['detected']} accepted={res['accepted']} "
              f"flagged={res['flagged']} rejected={res['rejected']}  drift={flag}")
        results[run_id] = res

    analysis = analyse(results, fov)
    (OUT_ROOT / "matrix_results.json").write_text(json.dumps(analysis, indent=2))
    print(f"\n[matrix] wrote {OUT_ROOT/'matrix_results.json'}")

    # Control gate.
    r0 = results["R0"]
    print("\n" + "=" * 60)
    if not r0["drift_pass"]:
        print(f"CONTROL FAIL: R0 drift {r0['offending_fields']} — baseline pin broken.")
    elif not (9 <= r0["accepted"] <= 13):
        print(f"CONTROL FAIL: R0 accepted={r0['accepted']} (expected ~11). "
              "Do NOT interpret variants.")
    else:
        print(f"CONTROL OK: R0 accepted={r0['accepted']} (~11), drift clean.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
