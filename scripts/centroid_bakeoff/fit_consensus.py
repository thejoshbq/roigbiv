"""Fit + leave-one-FOV-out (LOFO) cross-validate the Cellpose+Suite2p
consensus fusion model.

Offline training CLI, separate from ``run_centroid_bakeoff.py`` — fitting is
a one-time-ish step producing a persisted ``ConsensusModel`` artifact, not
per-run inference. See ``consensus.py``'s module docstring for the fusion
design.

n=5 real FOVs is genuinely small for fitting a logistic model, so this
script LOFO cross-validates rather than fitting on 100% of the data with no
held-out check (an explicit strengthening over
``roigbiv/registry/calibration.py``'s own precedent, which has no CV at
all). The synthetic FOV is excluded from fitting/LOFO entirely — its
Cellpose features are structurally degenerate (0 predictions across the
entire Phase 2 sweep grid), and including it would dilute an already-tiny
training set with rows where 2 of 5 features carry no signal. The frozen
fitted model is still run on the synthetic FOV afterward and reported
separately, clearly labeled as excluded from fitting.

Example
-------
::

    conda run -n roigbiv python scripts/centroid_bakeoff/fit_consensus.py \\
        --real-fov-dir data/BEGINNER_ROIS/LM_RoiSets/LM_RoiSets/TDT4_ENSURESA \\
        --fs 7.5 --synthetic --synthetic-seed 0

Output artifact (the model fit on all 5 real FOVs, for use by
``run_centroid_bakeoff.py --methods consensus``) lands at
``--out/consensus_model.json``. LOFO metrics are diagnostic only — the LOFO
folds are never what gets persisted.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from centroid_bakeoff.consensus import (  # noqa: E402
    CandidatePool, ConsensusModel, ConsensusScoreScaler, build_candidate_pool,
    collapse_predictions, fit_from_labels, label_candidate_pool, scale_pool_features,
)
from centroid_bakeoff.detector import CentroidDetectorInputs  # noqa: E402
from centroid_bakeoff.ground_truth import (  # noqa: E402
    build_synthetic_fov, discover_real_pairs, imagej_roiset_to_centroids,
)
from centroid_bakeoff.point_match import match_points  # noqa: E402
from centroid_bakeoff.report import print_consensus_lofo_summary  # noqa: E402

_DEFAULT_OUT = Path("experiments/runs/centroid_bakeoff_consensus")
_GRIN_FALLBACK_DIAMETER = 12.0  # roigbiv/pipeline/types.py's own GRIN-profile default

# Permissive pool-generation points — recall-maximizing, not each detector's
# own best-F1 point (Phase 2's best-swept values were -1.0/0.5 respectively).
# Any stricter point throws away recall the fusion model can never recover.
_POOL_CELLPROB_THRESHOLD = -6.0
_POOL_THRESHOLD_SCALING = 0.5

# Zero-parameter baseline thresholds — production defaults, not data-snooped
# from Phase 2's per-FOV best-swept points (which don't generalize to a
# single global constant): Suite2p's iscell_threshold CLI/PipelineConfig
# default (0.3), and the natural midpoint of Cellpose's unbounded
# mean-cellprob score (0.0) since no single best-swept cellprob rescore
# threshold generalized across FOVs in Phase 2.
_BASELINE_SUITE2P_ISCELL = 0.3
_BASELINE_CELLPOSE_MEAN_CELLPROB = 0.0

_CAVEAT = (
    "CONSENSUS MODEL -- n=5 real FOVs, leave-one-FOV-out CV. Point estimates "
    "are noisy at this N; treat per-fold spread, not just the average, as "
    "the result."
)


@dataclass
class FOVPoolData:
    """One real FOV's permissive-point candidate pool + labels + GT, kept
    together through the LOFO loop (fold membership is per-FOV)."""

    fov_stem: str
    pool: CandidatePool
    labels: np.ndarray
    gt: np.ndarray
    max_distance: float


def build_fov_pool(
    fov_stem: str,
    gt: np.ndarray,
    tif_path: Path,
    summary: dict,
    max_distance: float,
    *,
    fs: float,
    cpu: bool,
    work_dir: Path,
    cellprob_threshold: float = _POOL_CELLPROB_THRESHOLD,
    threshold_scaling: float = _POOL_THRESHOLD_SCALING,
) -> tuple[CandidatePool, np.ndarray]:
    """One permissive Cellpose run + one permissive Suite2p run -> pool + labels.

    Runs both detectors at their pool-generation (recall-maximizing) points,
    builds the raw-union candidate pool, and two-pass labels it against GT.
    ``pool.features`` are RAW (unscaled) — the caller applies
    :func:`~centroid_bakeoff.consensus.scale_pool_features` with whichever
    scaler is in effect for the current fold.
    """
    from centroid_bakeoff.detectors.cellpose_centroid import CellposeCentroidDetector
    from centroid_bakeoff.detectors.suite2p_centroid import Suite2pCentroidDetector
    from roigbiv.pipeline.types import PipelineConfig

    mean_M = summary.get("mean_M")
    shape = mean_M.shape if mean_M is not None else next(iter(summary.values())).shape
    inputs = CentroidDetectorInputs(
        summary=summary, fov_stem=fov_stem, shape=shape, fs=fs, raw_tif_path=tif_path,
    )

    cp_cfg = PipelineConfig(fs=fs, force_cpu=cpu)
    cp_det = CellposeCentroidDetector(cfg=cp_cfg, cellprob_threshold=cellprob_threshold)
    cp_result = cp_det.detect(inputs)

    s2p_det = Suite2pCentroidDetector(
        work_dir=work_dir / "suite2p_consensus", iscell_threshold=0.0, lean=True,
        threshold_scaling=threshold_scaling,
    )
    s2p_result = s2p_det.detect(inputs)

    pool = build_candidate_pool(cp_result, s2p_result, max_distance)
    labels = label_candidate_pool(pool, gt, max_distance)
    return pool, labels


def _sweep_accept_thresholds(p_all: np.ndarray, labels_all: np.ndarray, n: int = 41) -> list[dict]:
    """Row-level precision/recall/F1 (not point-matched — a direct
    classification metric over pool rows and their two-pass labels) at a
    grid of acceptance thresholds. Mirrors ``sweep.rescore_sweep``'s "vary a
    post-hoc cutoff, no extra work" shape, applied to candidate-row labels
    instead of point matches.
    """
    thresholds = sorted({round(float(v), 4) for v in np.linspace(0.0, 1.0, n)})
    rows = []
    for thr in thresholds:
        pred_pos = p_all >= thr
        tp = int(np.sum(pred_pos & (labels_all == 1)))
        fp = int(np.sum(pred_pos & (labels_all == 0)))
        fn = int(np.sum(~pred_pos & (labels_all == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        rows.append({"threshold": thr, "precision": precision, "recall": recall, "f1": f1, "n_tp": tp, "n_fp": fp, "n_fn": fn})
    return rows


def _best_f1_threshold(sweep: list[dict]) -> float:
    scored = [r for r in sweep if r["f1"] is not None]
    if not scored:
        return 0.5
    return max(scored, key=lambda r: r["f1"])["threshold"]


def _lowest_precision_floor_threshold(sweep: list[dict], floor: float = 0.5) -> float:
    """Lowest threshold that still holds precision >= floor -- a
    permissive "worth a human look" cutoff, distinct from accept_threshold."""
    ok = [r for r in sweep if r["precision"] is not None and r["precision"] >= floor]
    if not ok:
        return 1.0  # nothing clears the floor; review bucket collapses to "nothing"
    return min(r["threshold"] for r in ok)


def _micro_average(per_fold: dict[str, dict]) -> dict:
    tp = sum(m["n_tp"] for m in per_fold.values())
    fp = sum(m["n_fp"] for m in per_fold.values())
    fn = sum(m["n_fn"] for m in per_fold.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    loc_errs = [m["mean_localization_error"] for m in per_fold.values() if m.get("mean_localization_error") is not None]
    return {
        "n_tp": tp, "n_fp": fp, "n_fn": fn, "n_folds": len(per_fold),
        "precision": precision, "recall": recall, "f1": f1,
        "mean_localization_error": float(np.mean(loc_errs)) if loc_errs else None,
    }


def agreement_gated_baseline(
    pool: CandidatePool,
    suite2p_score_threshold: float = _BASELINE_SUITE2P_ISCELL,
    cellpose_score_threshold: float = _BASELINE_CELLPOSE_MEAN_CELLPROB,
) -> np.ndarray:
    """Zero-parameter sanity baseline (Josh's non-chosen alternative to the
    fitted logistic): accept a row if ``both_detected==1``, OR it's a solo
    whose own RAW score clears a fixed per-detector threshold. Evaluated on
    the same LOFO folds as the fitted model -- a fair, honest comparison, not
    a strawman.

    Returns an ``(N_pool,) float`` array of 0.0/1.0 -- usable directly as the
    ``p_consensus`` argument to ``collapse_predictions`` (threshold 0.5).
    """
    accept = np.zeros(pool.n, dtype=np.float32)
    for i, f in enumerate(pool.features):
        if f.both_detected == 1:
            accept[i] = 1.0
            continue
        if pool.origin[i] == "cellpose" and f.suite2p_present == 0:
            if pool.raw_score[i] >= cellpose_score_threshold:
                accept[i] = 1.0
        elif pool.origin[i] == "suite2p" and f.cellpose_present == 0:
            if pool.raw_score[i] >= suite2p_score_threshold:
                accept[i] = 1.0
    return accept


def lofo_cross_validate(fov_pools: dict[str, FOVPoolData]) -> dict:
    """Leave-one-FOV-out cross-validation across the real FOVs in *fov_pools*.

    Per held-out FOV: fit on the other N-1 FOVs' pooled rows (scaler fit on
    those same N-1, never the held-out FOV), score the held-out FOV's pool
    out-of-fold. Once every fold has an out-of-fold prediction, pool them
    ALL (no leakage -- each row's prediction came from a model that never
    saw it) to pick accept_threshold/review_threshold, then re-derive each
    fold's point-level P/R/F1 at that threshold by collapsing + matching
    against GT (reusing the already-computed out-of-fold scores, no re-fit,
    no re-detection).
    """
    fold_ids = sorted(fov_pools.keys())
    oof_p: dict[str, np.ndarray] = {}
    fold_models: dict[str, ConsensusModel] = {}
    baseline_accept: dict[str, np.ndarray] = {}

    for held_out in fold_ids:
        train = [fov_pools[k] for k in fold_ids if k != held_out]
        train_cp_raw = np.concatenate(
            [p.pool.raw_score[p.pool.origin == "cellpose"] for p in train]
        ) if train else np.zeros(0)
        train_s2p_raw = np.concatenate(
            [p.pool.raw_score[p.pool.origin == "suite2p"] for p in train]
        ) if train else np.zeros(0)
        scaler = ConsensusScoreScaler.fit(train_cp_raw, train_s2p_raw)

        train_samples = []
        for p in train:
            scaled_feats = scale_pool_features(p.pool, scaler)
            train_samples.extend(zip(scaled_feats, p.labels.tolist()))

        model = fit_from_labels(train_samples, scaler=scaler)
        fold_models[held_out] = model

        held = fov_pools[held_out]
        held_scaled = scale_pool_features(held.pool, scaler)
        oof_p[held_out] = (
            np.asarray([model.p_consensus(f) for f in held_scaled], dtype=np.float32)
            if held.pool.n else np.zeros(0, dtype=np.float32)
        )
        baseline_accept[held_out] = agreement_gated_baseline(held.pool)

    all_p = np.concatenate([oof_p[k] for k in fold_ids]) if fold_ids else np.zeros(0)
    all_labels = np.concatenate([fov_pools[k].labels for k in fold_ids]) if fold_ids else np.zeros(0, dtype=np.int32)
    threshold_sweep = _sweep_accept_thresholds(all_p, all_labels)
    accept_threshold = _best_f1_threshold(threshold_sweep)
    review_threshold = _lowest_precision_floor_threshold(threshold_sweep, floor=0.5)

    per_fold: dict[str, dict] = {}
    per_fold_baseline: dict[str, dict] = {}
    for held_out in fold_ids:
        held = fov_pools[held_out]
        centroids, _scores = collapse_predictions(held.pool, oof_p[held_out], accept_threshold, held.max_distance)
        match = match_points(held.gt, centroids, max_distance=held.max_distance)
        per_fold[held_out] = match.to_dict()

        b_centroids, _ = collapse_predictions(held.pool, baseline_accept[held_out], 0.5, held.max_distance)
        b_match = match_points(held.gt, b_centroids, max_distance=held.max_distance)
        per_fold_baseline[held_out] = b_match.to_dict()

    return {
        "fold_ids": fold_ids,
        "accept_threshold": accept_threshold,
        "review_threshold": review_threshold,
        "threshold_sweep": threshold_sweep,
        "per_fold": per_fold,
        "aggregate": _micro_average(per_fold),
        "baseline_per_fold": per_fold_baseline,
        "baseline_aggregate": _micro_average(per_fold_baseline),
        "caveat": _CAVEAT,
    }


def _run_foundation_for(tif_path: Path, fs: float, cpu: bool, work_dir: Path):
    from roigbiv.pipeline.foundation import run_foundation
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=fs, do_registration=False, force_cpu=cpu)
    fov_data = run_foundation(tif_path, cfg, work_dir / "foundation")
    return {"mean_M": fov_data.mean_M, "vcorr_S": fov_data.vcorr_S,
            "max_S": fov_data.max_S, "dog_map": fov_data.dog_map}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Fit + LOFO cross-validate the Cellpose+Suite2p consensus fusion model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--real-fov-dir", type=Path, required=True,
                   help="Root to recursively discover *_mc.tif + *_RoiSet[_FINAL].zip pairs.")
    p.add_argument("--synthetic", action="store_true",
                   help="Also score (not fit) the synthetic soma-injection FOV, reported "
                        "separately and clearly marked excluded from fit/LOFO.")
    p.add_argument("--synthetic-seed", type=int, default=0)
    p.add_argument("--synthetic-shape", default="300,512,512")
    p.add_argument("--fs", type=float, default=30.0)
    p.add_argument("--max-distance-px", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args(argv)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from roigbiv.pipeline.optics import measure_soma_scale

    pairs = discover_real_pairs([args.real_fov_dir])
    if not pairs:
        p.error(f"no real FOV pairs found under {args.real_fov_dir}")
    print(f"Discovered {len(pairs)} real FOV(s) under {args.real_fov_dir}", flush=True)

    fov_pools: dict[str, FOVPoolData] = {}
    for mc_tif, roi_zip, stem in pairs:
        print(f"\n== {stem} (real, building permissive pool) ==", flush=True)
        with tifffile.TiffFile(str(mc_tif)) as tf:
            shape = tf.pages[0].shape
        gt, _names = imagej_roiset_to_centroids(roi_zip, shape)

        work_dir = out_dir / "_work" / stem
        work_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_foundation_for(mc_tif, args.fs, args.cpu, work_dir)

        mean_M = summary.get("mean_M")
        soma_scale = measure_soma_scale(mean_M, summary.get("dog_map")) if mean_M is not None else None
        if args.max_distance_px is not None:
            max_distance = args.max_distance_px
        elif soma_scale is not None and soma_scale.ok:
            max_distance = soma_scale.diameter_med / 2.0
        else:
            max_distance = _GRIN_FALLBACK_DIAMETER / 2.0

        pool, labels = build_fov_pool(
            stem, gt, mc_tif, summary, max_distance, fs=args.fs, cpu=args.cpu, work_dir=work_dir,
        )
        print(f"  pool: n_cellpose={int((pool.origin == 'cellpose').sum())} "
              f"n_suite2p={int((pool.origin == 'suite2p').sum())} "
              f"n_positive_labels={int(labels.sum())}", flush=True)
        fov_pools[stem] = FOVPoolData(fov_stem=stem, pool=pool, labels=labels, gt=gt, max_distance=max_distance)

    print("\n== LOFO cross-validation (5 real FOVs) ==", flush=True)
    lofo = lofo_cross_validate(fov_pools)

    # Frozen model: fit on ALL 5 real FOVs (not a LOFO fold) -- this is the
    # artifact run_centroid_bakeoff.py --methods consensus actually loads.
    all_cp_raw = np.concatenate([fp.pool.raw_score[fp.pool.origin == "cellpose"] for fp in fov_pools.values()])
    all_s2p_raw = np.concatenate([fp.pool.raw_score[fp.pool.origin == "suite2p"] for fp in fov_pools.values()])
    full_scaler = ConsensusScoreScaler.fit(all_cp_raw, all_s2p_raw)
    full_samples = []
    for fp in fov_pools.values():
        scaled = scale_pool_features(fp.pool, full_scaler)
        full_samples.extend(zip(scaled, fp.labels.tolist()))
    full_model = fit_from_labels(full_samples, scaler=full_scaler)
    model_path = out_dir / "consensus_model.json"
    full_model.save(model_path)
    print(f"\nFitted model (all 5 real FOVs) saved to {model_path}", flush=True)

    synthetic_report = None
    if args.synthetic:
        stem = f"synthetic_seed{args.synthetic_seed}"
        print(f"\n== {stem} (synthetic -- NOT used for fit or LOFO) ==", flush=True)
        T, H, W = (int(x) for x in args.synthetic_shape.split(","))
        movie, gt, _specs = build_synthetic_fov(shape=(T, H, W), fs=args.fs, seed=args.synthetic_seed)
        work_dir = out_dir / "_work" / stem
        work_dir.mkdir(parents=True, exist_ok=True)
        synth_tif = work_dir / f"{stem}.tif"
        tifffile.imwrite(str(synth_tif), movie.astype(np.float32))
        summary = _run_foundation_for(synth_tif, args.fs, args.cpu, work_dir)

        mean_M = summary.get("mean_M")
        soma_scale = measure_soma_scale(mean_M, summary.get("dog_map")) if mean_M is not None else None
        max_distance = (
            args.max_distance_px if args.max_distance_px is not None
            else (soma_scale.diameter_med / 2.0 if soma_scale is not None and soma_scale.ok
                  else _GRIN_FALLBACK_DIAMETER / 2.0)
        )
        synth_pool, synth_labels = build_fov_pool(
            stem, gt, synth_tif, summary, max_distance, fs=args.fs, cpu=args.cpu, work_dir=work_dir,
        )
        synth_scaled = scale_pool_features(synth_pool, full_scaler)
        synth_p = (
            np.asarray([full_model.p_consensus(f) for f in synth_scaled], dtype=np.float32)
            if synth_pool.n else np.zeros(0, dtype=np.float32)
        )
        synth_centroids, _ = collapse_predictions(synth_pool, synth_p, lofo["accept_threshold"], max_distance)
        synth_match = match_points(gt, synth_centroids, max_distance=max_distance)
        synthetic_report = {"fov_stem": stem, **synth_match.to_dict(),
                             "n_raw_cellpose": int((synth_pool.origin == "cellpose").sum()),
                             "n_raw_suite2p": int((synth_pool.origin == "suite2p").sum())}

    print_consensus_lofo_summary(lofo, synthetic_report)

    payload = {
        "caveat": _CAVEAT,
        "lofo": lofo,
        "model_path": str(model_path),
        "synthetic": synthetic_report,
    }
    report_path = out_dir / "consensus_lofo_report.json"
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"\nLOFO report: {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
