"""Report generation — overlay PNG, JSON, printed table.

Conventions follow ``scripts/bench_motion_correction.py`` (JSON metrics + PNG
+ printed table, all under ``experiments/runs/``) and
``scripts/cv_bakeoff/grid.py`` (matplotlib multi-panel grid, one panel per
method) — this module borrows the panel-grid shape but plots points (GT
circles, per-method TP/FP markers) instead of mask contours, since centroids
have no boundary to draw.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _stretch_to_uint8(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Same percentile-stretch convention as roigbiv.overlay._stretch_to_uint8."""
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def render_overlay_grid(
    background: np.ndarray,
    gt: np.ndarray,
    method_results: dict,  # method_name -> (CentroidDetectorResult, PointMatchResult)
    *,
    fov_stem: str,
    gt_source: str,
    out_path: Path,
) -> Path:
    """One panel per method: background + GT (white circles, unfilled) +
    that method's TP (green +) / FP (red x) predictions. Unmatched GT (FN)
    show as GT circles with no nearby green +, visible by elimination.
    """
    bg = _stretch_to_uint8(background)
    n_panels = max(1, len(method_results))
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, (method, (det_result, match)) in zip(flat, method_results.items()):
        ax.imshow(bg, cmap="gray")
        if len(gt):
            ax.scatter(gt[:, 1], gt[:, 0], s=40, facecolors="none",
                       edgecolors="white", linewidths=1.0, label="GT")
        preds = det_result.centroids
        tp_pred_idx = {p for _, p, _ in match.tp}
        if len(preds):
            tp_mask = np.array([i in tp_pred_idx for i in range(len(preds))])
            fp_mask = ~tp_mask
            if tp_mask.any():
                ax.scatter(preds[tp_mask, 1], preds[tp_mask, 0], s=30, marker="+",
                           c="lime", linewidths=1.2, label="TP")
            if fp_mask.any():
                ax.scatter(preds[fp_mask, 1], preds[fp_mask, 0], s=30, marker="x",
                           c="red", linewidths=1.2, label="FP")
        f1 = match.f1
        f1_str = f"{f1:.2f}" if f1 is not None else "n/a"
        rt = det_result.meta.get("runtime_s", "?")
        ax.set_title(
            f"{method} · P={match.precision and round(match.precision,2)} "
            f"R={match.recall and round(match.recall,2)} F1={f1_str} · {rt}s",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
        ax.axis("off")

    for ax in flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"{fov_stem}  —  centroid bake-off ({gt_source} GT, n_gt={len(gt)})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def build_fov_report(
    fov_stem: str, gt_source: str, gt: np.ndarray, method_results: dict,
) -> dict:
    """Assemble one FOV's report dict from {method: (DetectorResult, PointMatchResult)}."""
    methods = {}
    for method, (det_result, match) in method_results.items():
        methods[method] = {
            **match.to_dict(),
            "n_pred": det_result.n,
            "runtime_s": det_result.meta.get("runtime_s"),
            "detector_meta": det_result.meta,
        }
    return {"fov_stem": fov_stem, "gt_source": gt_source, "n_gt": int(len(gt)),
            "methods": methods}


def build_aggregate(fov_reports: list[dict]) -> dict:
    """Micro-averaged aggregate across all FOVs: sum tp/fp/fn per method, then
    derive precision/recall/f1 from the sums (not a mean of per-FOV ratios,
    which would over-weight small/sparse FOVs).
    """
    per_method: dict[str, dict] = {}
    for rep in fov_reports:
        for method, m in rep["methods"].items():
            acc = per_method.setdefault(method, {
                "n_tp": 0, "n_fp": 0, "n_fn": 0,
                "_loc_errs": [], "_runtimes": [], "n_fovs": 0,
            })
            acc["n_tp"] += m["n_tp"]
            acc["n_fp"] += m["n_fp"]
            acc["n_fn"] += m["n_fn"]
            acc["n_fovs"] += 1
            if m.get("mean_localization_error") is not None:
                acc["_loc_errs"].append(m["mean_localization_error"])
            if m.get("runtime_s") is not None:
                acc["_runtimes"].append(m["runtime_s"])

    aggregate = {}
    for method, acc in per_method.items():
        tp, fp, fn = acc["n_tp"], acc["n_fp"], acc["n_fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and (precision + recall) > 0
              else None)
        aggregate[method] = {
            "n_tp": tp, "n_fp": fp, "n_fn": fn, "n_fovs": acc["n_fovs"],
            "precision": precision, "recall": recall, "f1": f1,
            "mean_localization_error": (
                float(np.mean(acc["_loc_errs"])) if acc["_loc_errs"] else None
            ),
            "mean_runtime_s": (
                float(np.mean(acc["_runtimes"])) if acc["_runtimes"] else None
            ),
        }
    return aggregate


def write_json_report(fov_reports: list[dict], aggregate: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "fov_reports": fov_reports,
        "aggregate": aggregate,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_summary_table(aggregate: dict) -> None:
    cols = ["precision", "recall", "f1", "mean_localization_error", "mean_runtime_s", "n_fovs"]
    hdr = f"{'method':<20}" + "".join(f"{c:>24}" for c in cols)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for method, m in aggregate.items():
        row = f"{method:<20}"
        for c in cols:
            v = m.get(c)
            row += f"{v:>24.4f}" if isinstance(v, (int, float)) else f"{'-':>24}"
        print(row, flush=True)


# ---------------------------------------------------------------------------
# Sweep reporting — PR curves + best-operating-point per (FOV, method)
# ---------------------------------------------------------------------------

def _pareto_frontier(pts: list) -> list:
    """Upper-right non-dominated subset (max recall AND max precision),
    ascending by recall — the classic PR-curve staircase. Sweeping a
    structural param and a rescore threshold together produces points that
    aren't monotonic in recall, so connecting all raw points in recall order
    zigzags; the frontier is what a reader actually wants to see."""
    ranked = sorted(pts, key=lambda p: (-p.match.recall, -p.match.precision))
    frontier, best_precision = [], -1.0
    for p in ranked:
        if p.match.precision > best_precision:
            frontier.append(p)
            best_precision = p.match.precision
    return list(reversed(frontier))


def render_pr_curve(
    sweep_results: dict,   # method -> sweep.SweepResult
    *, fov_stem: str, out_path: Path,
) -> Path:
    """One precision/recall line per method — the Pareto frontier of all
    swept points (structural-grid and free-rescore combined) — with all raw
    points shown as a light scatter underneath and the best-F1 point starred.
    """
    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = plt.get_cmap("tab10").colors

    for i, (method, sweep) in enumerate(sweep_results.items()):
        pts = [p for p in sweep.points
               if p.match.precision is not None and p.match.recall is not None]
        if not pts:
            continue
        color = colors[i % len(colors)]
        all_recalls = [p.match.recall for p in pts]
        all_precisions = [p.match.precision for p in pts]
        ax.scatter(all_recalls, all_precisions, color=color, s=10, alpha=0.25, zorder=2)

        frontier = _pareto_frontier(pts)
        ax.plot([p.match.recall for p in frontier], [p.match.precision for p in frontier],
                 "-o", color=color, markersize=4, alpha=0.9, label=method, zorder=3)

        best = sweep.best
        if best is not None:
            ax.scatter([best.match.recall], [best.match.precision], color=color,
                        marker="*", s=220, edgecolors="black", linewidths=0.8, zorder=5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{fov_stem}\nprecision/recall sweep (line = Pareto frontier, ★ = best F1)",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def build_sweep_report(fov_stem: str, gt_source: str, n_gt: int, sweep_results: dict) -> dict:
    """Assemble one FOV's sweep report from {method: SweepResult}."""
    return {
        "fov_stem": fov_stem, "gt_source": gt_source, "n_gt": int(n_gt),
        "methods": {method: sweep.to_dict() for method, sweep in sweep_results.items()},
    }


def write_sweep_json_report(fov_reports: list[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "fov_reports": fov_reports,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_sweep_best_table(fov_reports: list[dict]) -> None:
    """Per-FOV, per-method best-F1 point and the parameter value(s) that
    achieved it — the direct answer to "best achievable performance"."""
    hdr = f"{'fov':<26}{'method':<12}{'best_f1':>10}   swept params"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for rep in fov_reports:
        for method, m in rep["methods"].items():
            best = m.get("best")
            if best is None:
                print(f"{rep['fov_stem']:<26}{method:<12}{'n/a':>10}", flush=True)
                continue
            f1 = best.get("f1")
            f1_str = f"{f1:.4f}" if f1 is not None else "n/a"
            params_str = ", ".join(f"{k}={v}" for k, v in best["params"].items())
            print(f"{rep['fov_stem']:<26}{method:<12}{f1_str:>10}   {params_str}", flush=True)


def print_consensus_lofo_summary(lofo: dict, synthetic: Optional[dict] = None) -> None:
    """Print the consensus-fusion LOFO cross-validation summary.

    Ahead of any numbers: an unmissable small-N caveat banner (n=5 real FOVs
    for LOFO is genuinely small — per-fold spread matters as much as the
    average). Then per-fold P/R/F1 for the fitted model, the averaged LOFO
    P/R/F1, the same folds scored by a zero-parameter agreement-gated
    baseline (a fair, honest comparison — not a strawman), and, if provided,
    the synthetic FOV's result under a heading explicitly marked as excluded
    from fit/LOFO.
    """
    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "n/a"

    print(f"\n{lofo.get('caveat', '')}\n", flush=True)
    print(f"accept_threshold={lofo['accept_threshold']:.4f}  "
          f"review_threshold={lofo['review_threshold']:.4f}\n", flush=True)

    hdr = f"{'fov':<45}{'precision':>12}{'recall':>12}{'f1':>12}"
    print("== consensus model (LOFO out-of-fold) ==", flush=True)
    print(hdr, flush=True)
    for fov_stem, m in lofo["per_fold"].items():
        print(f"{fov_stem:<45}{_fmt(m['precision']):>12}{_fmt(m['recall']):>12}{_fmt(m['f1']):>12}", flush=True)
    agg = lofo["aggregate"]
    print(f"{'AVERAGE (micro)':<45}{_fmt(agg['precision']):>12}{_fmt(agg['recall']):>12}{_fmt(agg['f1']):>12}",
          flush=True)

    print("\n== agreement-gated baseline (same LOFO folds, zero parameters) ==", flush=True)
    print(hdr, flush=True)
    for fov_stem, m in lofo["baseline_per_fold"].items():
        print(f"{fov_stem:<45}{_fmt(m['precision']):>12}{_fmt(m['recall']):>12}{_fmt(m['f1']):>12}", flush=True)
    b_agg = lofo["baseline_aggregate"]
    print(f"{'AVERAGE (micro)':<45}{_fmt(b_agg['precision']):>12}{_fmt(b_agg['recall']):>12}{_fmt(b_agg['f1']):>12}",
          flush=True)

    if synthetic is not None:
        print(f"\n== {synthetic['fov_stem']} -- NOT used for fit or LOFO ==", flush=True)
        print("Cellpose contributes 0 usable candidates on this synthetic FOV; "
              "reported for qualitative sanity only.", flush=True)
        print(f"precision={_fmt(synthetic.get('precision'))} recall={_fmt(synthetic.get('recall'))} "
              f"f1={_fmt(synthetic.get('f1'))} (n_raw_cellpose={synthetic.get('n_raw_cellpose')}, "
              f"n_raw_suite2p={synthetic.get('n_raw_suite2p')})", flush=True)


def print_max_distance_sensitivity(sensitivity: dict) -> None:
    """``sensitivity``: fov_stem -> method -> {distance: PointMatchResult.to_dict()}.

    Shows whether the F1 ranking across methods survives a stricter/looser
    match tolerance than the soma-radius default, or is an artifact of one
    arbitrary cutoff.
    """
    for fov_stem, methods in sensitivity.items():
        print(f"\n{fov_stem} — max_distance sensitivity", flush=True)
        distances = sorted({d for m in methods.values() for d in m})
        hdr = f"{'method':<12}" + "".join(f"{f'd={d:g}':>14}" for d in distances)
        print(hdr, flush=True)
        for method, by_distance in methods.items():
            row = f"{method:<12}"
            for d in distances:
                f1 = by_distance.get(d, {}).get("f1")
                row += f"{f1:>14.4f}" if f1 is not None else f"{'-':>14}"
            print(row, flush=True)
