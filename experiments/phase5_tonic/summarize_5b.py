#!/usr/bin/env python
"""Phase-5b gate-aware threshold sweep + recall-first verdict.

Reads roi_records.json (one IoU-matched per-ROI record set per FOV, tier OFF)
and simulates the tonic accept tier at swept elevation thresholds. Because the
tier changes no mask, this post-hoc simulation is EXACT.

For each threshold τ, over the promotion population
(activity_type=="tonic" AND source_stage∈{1,2} AND currently review-routed):
  promoted        = elevation ≥ τ
  good_accept     = promoted AND matched-to-GT   (correctly skips review)
  risk_accept     = promoted AND NOT matched     (FP escaping review — but a
                    LOWER BOUND on true FP: anatomical GT under-represents
                    tonics, so some unmatched promotions are real tonic cells
                    GT missed, not false positives)
  auto_accept_precision = good_accept / promoted
  review_burden_delta   = -promoted   (ROIs removed from human review)

Recall is unaffected by construction (masks identical) — asserted via the OFF
arm being the only arm. The verdict bar is precision/burden:
  * auto_accept_precision ≥ PREC_BAR   (default 0.80; lower-bound-aware)
  * pooled risk_accept ≤ FP_MARGIN     (default: ≤ +2 FP across all FOVs)

Usage: python experiments/phase5_tonic/summarize_5b.py [roi_records.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("experiments/runs/phase5_5b/roi_records.json")

THRESHOLDS = [round(0.1 * i, 2) for i in range(0, 31)]   # 0.0 .. 3.0
PREC_BAR = 0.80
FP_MARGIN = 2          # pooled risk_accept ceiling across all FOVs
_REVIEW_OUTCOMES = {"flag"}
_REVIEW_CONFIDENCE = {"requires_review", "moderate", "low"}


def _review_routed(r: dict) -> bool:
    return r["gate_outcome"] in _REVIEW_OUTCOMES or r["confidence"] in _REVIEW_CONFIDENCE


def main() -> int:
    data = json.loads(RESULTS.read_text())
    fovs = {k: v for k, v in data.items() if "error" not in v}
    errs = {k: v for k, v in data.items() if "error" in v}

    # Promotion population: anatomical tonic ROIs currently routed to review.
    pop = []   # (stem, elevation, matched)
    for stem, v in fovs.items():
        for r in v.get("records", []):
            if (r["activity_type"] == "tonic" and r["source_stage"] in (1, 2)
                    and _review_routed(r)):
                pop.append((stem, r["elevation"], r["matched"]))

    print(f"=== Phase-5b tonic accept-tier — gate-aware sweep ===")
    print(f"FOVs scored: {len(fovs)}" + (f"  ({len(errs)} failed)" if errs else ""))
    for k, v in errs.items():
        print(f"  !! FAIL {k}: {v['error'][:120]}")

    # Population diagnostics
    n_tonic12_all = sum(
        1 for v in fovs.values() for r in v.get("records", [])
        if r["activity_type"] == "tonic" and r["source_stage"] in (1, 2))
    print(f"\nAnatomical tonic ROIs (source_stage∈{{1,2}}): {n_tonic12_all}")
    print(f"  of which currently review-routed (promotion population): {len(pop)}")
    if not pop:
        print("\nNo promotion candidates — tier would be inert on this set.")
        print("Either no anatomical tonics survive to review, or classify never "
              "labels stage-1/2 ROIs tonic on these FOVs. Report this and STOP.")
        return 0

    matched_pop = sum(1 for _, _, m in pop if m)
    print(f"  of which IoU-match GT: {matched_pop}/{len(pop)} "
          f"({matched_pop/len(pop):.2f} base precision)")
    elevs = sorted(e for _, e, _ in pop)
    print(f"  elevation distribution: min={elevs[0]:.3f} "
          f"med={elevs[len(elevs)//2]:.3f} max={elevs[-1]:.3f}")

    print(f"\n--- Threshold sweep (bar: precision≥{PREC_BAR}, "
          f"pooled risk_accept≤{FP_MARGIN}) ---")
    print(f"{'τ':>5s}{'promoted':>10s}{'good':>6s}{'risk':>6s}"
          f"{'precision':>11s}{'burden_Δ':>10s}{'pass':>6s}")
    passing = []
    for tau in THRESHOLDS:
        promoted = [(s, e, m) for s, e, m in pop if e >= tau]
        npro = len(promoted)
        good = sum(1 for _, _, m in promoted if m)
        risk = npro - good
        prec = good / npro if npro else float("nan")
        ok = npro > 0 and prec >= PREC_BAR and risk <= FP_MARGIN
        if ok:
            passing.append((tau, npro, good, risk, prec))
        flag = "✓" if ok else ""
        pstr = f"{prec:.3f}" if npro else "  -  "
        print(f"{tau:>5.1f}{npro:>10d}{good:>6d}{risk:>6d}{pstr:>11s}"
              f"{-npro:>10d}{flag:>6s}")

    print("\n--- Recommendation ---")
    if not passing:
        print("NO threshold satisfies the bar. The tier cannot auto-accept "
              "anatomical tonics at acceptable precision on this set → keep OFF.")
        return 0
    # Prefer the threshold that maximizes review-burden reduction (most promoted)
    # while still passing — i.e. the lowest passing τ.
    best = max(passing, key=lambda x: x[1])
    tau, npro, good, risk, prec = best
    print(f"Recommended τ = {tau:.2f}: promotes {npro} ROI(s), {good} GT-matched, "
          f"{risk} unmatched (lower-bound FP), precision={prec:.3f}.")
    print(f"Review-burden reduction: {npro} ROI(s) skip human review across "
          f"{len(fovs)} FOVs. CAVEAT: unmatched promotions are an FP UPPER bound "
          f"(GT under-represents tonics). Set cfg.tonic_accept_min_elevation="
          f"{tau:.2f}; flag stays OFF pending approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
