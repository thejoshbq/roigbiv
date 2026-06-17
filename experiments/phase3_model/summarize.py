#!/usr/bin/env python
"""Aggregate the Phase-3 3-arm A/B into a stratified, recall-first summary.

Pools TP/FP/FN across FOVs per arm per activity stratum (micro-averaged recall /
precision), reports the three decomposing pairwise contrasts, and lists per-FOV
recall so regressions in any single FOV are visible.

Usage: python experiments/phase3_model/summarize.py [ab_results.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("experiments/runs/phase3_model/ab_results.json")
ARMS = ["cp3", "cp3nd", "cpsam"]
STRATA = ["phasic", "sparse", "tonic", "silent", "ambiguous", "unknown"]


def micro(tp, fp, fn):
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return rec, prec, f1


def main() -> int:
    r = json.loads(RESULTS.read_text())
    # arm -> stratum(or 'overall') -> [tp,fp,fn]
    pool = {a: defaultdict(lambda: [0, 0, 0]) for a in ARMS}
    per_fov = defaultdict(dict)   # stem -> arm -> overall recall
    errors = []
    n_fov = set()
    for key, v in r.items():
        if "error" in v:
            errors.append((key, v["error"][:120])); continue
        arm = v["arm"]; stem = v["stem"]; n_fov.add(stem)
        d = v["detection"]
        ov = d["overall"]
        pool[arm]["overall"][0] += ov["tp"]
        pool[arm]["overall"][1] += ov["fp"]
        pool[arm]["overall"][2] += ov["fn"]
        per_fov[stem][arm] = ov["recall"]
        for s, m in d.get("by_type", {}).items():
            if isinstance(m, dict) and "tp" in m:
                pool[arm][s][0] += m.get("tp", 0)
                pool[arm][s][1] += m.get("fp", 0)
                pool[arm][s][2] += m.get("fn", 0)

    print(f"=== Phase-3 Stage-1 model A/B — {len(n_fov)} FOVs, arms={ARMS} ===")
    if errors:
        print(f"\n!! {len(errors)} failed arm(s):")
        for k, e in errors:
            print(f"   {k}: {e}")

    print("\n--- Micro-averaged (pooled TP/FP/FN) ---")
    hdr = f"{'stratum':12s}" + "".join(f"{a:>22s}" for a in ARMS)
    print(hdr)
    for s in ["overall"] + STRATA:
        cells = []
        any_data = False
        for a in ARMS:
            tp, fp, fn = pool[a][s]
            if tp + fp + fn:
                any_data = True
            rec, prec, _ = micro(tp, fp, fn)
            cells.append(f"R{rec:.3f} P{prec:.3f} tp{tp} fn{fn}")
        if any_data:
            print(f"{s:12s}" + "".join(f"{c:>22s}" for c in cells))

    print("\n--- Pairwise (recall, overall) ---")
    o = {a: micro(*pool[a]["overall"])[0] for a in ARMS}
    print(f"  cp3   ↔ cp3nd  (denoise):       {o['cp3']:.3f} → {o['cp3nd']:.3f}  Δ{o['cp3nd']-o['cp3']:+.3f}")
    print(f"  cp3nd ↔ cpsam  (architecture):  {o['cp3nd']:.3f} → {o['cpsam']:.3f}  Δ{o['cpsam']-o['cp3nd']:+.3f}")
    print(f"  cp3   ↔ cpsam  (as-deployed):   {o['cp3']:.3f} → {o['cpsam']:.3f}  Δ{o['cpsam']-o['cp3']:+.3f}")

    print("\n--- Per-FOV overall recall (regression check) ---")
    print(f"{'stem':52s}" + "".join(f"{a:>9s}" for a in ARMS))
    for stem in sorted(per_fov):
        row = per_fov[stem]
        print(f"{stem[:52]:52s}" + "".join(f"{row.get(a, float('nan')):>9.3f}" for a in ARMS))

    # FP burden (post-review): pooled FP per arm
    print("\n--- Post-review FP burden (pooled FP, overall) ---")
    for a in ARMS:
        print(f"  {a:6s}: fp={pool[a]['overall'][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
