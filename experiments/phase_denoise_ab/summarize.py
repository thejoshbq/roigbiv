#!/usr/bin/env python
"""Confirmatory denoise A/B summary — denoise-ON (Phase-4 fused) vs denoise-OFF.

denoise-ON  = experiments/runs/phase4_channel/ab_results.json  {stem}|fused
denoise-OFF = experiments/runs/phase_denoise_ab/off_results.json {stem}|off

Both arms share ch2=vcorr_max_fused, backend cellpose3 — the only variable is
use_denoise. Reports pooled micro-averaged recall/precision, the recall contrast,
the per-FOV regression check (the load-bearing test), and pooled FP burden.

Recall-first bar: no per-FOV recall regression AND pooled FP increase <= +15%.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/thejoshbq/Otis-Lab/Projects/Phoxel-Workbench/roigbiv")
ON = ROOT / "experiments/runs/phase4_channel/ab_results.json"
OFF = ROOT / "experiments/runs/phase_denoise_ab/off_results.json"


def micro(tp, fp, fn):
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return rec, prec


def main() -> int:
    on_raw = json.loads(ON.read_text())
    off_raw = json.loads(OFF.read_text())

    on = {k.split("|")[0]: v for k, v in on_raw.items()
          if k.endswith("|fused") and "error" not in v}
    off = {k.split("|")[0]: v for k, v in off_raw.items()
           if k.endswith("|off") and "error" not in v}

    stems = sorted(set(on) & set(off))
    miss_on = sorted(set(off) - set(on))
    miss_off = sorted(set(on) - set(off))

    print(f"=== Confirmatory denoise A/B (ch2=vcorr_max_fused fixed) — "
          f"{len(stems)} paired FOVs ===")
    if miss_off:
        print(f"  !! OFF arm missing/failed for {len(miss_off)}: "
              + ", ".join(s[:32] for s in miss_off))
    if miss_on:
        print(f"  !! ON arm missing for {len(miss_on)}: "
              + ", ".join(s[:32] for s in miss_on))

    pool = {"on": [0, 0, 0], "off": [0, 0, 0]}
    for s in stems:
        for arm, src in (("on", on), ("off", off)):
            ov = src[s]["detection"]["overall"]
            pool[arm][0] += ov["tp"]; pool[arm][1] += ov["fp"]; pool[arm][2] += ov["fn"]

    print("\n--- Pooled (micro-averaged) ---")
    for arm in ("on", "off"):
        tp, fp, fn = pool[arm]
        rec, prec = micro(tp, fp, fn)
        label = "denoise_on (deployed)" if arm == "on" else "denoise_off (candidate)"
        print(f"  {label:26s} R{rec:.3f} P{prec:.3f}  tp{tp} fp{fp} fn{fn}")
    r_on, _ = micro(*pool["on"]); r_off, _ = micro(*pool["off"])
    fp_on, fp_off = pool["on"][1], pool["off"][1]
    print(f"\n  recall delta (off - on): {r_off - r_on:+.3f}")
    fp_pct = (fp_off - fp_on) / fp_on * 100 if fp_on else 0.0
    print(f"  pooled FP: {fp_on} -> {fp_off}  ({fp_pct:+.1f}%)")

    print("\n--- Per-FOV overall recall (regression check) ---")
    print(f"{'stem':52s}{'on':>9s}{'off':>9s}{'Δ':>9s}")
    regressions = []
    for s in stems:
        ro = on[s]["detection"]["overall"]["recall"]
        rf = off[s]["detection"]["overall"]["recall"]
        d = rf - ro
        flag = "  <-- REGRESS" if d < -1e-9 else ""
        if d < -1e-9:
            regressions.append((s, d))
        print(f"{s[:52]:52s}{ro:>9.3f}{rf:>9.3f}{d:>+9.3f}{flag}")

    print("\n--- Verdict (recall-first bar: 0 FOV regressions, FP <= +15%) ---")
    bar_recall = not regressions
    bar_fp = fp_pct <= 15.0 + 1e-9
    print(f"  no-regression : {'PASS' if bar_recall else 'FAIL'} "
          f"({len(regressions)} FOV(s) regress)")
    print(f"  FP <= +15%    : {'PASS' if bar_fp else 'FAIL'} ({fp_pct:+.1f}%)")
    if bar_recall and bar_fp:
        print("  => OVERALL PASS: denoise-OFF is a clean default-flip candidate.")
    else:
        print("  => OVERALL FAIL: KEEP denoise ON.")
        for s, d in regressions:
            print(f"       regress {s[:48]} {d:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
