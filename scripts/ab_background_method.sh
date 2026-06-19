#!/usr/bin/env bash
# A/B the Foundation background method (svd vs rpca) on the held-out set.
#
# Sub-step-1 gate (see docs/.../algorithms.md §2.3 + the remediation plan):
#   rpca must be ≥ svd on overall recall, strictly better on the tonic stratum,
#   and not regress phasic/sparse beyond noise — before the default is flipped.
#
# Runs the FULL pipeline for each held-out FOV under both methods, then scores
# both arms with the stratified eval harness. Long-running (Suite2p MC + 4 stages
# × 13 FOVs × 2 arms). Resumable: re-running skips completed stages per --resume.
#
# Usage:  conda activate roigbiv && scripts/ab_background_method.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=experiments/harness/heldout_fovs.txt
ROOT=experiments/runs
FS=7.5   # 4×-averaged stacks (reference regime); override per dataset.

run_arm() {  # $1=method  $2=stem  $3=movie
  local method="$1" stem="$2" movie="$3"
  local out="$ROOT/${stem}_${method}"
  echo "── [$method] $stem ──"
  roigbiv-pipeline --input "$movie" --fs "$FS" \
    --background-method "$method" --no-viewer --resume \
    --output-dir "$out"
}

grep -v '^#' "$MANIFEST" | grep -v '^[[:space:]]*$' | while IFS='|' read -r stem movie gt; do
  [ -z "$stem" ] && continue
  run_arm svd  "$stem" "$movie"
  run_arm rpca "$stem" "$movie"
done

echo "=== scoring ==="
python -m roigbiv.eval.harness --batch "$MANIFEST" --pipeline-root "$ROOT" \
  --solver svd  --output "$ROOT/batch_svd_metrics.json"
python -m roigbiv.eval.harness --batch "$MANIFEST" --pipeline-root "$ROOT" \
  --solver rpca --output "$ROOT/batch_rpca_metrics.json"

echo "=== stratified deltas (rpca − svd) ==="
python - <<'PY'
import json
svd = json.load(open("experiments/runs/batch_svd_metrics.json"))
rpca = json.load(open("experiments/runs/batch_rpca_metrics.json"))
def rec(d, strat):
    det = d.get("detection", d)
    if strat == "overall":
        return det["overall"]["recall"]
    return det.get("by_activity_type", {}).get(strat, {}).get("recall", float("nan"))
for strat in ("overall", "phasic", "sparse", "tonic", "silent"):
    s, r = rec(svd, strat), rec(rpca, strat)
    print(f"  {strat:8s}  svd={s:.3f}  rpca={r:.3f}  Δ={r-s:+.3f}")
PY
