"""
Run Stage 3 + Gate 3 + subtraction on the virtual S₂ residual.

Picks up from where the previous scoped E2E left off: rebuilds the S₂ residual
view from svd_factors.npz + data.bin + the Stage 1/2 source sidecars (no dense
.dat under the virtual-residual model).
Run: conda run -n roigbiv python -u -m roigbiv.pipeline.tests.test_stage3_only
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from roigbiv.pipeline.gate3 import evaluate_gate3
from roigbiv.pipeline.residual import ResidualView, SourceLayer
from roigbiv.pipeline.stage3 import run_stage3
from roigbiv.pipeline.stage3_templates import build_template_bank
from roigbiv.pipeline.subtraction import compute_std_map, run_source_subtraction
from roigbiv.pipeline.types import FOVData, PipelineConfig


OUTPUT_DIR = Path("inference/pipeline/T1_230201_PrL-NAc-G6-6F_HI-D1_FOV1_PRE-000")


def _build_s2_view(shape) -> ResidualView:
    """Rebuild the S₂ residual view: foundation substrate + Stage 1/2 layers."""
    view = ResidualView.from_foundation(
        OUTPUT_DIR / "suite2p" / "plane0" / "data.bin",
        OUTPUT_DIR / "svd_factors.npz",
        shape,
        k_background=30,
    )
    for n in (1, 2):
        sp = OUTPUT_DIR / f"residual_S{n}.sources.npz"
        if sp.exists():
            layer = SourceLayer.load(sp)
            if layer.traces.shape[0] > 0:
                view = view.with_source(layer.flat_idx, layer.W_design,
                                        layer.traces, stage_idx=n)
    return view


def main():
    meta = json.loads((OUTPUT_DIR / "residual_S.meta.json").read_text())
    shape = tuple(meta["shape"])
    T, H, W = shape
    print(f"[stage3-test] shape={shape}", flush=True)

    view = _build_s2_view(shape)

    fov = FOVData(
        raw_path=Path("dummy.tif"),
        output_dir=OUTPUT_DIR,
        data_bin_path=OUTPUT_DIR / "suite2p" / "plane0" / "data.bin",
        shape=shape,
        residual_view=view,
        mean_M=np.zeros((H, W), dtype=np.float32),
        mean_S=np.zeros((H, W), dtype=np.float32),
        max_S=np.zeros((H, W), dtype=np.float32),
        std_S=np.zeros((H, W), dtype=np.float32),
        vcorr_S=np.zeros((H, W), dtype=np.float32),
        dog_map=np.zeros((H, W), dtype=np.float32),
        mean_L=np.zeros((H, W), dtype=np.float32),
        k_background=30,
    )
    cfg = PipelineConfig(fs=30.0, tau=1.0)

    template_bank = build_template_bank(cfg.fs, cfg.tau)
    print(f"[stage3-test] template bank: {[n for n, _ in template_bank]}", flush=True)

    print("[stage3-test] running Stage 3 FFT matched filter ...", flush=True)
    t0 = time.time()
    candidates = run_stage3(view, fov, template_bank, cfg, starting_label_id=100)
    print(f"[stage3-test] Stage 3 produced {len(candidates)} candidates "
          f"in {time.time()-t0:.1f}s", flush=True)

    print("[stage3-test] running Gate 3 (no prior ROIs for this isolated test) ...",
          flush=True)
    t0 = time.time()
    gated = evaluate_gate3(candidates, [], template_bank, cfg)
    n_acc = sum(1 for r in gated if r.gate_outcome == "accept")
    n_flag = sum(1 for r in gated if r.gate_outcome == "flag")
    n_rej = sum(1 for r in gated if r.gate_outcome == "reject")
    print(f"[stage3-test] Gate 3: {len(gated)} candidates → "
          f"{n_acc} accept, {n_flag} flag, {n_rej} reject in {time.time()-t0:.1f}s",
          flush=True)

    # Subtract to produce S₃ (virtual — appends a source layer)
    s3_subtract = [r for r in gated if r.gate_outcome in ("accept", "flag")]
    if s3_subtract:
        print(f"[stage3-test] subtracting {len(s3_subtract)} ROIs → S₃", flush=True)
        t0 = time.time()
        std_S2 = compute_std_map(view, chunk=500)
        view_s3, val3, _ = run_source_subtraction(
            view, std_S2, s3_subtract, OUTPUT_DIR, cfg,
            stage_idx=3, output_name="residual_S3",
        )
        n_pass = sum(1 for v in val3.values() if v.get("pass"))
        print(f"[stage3-test] S₃ layer written: {n_pass}/{len(s3_subtract)} passed "
              f"in {time.time()-t0:.1f}s", flush=True)
    else:
        print("[stage3-test] no accept/flag ROIs → skipping subtraction", flush=True)

    print("[stage3-test] DONE", flush=True)


if __name__ == "__main__":
    main()
