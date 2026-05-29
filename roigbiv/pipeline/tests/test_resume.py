"""Tests for :mod:`roigbiv.pipeline.resume`.

Covers:
- ``compute_cfg_fingerprint`` stability + sensitivity to cfg / input changes
- Manifest read/write round-trip and step idempotency
- ROI reconstruction from report + masks TIFF
- Residual size verification
- ``plan_resume`` decisions across the resume points enumerated in the spec
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline import resume
from roigbiv.pipeline.resume import (
    MANIFEST_FILENAME,
    ResumeError,
    ResumePlan,
    compute_cfg_fingerprint,
    plan_resume,
    read_manifest,
    update_manifest,
)
from roigbiv.pipeline.types import PipelineConfig, ROI


# ──────────────────────────── fixtures ────────────────────────────────────


SHAPE_T_H_W = (2, 4, 4)


def _write_tif(path: Path, content: bytes = b"raw_tif_payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_foundation(output_dir: Path) -> None:
    """Materialize all artifacts ``run_foundation`` would write."""
    output_dir = Path(output_dir)
    (output_dir / "summary").mkdir(parents=True, exist_ok=True)
    (output_dir / "suite2p" / "plane0").mkdir(parents=True, exist_ok=True)

    T, H, W = SHAPE_T_H_W

    # Summary TIFs (any float32 array of the right (H,W) shape).
    for name in ("mean_M", "mean_S", "max_S", "std_S",
                 "vcorr_S", "mean_L", "dog_map"):
        arr = np.full((H, W), float(hash(name) & 0xFF), dtype=np.float32)
        tifffile.imwrite(str(output_dir / "summary" / f"{name}.tif"), arr)

    # data.bin: int16 (T, H, W) — only its size needs to be plausible.
    data_bin = output_dir / "suite2p" / "plane0" / "data.bin"
    np.zeros((T, H, W), dtype=np.int16).tofile(str(data_bin))
    # ops.npy: minimal dict; resume reads only the path, never the file.
    np.save(str(output_dir / "suite2p" / "plane0" / "ops.npy"),
            np.array({"Ly": H, "Lx": W}, dtype=object), allow_pickle=True)

    # residual_S.dat + meta
    np.zeros((T, H, W), dtype=np.float32).tofile(
        str(output_dir / "residual_S.dat")
    )
    (output_dir / "residual_S.meta.json").write_text(
        json.dumps({"shape": [T, H, W], "dtype": "float32"})
    )

    # motion trace
    np.savez(
        str(output_dir / "motion_trace.npz"),
        xoff=np.zeros(T, dtype=np.float32),
        yoff=np.zeros(T, dtype=np.float32),
        fs=np.float32(7.5),
    )


def _write_residual(output_dir: Path, stage_idx: int) -> None:
    T, H, W = SHAPE_T_H_W
    np.zeros((T, H, W), dtype=np.float32).tofile(
        str(output_dir / f"residual_S{stage_idx}.dat")
    )
    (output_dir / f"residual_S{stage_idx}.meta.json").write_text(
        json.dumps({"shape": [T, H, W], "dtype": "float32"})
    )


def _write_stage(
    output_dir: Path,
    stage_idx: int,
    *,
    accept_label_ids: list[int],
    flag_label_ids: list[int] | None = None,
    reject_label_ids: list[int] | None = None,
) -> None:
    flag_label_ids = flag_label_ids or []
    reject_label_ids = reject_label_ids or []
    T, H, W = SHAPE_T_H_W

    stage_dir = output_dir / f"stage{stage_idx}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    label_image = np.zeros((H, W), dtype=np.uint16)
    rois_serialized = []
    # Place each accept+flag ROI as a single distinct pixel so masks are
    # round-trippable via (label_image == label_id).
    pos = 0
    for lid in accept_label_ids + flag_label_ids:
        y, x = divmod(pos, W)
        if y >= H:
            raise RuntimeError("test fixture: too many ROIs for fixture FOV")
        label_image[y, x] = lid
        outcome = "accept" if lid in accept_label_ids else "flag"
        rois_serialized.append({
            "label_id": int(lid),
            "source_stage": int(stage_idx),
            "confidence": "high" if outcome == "accept" else "moderate",
            "gate_outcome": outcome,
            "area": 1,
            "solidity": 1.0,
            "eccentricity": 0.0,
            "nuclear_shadow_score": 0.0,
            "soma_surround_contrast": 0.0,
            "cellpose_prob": None,
            "iscell_prob": None,
            "event_count": None,
            "corr_contrast": None,
            "activity_type": None,
            "gate_reasons": [],
            "features": {},
        })
        pos += 1
    for lid in reject_label_ids:
        rois_serialized.append({
            "label_id": int(lid),
            "source_stage": int(stage_idx),
            "confidence": "requires_review",
            "gate_outcome": "reject",
            "area": 1,
            "solidity": 0.0,
            "eccentricity": 1.0,
            "nuclear_shadow_score": 0.0,
            "soma_surround_contrast": 0.0,
            "cellpose_prob": None,
            "iscell_prob": None,
            "event_count": None,
            "corr_contrast": None,
            "activity_type": None,
            "gate_reasons": ["test"],
            "features": {},
        })

    tifffile.imwrite(str(stage_dir / f"stage{stage_idx}_masks.tif"), label_image)
    (stage_dir / f"stage{stage_idx}_report.json").write_text(json.dumps({
        "detected": len(rois_serialized),
        "accepted": len(accept_label_ids),
        "flagged": len(flag_label_ids),
        "rejected": len(reject_label_ids),
        "rois": rois_serialized,
    }))


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    tif_path = _write_tif(tmp_path / "input.tif")
    return {"output_dir": output_dir, "tif_path": tif_path}


# ─────────────────────── fingerprint tests ─────────────────────────────────


def test_fingerprint_is_stable(workspace):
    cfg = PipelineConfig(fs=7.5)
    fp1 = compute_cfg_fingerprint(cfg, workspace["tif_path"])
    fp2 = compute_cfg_fingerprint(cfg, workspace["tif_path"])
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_fingerprint_changes_on_cfg_change(workspace):
    cfg_a = PipelineConfig(fs=7.5)
    cfg_b = PipelineConfig(fs=30.0)
    fp_a = compute_cfg_fingerprint(cfg_a, workspace["tif_path"])
    fp_b = compute_cfg_fingerprint(cfg_b, workspace["tif_path"])
    assert fp_a != fp_b


def test_fingerprint_changes_on_input_size_change(workspace, tmp_path):
    cfg = PipelineConfig(fs=7.5)
    fp_a = compute_cfg_fingerprint(cfg, workspace["tif_path"])
    workspace["tif_path"].write_bytes(b"different_payload_with_different_size")
    fp_b = compute_cfg_fingerprint(cfg, workspace["tif_path"])
    assert fp_a != fp_b


def test_fingerprint_ignores_enable_stage_flags(workspace):
    """Toggling stage opt-in flags must NOT invalidate the fingerprint —
    the whole point is to enable a stage on a workspace produced by an
    earlier run that had it disabled."""
    cfg_off = PipelineConfig(fs=7.5, enable_stage_3=False, enable_stage_4=False)
    cfg_on = PipelineConfig(fs=7.5, enable_stage_3=True, enable_stage_4=True)
    fp_off = compute_cfg_fingerprint(cfg_off, workspace["tif_path"])
    fp_on = compute_cfg_fingerprint(cfg_on, workspace["tif_path"])
    assert fp_off == fp_on


def test_fingerprint_still_changes_for_parameter_knobs(workspace):
    """Per-stage *parameter* changes (not on/off flags) must invalidate."""
    cfg_a = PipelineConfig(fs=7.5, cellprob_threshold=-2.0)
    cfg_b = PipelineConfig(fs=7.5, cellprob_threshold=0.0)
    fp_a = compute_cfg_fingerprint(cfg_a, workspace["tif_path"])
    fp_b = compute_cfg_fingerprint(cfg_b, workspace["tif_path"])
    assert fp_a != fp_b


# ─────────────────────── manifest IO tests ─────────────────────────────────


def test_manifest_round_trip(workspace):
    cfg = PipelineConfig(fs=7.5)
    update_manifest(workspace["output_dir"], "foundation", cfg,
                    workspace["tif_path"])
    m = read_manifest(workspace["output_dir"])
    assert m is not None
    assert m["cfg_fingerprint"].startswith("sha256:")
    assert "foundation" in m["stages"]
    assert m["input_tif"] == str(workspace["tif_path"].resolve())


def test_manifest_step_idempotent(workspace):
    cfg = PipelineConfig(fs=7.5)
    update_manifest(workspace["output_dir"], "stage1", cfg,
                    workspace["tif_path"])
    first = read_manifest(workspace["output_dir"])["stages"]["stage1"]
    update_manifest(workspace["output_dir"], "stage1", cfg,
                    workspace["tif_path"])
    second = read_manifest(workspace["output_dir"])["stages"]["stage1"]
    # Both calls succeed; timestamp may refresh but the schema is stable.
    assert set(first.keys()) == set(second.keys()) == {
        "completed_at", "version", "status",
    }
    assert first["status"] == second["status"] == "completed"


def test_manifest_rejects_unknown_step(workspace):
    cfg = PipelineConfig(fs=7.5)
    with pytest.raises(ValueError, match="unknown manifest step"):
        update_manifest(workspace["output_dir"], "stage99", cfg,
                        workspace["tif_path"])


def test_manifest_rejects_unknown_status(workspace):
    cfg = PipelineConfig(fs=7.5)
    with pytest.raises(ValueError, match="status must be"):
        update_manifest(workspace["output_dir"], "stage1", cfg,
                        workspace["tif_path"], status="nope")


def test_manifest_skipped_status_persists(workspace):
    cfg = PipelineConfig(fs=7.5)
    update_manifest(workspace["output_dir"], "stage3", cfg,
                    workspace["tif_path"], status="skipped")
    entry = read_manifest(workspace["output_dir"])["stages"]["stage3"]
    assert entry["status"] == "skipped"


def test_read_manifest_returns_none_when_absent(workspace):
    assert read_manifest(workspace["output_dir"]) is None


# ───────────────── ROI reconstruction + residual checks ────────────────────


def test_load_rois_from_report_drops_rejects(workspace):
    _write_stage(
        workspace["output_dir"], 1,
        accept_label_ids=[1, 2],
        flag_label_ids=[3],
        reject_label_ids=[99, 100],
    )
    rois = resume._load_rois_from_report(
        workspace["output_dir"] / "stage1" / "stage1_report.json",
        workspace["output_dir"] / "stage1" / "stage1_masks.tif",
    )
    assert sorted(r.label_id for r in rois) == [1, 2, 3]
    assert all(r.gate_outcome != "reject" for r in rois)
    # Each accept/flag ROI's mask is recovered from the TIFF.
    assert all(r.mask.any() for r in rois)


def test_load_rois_raises_when_mask_missing_for_accept(workspace):
    """An accept/flag ROI whose label_id is not present in the masks TIFF
    indicates inconsistent state — must surface, not silently skip."""
    _write_stage(workspace["output_dir"], 1,
                 accept_label_ids=[1, 2])
    # Truncate the mask TIFF so label_id=2 disappears.
    masks_path = workspace["output_dir"] / "stage1" / "stage1_masks.tif"
    H, W = SHAPE_T_H_W[1], SHAPE_T_H_W[2]
    only_one = np.zeros((H, W), dtype=np.uint16)
    only_one[0, 0] = 1
    tifffile.imwrite(str(masks_path), only_one)
    with pytest.raises(ResumeError, match="label_id=2"):
        resume._load_rois_from_report(
            workspace["output_dir"] / "stage1" / "stage1_report.json",
            masks_path,
        )


def test_verify_residual_detects_truncation(workspace):
    output_dir = workspace["output_dir"]
    _write_residual(output_dir, 1)
    # Truncate the dat by 8 bytes; meta still claims full shape.
    dat = output_dir / "residual_S1.dat"
    dat.write_bytes(dat.read_bytes()[:-8])
    with pytest.raises(ResumeError, match="truncated"):
        resume._verify_residual(dat, output_dir / "residual_S1.meta.json")


# ─────────────────────────── plan_resume ──────────────────────────────────


def test_plan_resume_disabled_returns_foundation(workspace):
    cfg = PipelineConfig(fs=7.5)
    plan = plan_resume(workspace["output_dir"], workspace["tif_path"], cfg,
                       enable=False)
    assert plan.start_stage == "foundation"
    assert plan.enabled is False
    assert plan.fov is None


def test_plan_resume_no_manifest_starts_fresh(workspace):
    cfg = PipelineConfig(fs=7.5)
    plan = plan_resume(workspace["output_dir"], workspace["tif_path"], cfg,
                       enable=True)
    assert plan.start_stage == "foundation"
    assert plan.enabled is True
    assert plan.fov is None


def test_plan_resume_after_foundation_only(workspace):
    cfg = PipelineConfig(fs=7.5)
    _make_foundation(workspace["output_dir"])
    update_manifest(workspace["output_dir"], "foundation", cfg,
                    workspace["tif_path"])
    plan = plan_resume(workspace["output_dir"], workspace["tif_path"], cfg,
                       enable=True)
    assert plan.start_stage == "stage1"
    assert plan.fov is not None
    assert plan.fov.mean_M.shape == SHAPE_T_H_W[1:]
    assert plan.fov.shape == SHAPE_T_H_W


def test_plan_resume_after_stage1_with_residual_S1(workspace):
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    _write_stage(out, 1, accept_label_ids=[1], flag_label_ids=[2])
    _write_residual(out, 1)
    update_manifest(out, "stage1", cfg, workspace["tif_path"])
    update_manifest(out, "stage1_subtract", cfg, workspace["tif_path"])
    plan = plan_resume(out, workspace["tif_path"], cfg, enable=True)
    assert plan.start_stage == "stage2"
    assert plan.fov is not None
    assert plan.fov.residual_S1_path is not None
    assert sorted(r.label_id for r in plan.fov.rois) == [1, 2]
    assert plan.prior_reports[1]["accepted"] == 1


def test_plan_resume_replays_subtraction_when_only_detection_done(workspace):
    """Stage 1 detect ran (report exists), but subtraction died before
    writing residual_S1.dat. residual_S.dat is still present → resume from
    stage1_subtract."""
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    _write_stage(out, 1, accept_label_ids=[1])
    update_manifest(out, "stage1", cfg, workspace["tif_path"])
    # No residual_S1.dat yet.
    plan = plan_resume(out, workspace["tif_path"], cfg, enable=True)
    assert plan.start_stage == "stage1_subtract"
    assert plan.fov is not None


def test_plan_resume_advances_past_skipped_subtraction(workspace):
    """Stage 2 detect ran but Stage 2's subtraction was intentionally skipped
    (Stage 3/4 disabled in the prior run, so no consumer needed S₂). Manifest
    records stage2_subtract as status='skipped' and residual_S2.dat is absent.
    plan_resume must NOT refuse; it advances to Stage 3 with no S₂ residual,
    and run.py's _stage_input_residual will fall back to S₁."""
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    _write_stage(out, 1, accept_label_ids=[1])
    _write_residual(out, 1)
    _write_stage(out, 2, accept_label_ids=[10])
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    update_manifest(out, "stage1", cfg, workspace["tif_path"])
    update_manifest(out, "stage1_subtract", cfg, workspace["tif_path"])
    update_manifest(out, "stage2", cfg, workspace["tif_path"])
    update_manifest(out, "stage2_subtract", cfg, workspace["tif_path"],
                    status="skipped")
    plan = plan_resume(out, workspace["tif_path"], cfg, enable=True)
    assert plan.start_stage == "stage3"
    # residual_S2_path stayed None — Stage 3's input fallback handles it.
    assert plan.fov is not None
    assert plan.fov.residual_S2_path is None
    assert plan.fov.residual_S1_path == out / "residual_S1.dat"


def test_plan_resume_advances_past_skipped_detect(workspace):
    """Stage 3 detect was intentionally skipped on a prior run (enable_stage_3
    was False). No stage3_report.json exists but the manifest records it as
    skipped. plan_resume must advance to Stage 3 (which run.py will re-skip
    or run, depending on the new config) instead of refusing."""
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    _write_stage(out, 1, accept_label_ids=[1])
    _write_residual(out, 1)
    _write_stage(out, 2, accept_label_ids=[10])
    _write_residual(out, 2)
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    update_manifest(out, "stage1", cfg, workspace["tif_path"])
    update_manifest(out, "stage1_subtract", cfg, workspace["tif_path"])
    update_manifest(out, "stage2", cfg, workspace["tif_path"])
    update_manifest(out, "stage2_subtract", cfg, workspace["tif_path"])
    update_manifest(out, "stage3", cfg, workspace["tif_path"], status="skipped")
    plan = plan_resume(out, workspace["tif_path"], cfg, enable=True)
    assert plan.start_stage == "stage3"


def test_plan_resume_refuses_when_manifest_claims_subtract_done_but_residual_gone(workspace):
    """Manifest says stage1_subtract completed, but residual_S1.dat is gone.
    Something (user, cleanup script) deleted the residual after the fact —
    do not silently re-run subtraction; refuse and surface the inconsistency."""
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    _write_stage(out, 1, accept_label_ids=[1])
    _write_residual(out, 1)
    _write_stage(out, 2, accept_label_ids=[10])
    (out / "residual_S1.dat").unlink()
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    update_manifest(out, "stage1", cfg, workspace["tif_path"])
    update_manifest(out, "stage1_subtract", cfg, workspace["tif_path"])
    update_manifest(out, "stage2", cfg, workspace["tif_path"])
    with pytest.raises(ResumeError, match="residual_S1.dat"):
        plan_resume(out, workspace["tif_path"], cfg, enable=True)


def test_plan_resume_refuses_when_manifest_claims_foundation_but_artifacts_gone(workspace):
    """Manifest claims foundation completed but a Foundation artifact is
    missing — refuse rather than silently re-run Foundation."""
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    (out / "residual_S.dat").unlink()
    with pytest.raises(ResumeError, match="foundation"):
        plan_resume(out, workspace["tif_path"], cfg, enable=True)


def test_plan_resume_all_stages_complete(workspace):
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    _write_stage(out, 1, accept_label_ids=[1])
    _write_residual(out, 1)
    _write_stage(out, 2, accept_label_ids=[2])
    _write_residual(out, 2)
    _write_stage(out, 3, accept_label_ids=[3])
    _write_residual(out, 3)
    _write_stage(out, 4, accept_label_ids=[], flag_label_ids=[4])
    for step in ("stage1", "stage1_subtract", "stage2", "stage2_subtract",
                 "stage3", "stage3_subtract", "stage4"):
        update_manifest(out, step, cfg, workspace["tif_path"])
    plan = plan_resume(out, workspace["tif_path"], cfg, enable=True)
    assert plan.start_stage == "post_detection"
    assert sorted(r.label_id for r in plan.fov.rois) == [1, 2, 3, 4]
    assert plan.fov.residual_S3_path is not None


def test_plan_resume_refuses_on_fingerprint_mismatch(workspace):
    cfg_old = PipelineConfig(fs=7.5)
    cfg_new = PipelineConfig(fs=30.0)
    out = workspace["output_dir"]
    _make_foundation(out)
    update_manifest(out, "foundation", cfg_old, workspace["tif_path"])
    with pytest.raises(ResumeError, match=r"fs:.*7\.5.*30"):
        plan_resume(out, workspace["tif_path"], cfg_new, enable=True)


def test_plan_resume_refuses_on_input_change(workspace):
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    # Mutate the input file's content (changes size).
    workspace["tif_path"].write_bytes(b"a" * 4096)
    with pytest.raises(ResumeError, match="input_stat"):
        plan_resume(out, workspace["tif_path"], cfg, enable=True)


def test_plan_resume_missing_masks_tif_raises(workspace):
    cfg = PipelineConfig(fs=7.5)
    out = workspace["output_dir"]
    _make_foundation(out)
    update_manifest(out, "foundation", cfg, workspace["tif_path"])
    _write_stage(out, 1, accept_label_ids=[1])
    (out / "stage1" / "stage1_masks.tif").unlink()
    update_manifest(out, "stage1", cfg, workspace["tif_path"])
    with pytest.raises(ResumeError, match="masks.tif"):
        plan_resume(out, workspace["tif_path"], cfg, enable=True)


# ─────────────────────── ResumePlan.should_run ────────────────────────────


_ALL_STEPS = (
    "foundation", "stage1", "stage1_subtract",
    "stage2", "stage2_subtract", "stage3",
    "stage3_subtract", "stage4", "post_detection",
)


@pytest.mark.parametrize("start_stage,first_to_run", [
    ("foundation", "foundation"),
    ("stage1", "stage1"),
    ("stage1_subtract", "stage1_subtract"),
    ("stage2", "stage2"),
    ("stage3", "stage3"),
    ("stage4", "stage4"),
    ("post_detection", "post_detection"),
])
def test_should_run_skips_only_steps_before_start(start_stage, first_to_run):
    plan = ResumePlan(start_stage=start_stage, enabled=True)
    started = False
    for step in _ALL_STEPS:
        if step == first_to_run:
            started = True
        if started:
            assert plan.should_run(step), \
                f"start={start_stage}: should run {step}"
        else:
            assert not plan.should_run(step), \
                f"start={start_stage}: should NOT run {step}"


def test_should_run_with_disabled_plan_runs_everything():
    """When ``enabled=False`` (the default), start_stage is 'foundation'
    so should_run returns True for every step — full pipeline execution."""
    plan = ResumePlan()
    for step in _ALL_STEPS:
        assert plan.should_run(step) is True
