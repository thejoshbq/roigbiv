"""
Contract tests for the --foundation-only dry-run mode.

Run via:
    conda run -n roigbiv python -m roigbiv.pipeline.tests.test_run_modes

Covers:
  - run_pipeline short-circuits after Foundation (sentinel written, no stage
    dirs / ROI artifacts) when cfg.foundation_only is set
  - the UI/pipeline loader serves a summary-only FOVData for a dry-run dir
    instead of raising on the missing pipeline_log.json / merged_masks.tif
  - the resume fingerprint ignores foundation_only, so a dry run then a
    --resume run continues from Stage 1 without a mismatch
  - argparse rejects --foundation-only combined with --scout/--resume/--no-stage-N
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile


def _fake_fov(output_dir: Path, shape=(4, 16, 16)):
    from roigbiv.pipeline.types import FOVData
    return FOVData(
        raw_path=output_dir / "fov.tif",
        output_dir=output_dir,
        data_bin_path=output_dir / "suite2p" / "plane0" / "data.bin",
        shape=shape,
        mean_M=np.zeros(shape[1:], np.float32),
        vcorr_S=np.zeros(shape[1:], np.float32),
        k_background=30,
        rois=[],
    )


def test_foundation_only_short_circuit():
    """run_pipeline stops after Foundation: sentinel written, no detection."""
    from roigbiv.pipeline.run import run_pipeline
    from roigbiv.pipeline.types import PipelineConfig

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tif = td / "fov.tif"
        tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))
        out = td / "out"

        cfg = PipelineConfig(fs=7.5, foundation_only=True, no_viewer=True,
                             output_dir=out, force_cpu=True)

        def fake_run_foundation(tif_path, cfg, output_dir, gpu_lock=None):
            (Path(output_dir) / "summary").mkdir(parents=True, exist_ok=True)
            return _fake_fov(Path(output_dir))

        with patch("roigbiv.pipeline.foundation.run_foundation",
                   side_effect=fake_run_foundation):
            fov = run_pipeline(tif, cfg)

        sentinel = out / "foundation_only.json"
        assert sentinel.exists(), "foundation_only.json sentinel not written"
        meta = json.loads(sentinel.read_text())
        assert meta["mode"] == "foundation_only", meta
        for d in ("stage1", "stage2", "stage3", "stage4"):
            assert not (out / d).exists(), f"{d}/ created during a dry run"
        assert not (out / "merged_masks.tif").exists()
        assert not (out / "roi_metadata.json").exists()
        assert not (out / "pipeline_log.json").exists()
        assert fov.rois == []
    print("  [PASS] test_foundation_only_short_circuit "
          "(sentinel written, no stage dirs / ROI artifacts)")


def test_foundation_only_loader_guard():
    """The loader serves a summary-only FOVData for a dry-run dir, no crash."""
    from roigbiv.pipeline.loaders import load_fov_from_output_dir

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        (out / "summary").mkdir()
        mean_M = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        tifffile.imwrite(str(out / "summary" / "mean_M.tif"), mean_M)
        tifffile.imwrite(str(out / "summary" / "vcorr_S.tif"),
                         np.zeros((16, 16), np.float32))
        (out / "foundation_only.json").write_text(json.dumps(
            {"mode": "foundation_only", "shape": [4, 16, 16]}))

        # No pipeline_log.json / roi_metadata.json / merged_masks.tif exist.
        fov, review_queue = load_fov_from_output_dir(out)
        assert fov.mean_M is not None and fov.mean_M.shape == (16, 16)
        assert fov.rois == []
        assert review_queue == []
    print("  [PASS] test_foundation_only_loader_guard "
          "(summary-only FOVData, no FileNotFoundError)")


def test_foundation_only_fingerprint_stable():
    """Resume fingerprint ignores foundation_only → dry run then continue works."""
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    from roigbiv.pipeline.types import PipelineConfig

    with tempfile.TemporaryDirectory() as td:
        tif = Path(td) / "fov.tif"
        tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))

        dry = PipelineConfig(fs=7.5, foundation_only=True)
        cont = PipelineConfig(fs=7.5, foundation_only=False)
        assert compute_cfg_fingerprint(dry, tif) == compute_cfg_fingerprint(cont, tif), (
            "foundation_only must be excluded from the resume fingerprint so a "
            "dry run can be continued with --resume")

        # A genuine MC parameter change must still invalidate.
        changed = PipelineConfig(fs=7.5, mc_strip_height=48)
        assert compute_cfg_fingerprint(cont, tif) != compute_cfg_fingerprint(changed, tif), (
            "mc_strip_height change should invalidate the resume fingerprint")
    print("  [PASS] test_foundation_only_fingerprint_stable "
          "(dry-run continue works; real MC change still invalidates)")


def test_foundation_only_rejects_bad_combos():
    """argparse rejects --foundation-only with --scout/--resume/--no-stage-N."""
    from roigbiv.pipeline import run as run_mod

    bad = [
        ["--input", "/nonexistent", "--fs", "7.5", "--foundation-only", "--scout"],
        ["--input", "/nonexistent", "--fs", "7.5", "--foundation-only", "--resume"],
        ["--input", "/nonexistent", "--fs", "7.5", "--foundation-only", "--no-stage-2"],
    ]
    for argv in bad:
        raised = False
        try:
            run_mod.main(argv)
        except SystemExit as e:
            raised = e.code != 0
        assert raised, f"expected argparse error for: {argv}"
    print("  [PASS] test_foundation_only_rejects_bad_combos "
          "(scout / resume / stage-toggle combos rejected)")


def test_centroids_only_rejects_single_file_input():
    """--centroids without --foundation-only requires a directory --input
    (workspace mode) so centroids-only can resolve a prior {stem}_mc.tif —
    see roigbiv/pipeline/workspace.py::_run_centroids_only."""
    from roigbiv.pipeline import run as run_mod

    with tempfile.TemporaryDirectory() as td:
        tif = Path(td) / "fov.tif"
        tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))

        code = run_mod.main(["--input", str(tif), "--fs", "7.5", "--centroids"])
        assert code == 2, (
            "expected exit 2 for --centroids without --foundation-only on a "
            f"single-file --input, got {code}")
    print("  [PASS] test_centroids_only_rejects_single_file_input "
          "(single-file --input + --centroids alone is rejected)")


def test_centroids_with_foundation_only_reaches_run_single():
    """--centroids + --foundation-only ('both' mode) is a valid combo even for
    a single-file --input — it must dispatch to _run_single, not be rejected."""
    from unittest.mock import patch

    from roigbiv.pipeline import run as run_mod

    with tempfile.TemporaryDirectory() as td:
        tif = Path(td) / "fov.tif"
        tifffile.imwrite(str(tif), np.zeros((4, 16, 16), np.uint16))

        with patch.object(run_mod, "_run_single", return_value=0) as mock_single:
            code = run_mod.main([
                "--input", str(tif), "--fs", "7.5",
                "--centroids", "--foundation-only",
            ])
        assert mock_single.call_count == 1, (
            "expected --centroids --foundation-only to reach _run_single")
        assert code == 0
    print("  [PASS] test_centroids_with_foundation_only_reaches_run_single "
          "('both' mode on a single file is accepted, not rejected)")


if __name__ == "__main__":
    import traceback

    tests = [
        test_foundation_only_short_circuit,
        test_foundation_only_loader_guard,
        test_foundation_only_fingerprint_stable,
        test_foundation_only_rejects_bad_combos,
        test_centroids_only_rejects_single_file_input,
        test_centroids_with_foundation_only_reaches_run_single,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            traceback.print_exc()
            failed.append(test.__name__)
    print()
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")
