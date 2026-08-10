"""
Contract tests for the standalone centroid-discovery run modes wired into
:mod:`roigbiv.pipeline.workspace` (PipelineConfig.run_centroids /
foundation_only composability — see roigbiv/pipeline/centroids.py and the
truth table in workspace.py::run_with_workspace).

Suite2p and the full cascade (run_pipeline) are mocked out — these are
orchestration-contract tests, not pipeline-correctness tests (those live in
test_centroids.py and the existing foundation/stage tests).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.workspace import resolve_workspace, run_with_workspace
from roigbiv.registry.config import RegistryConfig


def _make_cfg(tmp_path: Path) -> RegistryConfig:
    return RegistryConfig(
        dsn=f"sqlite:///{tmp_path / 'registry.db'}",
        blob_backend="local",
        blob_root=tmp_path / "blobs",
        endpoint=None,
        api_key=None,
        calibration_path=tmp_path / "calibration.json",
    )


def _write_tif(path: Path, T: int = 6, H: int = 16, W: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((T, H, W), dtype=np.uint16))


def test_centroids_only_fails_fast_without_mc_on_disk(tmp_path):
    """No _mc-suffixed input and no prior {stem}_mc.tif → per-FOV error, not a crash."""
    _write_tif(tmp_path / "raw.tif")  # not _mc-suffixed, no prior output
    workspace = resolve_workspace(tmp_path)
    cfg = _make_cfg(tmp_path)

    with patch("roigbiv.pipeline.centroids.run_centroid_discovery") as mock_discover:
        results = run_with_workspace(
            workspace, {"run_centroids": True, "foundation_only": False},
            registry_config=cfg, skip_registry=True, skip_backfill=True,
        )

    mock_discover.assert_not_called()
    assert len(results) == 1
    assert results[0].error is not None
    assert "motion-corrected" in results[0].error
    assert results[0].centroid_count is None


def test_centroids_only_runs_on_precorrected_input(tmp_path):
    """An _mc-suffixed input is used directly — no Foundation/run_pipeline call."""
    tif = tmp_path / "fovA_mc.tif"
    _write_tif(tif)
    workspace = resolve_workspace(tmp_path)
    cfg = _make_cfg(tmp_path)

    from roigbiv.pipeline.centroids import CentroidResult

    with patch("roigbiv.pipeline.centroids.run_centroid_discovery") as mock_discover, \
         patch("roigbiv.pipeline.run.run_pipeline") as mock_pipeline:
        mock_discover.return_value = CentroidResult(
            output_path=workspace.output_root / "fovA" / "centroids.json", count=5)
        results = run_with_workspace(
            workspace, {"run_centroids": True, "foundation_only": False},
            registry_config=cfg, skip_registry=True, skip_backfill=True,
        )

    mock_pipeline.assert_not_called()
    mock_discover.assert_called_once()
    called_mc_tif = mock_discover.call_args[0][0]
    assert Path(called_mc_tif) == tif.resolve()
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].centroid_count == 5


def test_both_mode_chains_centroids_after_foundation(tmp_path):
    """foundation_only=True + run_centroids=True: run_pipeline then centroids
    on the {stem}_mc.tif it wrote, with centroid_count on the result."""
    tif = tmp_path / "fovA.tif"
    _write_tif(tif)
    workspace = resolve_workspace(tmp_path)
    cfg = _make_cfg(tmp_path)

    from roigbiv.pipeline.centroids import CentroidResult
    from roigbiv.pipeline.types import FOVData

    def fake_run_pipeline(tif_path, pcfg, **kwargs):
        out_dir = Path(pcfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "fovA_mc.tif").touch()
        return FOVData(
            raw_path=tif_path, output_dir=out_dir,
            data_bin_path=out_dir / "suite2p" / "plane0" / "data.bin",
            shape=(6, 16, 16),
            mean_M=np.zeros((16, 16), np.float32),
            vcorr_S=np.zeros((16, 16), np.float32),
            k_background=30, rois=[],
        )

    with patch("roigbiv.pipeline.run.run_pipeline",
               side_effect=fake_run_pipeline) as mock_pipeline, \
         patch("roigbiv.pipeline.centroids.run_centroid_discovery") as mock_discover:
        mock_discover.return_value = CentroidResult(
            output_path=workspace.output_root / "fovA" / "centroids.json", count=3)
        results = run_with_workspace(
            workspace, {"run_centroids": True, "foundation_only": True},
            registry_config=cfg, skip_registry=True, skip_backfill=True,
        )

    assert mock_pipeline.call_count == 1
    assert mock_discover.call_count == 1
    called_mc_tif = mock_discover.call_args[0][0]
    assert called_mc_tif.name == "fovA_mc.tif"
    assert len(results) == 1
    assert results[0].centroid_count == 3


def test_centroids_only_forces_sequential_even_with_n_workers(tmp_path):
    """Centroids-only is lightweight (no GPU registration) — always sequential,
    regardless of --n-workers; _run_parallel must never be invoked."""
    for name in ("a_mc.tif", "b_mc.tif"):
        _write_tif(tmp_path / name)
    workspace = resolve_workspace(tmp_path)
    cfg = _make_cfg(tmp_path)

    import roigbiv.pipeline.workspace as wsmod

    def _boom(*a, **k):
        raise AssertionError("_run_parallel must not be called for centroids-only")

    from roigbiv.pipeline.centroids import CentroidResult

    with patch.object(wsmod, "_run_parallel", side_effect=_boom), \
         patch("roigbiv.pipeline.centroids.run_centroid_discovery") as mock_discover:
        mock_discover.return_value = CentroidResult(
            output_path=tmp_path / "centroids.json", count=0)
        results = run_with_workspace(
            workspace, {"run_centroids": True, "foundation_only": False},
            registry_config=cfg, skip_registry=True, skip_backfill=True,
            n_workers=2,
        )

    assert len(results) == 2
    assert mock_discover.call_count == 2
