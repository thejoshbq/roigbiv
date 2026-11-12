"""Wiring test: drives ``scripts/bench/cli.py`` end to end on the synthetic benchmark.

Red until U1 (manifest), U2 (score) and U3 (split_audit) have all landed.
Expected values are hand-computed from ``synthetic.py``: every ROI is an
81-pixel square, so the match distance is ``0.5 * 2 * sqrt(81 / pi)``.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.tests.synthetic import centers, write_synthetic_benchmark  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CLI = REPO / "scripts" / "bench" / "cli.py"
MAX_D = math.sqrt(81 / math.pi)


def run_raw(*args, tmp_path):
    out = tmp_path / f"out_{args[0]}.json"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ROIGBIV_BENCH_ROOT")}
    proc = subprocess.run([sys.executable, str(CLI), *args, "--out", str(out)],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    return proc, (json.loads(out.read_text()) if out.exists() else None)


def run_cli(*args, tmp_path):
    proc, result = run_raw(*args, tmp_path=tmp_path)
    assert proc.returncode == 0, proc.stderr
    return result


def edit_manifest(root, fn):
    p = root / "manifest.json"
    m = json.loads(p.read_text())
    fn(m)
    p.write_text(json.dumps(m))


@pytest.fixture
def root(tmp_path):
    return write_synthetic_benchmark(tmp_path / "bench")


@pytest.fixture
def pred_dir(tmp_path):
    d = tmp_path / "pred"
    d.mkdir()
    np.save(d / "syn-a1.npy", centers("syn-a1"))                                     # 4 of 4
    a2 = centers("syn-a2/r1").copy()
    a2[3] = (60.0, 2.0)                                                               # 3 of 4, 1 FP
    np.save(d / "syn-a2.npy", a2)
    np.save(d / "syn-a3.npy", centers("syn-a3") + np.float32(1.0))                   # 3 of 3, shifted
    np.save(d / "syn-a4.npy", centers("syn-a4"))                                     # flagged
    return d


def test_score(root, pred_dir, tmp_path):
    r = run_cli("score", "--root", str(root), "--pred-dir", str(pred_dir), tmp_path=tmp_path)
    assert r["version"] == "syn0" and r["missing_predictions"] == []
    assert [f["fov_id"] for f in r["per_fov"]] == ["syn-a1", "syn-a2", "syn-a3"]
    for f in r["per_fov"]:
        assert f["max_distance"] == pytest.approx(MAX_D)
    assert [(f["n_gt"], f["n_pred"], f["n_tp"]) for f in r["per_fov"]] == [(4, 4, 4), (4, 4, 3), (3, 3, 3)]
    grin, prism = r["by_lens"]["GRIN"], r["by_lens"]["prism"]
    assert (grin["n_fovs"], grin["n_gt"], grin["n_pred"], grin["n_tp"]) == (2, 8, 8, 7)
    assert grin["recall"] == pytest.approx(0.875) and grin["macro_recall"] == pytest.approx(0.875)
    assert (prism["n_fovs"], prism["n_tp"], prism["recall"]) == (1, 3, 1.0)
    assert (r["all"]["n_gt"], r["all"]["n_tp"]) == (11, 10)


def test_score_filters(root, pred_dir, tmp_path):
    (pred_dir / "syn-a2.npy").unlink()
    r = run_cli("score", "--root", str(root), "--pred-dir", str(pred_dir), "--split", "dev",
                "--include-flagged", "--allow-missing", tmp_path=tmp_path)
    assert [f["fov_id"] for f in r["per_fov"]] == ["syn-a1", "syn-a4"]
    assert r["missing_predictions"] == ["syn-a2"]
    assert r["split"] == "dev" and r["include_flagged"] is True


def test_score_records_settings(root, pred_dir, tmp_path):
    r = run_cli("score", "--root", str(root), "--pred-dir", str(pred_dir), tmp_path=tmp_path)
    assert r["split"] is None and r["include_flagged"] is False and r["verified"] is True
    assert r["annotation_count_mismatches"] == []
    assert [f["n_rois_manifest"] for f in r["per_fov"]] == [4, 4, 3]


def test_missing_prediction_fails_unless_allowed(root, pred_dir, tmp_path):
    (pred_dir / "syn-a3.npy").unlink()
    proc, r = run_raw("score", "--root", str(root), "--pred-dir", str(pred_dir), tmp_path=tmp_path)
    assert proc.returncode == 1 and "syn-a3" in proc.stderr
    assert r["missing_predictions"] == ["syn-a3"]


def test_bad_prediction_shape(root, pred_dir, tmp_path):
    np.save(pred_dir / "syn-a1.npy", np.zeros((2, 3), dtype=np.float32))
    proc, _ = run_raw("score", "--root", str(root), "--pred-dir", str(pred_dir), tmp_path=tmp_path)
    assert proc.returncode == 2 and "syn-a1" in proc.stderr


def test_empty_prediction_is_allowed(root, pred_dir, tmp_path):
    np.save(pred_dir / "syn-a1.npy", np.zeros((0, 2), dtype=np.float32))
    r = run_cli("score", "--root", str(root), "--pred-dir", str(pred_dir), tmp_path=tmp_path)
    assert (r["per_fov"][0]["n_pred"], r["per_fov"][0]["n_tp"]) == (0, 0)


def test_score_verifies_hashes_by_default(root, pred_dir, tmp_path):
    zp = root / "fovs/syn-a3/annotation_a.zip"
    zp.write_bytes(zp.read_bytes() + b"\0")
    proc, _ = run_raw("score", "--root", str(root), "--pred-dir", str(pred_dir), tmp_path=tmp_path)
    assert proc.returncode == 2 and "sha256" in proc.stderr
    r = run_cli("score", "--root", str(root), "--pred-dir", str(pred_dir), "--no-verify", tmp_path=tmp_path)
    assert r["verified"] is False


def test_annotation_count_mismatch_is_reported(root, pred_dir, tmp_path):
    edit_manifest(root, lambda m: m["fovs"][0]["annotations"][0].update(n_rois=5))
    r = run_cli("score", "--root", str(root), "--pred-dir", str(pred_dir), tmp_path=tmp_path)
    assert r["annotation_count_mismatches"] == [
        {"fov_id": "syn-a1", "index": 0, "n_rois_manifest": 5, "n_parsed": 4}]


def test_audit(root, tmp_path):
    r = run_cli("audit", "--root", str(root), tmp_path=tmp_path)
    a = r["split_audit"]
    assert a["fovs_per_split"] == {"dev": 3, "test": 1} and a["clean"] is True
    assert (r["g0_core"]["n_fovs"], r["g0_core"]["n_animals"], r["g0_core"]["n_movies"]) == (3, 3, 2)
    assert r["g0_core"]["passed"] is False
    assert r["g0_incl_flagged"]["n_fovs"] == 4
    r = run_cli("audit", "--root", str(root), "--min-fovs", "3", "--min-movies", "2", tmp_path=tmp_path)
    assert r["g0_core"]["passed"] is True


def test_inter_rater(root, tmp_path):
    r = run_cli("inter-rater", "--root", str(root), tmp_path=tmp_path)
    assert list(r["per_fov"]) == ["syn-a2"]
    (pair,) = r["per_fov"]["syn-a2"]["pairs"]
    assert (pair["rater_a"], pair["rater_b"], pair["n_tp"]) == ("r1", "r2", 3)
    assert pair["f1"] == pytest.approx(0.75) and r["ceiling"] == pytest.approx(0.75)
    assert r["split"] is None and r["include_flagged"] is False and r["verified"] is True


def test_inter_rater_filters_like_score(root, tmp_path):
    r = run_cli("inter-rater", "--root", str(root), "--split", "test", tmp_path=tmp_path)
    assert r["per_fov"] == {} and r["ceiling"] is None and r["split"] == "test"


def test_inter_rater_same_rater_name_kept_apart(root, tmp_path):
    edit_manifest(root, lambda m: m["fovs"][1]["annotations"][1].update(rater="r1"))
    r = run_cli("inter-rater", "--root", str(root), "--no-verify", tmp_path=tmp_path)
    (pair,) = r["per_fov"]["syn-a2"]["pairs"]
    assert (pair["rater_a"], pair["rater_b"], pair["n_tp"]) == ("r1", "r1#1", 3)


def test_bad_root_exits_2(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, str(CLI), "audit", "--root", str(tmp_path / "none")],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "bench:" in proc.stderr
