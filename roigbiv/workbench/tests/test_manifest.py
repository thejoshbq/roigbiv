"""Run manifest backfilled from a saved run's FOV directory."""

import hashlib
import json
import os

import pytest

from roigbiv.workbench import manifest as mf


def _fov(tmp_path, *, movie_bytes=b"movie", cfg=None):
    movie = tmp_path / "bench" / "movie.tif"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(movie_bytes)
    st = movie.stat()
    fov = tmp_path / "run" / "fovA"
    fov.mkdir(parents=True)
    config = cfg or {
        "diameter": 28,
        "spatial_pool_radius": 14,
        "profile": "grin",
        "assume_optics": True,
        "explicit_fields": ["fs"],
        "auto_adapt": {"scale_source": "cpsam", "applied": {"diameter": 28}, "prior": {"profile": "grin"}},
        "cellpose_model": None,
    }
    (fov / "pipeline_log.json").write_text(
        json.dumps({"timestamp": "2026-11-19T14:11:22+00:00", "input": str(movie), "config": config})
    )
    (fov / ".roigbiv_manifest.json").write_text(
        json.dumps(
            {
                "roigbiv_version": "0.3.0",
                "input_tif": str(movie),
                "input_stat": {"size": st.st_size, "mtime_ns": st.st_mtime_ns},
                "cfg_fingerprint": "sha256:" + "b" * 64,
            }
        )
    )
    return fov, movie


def test_backfill_hashes_the_input_and_takes_commit_from_the_caller(tmp_path):
    fov, movie = _fov(tmp_path)
    man, problems = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="685440f", dirty=True)
    assert problems == []
    assert man.source == "backfill"
    assert (man.commit, man.dirty) == ("685440f", True)
    (inp,) = man.inputs
    assert inp.role == "movie" and inp.path == str(movie)
    assert inp.sha256 == hashlib.sha256(b"movie").hexdigest() and inp.size == 5
    # The saved fingerprint is the resume fingerprint (config minus stage flags, plus input size and mtime),
    # not a hash of the requested config, so it goes under its own name and the config hash is not recorded.
    assert man.requested_config_sha256 is None and "requested_config_sha256" in man.not_recorded
    assert dict(man.effective)["resume_fingerprint"] == "sha256:" + "b" * 64
    assert man.created_utc == "2026-11-19T14:11:22+00:00"
    assert {"environment", "model_weights"} <= set(man.not_recorded)


def test_effective_values_are_the_run_config_scalars_plus_the_scale_source(tmp_path):
    fov, _ = _fov(tmp_path)
    man, _ = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False)
    eff = dict(man.effective)
    assert eff["diameter"] == 28 and eff["spatial_pool_radius"] == 14 and eff["assume_optics"] is True
    assert eff["auto_adapt.scale_source"] == "cpsam"
    assert "explicit_fields" not in eff and "auto_adapt" not in eff and "cellpose_model" not in eff


def test_unknown_commit_is_listed_as_not_recorded(tmp_path):
    fov, _ = _fov(tmp_path)
    man, _ = mf.backfill_manifest(fov, run_id="outE", fov_id="fovA", commit=None, dirty=None)
    assert {"commit", "dirty"} <= set(man.not_recorded)


def test_a_movie_changed_since_the_run_is_a_problem_and_its_hash_is_not_recorded(tmp_path):
    fov, movie = _fov(tmp_path)
    movie.write_bytes(b"movie, edited")
    man, problems = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False)
    assert problems == [f"input {movie} differs from the run's record (size or mtime)"]
    assert man.inputs[0].sha256 is None  # today's file is not the run's input


def test_a_run_record_without_input_stat_says_so_instead_of_reporting_a_change(tmp_path):
    fov, movie = _fov(tmp_path)
    p = fov / ".roigbiv_manifest.json"
    d = json.loads(p.read_text())
    del d["input_stat"]
    p.write_text(json.dumps(d))
    man, problems = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False)
    assert problems == [f"input {movie}: the run recorded no size or mtime to compare"]
    assert man.inputs[0].sha256 is not None


def test_the_hash_cache_keys_on_the_resolved_path(tmp_path, monkeypatch):
    fov, _ = _fov(tmp_path)
    patch = tmp_path / "jobH.patch"
    patch.write_bytes(b"diff")
    cache = tmp_path / "hash_cache.json"
    monkeypatch.chdir(tmp_path)
    mf.backfill_manifest(
        fov, run_id="r", fov_id="f", commit="x", dirty=False, hash_cache=cache, extra_inputs=(("patch", "jobH.patch"),)
    )
    assert str(patch.resolve()) in json.loads(cache.read_text())


def test_a_missing_movie_is_a_problem_and_its_hash_reads_none(tmp_path):
    fov, movie = _fov(tmp_path)
    movie.unlink()
    man, problems = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False)
    assert problems == [f"input {movie} not found"]
    assert man.inputs[0].sha256 is None


def test_hashes_come_from_the_cache_when_size_and_mtime_match(tmp_path):
    fov, movie = _fov(tmp_path)
    cache = tmp_path / "hash_cache.json"
    mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False, hash_cache=cache)
    entry = json.loads(cache.read_text())[str(movie)]
    entry["sha256"] = "c" * 64
    cache.write_text(json.dumps({str(movie): entry}))
    man, _ = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False, hash_cache=cache)
    assert man.inputs[0].sha256 == "c" * 64
    os.utime(movie, ns=(1, 1))  # a touched file is hashed again (the run record is updated to match)
    rp = fov / ".roigbiv_manifest.json"
    rec = json.loads(rp.read_text())
    rec["input_stat"]["mtime_ns"] = 1
    rp.write_text(json.dumps(rec))
    man, _ = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False, hash_cache=cache)
    assert man.inputs[0].sha256 == hashlib.sha256(b"movie").hexdigest()


def test_extra_inputs_such_as_a_job_patch_are_hashed_too(tmp_path):
    fov, _ = _fov(tmp_path)
    patch = tmp_path / "jobH.patch"
    patch.write_bytes(b"diff")
    man, _ = mf.backfill_manifest(
        fov, run_id="outH", fov_id="fovA", commit="x", dirty=True, extra_inputs=(("patch", patch),)
    )
    assert [(i.role, i.sha256) for i in man.inputs] == [
        ("movie", hashlib.sha256(b"movie").hexdigest()),
        ("patch", hashlib.sha256(b"diff").hexdigest()),
    ]


def test_write_manifest_round_trips(tmp_path):
    fov, _ = _fov(tmp_path)
    man, _ = mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False)
    mf.write_manifest(man, tmp_path / "wb")
    assert mf.read_manifest(tmp_path / "wb") == man


def test_a_fingerprint_without_the_sha256_prefix_is_refused(tmp_path):
    fov, _ = _fov(tmp_path)
    p = fov / ".roigbiv_manifest.json"
    d = json.loads(p.read_text())
    d["cfg_fingerprint"] = "md5:abc"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="cfg_fingerprint"):
        mf.backfill_manifest(fov, run_id="outH", fov_id="fovA", commit="x", dirty=False)
