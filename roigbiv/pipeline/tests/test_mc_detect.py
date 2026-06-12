"""
Tests for content-first motion-correction detection + the Foundation stage label.

Covers:
  - detect_motion_corrected tiers: metadata > filename > none
  - metadata beats a non-_mc filename; strict suffix rejects substring traps
  - the real writer (_write_mc_tif) stamps a tag that reads back as "metadata"
  - _mc_stage_label reflects do_registration

Run via:
    conda run -n roigbiv python -m roigbiv.pipeline.tests.test_mc_detect
    pytest roigbiv/pipeline/tests/test_mc_detect.py
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.io import MC_SOFTWARE_TAG, detect_motion_corrected
from roigbiv.pipeline.foundation import _mc_stage_label


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _write_stack(path: Path, T=4, Ly=8, Lx=8, software=None) -> Path:
    """Write a flat (T, Ly, Lx) uint16 stack, optionally stamping the Software tag
    on the first page (mirrors the pipeline's _append_frames idiom)."""
    rng = np.random.default_rng(0)
    stack = (rng.random((T, Ly, Lx)) * 1000).astype(np.uint16)
    with tifffile.TiffWriter(str(path), bigtiff=True) as tw:
        for i, frame in enumerate(stack):
            if software is not None and i == 0:
                tw.write(frame, software=software)
            else:
                tw.write(frame, contiguous=True)
    return path


# ─────────────────────────────────────────────────────────────────────────
# detect_motion_corrected
# ─────────────────────────────────────────────────────────────────────────

def test_metadata_tag_detected():
    with tempfile.TemporaryDirectory() as d:
        p = _write_stack(Path(d) / "foo_mc.tif", software=MC_SOFTWARE_TAG)
        assert detect_motion_corrected(p) == (True, "metadata")


def test_metadata_overrides_missing_suffix():
    # A roigbiv-corrected movie that was renamed to drop the _mc suffix is still
    # recognised by content — metadata wins over filename.
    with tempfile.TemporaryDirectory() as d:
        p = _write_stack(Path(d) / "renamed_output.tif", software=MC_SOFTWARE_TAG)
        assert detect_motion_corrected(p) == (True, "metadata")


def test_filename_suffix_fallback():
    # External/legacy pre-corrected input: _mc suffix, no embedded tag.
    with tempfile.TemporaryDirectory() as d:
        p = _write_stack(Path(d) / "session01_mc.tif", software=None)
        assert detect_motion_corrected(p) == (True, "filename")


def test_raw_input_not_corrected():
    with tempfile.TemporaryDirectory() as d:
        p = _write_stack(Path(d) / "session01.tif", software=None)
        assert detect_motion_corrected(p) == (False, "none")


def test_strict_suffix_rejects_substring_traps():
    # Names that merely *contain* "mc" must not be misread as corrected.
    with tempfile.TemporaryDirectory() as d:
        for name in ("exp_mcg_001.tif", "foo_mc_raw.tif", "mcherry_stack.tif"):
            p = _write_stack(Path(d) / name, software=None)
            assert detect_motion_corrected(p) == (False, "none"), name


def test_unreadable_file_degrades_to_filename():
    # A non-TIFF / unreadable file named with the convention falls back cleanly.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "broken_mc.tif"
        p.write_bytes(b"not a tiff")
        assert detect_motion_corrected(p) == (True, "filename")
        q = Path(d) / "broken.tif"
        q.write_bytes(b"not a tiff")
        assert detect_motion_corrected(q) == (False, "none")


def test_write_mc_tif_roundtrip_stamps_metadata():
    # The real writer must produce a tag detect_motion_corrected reads as metadata,
    # and a flat (T, Ly, Lx) stack (the tag must not perturb page geometry).
    from roigbiv.pipeline.registration import _write_mc_tif
    with tempfile.TemporaryDirectory() as d:
        T, Ly, Lx = 6, 8, 8
        bin_path = Path(d) / "data.bin"
        arr = (np.arange(T * Ly * Lx) % 500).astype(np.int16).reshape(T, Ly, Lx)
        arr.tofile(str(bin_path))
        mc_path = _write_mc_tif(bin_path, Path(d) / "stem_mc.tif", Ly, Lx, chunk=4)
        assert detect_motion_corrected(mc_path) == (True, "metadata")
        back = tifffile.imread(str(mc_path))
        assert back.shape == (T, Ly, Lx), back.shape


# ─────────────────────────────────────────────────────────────────────────
# Stage label
# ─────────────────────────────────────────────────────────────────────────

def test_stage_label_registering():
    assert _mc_stage_label(True) == "Motion correction"


def test_stage_label_detection_only():
    label = _mc_stage_label(False)
    assert "detection-only" in label
    assert "pre-corrected" in label


# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_metadata_tag_detected,
        test_metadata_overrides_missing_suffix,
        test_filename_suffix_fallback,
        test_raw_input_not_corrected,
        test_strict_suffix_rejects_substring_traps,
        test_unreadable_file_degrades_to_filename,
        test_write_mc_tif_roundtrip_stamps_metadata,
        test_stage_label_registering,
        test_stage_label_detection_only,
    ]
    failed = []
    for test in tests:
        try:
            test()
            print(f"  [ok] {test.__name__}")
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
