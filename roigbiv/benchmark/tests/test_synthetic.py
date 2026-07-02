"""Tests for synthetic soma injection module (roigbiv.benchmark.synthetic).

Covers:
  - Exact mask recovery for all soma types
  - Deterministic seeding (reproducibility)
  - SNR band sampling
  - Save/reload roundtrip
  - In-place vs copy behavior
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.benchmark.synthetic import (
    SomaSpec,
    inject_somas,
    inject_from_tif,
    save_injection,
    default_spec,
    overlapping_pair,
)


def _synthetic_movie(seed=0, T=200, H=32, W=32):
    """Helper: generate a small synthetic background movie."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.5, size=(T, H, W)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Test 1: Exact mask recovery
# ─────────────────────────────────────────────────────────────────────────

def test_recover_exact_masks():
    """Inject one of each soma type + overlapping_pair; recover exact masks."""
    T, H, W = 200, 32, 32
    movie = np.zeros((T, H, W), dtype=np.float32)

    # Build specs: one of each type + overlapping_pair
    specs = [
        default_spec("dim", center=(10, 10), radius=4.0),
        default_spec("overlapping", center=(16, 16), radius=4.0),
        default_spec("sparse_transient", center=(22, 22), radius=4.0),
        default_spec("slow_modulation", center=(10, 22), radius=4.0),
        default_spec("elevated_baseline", center=(22, 10), radius=4.0),
    ]
    specs.extend(overlapping_pair(center=(16, 10), offset=(3, 3), radius=4.0))

    result = inject_somas(movie, specs, seed=0)

    # Check shape: 7 masks (5 singles + 2 from overlapping_pair)
    assert result.soma_masks.shape[0] == 7, \
        f"expected 7 masks, got {result.soma_masks.shape[0]}"

    # Each mask must be boolean and non-empty
    for i, mask in enumerate(result.soma_masks):
        assert mask.dtype == bool, f"mask {i}: expected bool, got {mask.dtype}"
        assert mask.any(), f"mask {i}: empty (no pixels set)"

    # Each mask must match the disk footprint computed from spec
    for i, (spec, mask) in enumerate(zip(specs, result.soma_masks)):
        cy, cx = spec.center
        radius = spec.radius
        dy_grid, dx_grid = np.ogrid[:H, :W]
        dy = dy_grid - cy
        dx = dx_grid - cx
        expected_disk = (dy * dy + dx * dx) <= (radius * radius)
        assert np.array_equal(mask, expected_disk), \
            f"mask {i} ({spec.soma_type} at {spec.center}): " \
            f"recovered mask doesn't match expected disk footprint"

    # Label IDs should be 1..7 in order
    assert result.label_mask.max() == 7, \
        f"expected max label_id=7, got {result.label_mask.max()}"

    print("  [PASS] test_recover_exact_masks")


# ─────────────────────────────────────────────────────────────────────────
# Test 2: Deterministic seeding
# ─────────────────────────────────────────────────────────────────────────

def test_deterministic_seed():
    """Same specs + seed → identical results; different seed → different results."""
    T, H, W = 200, 32, 32
    movie = _synthetic_movie(seed=0)

    # Mix of snr_target=None (samples from band) and soma types that use rng
    specs = [
        default_spec("dim", center=(10, 10), snr_target=None),
        default_spec("sparse_transient", center=(20, 20), event_rate_hz=0.1),
    ]

    # Deterministic: same seed should produce identical results
    result1 = inject_somas(movie.copy(), specs, seed=42)
    result2 = inject_somas(movie.copy(), specs, seed=42)

    assert np.array_equal(result1.movie, result2.movie), \
        "same seed produces different movies"
    assert np.array_equal(result1.soma_masks, result2.soma_masks), \
        "same seed produces different masks"
    assert result1.metadata == result2.metadata, \
        "same seed produces different metadata"

    # Sanity check: different seed should differ
    result3 = inject_somas(movie.copy(), specs, seed=99)
    assert not np.array_equal(result1.movie, result3.movie), \
        "different seed should produce different movies"

    print("  [PASS] test_deterministic_seed")


# ─────────────────────────────────────────────────────────────────────────
# Test 3: SNR band sampling
# ─────────────────────────────────────────────────────────────────────────

def test_snr_bands():
    """For each soma type, inject with snr_target=None; verify resolved SNR in band."""
    T, H, W = 200, 32, 32
    movie = _synthetic_movie(seed=0)

    snr_bands = {
        "dim": (1.5, 2.5),
        "overlapping": (3.0, 4.0),
        "sparse_transient": (5.0, 6.0),
        "slow_modulation": (4.0, 5.0),
        "elevated_baseline": (3.0, 4.0),
    }

    for offset, (soma_type, (lo, hi)) in enumerate(snr_bands.items()):
        spec = default_spec(soma_type, center=(16, 16), snr_target=None)
        result = inject_somas(movie.copy(), [spec], seed=42 + offset)

        resolved_snr = result.specs[0].snr_target
        assert lo <= resolved_snr <= hi, \
            f"{soma_type}: resolved SNR={resolved_snr} not in band ({lo}, {hi})"

    print("  [PASS] test_snr_bands")


# ─────────────────────────────────────────────────────────────────────────
# Test 4: Save and reload
# ─────────────────────────────────────────────────────────────────────────

def test_save_and_reload(tmp_path):
    """Save injection result to disk; verify file existence and content roundtrip."""
    T, H, W = 200, 32, 32
    movie = _synthetic_movie(seed=0)

    specs = [
        default_spec("dim", center=(10, 10)),
        default_spec("overlapping", center=(20, 20)),
        default_spec("sparse_transient", center=(25, 25)),
    ]

    result = inject_somas(movie, specs, seed=42)
    paths = save_injection(result, tmp_path)

    # Check files exist
    assert Path(tmp_path / "ground_truth_masks.tif").exists(), \
        "ground_truth_masks.tif not written"
    assert Path(tmp_path / "ground_truth_masks.npy").exists(), \
        "ground_truth_masks.npy not written"
    assert Path(tmp_path / "injection_metadata.json").exists(), \
        "injection_metadata.json not written"

    # Reload and verify content
    masks_tif = tifffile.imread(str(tmp_path / "ground_truth_masks.tif"))
    assert np.array_equal(masks_tif, result.label_mask), \
        "reloaded TIFF mask doesn't match label_mask"

    masks_npy = np.load(str(tmp_path / "ground_truth_masks.npy"))
    assert np.array_equal(masks_npy, result.soma_masks), \
        "reloaded NPY masks don't match soma_masks"

    with open(tmp_path / "injection_metadata.json") as f:
        metadata = json.load(f)

    assert "seed" in metadata, "metadata missing 'seed' key"
    assert "shape" in metadata, "metadata missing 'shape' key"
    assert "fs" in metadata, "metadata missing 'fs' key"
    assert "tau" in metadata, "metadata missing 'tau' key"
    assert "somas" in metadata, "metadata missing 'somas' key"
    assert len(metadata["somas"]) == len(specs), \
        f"metadata has {len(metadata['somas'])} somas, expected {len(specs)}"

    # Test save_movie=True
    save_injection(result, tmp_path, save_movie=True)
    injected_movie = np.load(str(tmp_path / "injected_movie.npy"))
    assert np.array_equal(injected_movie, result.movie), \
        "reloaded injected_movie doesn't match result.movie"

    print("  [PASS] test_save_and_reload")


# ─────────────────────────────────────────────────────────────────────────
# Test 5: In-place vs copy behavior
# ─────────────────────────────────────────────────────────────────────────

def test_in_place_vs_copy():
    """Test in_place=True mutates original; in_place=False preserves it."""
    T, H, W = 200, 32, 32

    specs = [default_spec("dim", center=(16, 16))]

    # ─────────────────────────────────────────────────────────────────────
    # in_place=True: should mutate the original array
    # ─────────────────────────────────────────────────────────────────────
    movie_float32 = _synthetic_movie(seed=0)
    original_before = movie_float32.copy()

    result_in_place = inject_somas(movie_float32, specs, in_place=True)

    # The returned movie should be the same object
    assert result_in_place.movie is movie_float32, \
        "in_place=True: result.movie is not the same object as input"

    # The original array should have been modified
    assert not np.array_equal(movie_float32, original_before), \
        "in_place=True: original array was not mutated"

    # ─────────────────────────────────────────────────────────────────────
    # in_place=False: should NOT mutate the original array
    # ─────────────────────────────────────────────────────────────────────
    movie_float32_2 = _synthetic_movie(seed=0)
    original_before_2 = movie_float32_2.copy()

    result_not_in_place = inject_somas(movie_float32_2, specs, in_place=False)

    # The returned movie should be a different object
    assert result_not_in_place.movie is not movie_float32_2, \
        "in_place=False: result.movie is the same object as input"

    # The original array should be unchanged
    assert np.array_equal(movie_float32_2, original_before_2), \
        "in_place=False: original array was mutated"

    # ─────────────────────────────────────────────────────────────────────
    # Error case: in_place=True with integer dtype should raise ValueError
    # ─────────────────────────────────────────────────────────────────────
    movie_int16 = np.zeros((T, H, W), dtype=np.int16)

    with pytest.raises(ValueError) as exc_info:
        inject_somas(movie_int16, specs, in_place=True)

    assert "in_place=True" in str(exc_info.value) or "float" in str(exc_info.value), \
        f"ValueError message doesn't explain the problem: {exc_info.value}"

    print("  [PASS] test_in_place_vs_copy")


# ─────────────────────────────────────────────────────────────────────────
# Test 6: Injected amplitude matches reported metadata (numeric SNR check)
# ─────────────────────────────────────────────────────────────────────────

def test_injected_amplitude_matches_metadata():
    """The actual pixel values written into the movie must match the
    amplitude the metadata claims was injected — catches formula bugs
    (wrong scaling, wrong weight, wrong noise floor) that a mask-only
    test cannot see."""
    movie = _synthetic_movie(seed=3, T=300, H=32, W=32)
    specs = [
        default_spec("dim", center=(10, 10), snr_target=3.0),
        default_spec("elevated_baseline", center=(20, 20), snr_target=3.0),
    ]
    result = inject_somas(movie.copy(), specs, fs=7.5, seed=1)
    diff = result.movie - movie

    for spec, meta in zip(result.specs, result.metadata["somas"]):
        cy, cx = spec.center
        peak = float(diff[:, cy, cx].max())
        assert peak == pytest.approx(meta["amplitude"], rel=1e-4), (
            f"{spec.soma_type}: peak injected signal at center ({peak}) "
            f"doesn't match metadata amplitude ({meta['amplitude']})"
        )

    # elevated_baseline profile is constant -> every frame at center should
    # equal the metadata amplitude, not just the peak.
    eb_center = specs[1].center
    eb_amplitude = result.metadata["somas"][1]["amplitude"]
    eb_trace = diff[:, eb_center[0], eb_center[1]]
    assert np.allclose(eb_trace, eb_amplitude, rtol=1e-4), \
        "elevated_baseline: injected signal is not constant over time at center pixel"

    print("  [PASS] test_injected_amplitude_matches_metadata")


# ─────────────────────────────────────────────────────────────────────────
# Test 7: slow_modulation period is in frames, not raw seconds (regression
# test for the seconds->frames unit bug caught in review)
# ─────────────────────────────────────────────────────────────────────────

def test_slow_modulation_period_is_seconds_not_frames():
    """mod_period_s must be converted to frames via *fs before use. A naive
    peak-amplitude check can't tell correct vs. wrong period apart (both
    reach the same peak); count oscillation cycles instead."""
    fs = 7.5
    T, H, W = 300, 16, 16
    movie = _synthetic_movie(seed=5, T=T, H=H, W=W)
    mod_period_s = 10.0
    spec = default_spec(
        "slow_modulation", center=(8, 8), snr_target=3.0, mod_period_s=mod_period_s,
    )
    result = inject_somas(movie.copy(), [spec], fs=fs, seed=0)
    diff = result.movie - movie
    trace = diff[:, 8, 8]

    crossings = int(np.sum(np.diff(np.sign(trace - trace.mean())) != 0))
    expected_period_frames = mod_period_s * fs
    expected_crossings = 2 * T / expected_period_frames

    assert abs(crossings - expected_crossings) <= 2, (
        f"observed {crossings} midline crossings, expected ~{expected_crossings:.1f} "
        f"for a {mod_period_s}s period at fs={fs}Hz "
        f"({expected_period_frames:.0f} frames/period). If mod_period_s were "
        f"used as a raw frame count instead of seconds, this would report "
        f"~{2 * T / mod_period_s:.0f} crossings instead."
    )

    print("  [PASS] test_slow_modulation_period_is_seconds_not_frames")


# ─────────────────────────────────────────────────────────────────────────
# Test 8: duplicate label_id raises
# ─────────────────────────────────────────────────────────────────────────

def test_duplicate_label_id_raises():
    """An explicit label_id colliding with an already-assigned one (auto or
    explicit) must raise, not silently merge two somas' ground truth."""
    movie = _synthetic_movie(seed=0)
    specs = [
        default_spec("dim", center=(10, 10)),        # auto-assigned label_id=1
        default_spec("elevated_baseline", center=(20, 20), label_id=1),  # collides
    ]
    with pytest.raises(ValueError, match="label_id"):
        inject_somas(movie.copy(), specs, seed=0)

    print("  [PASS] test_duplicate_label_id_raises")


# ─────────────────────────────────────────────────────────────────────────
# Test 9: label_mask overwrite behavior on genuine overlap
# ─────────────────────────────────────────────────────────────────────────

def test_label_mask_overlap_behavior():
    """Where two somas' footprints genuinely overlap, label_mask keeps only
    the later soma's label_id in the shared region (documented, lossy
    convenience layer), while soma_masks preserves each soma's full,
    independent footprint regardless of overlap."""
    movie = _synthetic_movie(seed=0, T=100, H=32, W=32)
    specs = overlapping_pair(center=(16, 16), offset=(3, 3), radius=4.0)
    result = inject_somas(movie.copy(), specs, seed=0)

    mask_a, mask_b = result.soma_masks
    overlap = mask_a & mask_b
    assert overlap.any(), "test setup: overlapping_pair did not actually overlap"

    label_a, label_b = result.specs[0].label_id, result.specs[1].label_id
    assert np.all(result.label_mask[overlap] == label_b), \
        "label_mask in the overlap region should carry the later soma's label_id"

    # soma_masks are unaffected by the overwrite in label_mask.
    cy, cx = specs[0].center
    dy_grid, dx_grid = np.ogrid[:32, :32]
    expected_disk_a = ((dy_grid - cy) ** 2 + (dx_grid - cx) ** 2) <= (specs[0].radius ** 2)
    assert np.array_equal(mask_a, expected_disk_a), \
        "soma_masks[0] should be the full independent disk despite the overlap"

    print("  [PASS] test_label_mask_overlap_behavior")


# ─────────────────────────────────────────────────────────────────────────
# Test 10: inject_from_tif round trip
# ─────────────────────────────────────────────────────────────────────────

def test_inject_from_tif(tmp_path):
    """inject_from_tif: TIFF load -> inject -> save round trip."""
    movie = _synthetic_movie(seed=0, T=50, H=16, W=16)
    input_tif = tmp_path / "input_movie.tif"
    tifffile.imwrite(str(input_tif), movie)

    specs = [default_spec("dim", center=(8, 8))]
    output_dir = tmp_path / "out"
    result = inject_from_tif(input_tif, specs, output_dir, seed=0)

    assert result.movie.shape == movie.shape
    assert (output_dir / "ground_truth_masks.tif").exists()
    assert (output_dir / "ground_truth_masks.npy").exists()
    assert (output_dir / "injection_metadata.json").exists()

    print("  [PASS] test_inject_from_tif")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    # Can't use tmp_path fixture in direct execution; use tempfile instead
    def test_save_and_reload_direct():
        with tempfile.TemporaryDirectory() as tmpdir:
            test_save_and_reload(Path(tmpdir))

    def test_inject_from_tif_direct():
        with tempfile.TemporaryDirectory() as tmpdir:
            test_inject_from_tif(Path(tmpdir))

    tests = [
        test_recover_exact_masks,
        test_deterministic_seed,
        test_snr_bands,
        test_save_and_reload_direct,
        test_in_place_vs_copy,
        test_injected_amplitude_matches_metadata,
        test_slow_modulation_period_is_seconds_not_frames,
        test_duplicate_label_id_raises,
        test_label_mask_overlap_behavior,
        test_inject_from_tif_direct,
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
