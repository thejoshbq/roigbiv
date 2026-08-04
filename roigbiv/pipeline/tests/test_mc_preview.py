"""Guards for the live motion-correction preview sidecar writer.

The writer runs inside the registration loop, so its two hard requirements are
(a) it never raises and never alters the registered data, and (b) the ``seq`` in
``state.json`` always names frame files that are already complete on disk —
otherwise the UI's A/B blink would compare two different frames, which is
exactly the artifact that would make a user distrust the registration.
"""
import json

import numpy as np
import pytest
from PIL import Image

from roigbiv.pipeline.mc_preview import (
    MCPreviewWriter,
    downsample,
    encode_png,
    preview_dir,
    writer_for,
)


def _frame(Ly=64, Lx=64, seed=0, level=500.0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:Ly, 0:Lx]
    blob = 200.0 * np.exp(-((y - Ly / 2) ** 2 + (x - Lx / 2) ** 2) / (2 * 8.0 ** 2))
    return (level + blob + rng.normal(0, 10, (Ly, Lx))).astype(np.float32)


def _writer(tmp_path, **kw):
    kw.setdefault("min_interval_s", 0.0)
    kw.setdefault("metrics", False)
    return MCPreviewWriter(tmp_path, stem="fov", backend="phasecorr", **kw)


def _state(tmp_path) -> dict:
    return json.loads((preview_dir(tmp_path) / "state.json").read_text())


# ── basics ──────────────────────────────────────────────────────────────────

def test_emit_writes_a_complete_record(tmp_path):
    w = _writer(tmp_path)
    w.set_total(500)
    w.set_phase("registering")
    w.emit(_frame(), _frame(seed=1), frame_index=7, n_done=64)

    st = _state(tmp_path)
    assert st["schema"] == 1
    assert st["phase"] == "registering"
    assert st["seq"] == 0
    assert st["frame_index"] == 7
    assert st["n_done"] == 64
    assert st["n_total"] == 500
    assert st["shape"] == [64, 64]
    for kind in ("raw", "corr", "avg"):
        assert (preview_dir(tmp_path) / f"{kind}_000000.png").exists()


def test_state_seq_always_names_files_on_disk(tmp_path):
    """The pairing invariant: whatever seq state.json advertises is complete."""
    w = _writer(tmp_path, max_records=4)
    for i in range(40):
        w.emit(_frame(seed=i), _frame(seed=i + 100), frame_index=i, n_done=i)
        st = _state(tmp_path)
        for kind in ("raw", "corr", "avg"):
            p = preview_dir(tmp_path) / f"{kind}_{st['seq']:06d}.png"
            assert p.exists() and p.stat().st_size > 0, (i, kind)


def test_no_avg_pane_when_disabled(tmp_path):
    w = _writer(tmp_path, avg=False)
    w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=1)
    assert not (preview_dir(tmp_path) / "avg_000000.png").exists()
    assert _state(tmp_path)["has_avg"] is False


def test_avg_pane_tracks_running_mean_of_raw_frames(tmp_path):
    """The pane must hold the running mean of previewed raw frames through the
    same frozen display window as the raw/corrected panes, not an independent
    stretch (encode_png with self._norm, not a per-frame percentile)."""
    w = _writer(tmp_path)
    raws = [_frame(seed=i, level=400.0 + 50.0 * i) for i in range(4)]
    corrs = [_frame(seed=i + 50) for i in range(4)]

    running_mean = None
    for i, (raw, corr) in enumerate(zip(raws, corrs)):
        w.emit(raw, corr, frame_index=i, n_done=i + 1)
        running_mean = (raw.copy() if running_mean is None
                         else running_mean + (raw - running_mean) / (i + 1))
        lo, hi = _state(tmp_path)["norm"]
        expected = encode_png(running_mean, lo, hi)
        actual = (preview_dir(tmp_path) / f"avg_{i:06d}.png").read_bytes()
        assert actual == expected, f"avg pane at seq={i} is not the running mean"


def test_preview_is_downsampled_to_max_dim(tmp_path):
    w = _writer(tmp_path, max_dim=32)
    w.emit(_frame(128, 256), _frame(128, 256, seed=1), frame_index=0, n_done=1)
    with Image.open(preview_dir(tmp_path) / "raw_000000.png") as im:
        assert max(im.size) <= 32
    assert _state(tmp_path)["shape"] == [128, 256]  # full-res shape recorded


def test_downsample_never_upsamples():
    assert downsample(np.zeros((1024, 1024)), 512).shape == (512, 512)
    assert downsample(np.zeros((300, 200)), 512).shape == (300, 200)


# ── throttling and normalization ────────────────────────────────────────────

def test_should_emit_respects_the_throttle(tmp_path):
    w = _writer(tmp_path, min_interval_s=10.0)
    assert w.should_emit()
    w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=1)
    assert not w.should_emit()
    assert _state(tmp_path)["seq"] == 0


def test_should_emit_is_false_when_disabled(tmp_path):
    w = _writer(tmp_path, enabled=False)
    assert not w.should_emit()
    w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=1)
    assert not preview_dir(tmp_path).exists()


def test_display_window_is_frozen_on_the_first_frame(tmp_path):
    """A per-frame stretch would make the A/B blink flicker on brightness,
    which reads as motion — the one thing the pane must not fake."""
    w = _writer(tmp_path)
    w.emit(_frame(level=500.0), _frame(level=500.0, seed=1),
           frame_index=0, n_done=1)
    first = _state(tmp_path)["norm"]

    w.emit(_frame(level=5000.0), _frame(level=5000.0, seed=1),
           frame_index=1, n_done=2)
    assert _state(tmp_path)["norm"] == first


def test_flat_frame_does_not_divide_by_zero(tmp_path):
    w = _writer(tmp_path)
    flat = np.full((32, 32), 7.0, dtype=np.float32)
    w.emit(flat, flat, frame_index=0, n_done=1)
    lo, hi = _state(tmp_path)["norm"]
    assert hi > lo
    assert (preview_dir(tmp_path) / "raw_000000.png").exists()


def test_encode_png_clips_to_the_window():
    img = np.array([[-100.0, 0.0], [50.0, 1000.0]], dtype=np.float32)
    png = encode_png(img, 0.0, 100.0)
    with Image.open(__import__("io").BytesIO(png)) as im:
        arr = np.asarray(im)
    assert arr[0, 0] == 0 and arr[1, 1] == 255
    assert 120 < arr[1, 0] < 135


# ── bounded, uniform recording ──────────────────────────────────────────────

def test_records_stay_within_budget_and_span_the_run(tmp_path):
    budget = 8
    w = _writer(tmp_path, max_records=budget)
    n = 200
    for i in range(n):
        w.emit(_frame(32, 32, seed=i), _frame(32, 32, seed=i + 1),
               frame_index=i, n_done=i)

    st = _state(tmp_path)
    records = st["records"]
    assert len(records) <= budget + 1        # +1 for the pinned live frame
    assert records == sorted(records)
    # Coverage must span the whole run, not collapse onto its tail.
    assert records[0] == 0
    assert records[-1] == n - 1
    # The retained timeline is evenly strided; the final entry is the pinned
    # live frame, which lands wherever the run happened to reach.
    strided = np.diff(records[:-1])
    assert strided.max() == strided.min()

    on_disk = {int(p.stem.split("_")[1])
               for p in preview_dir(tmp_path).glob("raw_*.png")}
    assert on_disk == set(records), "no orphaned or missing frame files"


def test_prior_run_records_are_cleared_at_open(tmp_path):
    w = _writer(tmp_path)
    for i in range(3):
        w.emit(_frame(seed=i), _frame(seed=i + 1), frame_index=i, n_done=i)
    w.__exit__(None, None, None)

    w2 = _writer(tmp_path)
    w2.emit(_frame(), _frame(seed=1), frame_index=0, n_done=1)
    # A stale timeline interleaved with a new one would make the scrubber lie.
    assert _state(tmp_path)["records"] == [0]
    assert not (preview_dir(tmp_path) / "raw_000002.png").exists()


# ── shift trace ─────────────────────────────────────────────────────────────

def test_record_shifts_keeps_full_resolution_in_the_npz(tmp_path):
    w = _writer(tmp_path)
    for b in range(4):
        w.record_shifts(b * 50, np.full(50, b * 1.0), np.full(50, -b * 1.0),
                        np.full(50, 0.5))
    w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=200)
    w.__exit__(None, None, None)

    z = np.load(preview_dir(tmp_path) / "shifts.npz")
    assert z["frame"].shape == (200,)
    assert z["y"].shape == z["x"].shape == z["cmax"].shape == (200,)
    assert z["frame"][0] == 0 and z["frame"][-1] == 199
    # state.json carries a decimated copy for drawing
    assert 0 < len(_state(tmp_path)["shifts"]["y"]) <= 500


def test_record_shifts_without_confidence_is_allowed(tmp_path):
    w = _writer(tmp_path)
    w.record_shifts(0, [1.0, 2.0], [0.0, 0.0])
    w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=2)
    assert _state(tmp_path)["shifts"]["cmax"] == [None, None]


def test_valid_crop_marks_the_roll_wrapped_edges(tmp_path):
    """np.roll wraps; the crop box tells the UI which strip is not real data."""
    w = _writer(tmp_path)
    w.record_shifts(0, [4.0, 6.0], [0.0, 0.0])   # dy > 0 → bottom rows wrap in
    w.emit(_frame(100, 100), _frame(100, 100, seed=1), frame_index=0, n_done=2)

    x0, y0, x1, y1 = _state(tmp_path)["valid_crop_frac"]
    assert (x0, x1) == (0.0, 1.0)
    assert y0 == 0.0
    assert y1 == pytest.approx(0.94)              # 6 of 100 rows invalid


def test_valid_crop_is_none_before_any_shift(tmp_path):
    w = _writer(tmp_path)
    w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=1)
    assert _state(tmp_path)["valid_crop_frac"] is None


# ── metrics ─────────────────────────────────────────────────────────────────

def test_metrics_are_computed_at_full_resolution(tmp_path):
    """Downsampled metrics would contradict the post-run per-FOV table, since
    lap_var_smooth and banding_score are both scale-dependent."""
    from roigbiv.pipeline.mc_metrics import compute_metrics

    corr = _frame(128, 128, seed=3)
    w = _writer(tmp_path, metrics=True, max_dim=32)
    w.emit(_frame(128, 128), corr, frame_index=0, n_done=1)

    live = _state(tmp_path)["live_metrics"]
    expected = compute_metrics(corr)
    assert live["lap_var_smooth"] == pytest.approx(expected["lap_var_smooth"])
    assert live["banding_score"] == pytest.approx(expected["banding_score"])


# ── failure handling ────────────────────────────────────────────────────────

def test_emit_survives_a_full_disk_and_gives_up(tmp_path, monkeypatch):
    w = _writer(tmp_path)
    import roigbiv.pipeline.mc_preview as mod

    def boom(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(mod.os, "replace", boom)
    for i in range(5):
        w.emit(_frame(), _frame(seed=1), frame_index=i, n_done=i)  # must not raise
    assert not w.enabled

    monkeypatch.undo()
    # Having self-disabled, it stays quiet rather than resuming mid-run.
    w.emit(_frame(), _frame(seed=1), frame_index=9, n_done=9)
    assert w.phase == "degraded"


def test_emit_survives_a_broken_frame(tmp_path):
    w = _writer(tmp_path)
    w.emit(None, None, frame_index=0, n_done=0)     # must not raise
    assert w.enabled


def test_context_manager_never_swallows_exceptions(tmp_path):
    with pytest.raises(ValueError):
        with _writer(tmp_path) as w:
            w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=1)
            raise ValueError("registration blew up")
    assert _state(tmp_path)["phase"] == "aborted"


def test_clean_exit_marks_done(tmp_path):
    with _writer(tmp_path) as w:
        w.emit(_frame(), _frame(seed=1), frame_index=0, n_done=1)
    assert _state(tmp_path)["phase"] == "done"


def test_terminal_phase_survives_exit(tmp_path):
    with _writer(tmp_path) as w:
        w.set_phase("skipped_precorrected")
    assert _state(tmp_path)["phase"] == "skipped_precorrected"


# ── rowwise-pcc integration (CPU, no GPU required) ──────────────────────────

def _shifted_stack(T=20, Ly=32, Lx=32, seed=4):
    import tifffile
    from scipy.ndimage import shift as ndi_shift

    rng = np.random.default_rng(seed)
    base = _frame(Ly, Lx, seed=seed)
    return np.stack([
        ndi_shift(base, (rng.uniform(-2, 2), rng.uniform(-2, 2)),
                  order=1, mode="reflect")
        for _ in range(T)
    ]).astype(np.uint16), tifffile


def test_rowwise_pcc_feeds_the_preview(tmp_path):
    from roigbiv.pipeline.registration import run_rowwise_pcc_register

    frames, tifffile = _shifted_stack()
    tif_path = tmp_path / "fov_test.tif"
    tifffile.imwrite(str(tif_path), frames)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _writer(out_dir) as w:
        mc_tif, mx, my = run_rowwise_pcc_register(
            tif_path, out_dir, fs=7.5, force_cpu=True, frame_batch=4,
            preview=w)

    assert mc_tif.exists()
    st = _state(out_dir)
    assert st["phase"] == "done"
    assert st["n_total"] == len(frames)
    assert st["seq"] >= 0
    # One rigid shift + confidence per frame, straight from the PCC peak.
    z = np.load(preview_dir(out_dir) / "shifts.npz")
    assert z["frame"].shape == (len(frames),)
    assert np.isfinite(z["cmax"]).all()
    for kind in ("raw", "corr", "avg"):
        assert (preview_dir(out_dir) / f"{kind}_{st['seq']:06d}.png").exists()


def test_rowwise_pcc_precorrected_input_is_labelled(tmp_path):
    """A pre-corrected stack has nothing to correct; say so rather than
    leaving the card empty forever."""
    from roigbiv.pipeline.registration import run_rowwise_pcc_register

    frames, tifffile = _shifted_stack(T=8)
    tif_path = tmp_path / "fov_mc.tif"
    tifffile.imwrite(str(tif_path), frames)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with _writer(out_dir) as w:
        run_rowwise_pcc_register(tif_path, out_dir, fs=7.5, force_cpu=True,
                                 do_registration=False, frame_batch=4,
                                 preview=w)

    assert _state(out_dir)["phase"] == "skipped_precorrected"
    assert _state(out_dir)["seq"] >= 0     # still shows the (unchanged) frame


def test_preview_does_not_change_the_registered_output(tmp_path):
    """Acceptance guard: the sidecar is diagnostic, so the corrected movie must
    be byte-identical with the preview on and off."""
    import hashlib

    from roigbiv.pipeline.registration import run_rowwise_pcc_register

    frames, tifffile = _shifted_stack(T=12)
    digests = []
    for i, use_preview in enumerate((False, True)):
        run_dir = tmp_path / f"run{i}"
        run_dir.mkdir()
        tif_path = run_dir / "fov_test.tif"
        tifffile.imwrite(str(tif_path), frames)
        out_dir = run_dir / "out"
        out_dir.mkdir()
        preview = _writer(out_dir) if use_preview else None
        mc_tif, _, _ = run_rowwise_pcc_register(
            tif_path, out_dir, fs=7.5, force_cpu=True, frame_batch=4,
            preview=preview)
        if preview is not None:
            preview.__exit__(None, None, None)
        digests.append(hashlib.sha256(mc_tif.read_bytes()).hexdigest())

    assert digests[0] == digests[1]


# ── config plumbing ─────────────────────────────────────────────────────────

def test_writer_for_honours_the_config(tmp_path):
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=7.5)
    w = writer_for(cfg, tmp_path, stem="fov", backend="phasecorr")
    assert w.enabled and w.max_dim == cfg.mc_preview_max_dim

    cfg2 = PipelineConfig(fs=7.5, mc_preview_enabled=False)
    w2 = writer_for(cfg2, tmp_path / "off", stem="fov", backend="phasecorr")
    assert not w2.enabled
    assert not preview_dir(tmp_path / "off").exists()


def test_preview_knobs_do_not_invalidate_resume(tmp_path):
    """Toggling a diagnostic sidecar must not force a pipeline re-run."""
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    from roigbiv.pipeline.types import PipelineConfig

    tif = tmp_path / "fov.tif"
    tif.write_bytes(b"stub")
    a = compute_cfg_fingerprint(PipelineConfig(fs=7.5), tif)
    b = compute_cfg_fingerprint(
        PipelineConfig(fs=7.5, mc_preview_enabled=False,
                       mc_preview_max_dim=128), tif)
    assert a == b
