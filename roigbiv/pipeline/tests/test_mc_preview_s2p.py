"""Guards for the Suite2p (``phasecorr``) live-preview hooks.

These run against a *fake* Suite2p, so they need neither suite2p nor a GPU. The
fake deliberately reproduces the two behaviours that make the real hook subtle:
``register_frames`` mutates the caller's array in place
(``suite2p/registration/register.py:313``), and the array handed to it is a
memmap view rather than a copy (``suite2p/io/binary.py:122``).
"""
import io
import json
import sys
import types

import numpy as np
import pytest

from roigbiv.pipeline import mc_preview_s2p as hooks
from roigbiv.pipeline.mc_preview import MCPreviewWriter, preview_dir

RAW_LEVEL = 100.0
SHIFT = 50.0


@pytest.fixture
def fake_suite2p(monkeypatch):
    """Install a minimal fake ``suite2p.registration.register`` module."""
    reg = types.ModuleType("suite2p.registration.register")

    def register_frames(refAndMasks, frames, rmin=-np.inf, rmax=np.inf,
                        bidiphase=0, ops=None, nZ=1):
        # In place, exactly like the real one.
        frames[:] = frames + SHIFT
        n = len(frames)
        return (frames, np.arange(n) * 1.0, np.arange(n) * -1.0,
                np.full(n, 0.75), None, None, None, None)

    def compute_reference_and_register_frames(f_align_in, *a, **k):
        return None

    reg.register_frames = register_frames
    reg.compute_reference_and_register_frames = compute_reference_and_register_frames

    pkg = types.ModuleType("suite2p.registration")
    pkg.register = reg
    monkeypatch.setitem(sys.modules, "suite2p", types.ModuleType("suite2p"))
    monkeypatch.setitem(sys.modules, "suite2p.registration", pkg)
    monkeypatch.setitem(sys.modules, "suite2p.registration.register", reg)
    return reg


def _writer(tmp_path, **kw):
    kw.setdefault("min_interval_s", 0.0)
    kw.setdefault("metrics", False)
    return MCPreviewWriter(tmp_path, stem="fov", backend="phasecorr", **kw)


def _state(tmp_path) -> dict:
    return json.loads((preview_dir(tmp_path) / "state.json").read_text())


def _batch(n=4, Ly=16, Lx=16):
    return np.full((n, Ly, Lx), RAW_LEVEL, dtype=np.float32)


# ── the in-place mutation trap ──────────────────────────────────────────────

def test_raw_pane_is_snapshotted_before_registration_mutates_it(
        tmp_path, fake_suite2p):
    """The single highest-value guard here.

    ``register_frames`` overwrites the array it is given, and under Suite2p that
    array is a memmap view. Without a copy taken *before* delegating, the "raw"
    pane would show the corrected frame — a preview that looks like it works
    while proving nothing.
    """
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        frames = _batch()
        out = sys.modules["suite2p.registration.register"].register_frames(
            None, frames)

    # The fake really did mutate in place...
    assert out[0] is frames
    assert frames.max() == pytest.approx(RAW_LEVEL + SHIFT)
    # ...and the frozen display window came from the pre-mutation values.
    lo, hi = _state(tmp_path)["norm"]
    assert lo < RAW_LEVEL + SHIFT / 2, (
        "raw pane captured post-mutation pixels")


def test_raw_and_corrected_panes_differ(tmp_path, fake_suite2p):
    from PIL import Image

    w = _writer(tmp_path, max_dim=64)
    with hooks.suite2p_preview_hooks(w):
        rng = np.random.default_rng(0)
        frames = rng.normal(500, 50, (4, 32, 32)).astype(np.float32)
        sys.modules["suite2p.registration.register"].register_frames(None, frames)

    seq = _state(tmp_path)["seq"]
    with Image.open(preview_dir(tmp_path) / f"raw_{seq:06d}.png") as im:
        raw = np.asarray(im)
    with Image.open(preview_dir(tmp_path) / f"corr_{seq:06d}.png") as im:
        corr = np.asarray(im)
    assert not np.array_equal(raw, corr)


# ── shifts and progress ─────────────────────────────────────────────────────

def test_shifts_and_confidence_are_recorded_per_frame(tmp_path, fake_suite2p):
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        reg = sys.modules["suite2p.registration.register"]
        reg.register_frames(None, _batch(n=8))
        w.set_progress(8)
        reg.register_frames(None, _batch(n=8))
    w.__exit__(None, None, None)

    z = np.load(preview_dir(tmp_path) / "shifts.npz")
    assert z["frame"].shape == (16,)
    assert list(z["frame"][:3]) == [0, 1, 2]
    assert list(z["frame"][8:11]) == [8, 9, 10]     # second batch offset
    assert np.allclose(z["cmax"], 0.75)


def test_recursive_calls_are_not_double_counted(tmp_path, fake_suite2p):
    """register_frames recurses into itself when nZ > 1; only the outer call
    is a batch."""
    reg = sys.modules["suite2p.registration.register"]
    inner = reg.register_frames

    def recursive(refAndMasks, frames, *a, **k):
        if k.pop("_outer", True):
            return reg.register_frames(refAndMasks, frames, *a, _outer=False, **k)
        return inner(refAndMasks, frames, *a, **k)

    reg.register_frames = recursive
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        reg.register_frames(None, _batch(n=4))
    w.__exit__(None, None, None)

    z = np.load(preview_dir(tmp_path) / "shifts.npz")
    assert z["frame"].shape == (4,), "recursive call recorded twice"


# ── stdout tap ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line,expected", [
    ("Registered 500/2000 in 3.10s", ("registering", (500, 2000))),
    ("----------- REGISTRATION", ("registering", None)),
    ("----------- REGISTRATION STEP 2", ("registering", None)),
    ("Reference frame, 1.20 sec.", ("registering", None)),
    ("1200 frames of binary, time 4.00 sec.", ("converting", None)),
    ("NOTE: not running registration, ops['do_registration']=0",
     ("skipped_precorrected", None)),
    ("NOTE: not running registration, plane already registered",
     ("skipped_resume", None)),
    ("----------- ROI DETECTION", ("done", None)),
    # Announces that registration *will* run — must not read as a skip.
    ("NOTE: not registered / registration forced with ops['do_registration']>1",
     (None, None)),
    ("some unrelated chatter", (None, None)),
    ("", (None, None)),
])
def test_classify_line(line, expected):
    assert hooks.classify_line(line) == expected


def test_tap_forwards_everything_and_restores_by_identity(
        tmp_path, fake_suite2p, monkeypatch):
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        assert sys.stdout is not sink
        print("----------- REGISTRATION")
        print("Registered 128/512 in 1.00s")
    assert sys.stdout is sink
    assert "Registered 128/512" in sink.getvalue()
    assert "REGISTRATION" in sink.getvalue()
    st = _state(tmp_path)
    assert (st["n_done"], st["n_total"]) == (128, 512)


def test_tap_restores_stdout_on_exception(tmp_path, fake_suite2p, monkeypatch):
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)
    w = _writer(tmp_path)
    with pytest.raises(RuntimeError):
        with hooks.suite2p_preview_hooks(w):
            raise RuntimeError("suite2p exploded")
    assert sys.stdout is sink


def test_tap_buffers_split_writes(tmp_path, fake_suite2p, monkeypatch):
    """print() issues the payload and the newline as two separate writes."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        sys.stdout.write("Registered 64/256 in 0.5s")
        assert _state(tmp_path)["n_done"] == 0      # not yet a complete line
        sys.stdout.write("\n")
        assert _state(tmp_path)["n_done"] == 64


def test_detection_banner_does_not_overwrite_a_skip(tmp_path, fake_suite2p,
                                                    monkeypatch):
    """Suite2p prints ROI DETECTION after any outcome; relabelling a skipped
    run as 'done' would discard the reason the card is empty."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        print("NOTE: not running registration, ops['do_registration']=0")
        print("----------- ROI DETECTION")
    assert w.phase == "skipped_precorrected"


# ── graceful degradation ────────────────────────────────────────────────────

def test_unrecognised_signature_disables_the_hook(tmp_path, fake_suite2p):
    """A Suite2p upgrade must degrade the preview, not break registration."""
    reg = sys.modules["suite2p.registration.register"]
    original = reg.register_frames
    reg.register_frames = lambda *a, **k: None      # no (refAndMasks, frames)

    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        assert reg.register_frames is not original  # untouched by us
    assert _state(tmp_path)["phase"] == "unsupported"
    assert "signature" in (_state(tmp_path)["note"] or "")


def test_missing_suite2p_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "suite2p.registration", None)
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        pass
    assert _state(tmp_path)["phase"] == "unsupported"


def test_disabled_writer_installs_nothing(tmp_path, fake_suite2p):
    reg = sys.modules["suite2p.registration.register"]
    original = reg.register_frames
    with hooks.suite2p_preview_hooks(None):
        assert reg.register_frames is original
    w = _writer(tmp_path, enabled=False)
    with hooks.suite2p_preview_hooks(w):
        assert reg.register_frames is original


def test_hooks_are_restored_after_use(tmp_path, fake_suite2p):
    reg = sys.modules["suite2p.registration.register"]
    original = reg.register_frames
    original_compute = reg.compute_reference_and_register_frames
    w = _writer(tmp_path)
    with hooks.suite2p_preview_hooks(w):
        assert reg.register_frames is not original
    assert reg.register_frames is original
    assert reg.compute_reference_and_register_frames is original_compute
    assert not hooks._installed
