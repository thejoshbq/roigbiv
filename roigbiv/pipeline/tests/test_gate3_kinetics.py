"""Gate 3 rise/decay kinetics at the lab's frame rate (7.5 Hz).

At 7.5 Hz a GCaMP6s transient rises in under one frame. The rise is measured from the event onset (the last
sub-10% sample before the peak), so the 10% crossing is not taken from baseline noise at the window start, and
a sub-frame rise is a finite, passing value rather than "unresolved".
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.gate3 import _extract_event_waveform, _rise_decay_ratio
from roigbiv.pipeline.stage3_templates import build_template_bank

THRESHOLD = 0.5  # PipelineConfig.gate3_max_rise_decay_ratio


def _event_window(fs: float, tau: float, noise_sd: float, rng) -> np.ndarray:
    """A unit-peak copy of the pipeline's own single-event template at onset frame 150 of a 400-frame trace,
    cut with Gate 3's own window (5 tau), event frame = onset."""
    tmpl = build_template_bank(fs, tau)[0][1]
    tmpl = tmpl / tmpl.max()
    tr = np.zeros(400, np.float32)
    tr[150:150 + len(tmpl)] += tmpl
    if noise_sd:
        tr += rng.normal(0, noise_sd, tr.size).astype(np.float32)
    return _extract_event_waveform(tr, 150, int(5.0 * tau * fs))


def test_noiseless_template_passes_at_7_5_hz():
    rd = _rise_decay_ratio(_event_window(7.5, 1.0, 0.0, None))
    assert np.isfinite(rd), "a rise inside one frame must be a finite ratio, not unresolved"
    assert rd < THRESHOLD


def test_noiseless_template_passes_at_30_hz():
    assert _rise_decay_ratio(_event_window(30.0, 1.0, 0.0, None)) < THRESHOLD


def test_snr20_template_passes_most_draws_at_7_5_hz():
    rng = np.random.default_rng(0)
    rd = np.array([_rise_decay_ratio(_event_window(7.5, 1.0, 1 / 20, rng)) for _ in range(200)])
    assert np.mean(rd < THRESHOLD) >= 0.9


def test_baseline_noise_does_not_set_the_onset():
    """With a noisy baseline before a sharp event, the first sample above 10% can sit at the window start, which
    would make the rise the distance from window start to peak. Rise must be measured from the onset."""
    rng = np.random.default_rng(1)
    rd = np.array([_rise_decay_ratio(_event_window(7.5, 1.0, 1 / 8, rng)) for _ in range(200)])
    finite = rd[np.isfinite(rd)]
    assert np.median(finite) < THRESHOLD


def test_symmetric_bump_is_rejected():
    t = np.arange(37, dtype=np.float32)
    bump = np.exp(-((t - 18) ** 2) / (2 * 4.0 ** 2)).astype(np.float32)
    assert not _rise_decay_ratio(bump) < THRESHOLD


def test_slow_ramp_is_rejected():
    """Rise as long as the decay (10 frames each) is not a calcium transient."""
    w = np.concatenate([np.zeros(5), np.linspace(0, 1, 11)[1:], np.linspace(1, 0, 11)[1:], np.zeros(6)])
    assert not _rise_decay_ratio(w.astype(np.float32)) < THRESHOLD


def test_peak_without_decay_in_window_is_unresolved():
    w = np.concatenate([np.zeros(9), np.linspace(0.2, 1.0, 28)]).astype(np.float32)
    assert not np.isfinite(_rise_decay_ratio(w))
