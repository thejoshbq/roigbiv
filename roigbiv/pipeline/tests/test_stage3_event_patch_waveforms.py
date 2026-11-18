"""_event_patch_waveforms matches a direct computation on a dense residual, across chunk boundaries, image
edges and clamped windows."""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.gate3 import _extract_event_waveform
from roigbiv.pipeline.residual import ResidualView
from roigbiv.pipeline.stage3 import _event_patch_waveforms


def _direct(res, ev, window):
    T, H, W = res.shape
    ys = np.clip(np.arange(ev["y"] - 1, ev["y"] + 2), 0, H - 1)
    xs = np.clip(np.arange(ev["x"] - 1, ev["x"] + 2), 0, W - 1)
    flat = np.unique((ys[:, None] * W + xs[None, :]).ravel())
    tr = res.reshape(T, H * W)[:, flat].mean(1)
    return _extract_event_waveform(tr, ev["frame"], window)


def test_patch_waveforms_match_direct_computation():
    rng = np.random.default_rng(3)
    res = rng.normal(size=(230, 20, 24)).astype(np.float32)
    events = [
        [{"y": 10, "x": 12, "frame": 95}, {"y": 0, "x": 0, "frame": 3}],     # chunk boundary; corner + window start
        [{"y": 19, "x": 23, "frame": 226}],                                  # far corner + window end
        [],
        [{"y": 10, "x": 12, "frame": 140}, {"y": 5, "x": 23, "frame": 50}],  # shared centre; right edge
    ]
    window = 37
    got = _event_patch_waveforms(ResidualView.from_dense(res), events, window, chunk=50)
    assert [len(g) for g in got] == [len(e) for e in events]
    for evs, ws in zip(events, got):
        for ev, w in zip(evs, ws):
            np.testing.assert_allclose(w, _direct(res, ev, window), rtol=1e-5, atol=1e-5)
