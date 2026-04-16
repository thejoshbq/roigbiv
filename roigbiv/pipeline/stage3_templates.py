"""
ROI G. Biv pipeline — Template bank for Stage 3 matched filtering (spec §9.1).

A bank of 3 normalized calcium-transient waveforms per indicator family.
Each template is A * (1 - exp(-t/tau_rise)) * exp(-t/tau_decay), truncated at
5 * tau_decay, sampled at fs, then L2-normalized to unit energy so match
scores are directly comparable across templates.

Shared by stage3.run_stage3 (for matched filtering) and gate3.evaluate_gate3
(for waveform R² validation).
"""
from __future__ import annotations

import numpy as np


# Indicator-specific kinetics (spec §9.1 table)
_KINETICS = {
    "gcamp6s": [
        # (name,        tau_rise_s, tau_decay_s)
        ("single",      0.05, 1.0),
        ("doublet",     0.075, 1.2),
        ("burst",       0.10, 1.5),
    ],
    "jgcamp8f": [
        ("single",      0.04, 0.5),
        ("doublet",     0.06, 0.6),
        ("burst",       0.08, 0.75),
    ],
}


def _pick_indicator(tau: float) -> str:
    """Select indicator family by decay constant."""
    # Threshold halfway between the two indicators' decays
    return "jgcamp8f" if tau < 0.75 else "gcamp6s"


def _build_waveform(tau_rise: float, tau_decay: float, fs: float) -> np.ndarray:
    """Generate a single L2-normalized calcium transient waveform.

    Truncate at 5 * tau_decay (where amplitude < 1% of peak).
    """
    T_end = 5.0 * tau_decay
    t = np.arange(0.0, T_end, 1.0 / fs, dtype=np.float32)
    w = (1.0 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    norm = float(np.linalg.norm(w))
    if norm > 0:
        w = w / norm
    return w.astype(np.float32)


def build_template_bank(fs: float, tau: float) -> list[tuple[str, np.ndarray]]:
    """Return [(name, waveform), ...] for the indicator family matching `tau`.

    Parameters
    ----------
    fs  : acquisition frame rate (Hz)
    tau : indicator decay constant (s); < 0.75 s → jGCaMP8f, else GCaMP6s.

    Returns
    -------
    list of (name, waveform) tuples. Each waveform is a 1-D float32 array,
    L2-normalized. Length varies across templates within the bank (shorter
    for faster kinetics). Callers that need equal-length templates should
    zero-pad to the longest.
    """
    family = _pick_indicator(tau)
    spec = _KINETICS[family]
    bank = []
    for name, tr, td in spec:
        wf = _build_waveform(tr, td, fs)
        bank.append((f"{name}_{family}", wf))
    return bank
