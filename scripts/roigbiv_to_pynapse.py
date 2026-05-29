#!/usr/bin/env python
"""Minimal example: load a ROIGBIV traces bundle into pynapse.

ROIGBIV writes:
    {fov_output_dir}/traces/
        traces.npy           # float32 [n_rois, n_frames] — neuropil-corrected (PRIMARY)
        traces_raw.npy       # raw fluorescence
        traces_neuropil.npy  # neuropil estimate
        traces_meta.json     # row-to-ID + fs + frame_averaging + session/FOV IDs

`pynapse.SignalRecording` loads `.npy` directly and identifies neurons by
row index only. The row-to-ID mapping lives in the sidecar, so you join
it back to traces by `row_index`.

Frame-rate convention: roigbiv's ``fs`` is the EFFECTIVE rate (already
post-binning; e.g. 7.5 Hz for a 4x-averaged 30 Hz stack). Pynapse expects
``fps`` as the RAW acquisition rate and computes
``effective_fps = fps / frame_averaging``. So the handoff multiplies
``fs * frame_averaging`` to reconstruct the raw rate pynapse expects.

Usage
-----
    python scripts/roigbiv_to_pynapse.py <fov_output_dir> [<event_log>]

If ``<event_log>`` is omitted, only the ``SignalRecording`` is printed —
building a full ``Sample`` requires a REACHER event log.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_signals(fov_output_dir: Path):
    """Load the primary traces bundle as a pynapse ``SignalRecording`` +
    its sidecar metadata.

    Returns
    -------
    (signal, meta) : tuple[SignalRecording, dict]
    """
    from pynapse.core.io.microscopy import SignalRecording

    traces_dir = Path(fov_output_dir) / "traces"
    traces_npy = traces_dir / "traces.npy"
    sidecar_path = traces_dir / "traces_meta.json"
    if not traces_npy.exists():
        raise FileNotFoundError(
            f"{traces_npy} not found — run the pipeline to produce a traces/ "
            "bundle, or point at the right FOV output directory."
        )
    if not sidecar_path.exists():
        raise FileNotFoundError(f"{sidecar_path} not found")

    meta = json.loads(sidecar_path.read_text())
    signal = SignalRecording(source=str(traces_npy), name=fov_output_dir.name)
    return signal, meta


def build_sample(
    fov_output_dir: Path,
    event_log_source,
    *,
    name: str | None = None,
):
    """Build a pynapse ``Sample`` from a ROIGBIV traces bundle + a REACHER
    event log source.

    The ``fps`` passed to ``Sample`` is the *raw* acquisition rate
    (``meta["fs"] * meta["frame_averaging"]``). Pynapse computes
    ``effective_fps = fps / frame_averaging``, which equals ``meta["fs"]``
    — i.e. the rate of the frames actually stored in ``traces.npy``.
    """
    from pynapse.core.sample import Sample

    signal, meta = load_signals(fov_output_dir)
    fs = float(meta["fs"])
    frame_averaging = int(meta.get("frame_averaging", 1))
    raw_fps = fs * frame_averaging  # pynapse's fps is the un-averaged rate

    sample = Sample(
        event_data=event_log_source,
        signal_data=signal,
        fps=raw_fps,
        frame_averaging=frame_averaging,
        name=name or fov_output_dir.name,
    )
    return sample, meta


def roi_row_map(meta: dict) -> list[dict]:
    """Return the ``rois[]`` block — canonical row-to-ID mapping.

    Each entry has ``row_index`` (int), ``local_label_id`` (int), and,
    when the FOV was registered, ``global_cell_id`` (UUID string). Use
    ``row_index`` to index into ``traces.npy`` for a specific neuron.
    """
    return list(meta.get("rois") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fov_output_dir", type=Path,
                        help="FOV output directory containing traces/")
    parser.add_argument("event_log", nargs="?", default=None,
                        help="REACHER event log path (optional)")
    args = parser.parse_args()

    signal, meta = load_signals(args.fov_output_dir)
    print(signal)
    print("\nSidecar summary:")
    print(f"  fs               = {meta['fs']} Hz (effective)")
    print(f"  frame_averaging  = {meta['frame_averaging']}")
    print(f"  effective_fps    = {meta['effective_fps']} Hz")
    print(f"  n_rois           = {meta['n_rois']}")
    print(f"  n_frames         = {meta['n_frames']}")
    print(f"  session_id       = {meta.get('session_id')}")
    print(f"  fov_id           = {meta.get('fov_id')}")
    print(f"  registry_decision= {meta.get('registry_decision')}")
    matched = sum(1 for r in meta.get("rois", []) if "global_cell_id" in r)
    print(f"  rois with global_cell_id: {matched}/{meta['n_rois']}")

    if args.event_log:
        sample, _ = build_sample(args.fov_output_dir, args.event_log)
        print(
            f"\nBuilt Sample: num_neurons={sample.num_neurons}, "
            f"num_frames={sample.num_frames}, "
            f"effective_fps={sample.effective_fps}"
        )


if __name__ == "__main__":
    main()
