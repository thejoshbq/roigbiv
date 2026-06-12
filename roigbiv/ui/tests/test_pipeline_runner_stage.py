"""The Run-status banner derives the current stage from log markers.

``PipelineRunner._append_and_tally`` parses ``fmt.stage_header`` lines streamed
by the pipeline and exposes the latest as ``RunSnapshot.current_stage`` so the
UI can name what the pipeline is doing instead of a static "run started".
"""
import threading

import pytest

from roigbiv.pipeline import fmt
from roigbiv.ui.services.pipeline_runner import (
    PipelineRunner,
    _derive_stage,
)


def _runner():
    return PipelineRunner(threading.Lock())


def test_derive_stage_maps_each_marker():
    cases = {
        fmt.stage_header(1, "Cellpose detection"): "Stage 1 · Cellpose detection",
        fmt.stage_header("1→S", "Source subtraction"): "Source subtraction",
        fmt.stage_header(2, "Suite2p temporal detection"):
            "Stage 2 · Temporal detection",
        fmt.stage_header(3, "Template sweep on residual view"):
            "Stage 3 · Template sweep",
        fmt.stage_header(4, "Tonic neuron search on residual view"):
            "Stage 4 · Tonic search",
        fmt.stage_header("Post", "Trace extraction + QC"): "Trace extraction + QC",
    }
    for marker, expected in cases.items():
        # stage_header returns a leading newline + the rule line.
        line = marker.strip()
        assert _derive_stage(line) == expected


def test_derive_stage_keeps_fov_prefix():
    line = "[FOV 2/5] " + fmt.stage_header(3, "Template sweep").strip()
    assert _derive_stage(line) == "FOV 2/5 · Stage 3 · Template sweep"


def test_derive_stage_ignores_non_markers():
    for line in (fmt.gate_outcome(1, 10, 8, 1, 1),
                 fmt.sub_phase("Foundation complete. k_background=30"),
                 "pipeline OK (12.3s) — accept=5 flag=1 reject=2",
                 "--- Stage 99: unknown phase ---"):
        assert _derive_stage(line) is None


def test_current_stage_transitions_through_a_run():
    runner = _runner()
    assert runner.snapshot().current_stage is None
    sequence = [
        (fmt.sub_phase("Foundation complete."), None),  # not a stage marker
        (fmt.stage_header(1, "Cellpose detection"), "Stage 1 · Cellpose detection"),
        (fmt.gate_outcome(1, 10, 8, 1, 1), "Stage 1 · Cellpose detection"),  # sticky
        (fmt.stage_header("1→S", "Source subtraction"), "Source subtraction"),
        (fmt.stage_header(2, "Suite2p temporal detection"),
         "Stage 2 · Temporal detection"),
        (fmt.stage_header(3, "Template sweep on residual view"),
         "Stage 3 · Template sweep"),
        (fmt.stage_header(4, "Tonic neuron search on residual view"),
         "Stage 4 · Tonic search"),
        (fmt.stage_header("Post", "Trace extraction + QC"), "Trace extraction + QC"),
    ]
    for marker, expected_after in sequence:
        for line in marker.splitlines():
            runner._append_and_tally(line)
        if expected_after is not None:
            assert runner.snapshot().current_stage == expected_after


def test_pipeline_ok_counts_done():
    runner = _runner()
    runner._append_and_tally("pipeline OK (1.0s) — accept=1 flag=0 reject=0")
    assert runner.snapshot().n_done == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
