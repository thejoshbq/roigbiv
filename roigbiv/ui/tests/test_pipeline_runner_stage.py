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


# ── Stop (cooperative abort) ─────────────────────────────────────────────────

def test_abort_noop_when_no_run_active():
    runner = _runner()
    assert runner.abort() is False
    snap = runner.snapshot()
    assert snap.stopping is False
    assert snap.stopped is False


def test_abort_sets_stopping_then_stopped():
    runner = _runner()
    runner._active = True              # simulate an in-flight run
    assert runner.abort() is True
    assert runner._abort_event.is_set()
    # While still active: stopping is True, stopped is False.
    snap = runner.snapshot()
    assert snap.stopping is True
    assert snap.stopped is False
    # When the run ends with the event set: stopped flips True, stopping clears.
    runner._active = False
    snap = runner.snapshot()
    assert snap.stopping is False
    assert snap.stopped is True


def test_stopped_is_independent_of_awaiting():
    # A stopped run is distinct from a run paused for optics confirmation.
    runner = _runner()
    runner._active = True
    runner.abort()
    runner._active = False
    snap = runner.snapshot()
    assert snap.stopped is True
    assert snap.n_awaiting == 0


# ── Launched-config echo (overrides capture) ─────────────────────────────────

def test_banner_error_wins_over_stopped():
    # A crash on the post-stop path still leaves the abort event set; the banner
    # must surface the failure, not mask it as a clean "Run stopped."
    import time

    from roigbiv.ui.pages.process import _render_banner
    from roigbiv.ui.services.pipeline_runner import RunSnapshot

    snap = RunSnapshot(
        active=False, started_at=time.time(), completed_at=time.time(),
        n_fovs=1, n_done=0, n_failed=1, logs=[],
        error="RuntimeError: backfill blew up", stopped=True, stopping=False)
    alert = _render_banner(snap)
    assert "failed" in str(alert.children).lower()
    assert alert.color == "danger"


def test_overrides_exposed_in_snapshot_and_cleared_on_reset():
    runner = _runner()
    runner._overrides = {"fs": 7.5, "tau": 1.0}
    assert runner.snapshot().overrides == {"fs": 7.5, "tau": 1.0}
    # Snapshot returns a copy — callers can't mutate runner state.
    runner.snapshot().overrides["fs"] = 30.0
    assert runner._overrides["fs"] == 7.5
    runner._reset_locked()
    assert runner.snapshot().overrides is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
