"""Guards for the shared run-status panel.

One runner per browser session behind one GPU gate, so the Motion-correction
and Centroids pages both look at this. Two consequences these cases pin:

* the banner has to *name* the run, or a centroid run started next door reads
  on the motion page as the motion run having stalled;
* the results table has to drop columns the run did not produce, or a
  centroids-only run shows four em-dashed MC metrics that read as failure.
"""
from __future__ import annotations

import time

import pytest

from roigbiv.ui.components import run_panel
from roigbiv.ui.services.pipeline_runner import RunSnapshot
from roigbiv.ui.tests._tree import text, walk


def _snap(**kw) -> RunSnapshot:
    base = dict(active=False, started_at=None, completed_at=None,
                n_fovs=0, n_done=0, n_failed=0, logs=[], error=None)
    base.update(kw)
    return RunSnapshot(**base)


def _headers(table) -> list:
    return [c.children for c in walk(table) if type(c).__name__ == "Th"]


# ── naming the run ─────────────────────────────────────────────────────────


def test_run_mode_reads_off_the_overrides():
    assert run_panel.run_mode({"foundation_only": True}) == "motion correction"
    assert run_panel.run_mode({"run_centroids": True}) == "centroid discovery"
    assert run_panel.run_mode(None) == "pipeline"


def test_the_banner_names_which_run_is_active():
    """A page showing a foreign run must say whose it is."""
    snap = _snap(active=True, started_at=time.time(),
                 overrides={"run_centroids": True},
                 current_stage="Stage 1 · Cellpose detection")
    assert "centroid discovery" in text(run_panel.render_banner(snap))


def test_the_banner_is_absent_before_any_run():
    assert run_panel.render_banner(_snap()) is None
    assert run_panel.render_banner(None) is None


def test_an_error_outranks_a_stop():
    """A crash on the post-stop path still sets the abort event; without the
    guard the failure would be masked as a clean "Run stopped."."""
    snap = _snap(started_at=time.time(), error="boom", stopped=True)
    assert "failed" in text(run_panel.render_banner(snap)).lower()


def test_a_stop_outranks_a_clean_completion():
    snap = _snap(started_at=time.time(), completed_at=time.time(), stopped=True)
    assert "stopped" in text(run_panel.render_banner(snap)).lower()


def test_stopping_is_distinguished_from_stopped():
    snap = _snap(active=True, started_at=time.time(), stopping=True,
                 current_stage="Foundation")
    body = text(run_panel.render_banner(snap))
    assert "Stopping" in body and "Foundation" in body


# ── progress ───────────────────────────────────────────────────────────────


def test_progress_counts_failures_as_done():
    """A failed FOV is finished with, and a bar that never fills reads as hung."""
    assert run_panel.progress_for(_snap(n_fovs=4, n_done=1, n_failed=1)) == (50, "2 / 4")


def test_progress_is_zero_before_a_run():
    assert run_panel.progress_for(None) == (0, "")
    assert run_panel.progress_for(_snap()) == (0, "")


# ── the results table ──────────────────────────────────────────────────────


def test_no_results_says_so_rather_than_rendering_an_empty_table():
    assert "No FOV results" in text(run_panel.render_results([]))


def test_a_motion_run_shows_metrics_and_no_centroid_column():
    table = run_panel.render_results([
        {"stem": "a", "duration_s": 1.0,
         "mc_metrics": {"lap_var_smooth": 0.5, "banding_score": 0.1,
                        "grad_anisotropy_xy": 0.2, "contrast_rms": 0.3}},
    ])
    heads = _headers(table)
    assert "Sharpness" in heads and "Contrast" in heads
    assert "Centroids" not in heads


def test_a_centroids_run_shows_counts_and_no_metric_columns():
    table = run_panel.render_results([
        {"stem": "a", "duration_s": 1.0, "centroid_count": 20},
    ])
    heads = _headers(table)
    assert "Centroids" in heads
    for metric in ("Sharpness", "Banding", "Anisotropy", "Contrast"):
        assert metric not in heads, (
            f"{metric} has no value in a centroids-only run and reads as failure")


def test_a_combined_run_shows_both():
    table = run_panel.render_results([
        {"stem": "a", "duration_s": 1.0, "centroid_count": 20,
         "mc_metrics": {"lap_var_smooth": 0.5}},
    ])
    heads = _headers(table)
    assert "Sharpness" in heads and "Centroids" in heads


def test_a_failed_fov_is_marked():
    assert "FAILED" in text(run_panel.render_results(
        [{"stem": "a", "duration_s": 1.0, "error": "boom"}]))


# ── the launched-config echo ───────────────────────────────────────────────


def test_launched_config_echoes_the_overrides_that_started_the_run():
    """Read from the snapshot, not the form: the form is persisted and editable
    mid-run, and would misrepresent what is actually executing."""
    card = run_panel.launched_config(_snap(
        started_at=time.time(), completed_at=time.time(), n_fovs=3, n_done=3,
        overrides={"fs": 7.5, "tau": 1.0, "foundation_only": True,
                   "motion_correction_backend": "phasecorr"}))
    body = text(card)
    assert "phasecorr" in body and "motion correction" in body


def test_launched_config_is_absent_before_a_run():
    assert run_panel.launched_config(_snap()) is None
    assert run_panel.launched_config(None) is None


# ── the timer ──────────────────────────────────────────────────────────────


def test_timer_is_blank_before_a_run():
    assert run_panel.format_timer(None, None) == ""


def test_timer_reports_elapsed_against_completion_once_finished():
    start = time.time() - 3661
    body = text(run_panel.format_timer(start, start + 3661))
    assert "01:01:01" in body


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
