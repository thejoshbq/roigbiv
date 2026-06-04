"""Unit tests for roigbiv.pipeline.fmt terminal formatting helpers."""
import re

import pytest

from roigbiv.pipeline import fmt


def test_fov_banner_structure():
    out = fmt.fov_banner("fov_A", 1, 3)
    lines = out.splitlines()
    assert len(lines) == 4  # leading \n produces empty first line
    rule_line = lines[1]
    label_line = lines[2]
    end_rule = lines[3]
    assert rule_line == "=" * 72
    assert end_rule == "=" * 72
    assert "FOV 1/3" in label_line
    assert "fov_A" in label_line


def test_fov_banner_custom_width():
    out = fmt.fov_banner("fov_B", 2, 5, width=40)
    lines = out.splitlines()
    assert lines[1] == "=" * 40


def test_fov_separator_length():
    out = fmt.fov_separator()
    assert out.startswith("\n")
    assert "-" * 72 in out


def test_stage_header_structure():
    out = fmt.stage_header(1, "Cellpose detection")
    assert "Stage 1:" in out
    assert "Cellpose detection" in out
    assert out.startswith("\n")
    assert "---" in out


def test_stage_header_fits_80col():
    # Simulate worst-case batch prefix: "[FOV 10/10] " = 13 chars
    out = fmt.stage_header(3, "Template sweep")
    content_line = out.splitlines()[-1]
    assert len(content_line) + 13 <= 80


def test_stage_header_long_label_no_exception():
    # Long label should not crash — just clips the trailing dashes
    out = fmt.stage_header("Post", "A" * 80)
    assert isinstance(out, str)


def test_gate_outcome_tokens():
    out = fmt.gate_outcome(1, 87, 72, 8, 7)
    assert "Gate 1" in out
    assert "87" in out
    assert "72" in out
    assert "8" in out
    assert "7" in out
    assert "|" in out
    assert out.startswith("  ")


def test_gate_outcome_stage4_zero_accept():
    out = fmt.gate_outcome(4, 12, 0, 5, 7)
    assert "0 accept" in out


def test_sub_phase_no_timing():
    out = fmt.sub_phase("motion correction")
    assert out == "  motion correction"


def test_sub_phase_with_timing():
    out = fmt.sub_phase("SVD", elapsed_s=8.4)
    assert "[8.4s]" in out
    assert out.startswith("  SVD")


def test_sub_phase_timing_format():
    out = fmt.sub_phase("x", elapsed_s=0.05)
    assert "[0.1s]" in out or "[0.0s]" in out  # rounds to 1 decimal


def test_stage_done():
    out = fmt.stage_done(12.3)
    assert "done" in out
    assert "[12.3s]" in out


def test_pipeline_complete_no_timing():
    out = fmt.pipeline_complete("fov_test")
    lines = out.splitlines()
    assert lines[1] == "=" * 72
    assert "Pipeline complete" in lines[2]
    assert "fov_test" in lines[2]
    assert lines[3] == "=" * 72


def test_pipeline_complete_with_timing_minutes():
    out = fmt.pipeline_complete("fov_test", total_s=134.0)
    assert "[2m 14s]" in out


def test_pipeline_complete_with_timing_seconds_only():
    out = fmt.pipeline_complete("fov_test", total_s=45.0)
    assert "[45s]" in out
    assert "m" not in out.split("[")[-1]


def test_no_ansi_sequences():
    ansi_re = re.compile(r"\x1b\[")
    outputs = [
        fmt.fov_banner("test", 1, 1),
        fmt.fov_separator(),
        fmt.stage_header(1, "Cellpose detection"),
        fmt.gate_outcome(1, 10, 8, 1, 1),
        fmt.sub_phase("step", 1.0),
        fmt.sub_phase("step"),
        fmt.stage_done(5.0),
        fmt.pipeline_complete("fov", 90.0),
    ]
    for out in outputs:
        assert not ansi_re.search(out), f"ANSI escape found in: {out!r}"


def test_all_return_str():
    assert isinstance(fmt.fov_banner("x", 1, 1), str)
    assert isinstance(fmt.fov_separator(), str)
    assert isinstance(fmt.stage_header(1, "label"), str)
    assert isinstance(fmt.gate_outcome(1, 5, 4, 0, 1), str)
    assert isinstance(fmt.sub_phase("label"), str)
    assert isinstance(fmt.sub_phase("label", 1.0), str)
    assert isinstance(fmt.stage_done(1.0), str)
    assert isinstance(fmt.pipeline_complete("name"), str)
    assert isinstance(fmt.pipeline_complete("name", 60.0), str)
