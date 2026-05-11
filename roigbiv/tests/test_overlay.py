"""Tests for `roigbiv.overlay.render_overlay` and the CLI parser.

Synthetic fixtures only — no pipeline run, no GPU. Exercises the new
`outcomes=` keyword and the corresponding `--overlay-outcomes` parser.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pytest

from roigbiv import overlay as _ov
from roigbiv.pipeline.run import _parse_overlay_outcomes
from roigbiv.pipeline.types import FOVData, ROI


_TS = _dt.datetime(2026, 5, 1, 12, 0, 0)
_H = _W = 64


def _disk_mask(cy: int, cx: int, radius: int = 4) -> np.ndarray:
    yy, xx = np.ogrid[:_H, :_W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius


def _build_fov(tmp_path: Path) -> FOVData:
    rng = np.random.default_rng(0)
    mean_M = rng.uniform(0.0, 1.0, size=(_H, _W)).astype(np.float32)
    rois = [
        ROI(
            mask=_disk_mask(16, 16),
            label_id=1, source_stage=1, confidence="high", gate_outcome="accept",
        ),
        ROI(
            mask=_disk_mask(16, 48),
            label_id=2, source_stage=1, confidence="moderate", gate_outcome="flag",
        ),
        ROI(
            mask=_disk_mask(48, 32),
            label_id=3, source_stage=1, confidence="requires_review", gate_outcome="reject",
        ),
    ]
    return FOVData(
        raw_path=tmp_path / "fake.tif",
        output_dir=tmp_path,
        data_bin_path=tmp_path / "data.bin",
        shape=(1, _H, _W),
        residual_S_path=tmp_path / "residual_S.bin",
        mean_M=mean_M,
        rois=rois,
    )


def _color_present(arr: np.ndarray, hex_color: str, region: tuple[slice, slice]) -> bool:
    """True iff any pixel in `arr[region]` matches `hex_color` exactly.

    PIL `draw.line` at width=1 along integer-rounded contour points
    produces unanti-aliased pixels, so an exact RGB match is reliable.
    """
    rgb = (
        int(hex_color[1:3], 16),
        int(hex_color[3:5], 16),
        int(hex_color[5:7], 16),
    )
    sub = arr[region]
    matches = (
        (sub[..., 0] == rgb[0])
        & (sub[..., 1] == rgb[1])
        & (sub[..., 2] == rgb[2])
    )
    return bool(matches.any())


def _read_png_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


# Per-ROI bounding regions, padded by 2px so the contour ring is fully inside.
_ACCEPT_REGION = (slice(10, 23), slice(10, 23))   # (cy=16, cx=16, r=4)
_FLAG_REGION = (slice(10, 23), slice(42, 55))     # (cy=16, cx=48, r=4)
_REJECT_REGION = (slice(42, 55), slice(26, 39))   # (cy=48, cx=32, r=4)


def _stub_annotation(monkeypatch) -> None:
    """Disable the annotation overlay so exact-color contour checks aren't
    dimmed by the alpha-170 black rectangle that normally covers the
    annotation region (which spans the full image on these small fixtures).
    """
    monkeypatch.setattr(_ov, "_draw_annotation", lambda img, lines: img)


def test_default_draws_all_three_outcomes(tmp_path: Path, monkeypatch) -> None:
    _stub_annotation(monkeypatch)
    fov = _build_fov(tmp_path)
    out_path = _ov.render_overlay(
        fov, tmp_path, model_name="test_model", fov_stem="fov_a",
        timestamp=_TS,
    )
    assert out_path.exists()
    rgb = _read_png_rgb(out_path)
    assert _color_present(rgb, _ov._ACCEPT_COLOR, _ACCEPT_REGION)
    assert _color_present(rgb, _ov._FLAG_COLOR, _FLAG_REGION)
    assert _color_present(rgb, _ov._REJECT_COLOR, _REJECT_REGION)


def test_filter_excludes_reject(tmp_path: Path, monkeypatch) -> None:
    _stub_annotation(monkeypatch)
    fov = _build_fov(tmp_path)
    out_path = _ov.render_overlay(
        fov, tmp_path, model_name="test_model", fov_stem="fov_b",
        timestamp=_TS, outcomes=("accept", "flag"),
    )
    rgb = _read_png_rgb(out_path)
    assert _color_present(rgb, _ov._ACCEPT_COLOR, _ACCEPT_REGION)
    assert _color_present(rgb, _ov._FLAG_COLOR, _FLAG_REGION)
    assert not _color_present(rgb, _ov._REJECT_COLOR, _REJECT_REGION)


def test_annotation_always_reports_all_three_counts(tmp_path: Path, monkeypatch) -> None:
    fov = _build_fov(tmp_path)
    captured: list[list[str]] = []

    real_draw = _ov._draw_annotation

    def _capture(img, lines):
        captured.append(list(lines))
        return real_draw(img, lines)

    monkeypatch.setattr(_ov, "_draw_annotation", _capture)

    _ov.render_overlay(
        fov, tmp_path, model_name="m", fov_stem="fov_c",
        timestamp=_TS,
    )
    _ov.render_overlay(
        fov, tmp_path, model_name="m", fov_stem="fov_d",
        timestamp=_TS, outcomes=("accept",),
    )

    assert len(captured) == 2
    for lines in captured:
        count_line = next(line for line in lines if "ROIs:" in line)
        assert "1 accept" in count_line
        assert "1 flag" in count_line
        assert "1 reject" in count_line


def test_drawn_suffix_only_when_filtered(tmp_path: Path, monkeypatch) -> None:
    fov = _build_fov(tmp_path)
    captured: list[list[str]] = []
    real_draw = _ov._draw_annotation

    def _capture(img, lines):
        captured.append(list(lines))
        return real_draw(img, lines)

    monkeypatch.setattr(_ov, "_draw_annotation", _capture)

    _ov.render_overlay(
        fov, tmp_path, model_name="m", fov_stem="fov_e",
        timestamp=_TS,
    )
    _ov.render_overlay(
        fov, tmp_path, model_name="m", fov_stem="fov_f",
        timestamp=_TS, outcomes=("accept", "flag"),
    )

    default_lines, filtered_lines = captured
    assert not any(line.startswith("drawn:") for line in default_lines)
    drawn_line = next(line for line in filtered_lines if line.startswith("drawn:"))
    assert drawn_line == "drawn: accept,flag"


# ── _parse_overlay_outcomes ─────────────────────────────────────────────


def test_parser_accepts_canonical_form() -> None:
    assert _parse_overlay_outcomes("accept,flag,reject") == (
        "accept", "flag", "reject",
    )


def test_parser_lowercases_and_strips() -> None:
    assert _parse_overlay_outcomes(" Accept , FLAG ") == ("accept", "flag")


def test_parser_deduplicates() -> None:
    assert _parse_overlay_outcomes("accept,accept,flag") == ("accept", "flag")


def test_parser_rejects_empty() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_overlay_outcomes("")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_overlay_outcomes(" , ")


def test_parser_rejects_unknown_tokens() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="invalid"):
        _parse_overlay_outcomes("accept,maybe")
