from __future__ import annotations

from datetime import date

from roigbiv.registry.filename import parse_filename_metadata


def test_canonical_lab_filename():
    meta = parse_filename_metadata("T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc")
    assert meta.session_date == date(2022, 12, 9)
    assert meta.animal_id == "PrL-NAc-G6-5M_HI-D1"
    assert meta.region == "PrL-NAc"
    assert meta.fov_number == 1


def test_fov_number_parse():
    meta = parse_filename_metadata("T1_240101_PVT-G6-3M_DAY2_FOV7_BEH")
    assert meta.fov_number == 7
    assert meta.region == "PVT"


def test_missing_date_falls_back_to_unknown():
    meta = parse_filename_metadata("some_random_fov_name")
    assert meta.session_date is None
    assert meta.animal_id != ""
