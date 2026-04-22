from __future__ import annotations

from datetime import date

from roigbiv.registry.filename import parse_filename_metadata


def test_canonical_lab_filename():
    meta = parse_filename_metadata("T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc")
    assert meta.session_date == date(2022, 12, 9)
    assert meta.animal_id == "T1"
    assert meta.region == "PrL-NAc"
    assert meta.fov_number == 1


def test_fov_number_parse():
    meta = parse_filename_metadata("T1_240101_PVT-G6-3M_DAY2_FOV7_BEH")
    assert meta.animal_id == "T1"
    assert meta.fov_number == 7
    assert meta.region == "PVT"


def test_cross_session_same_animal():
    stems = [
        "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002_mc",
        "T1_221215_PrL-NAc-G6-5M_LOW-D1_FOV1_PRE-000_mc",
        "T1_230116_PrL-NAc-G6-5M_EXT-D9_FOV1_EXT-D9_PRE-000_mc",
    ]
    metas = [parse_filename_metadata(s) for s in stems]
    assert {m.animal_id for m in metas} == {"T1"}
    assert {m.region for m in metas} == {"PrL-NAc"}
    assert {m.fov_number for m in metas} == {1}


def test_missing_date_falls_back_to_unknown():
    meta = parse_filename_metadata("some_random_fov_name")
    assert meta.session_date is None
    assert meta.animal_id != ""
