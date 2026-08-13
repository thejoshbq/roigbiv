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
    assert meta.date_source == "unparsed"
    assert meta.animal_id != ""


def test_yymmdd_stems_report_their_source():
    meta = parse_filename_metadata("T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc")
    # Month 22 is not a real month, so YYMMDD is the only valid reading.
    assert meta.date_source == "yymmdd"


def test_prism_stem_reads_as_mmddyy():
    # 052126 -> YYMMDD would be month 21; MMDDYY is the only valid reading.
    meta = parse_filename_metadata("052126_DS-Prism-3_VI15_D2_FOV2_beh-006")
    assert meta.session_date == date(2026, 5, 21)
    assert meta.date_source == "mmddyy"
    assert meta.region == "DS-Prism"
    assert meta.fov_number == 2


def test_prism_stem_that_used_to_silently_misdate():
    # Regression: 060126 parsed as YYMMDD gives 2006-01-26 — a valid date, and
    # wrong by twenty years. MMDDYY is the only reading that is *not* also
    # valid as YYMMDD... it is, so this lands in the ambiguous bucket rather
    # than silently taking the wrong one.
    meta = parse_filename_metadata("060126_DS-Prism-3_VI15_D3_FOV2_beh-007")
    assert meta.date_source == "ambiguous"
    assert meta.session_date == date(2026, 6, 1)


def test_date_leading_stem_names_the_animal():
    # Without this, every prism FOV in the lab parses to animal_id="unknown"
    # and they all become each other's match candidates.
    meta = parse_filename_metadata("052126_DS-Prism-3_VI15_D2_FOV2_beh-006")
    assert meta.animal_id == "DS-Prism-3"


def test_distinct_prism_animals_do_not_share_a_candidate_pool():
    a = parse_filename_metadata("052126_DS-Prism-3_VI15_D2_FOV2_beh-006")
    b = parse_filename_metadata("052126_DS-Prism-4_VI15_D2_FOV2_beh-006")
    assert a.animal_id != b.animal_id


def test_fully_ambiguous_six_digit_group_is_flagged():
    meta = parse_filename_metadata("010203_DS-Prism-3_VI15_D2_FOV1_beh-001")
    assert meta.date_source == "ambiguous"
    # The MMDDYY reading is taken as the guess; the flag is what matters.
    assert meta.session_date == date(2003, 1, 2)


def test_impossible_six_digit_group_is_unparsed():
    meta = parse_filename_metadata("999999_DS-Prism-3_VI15_D2_FOV1_beh-001")
    assert meta.session_date is None
    assert meta.date_source == "unparsed"
