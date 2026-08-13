"""Parse animal_id / region / session_date / fov_number from a FOV filename stem.

Two date conventions coexist in this lab, and a bare six-digit group is not
self-describing:

    T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc   (YYMMDD)
    052126_DS-Prism-3_VI15_D2_FOV2_beh-006              (MMDDYY)

Rather than pick one and silently misdate the other (``060126`` read as YYMMDD
is 2006-01-26 — wrong by twenty years), each reading is tried and only an
unambiguously valid calendar date is accepted. Genuinely ambiguous stems like
``010203`` are reported via ``date_source`` so the caller — in practice the
HITL session-ordering step — can put the decision in front of a human.

Returns default values (animal_id="unknown", region="unknown", fov_number=1,
session_date=None) when the pattern cannot be matched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

# How ``session_date`` was arrived at. "ambiguous" means both readings were
# valid calendar dates and the MMDDYY one was taken as a guess; "manual" is set
# by callers that override the parsed date outright.
DateSource = str
DATE_SOURCES = ("mmddyy", "yymmdd", "ambiguous", "manual", "unparsed")


@dataclass
class FilenameMetadata:
    animal_id: str
    region: str
    session_date: Optional[date]
    fov_number: int
    date_source: DateSource = "unparsed"


_DATE_RE = re.compile(r"(?:^|_)(\d{6})(?:_|$)")
_FOV_RE = re.compile(r"_FOV(\d+)(?:_|$)")


def _as_date(year2: str, month: str, day: str) -> Optional[date]:
    try:
        return date(2000 + int(year2), int(month), int(day))
    except ValueError:
        return None


def resolve_six_digit_date(digits: str) -> tuple[Optional[date], DateSource]:
    """Resolve a six-digit group to a date, preferring the unambiguous reading.

    Returns ``(None, "unparsed")`` when neither MMDDYY nor YYMMDD names a real
    calendar date, and ``(mmddyy_reading, "ambiguous")`` when both do.
    """
    mmddyy = _as_date(digits[4:6], digits[0:2], digits[2:4])
    yymmdd = _as_date(digits[0:2], digits[2:4], digits[4:6])

    if mmddyy is not None and yymmdd is not None:
        return mmddyy, "ambiguous"
    if mmddyy is not None:
        return mmddyy, "mmddyy"
    if yymmdd is not None:
        return yymmdd, "yymmdd"
    return None, "unparsed"


def parse_filename_metadata(stem: str) -> FilenameMetadata:
    clean = stem.replace("_mc", "").strip("_")

    session_date: Optional[date] = None
    date_source: DateSource = "unparsed"
    animal_part = ""
    after_date = ""

    m_date = _DATE_RE.search(clean)
    if m_date:
        session_date, date_source = resolve_six_digit_date(m_date.group(1))
        animal_part = clean[:m_date.start(1)].rstrip("_")
        after_date = clean[m_date.end():].lstrip("_")
    else:
        animal_part = clean

    fov_number = 1
    indicator_part = after_date
    m_fov = _FOV_RE.search("_" + after_date) if after_date else None
    if m_fov:
        fov_number = int(m_fov.group(1))
        indicator_part = after_date[:m_fov.start() - 1] if m_fov.start() > 0 else ""

    # Date-leading stems (the prism convention) have nothing before the date to
    # name the animal. Falling back to "unknown" would collapse every animal's
    # prism FOVs into one candidate pool in `find_candidates`, so take the first
    # post-date segment ("DS-Prism-3") as the subject instead.
    if not animal_part and after_date:
        animal_part = after_date.split("_")[0]

    animal_id = animal_part or "unknown"
    region = _extract_region(indicator_part) if indicator_part else "unknown"

    return FilenameMetadata(
        animal_id=animal_id,
        region=region,
        session_date=session_date,
        fov_number=fov_number,
        date_source=date_source,
    )


def _extract_region(indicator: str) -> str:
    if not indicator or indicator == "unknown":
        return "unknown"
    first_segment = indicator.split("_")[0]
    tokens = first_segment.split("-")
    region_tokens: list[str] = []
    for tok in tokens:
        if any(ch.isdigit() for ch in tok):
            break
        region_tokens.append(tok)
    return "-".join(region_tokens) if region_tokens else first_segment
