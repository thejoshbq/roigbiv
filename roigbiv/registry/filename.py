"""Parse animal_id / region / session_date / fov_number from a FOV filename stem.

Lab convention example:
    T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002_mc

Returns default values (animal_id="unknown", region="unknown", fov_number=1,
session_date=None) when the pattern cannot be matched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class FilenameMetadata:
    animal_id: str
    region: str
    session_date: Optional[date]
    fov_number: int


_DATE_RE = re.compile(r"(?:^|_)(\d{6})(?:_|$)")
_FOV_RE = re.compile(r"_FOV(\d+)(?:_|$)")


def parse_filename_metadata(stem: str) -> FilenameMetadata:
    clean = stem.replace("_mc", "").strip("_")

    session_date: Optional[date] = None
    after_date = clean
    m_date = _DATE_RE.search(clean)
    if m_date:
        yymmdd = m_date.group(1)
        try:
            year = 2000 + int(yymmdd[:2])
            session_date = date(year, int(yymmdd[2:4]), int(yymmdd[4:6]))
        except ValueError:
            session_date = None
        after_date = clean[m_date.end():]

    fov_number = 1
    animal_part = after_date
    m_fov = _FOV_RE.search("_" + after_date)
    if m_fov:
        fov_number = int(m_fov.group(1))
        animal_part = after_date[:m_fov.start() - 1] if m_fov.start() > 0 else ""
    animal_id = animal_part.strip("_") or "unknown"

    region = _extract_region(animal_id)

    return FilenameMetadata(
        animal_id=animal_id,
        region=region,
        session_date=session_date,
        fov_number=fov_number,
    )


def _extract_region(animal_id: str) -> str:
    if not animal_id or animal_id == "unknown":
        return "unknown"
    first_segment = animal_id.split("_")[0]
    tokens = first_segment.split("-")
    region_tokens: list[str] = []
    for tok in tokens:
        if any(ch.isdigit() for ch in tok):
            break
        region_tokens.append(tok)
    return "-".join(region_tokens) if region_tokens else first_segment
