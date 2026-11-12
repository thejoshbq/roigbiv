"""Split audit and the G0 size gate.

A FOV entry is one imaging session. Animals and split groups must not appear
on both sides of the dev/test split, and flagged FOVs stay out of test.
"""
# contract stub: U3 replaces the function bodies below; signatures are frozen.
from __future__ import annotations

from typing import Sequence

from bench.schema import FOV, G0Result, SplitAudit


def audit_split(fovs: Sequence[FOV]) -> SplitAudit:
    """Count and leak-check the given FOVs (all tiers).

    - ``fovs_per_split``: FOVs per split value, e.g. ``{"dev": 3, "test": 1}``;
      only splits that occur are keys.
    - ``animals_per_split``: distinct animals per split, same keys.
    - ``sessions_per_animal``: FOV entries per animal.
    - ``animal_leaks``: animals with FOVs in both ``dev`` and ``test``, sorted.
    - ``group_leaks``: split groups with FOVs in both, sorted.
    - ``flagged_in_test``: ids of ``tier == "flagged"`` FOVs whose split is
      ``test``, sorted.
    """
    return SplitAudit(
        fovs_per_split={}, animals_per_split={}, sessions_per_animal={},
        animal_leaks=(), group_leaks=(), flagged_in_test=(),
    )


def check_g0(
    fovs: Sequence[FOV],
    *,
    min_fovs: int = 8,
    min_animals: int = 3,
    min_movies: int = 3,
    include_flagged: bool = False,
) -> G0Result:
    """Count what G0 counts.

    Only ``tier == "core"`` FOVs count unless ``include_flagged``. A FOV
    counts when it has at least one image and at least one annotation.
    ``n_animals`` is the number of distinct animals among the counted FOVs;
    ``n_movies`` the counted FOVs whose ``movie`` is not ``None``.
    """
    return G0Result(
        n_fovs=0, n_animals=0, n_movies=0,
        min_fovs=min_fovs, min_animals=min_animals, min_movies=min_movies,
    )
