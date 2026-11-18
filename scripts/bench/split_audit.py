"""Split audit and the G0 size gate.

A FOV entry is one imaging session. Animals and split groups must not appear
on both sides of the dev/test split, and flagged FOVs stay out of test.
"""
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
    fovs_per_split: dict[str, int] = {}
    animals_by_split: dict[str, set[str]] = {}
    sessions_per_animal: dict[str, int] = {}
    splits_by_animal: dict[str, set[str]] = {}
    splits_by_group: dict[str, set[str]] = {}
    for fov in fovs:
        fovs_per_split[fov.split] = fovs_per_split.get(fov.split, 0) + 1
        animals_by_split.setdefault(fov.split, set()).add(fov.animal)
        sessions_per_animal[fov.animal] = sessions_per_animal.get(fov.animal, 0) + 1
        splits_by_animal.setdefault(fov.animal, set()).add(fov.split)
        splits_by_group.setdefault(fov.split_group, set()).add(fov.split)

    animal_leaks = sorted(a for a, s in splits_by_animal.items() if {"dev", "test"} <= s)
    group_leaks = sorted(g for g, s in splits_by_group.items() if {"dev", "test"} <= s)
    flagged_in_test = sorted(
        f.id for f in fovs if f.tier == "flagged" and f.split == "test"
    )
    return SplitAudit(
        fovs_per_split=fovs_per_split,
        animals_per_split={s: len(a) for s, a in animals_by_split.items()},
        sessions_per_animal=sessions_per_animal,
        animal_leaks=tuple(animal_leaks),
        group_leaks=tuple(group_leaks),
        flagged_in_test=tuple(flagged_in_test),
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
    counted = [
        f for f in fovs
        if (include_flagged or f.tier == "core") and f.images and f.annotations
    ]
    return G0Result(
        n_fovs=len(counted),
        n_animals=len({f.animal for f in counted}),
        n_movies=sum(f.movie is not None for f in counted),
        min_fovs=min_fovs, min_animals=min_animals, min_movies=min_movies,
    )
