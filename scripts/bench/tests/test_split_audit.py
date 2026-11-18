"""Accept tests for U3 (``bench.split_audit``). Written by the contract; workers do not edit them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.schema import FOV, AnnotationRef, ImageRef, MovieRef  # noqa: E402
from bench.split_audit import audit_split, check_g0  # noqa: E402


def make_fov(fov_id, animal, split="dev", tier="core", group=None, movie=False, images=True, annotations=1):
    return FOV(
        id=fov_id, animal=animal, split_group=group or animal, split=split, tier=tier,
        lens="GRIN", region="r", cell_type="neuron", image_shape=(8, 8),
        images={"std": ImageRef("std.tif", "0" * 64)} if images else {},
        annotations=tuple(AnnotationRef(f"a{i}.zip", "0" * 64, None, 3) for i in range(annotations)),
        movie=MovieRef("tif", "m.tif", 10, "0" * 64) if movie else None,
        fs_hz=None, averaging=None,
    )


def test_clean_split():
    fovs = [
        make_fov("a1", "A"), make_fov("a2", "A"), make_fov("b1", "B", split="test"),
        make_fov("c1", "C", tier="flagged"),
    ]
    a = audit_split(fovs)
    assert a.fovs_per_split == {"dev": 3, "test": 1}
    assert a.animals_per_split == {"dev": 2, "test": 1}
    assert a.sessions_per_animal == {"A": 2, "B": 1, "C": 1}
    assert a.animal_leaks == () and a.group_leaks == () and a.flagged_in_test == ()
    assert a.clean


def test_leaks():
    fovs = [
        make_fov("x1", "A", "dev", group="GA"),
        make_fov("x2", "A", "test", group="GA"),
        make_fov("y1", "B", "dev", group="GX"),
        make_fov("z1", "C", "test", group="GX"),
        make_fov("f1", "F", "test", tier="flagged", group="GF"),
    ]
    a = audit_split(fovs)
    assert a.animal_leaks == ("A",)
    assert a.group_leaks == ("GA", "GX")
    assert a.flagged_in_test == ("f1",)
    assert a.fovs_per_split == {"dev": 2, "test": 3}
    assert a.animals_per_split == {"dev": 2, "test": 3}
    assert not a.clean


def test_leaks_are_sorted():
    fovs = [make_fov(i, i.upper(), s) for i in ("q", "b", "m") for s in ("dev", "test")]
    assert audit_split(fovs).animal_leaks == ("B", "M", "Q")


def _eight_core():
    animals = ["A", "A", "A", "B", "B", "B", "C", "C"]
    return [make_fov(f"f{i}", a, movie=i < 3) for i, a in enumerate(animals)]


def test_g0_pass():
    g = check_g0(_eight_core())
    assert (g.n_fovs, g.n_animals, g.n_movies) == (8, 3, 3)
    assert (g.min_fovs, g.min_animals, g.min_movies) == (8, 3, 3)
    assert g.fovs_ok and g.animals_ok and g.movies_ok and g.passed


def test_g0_counts_only_fovs_with_image_and_annotation():
    fovs = _eight_core()
    fovs[7] = make_fov("f7", "C", annotations=0)
    fovs[6] = make_fov("f6", "C", images=False)
    g = check_g0(fovs)
    assert g.n_fovs == 6 and g.n_animals == 2 and not g.fovs_ok and not g.animals_ok and not g.passed


def test_g0_flagged():
    fovs = [make_fov("a", "A", movie=True), make_fov("b", "B", movie=True),
            make_fov("c", "C", tier="flagged", movie=True)]
    g = check_g0(fovs, min_fovs=2, min_animals=2, min_movies=3)
    assert (g.n_fovs, g.n_animals, g.n_movies) == (2, 2, 2) and not g.movies_ok and not g.passed
    g = check_g0(fovs, min_fovs=2, min_animals=2, min_movies=3, include_flagged=True)
    assert (g.n_fovs, g.n_animals, g.n_movies) == (3, 3, 3) and g.passed


def test_g0_empty():
    g = check_g0([])
    assert (g.n_fovs, g.n_animals, g.n_movies) == (0, 0, 0) and not g.passed
