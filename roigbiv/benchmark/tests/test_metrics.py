"""Tests for roigbiv.benchmark.metrics — JSON round-trip coverage."""
from __future__ import annotations

import json

import pytest

from roigbiv.benchmark.metrics import (
    DetectionMetrics,
    HitlMetrics,
    RuntimeMetrics,
    TraceMetrics,
    TrackingMetrics,
)

METRIC_CLASSES = [
    DetectionMetrics,
    TrackingMetrics,
    RuntimeMetrics,
    HitlMetrics,
    TraceMetrics,
]


def _populated_kwargs(cls) -> dict:
    """Build a populated instance for `cls` using distinct, type-appropriate values.

    Relies on `from __future__ import annotations` in metrics.py, which stores
    each field's `.type` as its literal source string (e.g. "Optional[int]").
    Matched by exact membership rather than substring to avoid misclassifying
    a future field whose type name happens to contain "int".
    """
    kwargs = {}
    for i, field_name in enumerate(cls.__dataclass_fields__):
        field_type = cls.__dataclass_fields__[field_name].type
        if field_type == "Optional[int]":
            kwargs[field_name] = i + 1
        elif field_type == "Optional[float]":
            kwargs[field_name] = round(0.1 * (i + 1), 3)
        else:
            raise AssertionError(f"unhandled field type {field_type!r} on {cls.__name__}.{field_name}")
    return kwargs


@pytest.mark.parametrize("cls", METRIC_CLASSES)
def test_json_roundtrip_populated(cls):
    obj = cls(**_populated_kwargs(cls))
    payload = obj.to_dict()
    rebuilt = cls.from_dict(json.loads(json.dumps(payload)))
    assert rebuilt == obj


@pytest.mark.parametrize("cls", METRIC_CLASSES)
def test_empty_all_none(cls):
    obj = cls.empty()
    for field_name in cls.__dataclass_fields__:
        assert getattr(obj, field_name) is None


@pytest.mark.parametrize("cls", METRIC_CLASSES)
def test_empty_roundtrips(cls):
    obj = cls.empty()
    payload = obj.to_dict()
    rebuilt = cls.from_dict(json.loads(json.dumps(payload)))
    assert rebuilt == obj
    assert rebuilt == cls()


@pytest.mark.parametrize("cls", METRIC_CLASSES)
def test_from_dict_reconstructs_equivalent_instance(cls):
    obj = cls(**_populated_kwargs(cls))
    rebuilt = cls.from_dict(obj.to_dict())
    assert rebuilt == obj
    assert rebuilt is not obj
