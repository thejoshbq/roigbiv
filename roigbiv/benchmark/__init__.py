"""Benchmark / guardrail harness for roigbiv — dataset manifest schema, metrics, synthetic soma injection."""

from roigbiv.benchmark.schema import (
    LENS_TYPES,
    QUALITY_TIERS,
    ManifestEntry,
    BenchmarkManifest,
    ValidationError,
    ManifestError,
    load_manifest,
    validate_manifest,
)

__all__ = [
    "LENS_TYPES",
    "QUALITY_TIERS",
    "ManifestEntry",
    "BenchmarkManifest",
    "ValidationError",
    "ManifestError",
    "load_manifest",
    "validate_manifest",
]
