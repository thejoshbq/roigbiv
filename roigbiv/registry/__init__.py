"""Cross-session FOV + cell tracking registry.

Public entry points:
    register_or_match   — orchestrator called by the pipeline with --registry
    build_store         — resolve a RegistryStore from env config
    build_blob_store    — resolve a BlobStore from env config
    build_adapter_config — build an AdapterConfig from env config
    load_calibration    — load the FOV-logistic calibration
"""
from __future__ import annotations

from roigbiv.registry.config import (
    RegistryConfig,
    build_adapter_config,
    build_blob_store,
    build_store,
    load_calibration,
)
from roigbiv.registry.orchestrator import register_or_match

__all__ = [
    "RegistryConfig",
    "build_store",
    "build_blob_store",
    "build_adapter_config",
    "load_calibration",
    "register_or_match",
]
