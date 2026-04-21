"""Cross-session FOV + cell tracking registry.

Public entry points:
    register_or_match  — orchestrator called by the pipeline with --registry
    build_store        — resolve a RegistryStore from env config
    build_blob_store   — resolve a BlobStore from env config
"""
from __future__ import annotations

from roigbiv.registry.config import RegistryConfig, build_blob_store, build_store
from roigbiv.registry.orchestrator import register_or_match

__all__ = [
    "RegistryConfig",
    "build_store",
    "build_blob_store",
    "register_or_match",
]
