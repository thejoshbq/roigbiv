"""Env-driven config for the registry.

Defaults are safe for single-user local use: SQLite under inference/registry.db
and a local filesystem BlobStore under inference/fingerprints/. Override any of
the environment variables below to point at a different backend.

Env vars
--------
ROIGBIV_REGISTRY_DSN
    SQLAlchemy DSN. Default: sqlite:///<cwd>/inference/registry.db
ROIGBIV_BLOB_BACKEND
    "local" (default) or "s3" (reserved).
ROIGBIV_BLOB_ROOT
    Filesystem root for LocalBlobStore. Default: <cwd>/inference/fingerprints
ROIGBIV_REGISTRY_ENDPOINT
    If set, HTTPStore is used (not yet implemented). Reserved for Phase C.
ROIGBIV_REGISTRY_API_KEY
    Auth token for HTTPStore. Reserved for Phase C.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _default_db_path() -> Path:
    cwd = Path.cwd()
    return cwd / "inference" / "registry.db"


def _default_blob_root() -> Path:
    return Path.cwd() / "inference" / "fingerprints"


@dataclass
class RegistryConfig:
    dsn: str
    blob_backend: str
    blob_root: Path
    endpoint: Optional[str]
    api_key: Optional[str]

    @classmethod
    def from_env(cls) -> "RegistryConfig":
        dsn = os.environ.get("ROIGBIV_REGISTRY_DSN")
        if not dsn:
            db_path = _default_db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            dsn = f"sqlite:///{db_path}"

        blob_backend = os.environ.get("ROIGBIV_BLOB_BACKEND", "local").lower()
        blob_root = Path(os.environ.get("ROIGBIV_BLOB_ROOT", str(_default_blob_root())))
        endpoint = os.environ.get("ROIGBIV_REGISTRY_ENDPOINT")
        api_key = os.environ.get("ROIGBIV_REGISTRY_API_KEY")
        return cls(
            dsn=dsn,
            blob_backend=blob_backend,
            blob_root=blob_root,
            endpoint=endpoint,
            api_key=api_key,
        )


def build_store(cfg: Optional[RegistryConfig] = None):
    """Resolve a RegistryStore implementation from config."""
    cfg = cfg or RegistryConfig.from_env()
    if cfg.endpoint:
        raise NotImplementedError(
            "HTTPStore (ROIGBIV_REGISTRY_ENDPOINT) is reserved for Phase C; "
            "no implementation yet."
        )
    from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore
    return SQLAlchemyStore(dsn=cfg.dsn)


def build_blob_store(cfg: Optional[RegistryConfig] = None):
    """Resolve a BlobStore implementation from config."""
    cfg = cfg or RegistryConfig.from_env()
    if cfg.blob_backend == "s3":
        raise NotImplementedError(
            "S3BlobStore is reserved for Phase C; set ROIGBIV_BLOB_BACKEND=local."
        )
    from roigbiv.registry.blob.local import LocalBlobStore
    return LocalBlobStore(root=cfg.blob_root)
