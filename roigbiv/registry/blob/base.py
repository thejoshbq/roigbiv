"""BlobStore protocol — put/get/exists for opaque payloads.

URIs are the portable handle stored in the registry DB. Local backend returns
file:// URIs; future S3 backend returns s3:// URIs. Registry schema never
changes; only the URI scheme flips.
"""
from __future__ import annotations

from typing import Protocol


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> str:
        """Store `data` under `key`. Returns the URI by which it can be fetched."""
        ...

    def get(self, uri: str) -> bytes:
        """Retrieve bytes previously stored at `uri`."""
        ...

    def exists(self, uri: str) -> bool:
        """True iff `uri` can be fetched."""
        ...
