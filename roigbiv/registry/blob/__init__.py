"""BlobStore protocol + backends."""
from roigbiv.registry.blob.base import BlobStore
from roigbiv.registry.blob.local import LocalBlobStore

__all__ = ["BlobStore", "LocalBlobStore"]
