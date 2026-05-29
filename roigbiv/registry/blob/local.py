"""Local-filesystem BlobStore.

URIs are file:// absolute paths. Keys are treated as POSIX-style relative paths
beneath `root` (arbitrary subdirectories allowed; they're created on put).
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


class LocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> str:
        key_path = self._resolve_key(key)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(data)
        return key_path.as_uri()

    def get(self, uri: str) -> bytes:
        return self._uri_to_path(uri).read_bytes()

    def exists(self, uri: str) -> bool:
        try:
            return self._uri_to_path(uri).exists()
        except ValueError:
            return False

    def _resolve_key(self, key: str) -> Path:
        key = key.lstrip("/")
        candidate = (self.root / key).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError(f"key {key!r} escapes blob root {self.root}")
        return candidate

    @staticmethod
    def _uri_to_path(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"LocalBlobStore cannot handle URI scheme {parsed.scheme!r}")
        return Path(unquote(parsed.path))
