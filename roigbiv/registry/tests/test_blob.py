from __future__ import annotations

from pathlib import Path

import pytest

from roigbiv.registry.blob.local import LocalBlobStore


def test_local_put_get_exists(tmp_path: Path):
    store = LocalBlobStore(root=tmp_path)
    uri = store.put("fov1/mean.npy", b"hello")
    assert store.exists(uri)
    assert store.get(uri) == b"hello"


def test_local_rejects_traversal(tmp_path: Path):
    store = LocalBlobStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.put("../escape.npy", b"x")


def test_local_returns_file_uri(tmp_path: Path):
    store = LocalBlobStore(root=tmp_path)
    uri = store.put("a/b/c.npy", b"x")
    assert uri.startswith("file://")
