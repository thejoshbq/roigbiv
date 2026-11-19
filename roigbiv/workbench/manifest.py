"""Run manifest backfilled from a saved run's FOV directory.

A saved run records its input path with size and mtime (``.roigbiv_manifest.json``), a config fingerprint and the
effective config (``pipeline_log.json``). It does not record the commit, the environment or the model weights,
so the caller supplies the commit and dirty flag from the job script, and the rest is listed in ``not_recorded``.
The saved ``cfg_fingerprint`` is the resume fingerprint, recorded as the effective value ``resume_fingerprint``.
Input hashes are computed once per file and cached by path, size and mtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from roigbiv.workbench.models import FileHash, RunManifest

MANIFEST_FILE = "run_manifest.json"
_CHUNK = 1 << 22


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _hash(path: Path, cache: dict | None) -> FileHash | None:
    """FileHash for ``path``; None when it does not exist."""
    if not path.exists():
        return None
    st = path.stat()
    key = str(path.resolve())
    hit = (cache or {}).get(key)
    if hit and hit["size"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
        sha = hit["sha256"]
    else:
        sha = _sha256(path)
        if cache is not None:
            cache[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
    return FileHash(role="", path=key, sha256=sha, size=st.st_size)


def _effective(config: dict) -> tuple:
    """Scalar config values as the run used them, plus where the scale came from."""
    eff = [(k, v) for k, v in config.items() if isinstance(v, (bool, int, float, str))]
    scale_source = (config.get("auto_adapt") or {}).get("scale_source")
    if scale_source is not None:
        eff.append(("auto_adapt.scale_source", scale_source))
    return tuple(eff)


def backfill_manifest(
    fov_dir: Path,
    *,
    run_id: str,
    fov_id: str,
    commit: str | None,
    dirty: bool | None,
    extra_inputs: Iterable[tuple[str, Path]] = (),
    hash_cache: Path | None = None,
) -> tuple[RunManifest, list]:
    """Build a manifest for one saved FOV (read-only). Returns it with a list of problems found."""
    fov_dir = Path(fov_dir)
    log = json.loads((fov_dir / "pipeline_log.json").read_text())
    rec = json.loads((fov_dir / ".roigbiv_manifest.json").read_text())
    fp = rec.get("cfg_fingerprint") or ""
    if not fp.startswith("sha256:"):
        raise ValueError(f"{fov_dir}: cfg_fingerprint {fp!r} is not sha256:<hex>")
    cache = json.loads(hash_cache.read_text()) if hash_cache and hash_cache.exists() else ({} if hash_cache else None)

    problems: list = []
    inputs = []
    movie = Path(rec["input_tif"])
    then = rec.get("input_stat") or {}
    changed = False
    if movie.exists():
        st = movie.stat()
        if then.get("size") is None or then.get("mtime_ns") is None:
            problems.append(f"input {movie}: the run recorded no size or mtime to compare")
        elif (then["size"], then["mtime_ns"]) != (st.st_size, st.st_mtime_ns):
            problems.append(f"input {movie} differs from the run's record (size or mtime)")
            changed = True
    for role, path in (("movie", movie), *((r, Path(p)) for r, p in extra_inputs)):
        if role == "movie" and changed:  # today's bytes are not the run's input; do not record their hash
            inputs.append(FileHash(role=role, path=str(path), sha256=None))
            continue
        h = _hash(path, cache)
        if h is None:
            problems.append(f"input {path} not found")
            inputs.append(FileHash(role=role, path=str(path), sha256=None))
            continue
        inputs.append(FileHash(role=role, path=h.path, sha256=h.sha256, size=h.size))
    if cache is not None and hash_cache is not None:
        hash_cache.parent.mkdir(parents=True, exist_ok=True)
        hash_cache.write_text(json.dumps(cache, indent=1, sort_keys=True))

    # The saved cfg_fingerprint is the resume fingerprint (config minus stage flags and scale-derived fields,
    # plus the input's size and mtime), not a hash of the requested config; it is kept under its own name.
    not_recorded = ["environment", "model_weights", "annotations", "requested_config_sha256"]
    not_recorded += [k for k, v in (("commit", commit), ("dirty", dirty)) if v is None]
    man = RunManifest(
        run_id=run_id,
        fov_id=fov_id,
        source="backfill",
        created_utc=log["timestamp"],
        commit=commit,
        dirty=dirty,
        environment=(),
        inputs=tuple(inputs),
        model_weights=(),
        annotations=(),
        effective=(*_effective(log.get("config") or {}), ("resume_fingerprint", fp)),
        requested_config_sha256=None,
        not_recorded=tuple(not_recorded),
    )
    return man, problems


def write_manifest(man: RunManifest, dest: Path) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MANIFEST_FILE).write_text(man.to_json())


def read_manifest(src: Path) -> RunManifest:
    return RunManifest.from_json((Path(src) / MANIFEST_FILE).read_text())
