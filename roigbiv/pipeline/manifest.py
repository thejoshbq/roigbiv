from __future__ import annotations

import datetime
import hashlib
import json
import logging
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from roigbiv.pipeline.resume import _json_ready
from roigbiv.pipeline.types import PipelineConfig

# Constants
MANIFEST_FILENAME = "run_manifest.json"


def run_manifest_path(output_dir: Path) -> Path:
    """Return the path to the run manifest file in output_dir."""
    return Path(output_dir) / MANIFEST_FILENAME


def _git_state(repo_root: Path) -> dict:
    """Return {commit: str|None, dirty: bool|None, branch: str|None}.
    All keys are None if git binary is absent, repo is not a checkout, or any
    command fails. Never raises."""
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            timeout=5,
            check=False,
            capture_output=True,
            text=True,
        )
        commit = commit_result.stdout.strip() or None

        dirty_result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            timeout=5,
            check=False,
            capture_output=True,
            text=True,
        )
        dirty = bool(dirty_result.stdout.strip()) if dirty_result.returncode == 0 else None

        branch_result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            timeout=5,
            check=False,
            capture_output=True,
            text=True,
        )
        branch = branch_result.stdout.strip() or None

        return {"commit": commit, "dirty": dirty, "branch": branch}
    except Exception:
        return {"commit": None, "dirty": None, "branch": None}


def _python_info() -> dict:
    """Return {version: str, implementation: str, executable: str}."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def _cuda_info() -> dict:
    """Return {available: bool, torch_version: str|None, cuda_version: str|None,
    device_count: int, device_names: list[str]}.
    Returns all-false/null/empty if torch is not importable or CUDA unavailable."""
    try:
        import torch

        available = torch.cuda.is_available()
        torch_version = torch.__version__
        cuda_version = torch.version.cuda if available else None
        device_count = torch.cuda.device_count() if available else 0
        device_names = (
            [torch.cuda.get_device_name(i) for i in range(device_count)]
            if device_count > 0
            else []
        )

        return {
            "available": available,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "device_count": device_count,
            "device_names": device_names,
        }
    except Exception:
        return {
            "available": False,
            "torch_version": None,
            "cuda_version": None,
            "device_count": 0,
            "device_names": [],
        }


def _package_versions(names: list[str]) -> dict:
    """Return {name: version_str|None} for each package in names.
    Uses importlib.metadata.version(). Catches PackageNotFoundError -> None."""
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
        except Exception:
            result[name] = None
    return result


def _hash_file(path: Path) -> str | None:
    """SHA256 hash of file at path, streamed in 1 MiB chunks.
    Returns "sha256:<hex>" on success, None if file missing/unreadable. Never raises."""
    try:
        hash_sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hash_sha256.update(chunk)
        return "sha256:" + hash_sha256.hexdigest()
    except Exception:
        return None


def _seeds() -> dict:
    """Return {torch_manual_seed: RNG_SEED, torch_cuda_manual_seed_all: RNG_SEED},
    importing RNG_SEED from roigbiv.pipeline.foundation (already defined there as
    RNG_SEED = 0, module-level constant, right after the imports)."""
    from roigbiv.pipeline.foundation import RNG_SEED

    return {
        "torch_manual_seed": RNG_SEED,
        "torch_cuda_manual_seed_all": RNG_SEED,
    }


def build_manifest(cfg: PipelineConfig, tif_path: Path, output_dir: Path) -> dict:
    """Pure function: assemble the reproducibility manifest dict.
    Returns dict matching the JSON schema exactly (all top-level keys present)
    even when git/torch/packages unavailable. Config is serialized via
    _json_ready(cfg.summary_for_log()). The git/cuda/packages/hash sub-fields
    are individually fail-open, but this function does not itself guard against
    cfg.summary_for_log()/platform failures — use write_manifest() for a fully
    fail-open call."""
    from roigbiv import __version__

    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    repo_root = Path(__file__).resolve().parent.parent.parent

    return {
        "schema_version": "1.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "roigbiv_version": __version__,
        "git": _git_state(repo_root),
        "python": _python_info(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cuda": _cuda_info(),
        "packages": _package_versions(["torch", "cellpose", "suite2p", "numpy", "roicat"]),
        "seeds": _seeds(),
        "config": _json_ready(cfg.summary_for_log()),
        "input": {
            "path": str(tif_path),
            "tif_hashes": {tif_path.name: _hash_file(tif_path)},
        },
        "output_dir": str(output_dir),
    }


def write_manifest(
    cfg: PipelineConfig, tif_path: Path, output_dir: Path
) -> Path | None:
    """Create output_dir, call build_manifest(), write JSON with indent=2.
    Returns Path to written manifest on success, None on any exception.
    Logs a warning (via logging.warning()) on failure. Never raises."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(cfg, tif_path, output_dir)
        path = run_manifest_path(output_dir)
        path.write_text(json.dumps(manifest, indent=2))
        return path
    except Exception as e:
        logging.warning(f"Failed to write manifest: {e}")
        return None
