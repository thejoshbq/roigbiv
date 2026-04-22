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

v3 matcher env vars
-------------------
ROIGBIV_ROICAT_DEVICE
    "cuda" or "cpu". Default: auto-detect (cuda if torch.cuda.is_available()
    else cpu). RoMa on CPU is minutes-per-session; cuda is strongly preferred.
ROIGBIV_ROINET_CACHE
    Directory for ROInet weights. Default: ~/.cache/roigbiv/roinet.
ROIGBIV_UM_PER_PIXEL
    Physical scale for ROICaT's resizer. Default: 1.0.
ROIGBIV_ROICAT_ALIGNMENT
    Aligner.fit_geometric method (RoMa|PhaseCorrelation|ECC_cv2|…).
    Default: RoMa (validated pass on the T1 three-session dataset; triggers
    ~1.5 GB one-time weight download on first use).
ROIGBIV_ROICAT_ALL_TO_ALL
    "1" to enable O(n²) all-pairs alignment; "0" (default) uses sequential.
ROIGBIV_ROICAT_NONRIGID
    "1" to enable ROICaT non-rigid alignment after geometric fit.
ROIGBIV_ROICAT_HUNGARIAN_THRESH
    Cost threshold for ``Clusterer.fit_sequentialHungarian``. Default: 0.6.
ROIGBIV_ROICAT_ROI_MIXING
    Weight of ROI-footprint density in Aligner.augment_FOV_images
    (0.0 = mean projection only, 1.0 = footprints only). Default: 0.5
    (ROICaT's own default; tested 0.9 on the T1 three-session dataset and
    it regressed alignment inlier rate — mean projection has richer spatial
    features than footprint density for phase correlation).
ROIGBIV_CALIBRATION_PATH
    JSON calibration file. Default: <cwd>/inference/registry_calibration.json
    (falls back to hand-priors when absent).
ROIGBIV_FOV_ACCEPT_THRESHOLD
    Posterior cutoff for auto-match. Default: 0.9.
ROIGBIV_FOV_REVIEW_THRESHOLD
    Posterior cutoff for review. Default: 0.5.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _default_db_path() -> Path:
    cwd = Path.cwd()
    return cwd / "inference" / "registry.db"


def _default_blob_root() -> Path:
    return Path.cwd() / "inference" / "fingerprints"


def _default_calibration_path() -> Path:
    return Path.cwd() / "inference" / "registry_calibration.json"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _auto_device() -> str:
    """CUDA if available, else CPU. Deferred torch import."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class RegistryConfig:
    dsn: str
    blob_backend: str
    blob_root: Path
    endpoint: Optional[str]
    api_key: Optional[str]
    # v3 adapter knobs (formerly v2 embedder knobs).
    roinet_cache: Optional[Path] = None
    roicat_device: str = field(default_factory=lambda: _auto_device())
    um_per_pixel: float = 1.0
    roicat_alignment: str = "RoMa"
    roicat_all_to_all: bool = False
    roicat_nonrigid: bool = False
    roicat_hungarian_thresh: float = 0.6
    roicat_roi_mixing: float = 0.5
    calibration_path: Path = field(default_factory=_default_calibration_path)
    fov_accept_threshold: float = 0.9
    fov_review_threshold: float = 0.5

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

        roinet_cache_env = os.environ.get("ROIGBIV_ROINET_CACHE")
        roinet_cache = Path(roinet_cache_env) if roinet_cache_env else None
        calibration_path = Path(
            os.environ.get(
                "ROIGBIV_CALIBRATION_PATH", str(_default_calibration_path())
            )
        )
        return cls(
            dsn=dsn,
            blob_backend=blob_backend,
            blob_root=blob_root,
            endpoint=endpoint,
            api_key=api_key,
            roinet_cache=roinet_cache,
            roicat_device=os.environ.get("ROIGBIV_ROICAT_DEVICE") or _auto_device(),
            um_per_pixel=float(os.environ.get("ROIGBIV_UM_PER_PIXEL", "1.0")),
            roicat_alignment=os.environ.get(
                "ROIGBIV_ROICAT_ALIGNMENT", "RoMa"
            ),
            roicat_all_to_all=_env_bool("ROIGBIV_ROICAT_ALL_TO_ALL", False),
            roicat_nonrigid=_env_bool("ROIGBIV_ROICAT_NONRIGID", False),
            roicat_hungarian_thresh=float(
                os.environ.get("ROIGBIV_ROICAT_HUNGARIAN_THRESH", "0.6")
            ),
            roicat_roi_mixing=float(
                os.environ.get("ROIGBIV_ROICAT_ROI_MIXING", "0.5")
            ),
            calibration_path=calibration_path,
            fov_accept_threshold=float(
                os.environ.get("ROIGBIV_FOV_ACCEPT_THRESHOLD", "0.9")
            ),
            fov_review_threshold=float(
                os.environ.get("ROIGBIV_FOV_REVIEW_THRESHOLD", "0.5")
            ),
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


def build_adapter_config(cfg: Optional[RegistryConfig] = None):
    """Translate :class:`RegistryConfig` into a
    :class:`roigbiv.registry.roicat_adapter.AdapterConfig`.
    """
    from roigbiv.registry.roicat_adapter import AdapterConfig

    cfg = cfg or RegistryConfig.from_env()
    return AdapterConfig(
        um_per_pixel=cfg.um_per_pixel,
        device=cfg.roicat_device,
        all_to_all=cfg.roicat_all_to_all,
        nonrigid=cfg.roicat_nonrigid,
        alignment_method=cfg.roicat_alignment,
        sequential_hungarian_thresh_cost=cfg.roicat_hungarian_thresh,
        roi_mixing_factor=cfg.roicat_roi_mixing,
        roinet_cache_dir=cfg.roinet_cache,
    )


def load_calibration(cfg: Optional[RegistryConfig] = None):
    """Load (or default-fallback to) a :class:`CalibrationModel`."""
    from roigbiv.registry.calibration import CalibrationModel

    cfg = cfg or RegistryConfig.from_env()
    return CalibrationModel.load(cfg.calibration_path)
