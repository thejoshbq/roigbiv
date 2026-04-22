"""RegistryStore Protocol + record dataclasses.

All callers (CLI, Streamlit, orchestrator, backfill) talk to the Protocol,
never a concrete backend. Adding HTTPStore later is purely additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Protocol

import numpy as np


@dataclass
class FOVRecord:
    fov_id: str
    fingerprint_hash: str
    animal_id: str
    region: str
    mean_m_uri: str
    centroid_table_uri: str
    created_at: datetime
    latest_session_date: Optional[date] = None
    # v2 — present when the FOV was registered with an embedding-aware pipeline.
    fingerprint_version: int = 1
    fov_embedding_uri: Optional[str] = None
    roi_embeddings_uri: Optional[str] = None


@dataclass
class CellRecord:
    global_cell_id: str
    fov_id: str
    first_seen_session_id: Optional[str] = None
    morphology_summary: dict = field(default_factory=dict)


@dataclass
class SessionRecord:
    session_id: str
    fov_id: str
    session_date: date
    output_dir: str
    fov_sim: Optional[float] = None
    n_matched: int = 0
    n_new: int = 0
    n_missing: int = 0
    created_at: Optional[datetime] = None
    fov_posterior: Optional[float] = None
    # v3: blob URI for this session's per-ROI ROICaT cluster label array
    # (int32, length = n_rois_session).
    cluster_labels_uri: Optional[str] = None


@dataclass
class ObservationRecord:
    global_cell_id: str
    session_id: str
    local_label_id: int
    match_score: Optional[float] = None
    observation_id: Optional[str] = None
    # v3: ROICaT cluster label for this observation (nullable for legacy rows).
    cluster_label: Optional[int] = None


class RegistryStore(Protocol):
    """Every read/write the rest of the codebase needs."""

    def ensure_schema(self) -> None: ...

    def get_fov_by_hash(self, fingerprint_hash: str) -> Optional[FOVRecord]: ...
    def get_fov(self, fov_id: str) -> Optional[FOVRecord]: ...
    def find_candidates(self, animal_id: str, region: str) -> list[FOVRecord]: ...
    def find_candidates_by_embedding(
        self,
        animal_id: str,
        region: str,
        fov_embedding: np.ndarray,
        blob_store,
        top_k: int = 10,
        min_cosine: float = 0.0,
    ) -> list[FOVRecord]: ...
    def list_fovs(self, filters: Optional[dict] = None) -> list[FOVRecord]: ...
    def insert_fov(self, fov: FOVRecord) -> None: ...
    def update_fov_latest_session(self, fov_id: str, session_date: date) -> None: ...

    def insert_session(self, session: SessionRecord) -> None: ...
    def list_sessions(self, fov_id: str) -> list[SessionRecord]: ...
    def update_session_cluster_labels(
        self, session_id: str, cluster_labels_uri: str
    ) -> None: ...

    def insert_cell(self, cell: CellRecord) -> None: ...
    def list_cells(self, fov_id: str) -> list[CellRecord]: ...

    def insert_observations(self, observations: list[ObservationRecord]) -> None: ...
    def list_observations_for_cell(self, global_cell_id: str) -> list[ObservationRecord]: ...
    def list_observations_for_session(self, session_id: str) -> list[ObservationRecord]: ...
