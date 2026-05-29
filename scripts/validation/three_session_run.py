"""Phase-4 validation: ingest the three T1 sessions into a fresh v3 registry.

Runs register_or_match in chronological order for
  T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002
  T1_221215_PrL-NAc-G6-5M_LOW-D1_FOV1_PRE-000
  T1_230116_PrL-NAc-G6-5M_EXT-D9_FOV1_EXT-D9_PRE-000

Captures per-step feature values, posteriors, and decisions into a JSON
transcript, then prints a final DB-state summary. Does not overwrite
inference/registry.db; targets inference/registry_roicat.db / inference/fingerprints_v3.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    project_root = Path("/home/thejoshbq/Otis-Lab/Projects/roigbiv")
    fresh_db = project_root / "inference" / "registry_roicat.db"
    fresh_blob = project_root / "inference" / "fingerprints_v3"

    # Clean fresh slate (do NOT touch registry.db or fingerprints/).
    if fresh_db.exists():
        fresh_db.unlink()
    if fresh_blob.exists():
        import shutil

        shutil.rmtree(fresh_blob)
    fresh_blob.mkdir(parents=True, exist_ok=True)

    os.environ["ROIGBIV_REGISTRY_DSN"] = f"sqlite:///{fresh_db}"
    os.environ["ROIGBIV_BLOB_ROOT"] = str(fresh_blob)

    # Import only AFTER env vars are set so config.from_env picks them up.
    from roigbiv.registry import (
        RegistryConfig,
        build_adapter_config,
        build_blob_store,
        build_store,
        load_calibration,
        register_or_match,
    )
    from roigbiv.registry.roicat_adapter import load_session_input

    sessions = [
        "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002",
        "T1_221215_PrL-NAc-G6-5M_LOW-D1_FOV1_PRE-000",
        "T1_230116_PrL-NAc-G6-5M_EXT-D9_FOV1_EXT-D9_PRE-000",
    ]
    out_root = project_root / "data" / "output"

    cfg = RegistryConfig.from_env()
    store = build_store(cfg)
    store.ensure_schema()
    blob_store = build_blob_store(cfg)
    adapter_cfg = build_adapter_config(cfg)
    calibration = load_calibration(cfg)

    transcript = {
        "db_path": str(fresh_db),
        "blob_root": str(fresh_blob),
        "adapter_config": {
            "um_per_pixel": adapter_cfg.um_per_pixel,
            "device": adapter_cfg.device,
            "alignment_method": adapter_cfg.alignment_method,
            "all_to_all": adapter_cfg.all_to_all,
            "nonrigid": adapter_cfg.nonrigid,
            "sequential_hungarian_thresh_cost": adapter_cfg.sequential_hungarian_thresh_cost,
            "d_cutoff": adapter_cfg.d_cutoff,
        },
        "accept_threshold": cfg.fov_accept_threshold,
        "review_threshold": cfg.fov_review_threshold,
        "calibration_trained": calibration.trained,
        "steps": [],
    }

    for i, stem in enumerate(sessions):
        print(f"\n=== [{i+1}/{len(sessions)}] ingesting {stem} ===", flush=True)
        output_dir = out_root / stem
        query = load_session_input(output_dir, session_key=stem)
        t0 = time.time()
        report = register_or_match(
            fov_stem=stem,
            query=query,
            output_dir=output_dir,
            store=store,
            blob_store=blob_store,
            adapter_config=adapter_cfg,
            calibration=calibration,
            accept_threshold=cfg.fov_accept_threshold,
            review_threshold=cfg.fov_review_threshold,
        )
        elapsed = time.time() - t0

        import numpy as np

        n_rois_true = int(len(set(int(v) for v in np.unique(query.merged_masks)) - {0}))
        step = {
            "index": i,
            "stem": stem,
            "elapsed_seconds": round(elapsed, 2),
            "n_query_rois": n_rois_true,
            "decision": report.get("decision"),
            "fov_id": report.get("fov_id"),
            "candidate_fov_id": (
                report.get("candidate_fov_id") or report.get("best_candidate_fov_id")
            ),
            # auto_match / review reports carry `fov_posterior`; new_fov carries
            # `best_candidate_posterior` when a candidate was actually clustered.
            "fov_posterior": report.get("fov_posterior"),
            "best_candidate_posterior": report.get("best_candidate_posterior"),
            "best_candidate_decision": report.get("best_candidate_decision"),
            # Feature vector — present on auto_match / review AND on new_fov with
            # best-candidate diagnostics.
            "alignment_method": (
                report.get("alignment_method")
                or report.get("best_candidate_alignment_method")
            ),
            "alignment_inlier_rate": (
                report.get("alignment_inlier_rate")
                or report.get("best_candidate_alignment_inlier_rate")
            ),
            "n_shared_clusters": (
                report.get("n_shared_clusters")
                or report.get("best_candidate_n_shared_clusters")
            ),
            "fraction_query_clustered": (
                report.get("fraction_query_clustered")
                or report.get("best_candidate_fraction_query_clustered")
            ),
            "mean_cluster_cohesion": (
                report.get("mean_cluster_cohesion")
                or report.get("best_candidate_mean_cluster_cohesion")
            ),
            "n_matched": report.get("n_matched"),
            "n_new": report.get("n_new") or report.get("n_new_cells"),
            "n_missing": report.get("n_missing"),
            "fingerprint_hash": report.get("fingerprint_hash"),
            "fingerprint_version": report.get("fingerprint_version"),
        }
        transcript["steps"].append(step)
        print(json.dumps(step, indent=2), flush=True)

    # Final DB state.
    fovs = store.list_fovs()
    fov_states = []
    for fov in fovs:
        sessions_in_fov = store.list_sessions(fov.fov_id)
        cells_in_fov = store.list_cells(fov.fov_id)
        fov_states.append({
            "fov_id": fov.fov_id,
            "animal_id": fov.animal_id,
            "region": fov.region,
            "fingerprint_version": fov.fingerprint_version,
            "fingerprint_hash": fov.fingerprint_hash,
            "latest_session_date": (
                fov.latest_session_date.isoformat() if fov.latest_session_date else None
            ),
            "n_sessions": len(sessions_in_fov),
            "n_cells": len(cells_in_fov),
            "sessions": [
                {
                    "session_date": s.session_date.isoformat(),
                    "output_dir": s.output_dir,
                    "fov_posterior": s.fov_posterior,
                    "n_matched": s.n_matched,
                    "n_new": s.n_new,
                    "n_missing": s.n_missing,
                    "cluster_labels_uri": s.cluster_labels_uri,
                }
                for s in sessions_in_fov
            ],
        })
    transcript["final_db"] = {
        "n_fovs": len(fovs),
        "fovs": fov_states,
    }

    expected_n_fov = 1
    transcript["verdict"] = {
        "pass": len(fovs) == expected_n_fov and len(fovs) > 0 and fov_states[0]["n_sessions"] == 3,
        "expected_n_fov": expected_n_fov,
        "expected_n_sessions_in_fov": 3,
        "actual_n_fov": len(fovs),
        "actual_n_sessions": fov_states[0]["n_sessions"] if fov_states else 0,
    }

    transcript_path = project_root / "docs" / "validation" / "three_session_transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(transcript, indent=2, default=str))

    print("\n=== FINAL DB STATE ===", flush=True)
    print(json.dumps(transcript["final_db"], indent=2, default=str), flush=True)
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(transcript["verdict"], indent=2), flush=True)
    print(f"\ntranscript written to: {transcript_path}", flush=True)
    return 0 if transcript["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
