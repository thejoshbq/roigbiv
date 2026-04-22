"""Terminal entry point for the cross-session FOV registry.

Subcommands:
  list        — FOVs in the registry
  show        — one FOV's sessions + cell counts
  match       — retroactively match a single output dir (no re-run)
  track       — longitudinal trace across sessions for one global_cell_id
  backfill    — walk inference/pipeline/ and register every FOV present
  migrate     — run `alembic upgrade head` against the active DSN
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from roigbiv.registry import (
    RegistryConfig,
    build_adapter_config,
    build_blob_store,
    build_store,
    load_calibration,
    register_or_match,
)
from roigbiv.registry.backfill import run_backfill
from roigbiv.registry.filename import parse_filename_metadata
from roigbiv.registry.roicat_adapter import load_session_input


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roigbiv-registry",
        description="Cross-session FOV + cell registry for the roigbiv pipeline.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all FOVs in the registry.")

    show = sub.add_parser("show", help="Show sessions + cell counts for one FOV.")
    show.add_argument("fov_id")

    match = sub.add_parser(
        "match",
        help="Retroactively register/match a single pipeline output dir.",
    )
    match.add_argument("output_dir", type=Path)

    track = sub.add_parser(
        "track",
        help="List every session observation for one global_cell_id.",
    )
    track.add_argument("global_cell_id")

    bf = sub.add_parser(
        "backfill",
        help="Walk a root directory and register each FOV.",
    )
    bf.add_argument("--root", type=Path, default=Path("inference/pipeline"),
                    help="Root directory to scan (default: inference/pipeline).")
    bf.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without mutating state.")

    sub.add_parser("migrate", help="Run `alembic upgrade head` against the DSN.")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        return _cmd_list()
    if args.cmd == "show":
        return _cmd_show(args.fov_id)
    if args.cmd == "match":
        return _cmd_match(args.output_dir)
    if args.cmd == "track":
        return _cmd_track(args.global_cell_id)
    if args.cmd == "backfill":
        return _cmd_backfill(args.root, args.dry_run)
    if args.cmd == "migrate":
        return _cmd_migrate()
    return 2


def _cmd_list() -> int:
    store = build_store()
    store.ensure_schema()
    rows = store.list_fovs()
    if not rows:
        print("(registry is empty)")
        return 0
    for f in rows:
        last = f.latest_session_date.isoformat() if f.latest_session_date else "-"
        print(f"{f.fov_id}  animal={f.animal_id}  region={f.region}  "
              f"last={last}  hash={f.fingerprint_hash[:12]}  "
              f"v={f.fingerprint_version}")
    return 0


def _cmd_show(fov_id: str) -> int:
    store = build_store()
    store.ensure_schema()
    fov = store.get_fov(fov_id)
    if fov is None:
        print(f"fov_id {fov_id!r} not found", file=sys.stderr)
        return 1
    sessions = store.list_sessions(fov_id)
    cells = store.list_cells(fov_id)
    print(f"FOV {fov.fov_id}")
    print(f"  animal_id: {fov.animal_id}  region: {fov.region}")
    print(f"  fingerprint: {fov.fingerprint_hash}  v={fov.fingerprint_version}")
    print(f"  created_at: {fov.created_at.isoformat() if fov.created_at else '-'}")
    print(f"  cells: {len(cells)}")
    print(f"  sessions: {len(sessions)}")
    for s in sessions:
        posterior = s.fov_posterior if s.fov_posterior is not None else s.fov_sim
        posterior_str = f"{posterior:.3f}" if posterior is not None else "-"
        print(f"    {s.session_date.isoformat()}  posterior={posterior_str}  "
              f"matched={s.n_matched} new={s.n_new} missing={s.n_missing}  "
              f"{s.output_dir}")
    return 0


def _cmd_match(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    try:
        query = load_session_input(output_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cfg = RegistryConfig.from_env()
    store = build_store(cfg)
    blob_store = build_blob_store(cfg)
    adapter_cfg = build_adapter_config(cfg)
    calibration = load_calibration(cfg)
    fname = parse_filename_metadata(output_dir.name)
    report = register_or_match(
        fov_stem=output_dir.name,
        query=query,
        output_dir=output_dir,
        store=store,
        blob_store=blob_store,
        session_date_override=fname.session_date,
        adapter_config=adapter_cfg,
        calibration=calibration,
        accept_threshold=cfg.fov_accept_threshold,
        review_threshold=cfg.fov_review_threshold,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


def _cmd_track(global_cell_id: str) -> int:
    store = build_store()
    store.ensure_schema()
    obs = store.list_observations_for_cell(global_cell_id)
    if not obs:
        print(f"no observations for global_cell_id {global_cell_id!r}",
              file=sys.stderr)
        return 1
    sessions_by_id: dict = {}
    for o in obs:
        if o.session_id not in sessions_by_id:
            fov_id = _fov_id_from_obs(store, o)
            if fov_id:
                for s in store.list_sessions(fov_id):
                    sessions_by_id[s.session_id] = s
    print(f"global_cell_id: {global_cell_id}")
    for o in obs:
        s = sessions_by_id.get(o.session_id)
        date_s = s.session_date.isoformat() if s else "-"
        out_dir = s.output_dir if s else "-"
        cluster_str = (
            f" cluster={o.cluster_label}" if o.cluster_label is not None else ""
        )
        print(f"  {date_s}  local_label_id={o.local_label_id}  "
              f"score={o.match_score}{cluster_str}  {out_dir}")
    return 0


def _fov_id_from_obs(store, obs) -> str:
    # Observations know session_id; one DB roundtrip is fine here.
    for fov in store.list_fovs():
        for s in store.list_sessions(fov.fov_id):
            if s.session_id == obs.session_id:
                return fov.fov_id
    return ""


def _cmd_backfill(root: Path, dry_run: bool) -> int:
    reports = run_backfill(root.resolve(), dry_run=dry_run)
    if not reports:
        print(f"no candidates under {root}")
        return 0
    for r in reports:
        if "error" in r:
            print(f"ERR  {r.get('stem', '?')}: {r['error']}")
        else:
            dec = r.get("decision") or r.get("action")
            posterior = r.get("fov_posterior") or r.get("fov_sim") or 1.0
            if dec == "new_fov":
                print(f"new  {r['stem']}  fov_id={r['fov_id']}  "
                      f"cells={r['n_new_cells']}")
            elif dec in ("auto_match", "hash_match"):
                print(f"{dec}  {r['stem']}  fov_id={r['fov_id']}  "
                      f"p={posterior:.3f}  "
                      f"m={r.get('n_matched', 0)} n={r.get('n_new', 0)} "
                      f"x={r.get('n_missing', 0)}")
            elif dec == "review":
                print(f"rev  {r['stem']}  p={posterior:.3f}")
            else:
                print(f"{dec}  {r.get('stem', '?')}")
    return 0


def _cmd_migrate() -> int:
    from roigbiv.registry.migrate import ensure_alembic_head

    result = ensure_alembic_head()
    print(f"migrate: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
