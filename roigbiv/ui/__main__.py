"""Launch the ROIGBIV Dash UI.

Usage::

    roigbiv-ui [--workspace PATH] [--host 127.0.0.1] [--port 8050] [--debug]
    python -m roigbiv.ui
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roigbiv-ui",
        description="ROIGBIV Dash interface — processing, registry, viewer, HITL review.",
    )
    parser.add_argument("--workspace", type=Path, default=None,
                        help="Workspace directory to bind the UI to. "
                             "Sets ROIGBIV_REGISTRY_DSN / BLOB_ROOT so Registry, "
                             "Viewer, and HITL read the workspace's registry.db "
                             "instead of the default inference/registry.db.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1). "
                             "Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port to bind (default: 8050).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Dash debug mode (hot-reload, dev tools).")
    args = parser.parse_args(argv)

    if args.workspace is not None:
        try:
            from roigbiv.pipeline.workspace import (
                configure_registry_env,
                resolve_workspace,
            )
        except ImportError as exc:
            print(f"ERROR: cannot import workspace helpers: {exc}", file=sys.stderr)
            return 2
        try:
            workspace = resolve_workspace(args.workspace)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        configure_registry_env(workspace)
        print(f"Workspace: {workspace.input_root}", flush=True)
        print(f"Registry:  {workspace.db_path}", flush=True)

    try:
        from roigbiv.ui.app import build_app
    except ImportError as exc:
        print(
            "ERROR: Dash UI dependencies are not installed. "
            "Run:  pip install 'roigbiv[ui]'",
            file=sys.stderr,
        )
        print(f"  ({exc})", file=sys.stderr)
        return 2

    app = build_app()
    print(f"ROIGBIV UI → http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
