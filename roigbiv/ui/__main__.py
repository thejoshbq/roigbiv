"""Launch the ROIGBIV Dash UI.

Usage::

    roigbiv-ui [--host 127.0.0.1] [--port 8050] [--debug]
    python -m roigbiv.ui
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roigbiv-ui",
        description="ROIGBIV Dash interface — processing, registry, viewer, HITL review.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1). "
                             "Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port to bind (default: 8050).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Dash debug mode (hot-reload, dev tools).")
    args = parser.parse_args(argv)

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
