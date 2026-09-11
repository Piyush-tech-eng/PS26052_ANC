#!/usr/bin/env python3
"""PS26052 ANC — Unified Command Center & Deliverable Launcher.

Single-command root entrypoint to launch the full-featured, interactive
Adaptive Noise Cancellation and Neural Speech Enhancement application.

Usage:
    python app.py
    python app.py --port 8080
    python app.py --no-browser
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on Python search path
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PS26052 ANC — Unified Tactical Command Center & Deliverable"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port for the web interface (default: 8080)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default web browser",
    )
    args = parser.parse_args()

    # Pre-flight environment check
    venv_active = "VIRTUAL_ENV" in os.environ or hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    if not venv_active:
        print("[Notice] Running outside an explicit virtual environment. Recommended: activate .venv")

    from anc.interface.server import start_server

    start_server(
        port=args.port,
        open_browser=not args.no_browser,
        repo_root=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
