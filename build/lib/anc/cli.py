"""PS26052 ANC — Command Line Interface.

Provides the packaged command-line entrypoint for the PS26052 ANC system.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Packaged CLI entrypoint forwarding to live demo orchestrator."""
    try:
        from anc.demo import main as demo_main
    except ImportError:
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import run_live_demo
        demo_main = run_live_demo.main

    demo_main()


if __name__ == "__main__":
    main()
