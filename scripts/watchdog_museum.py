"""Restart wrapper for the VCL museum runtime.

This script launches the museum startup command, waits for it to exit, and
restarts it after a short delay. It is intended to be run under launchd or
manually from a terminal on the museum machine.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
START_SCRIPT = REPO_DIR / "scripts" / "start_museum.py"
RESTART_DELAY_SECONDS = float(os.environ.get("VCL_RESTART_DELAY", "5"))

_shutdown_requested = False
_child_process: subprocess.Popen[str] | None = None


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    if _child_process is not None and _child_process.poll() is None:
        _child_process.terminate()


def main() -> int:
    global _child_process

    if not START_SCRIPT.exists():
        print(f"Startup script not found: {START_SCRIPT}", file=sys.stderr)
        return 1

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    while not _shutdown_requested:
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{started_at}] Starting museum runtime: {START_SCRIPT}", flush=True)

        _child_process = subprocess.Popen(
            [sys.executable, str(START_SCRIPT)],
            cwd=str(REPO_DIR),
            text=True,
        )

        exit_code = _child_process.wait()
        finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{finished_at}] Museum runtime exited with code {exit_code}",
            flush=True,
        )

        if _shutdown_requested:
            break

        time.sleep(RESTART_DELAY_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
