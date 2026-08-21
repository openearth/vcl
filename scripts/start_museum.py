"""Cross-platform launcher for the VCL museum runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent


def _candidate_python_binaries() -> list[Path]:
    candidates: list[Path] = []

    env_python = os.environ.get("PYTHON_BIN")
    if env_python:
        candidates.append(Path(env_python))

    if sys.executable:
        candidates.append(Path(sys.executable))

    candidates.append(REPO_DIR / ".venv" / "bin" / "python")
    candidates.append(REPO_DIR / ".venv" / "Scripts" / "python.exe")

    return candidates


def _resolve_python_binary() -> Path:
    for candidate in _candidate_python_binaries():
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a Python interpreter. Set PYTHON_BIN or create the local virtual environment."
    )


def build_command() -> list[str]:
    python_bin = _resolve_python_binary()

    default_layer = os.environ.get("DEFAULT_LAYER", "overview")
    inactivity_timeout = os.environ.get("INACTIVITY_TIMEOUT", "120")
    render_fps = os.environ.get("RENDER_FPS", "30")
    year_loop_fps = os.environ.get("YEAR_LOOP_FPS", "1")

    return [
        str(python_bin),
        "-m",
        "vcl.cli",
        "--museum",
        "--default-layer",
        default_layer,
        "--inactivity-timeout",
        inactivity_timeout,
        "--render-fps",
        render_fps,
        "--year-loop-fps",
        year_loop_fps,
    ]


def main() -> int:
    try:
        command = build_command()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    completed = subprocess.run(command, cwd=REPO_DIR)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())