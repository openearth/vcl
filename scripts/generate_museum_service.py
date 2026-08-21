"""Generate OS service definitions for the VCL museum watchdog.

Service managers require absolute paths. This helper resolves the current
repository path and Python interpreter and writes a matching launchd plist or
systemd unit file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent


def resolve_python() -> Path:
    env_python = os.environ.get("PYTHON_BIN")
    if env_python:
        return Path(env_python).resolve()
    return Path(sys.executable).resolve()


def launchd_template(repo_dir: Path, python_bin: Path, log_dir: Path) -> str:
    stdout_log = log_dir / "vcl-museum.stdout.log"
    stderr_log = log_dir / "vcl-museum.stderr.log"
    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>Label</key>
    <string>com.openearth.vcl.museum</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_bin}</string>
        <string>{repo_dir / 'scripts' / 'watchdog_museum.py'}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{repo_dir}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{stdout_log}</string>

    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>DEFAULT_LAYER</key>
        <string>overview</string>
        <key>INACTIVITY_TIMEOUT</key>
        <string>120</string>
        <key>RENDER_FPS</key>
        <string>30</string>
        <key>YEAR_LOOP_FPS</key>
        <string>1</string>
        <key>VCL_RESTART_DELAY</key>
        <string>5</string>
    </dict>
</dict>
</plist>
"""


def systemd_template(repo_dir: Path, python_bin: Path) -> str:
    return f"""[Unit]
Description=Virtual Climate Lab Museum Mode
After=network.target graphical.target

[Service]
Type=simple
WorkingDirectory={repo_dir}
Environment=DEFAULT_LAYER=overview
Environment=INACTIVITY_TIMEOUT=120
Environment=RENDER_FPS=30
Environment=YEAR_LOOP_FPS=1
Environment=VCL_RESTART_DELAY=5
ExecStart={python_bin} {repo_dir / 'scripts' / 'watchdog_museum.py'}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["launchd", "systemd"])
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path.home() / "Library" / "Logs",
        help="Directory for launchd stdout/stderr logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = REPO_DIR.resolve()
    python_bin = resolve_python()

    if args.target == "launchd":
        args.log_dir.mkdir(parents=True, exist_ok=True)
        content = launchd_template(
            repo_dir=repo_dir,
            python_bin=python_bin,
            log_dir=args.log_dir,
        )
    else:
        content = systemd_template(repo_dir=repo_dir, python_bin=python_bin)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Wrote {args.target} config to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())