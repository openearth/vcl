#!/bin/zsh

set -eu

SCRIPT_DIR=${0:A:h}
REPO_DIR=${SCRIPT_DIR:h}
PYTHON_BIN=${PYTHON_BIN:-"$REPO_DIR/.venv/bin/python"}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found at $PYTHON_BIN" >&2
    echo "Create the virtual environment first, for example with: uv sync" >&2
    exit 1
fi

DEFAULT_LAYER=${DEFAULT_LAYER:-overview}
INACTIVITY_TIMEOUT=${INACTIVITY_TIMEOUT:-120}
RENDER_FPS=${RENDER_FPS:-30}
YEAR_LOOP_FPS=${YEAR_LOOP_FPS:-1}

cd "$REPO_DIR"

exec "$PYTHON_BIN" -m vcl.cli \
    --museum \
    --default-layer "$DEFAULT_LAYER" \
    --inactivity-timeout "$INACTIVITY_TIMEOUT" \
    --render-fps "$RENDER_FPS" \
    --year-loop-fps "$YEAR_LOOP_FPS"