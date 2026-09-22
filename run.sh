#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
fi
.venv/bin/python -c "import pygame" 2>/dev/null || .venv/bin/pip install -q -r requirements.txt
export PYGAME_HIDE_SUPPORT_PROMPT=1
exec .venv/bin/python main.py "$@"
