#!/usr/bin/env bash
# any arguments go straight through to live.py, e.g. --demo or --reward.w_cohesion 0
# every launch is a new run, except --demo, which needs a trained policy to show (it loads the latest run)
set -e
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/pip ]; then
    python3 -m venv --clear .venv
    .venv/bin/python -m pip install -q --upgrade pip
fi
.venv/bin/python -c "import torch" 2>/dev/null || .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python -c "import pygame, matplotlib" 2>/dev/null || .venv/bin/pip install -q -r boids_rl/requirements.txt
export PYGAME_HIDE_SUPPORT_PROMPT=1
fresh=--fresh
for arg in "$@"; do
    [ "$arg" = "--demo" ] && fresh=
done
exec .venv/bin/python -m boids_rl.live $fresh "$@"
