#!/usr/bin/env bash
set -euo pipefail

export PATH="/root/.local/bin:${PATH}"

cd /workspace/mmds

if [[ ! -f pyproject.toml ]]; then
  echo "Expected a uv project at /workspace/mmds, but pyproject.toml was not found." >&2
  exit 1
fi

# Repair the bind-mounted environment when it is missing or points to a host-only interpreter.
if [[ ! -x .venv/bin/python ]]; then
  uv sync --locked
fi

if [[ $# -eq 0 ]]; then
  exec sleep infinity
fi

exec "$@"
