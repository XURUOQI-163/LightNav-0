#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/.conda/envs/omninavbench/bin/python}"
if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Python missing: $ISAAC_PYTHON; set ISAAC_PYTHON to the configured environment." >&2
  exit 1
fi
# Do not inherit ROS Python 3.12 packages into Isaac's Python 3.11 process.
export PYTHONPATH="$PWD:$PWD/../habitat_demo"
exec "$ISAAC_PYTHON" -m isaac_web "$@"
