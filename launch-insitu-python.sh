#!/usr/bin/env bash
set -euxo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Put BOTH the repo root AND the outer 'doreisa' folder on PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/doreisa"

# Do everything with a single ray head node
ray start --head --node-ip-address=10.3.77.167 --port=6379 

# analytics
python3 -m analytics.avg &
ANALYTICS_PID=$!

# simulation
mpirun -n 2 python3 python/sim-doreisa.py  --steps 10 --print-every 1 --seed-mode local --periodic --viz-every 1 --viz-gif

wait $ANALYTICS_PID
ray stop