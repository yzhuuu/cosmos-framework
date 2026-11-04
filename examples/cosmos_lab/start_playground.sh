#!/usr/bin/env bash
# Keep this shell running. GPU, relay and UI stop together on exit or after one hour.
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-serve
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export GRADIO_TEMP_DIR="$COSMOS_LAB_ROOT/cache/gradio"
export GRADIO_ANALYTICS_ENABLED=False
JOB_ID=$(sbatch --parsable examples/cosmos_lab/serve.sbatch)
export COSMOS_SERVE_RUN_DIR="$COSMOS_LAB_ROOT/runs/nano-serve-${JOB_ID}"
mkdir -p "$COSMOS_SERVE_RUN_DIR"
RELAY_PID= UI_PID=
cleanup() {
  [[ -z "$UI_PID" ]] || kill "$UI_PID" 2>/dev/null || true
  [[ -z "$RELAY_PID" ]] || kill "$RELAY_PID" 2>/dev/null || true
  scancel "$JOB_ID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
python examples/cosmos_lab/serve_relay.py "$JOB_ID" > "$COSMOS_SERVE_RUN_DIR/relay.log" 2>&1 &
RELAY_PID=$!
python examples/cosmos_lab/serve_ui.py > "$COSMOS_SERVE_RUN_DIR/ui.log" 2>&1 &
UI_PID=$!
printf 'Job %s; UI http://localhost:7861; login saved in %s/secrets/playground.json\n' "$JOB_ID" "$COSMOS_LAB_ROOT"
while [[ -n "$(squeue -h -j "$JOB_ID" -o %i)" ]]; do
  kill -0 "$RELAY_PID" "$UI_PID"
  sleep 10
done
