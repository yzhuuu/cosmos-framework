#!/usr/bin/env bash
#SBATCH --job-name=yuzh-policy-probe
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/policy-probe-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-preferences
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/runtime/libero/venv/bin:$PATH"
export LIBERO_CONFIG_PATH="$COSMOS_LAB_ROOT/runtime/libero/config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$COSMOS_LAB_ROOT/runtime/libero/egl-nvidia.json"
export LD_LIBRARY_PATH="$COSMOS_LAB_ROOT/runtime/libero/system/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export NUMBA_CACHE_DIR="$COSMOS_LAB_ROOT/cache/numba-libero"
export MPLCONFIGDIR="$COSMOS_LAB_ROOT/cache/matplotlib-libero"
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/pi05-closed-loop-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
cp "${YUZH_POLICY_PROTOCOL:-$COSMOS_LAB_ROOT/runs/r3-libero-preflight/policy-protocol.json}" "$OUTPUT_DIR/protocol.json"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
git -C "$COSMOS_LAB_ROOT/runtime/libero/RLinf-LIBERO" rev-parse HEAD > "$OUTPUT_DIR/assets-commit.txt"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/simulator-packages.txt"
POLICY_PYTHON=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("policy_python", "repos/cosmos-framework-sft/.venv/bin/python"))' "$OUTPUT_DIR/protocol.json")
uv pip freeze --python "$COSMOS_LAB_ROOT/$POLICY_PYTHON" > "$OUTPUT_DIR/policy-packages.txt"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun python -m examples.cosmos_lab.pi05_closed_loop "$COSMOS_LAB_ROOT" "$OUTPUT_DIR/probe" "$OUTPUT_DIR/protocol.json"
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
