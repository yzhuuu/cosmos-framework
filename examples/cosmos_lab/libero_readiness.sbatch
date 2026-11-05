#!/usr/bin/env bash
#SBATCH --job-name=yuzh-sim-probe
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/sim-probe-%j.log
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
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/libero-readiness-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
cp "$COSMOS_LAB_ROOT/runs/r3-libero-preflight/protocol.json" "$OUTPUT_DIR/protocol.json"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
git -C "$COSMOS_LAB_ROOT/runtime/libero/RLinf-LIBERO" rev-parse HEAD > "$OUTPUT_DIR/assets-commit.txt"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun python -m cosmos_framework.posttrain.embodied.sims.libero.preflight \
  --train-manifest "$COSMOS_LAB_ROOT/runtime/libero/manifests/train.json" \
  --eval-manifest "$COSMOS_LAB_ROOT/runtime/libero/manifests/eval.json" \
  --output "$OUTPUT_DIR/native-preflight.json"
srun python -m examples.cosmos_lab.libero_readiness \
  "$OUTPUT_DIR/protocol.json" "$OUTPUT_DIR/probe"
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
