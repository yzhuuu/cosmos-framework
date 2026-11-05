#!/usr/bin/env bash
#SBATCH --job-name=yuzh-ppo-smoke
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/ppo-smoke-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-preferences
source examples/cosmos_lab/environment.sh
export COSMOS_TRAINING=1
export PATH="$COSMOS_LAB_ROOT/runtime/pi05-ppo/venv/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export POSTTRAIN_RESOURCE_ROOT="$COSMOS_LAB_ROOT/runtime/pi05-ppo/resources"
export LIBERO_MANIFEST_ROOT="$COSMOS_LAB_ROOT/runtime/pi05-ppo/manifests"
export POSTTRAIN_SIM_PYTHON="$COSMOS_LAB_ROOT/runtime/libero/venv/bin/python"
export LIBERO_CONFIG_PATH="$COSMOS_LAB_ROOT/runtime/libero/config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$COSMOS_LAB_ROOT/runtime/libero/egl-nvidia.json"
export LD_LIBRARY_PATH="$COSMOS_LAB_ROOT/runtime/libero/system/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export NUMBA_CACHE_DIR="$COSMOS_LAB_ROOT/cache/numba-libero"
export MPLCONFIGDIR="$COSMOS_LAB_ROOT/cache/matplotlib-libero"
export NNODES=1 GPUS_PER_NODE=4
export WANDB_ENTITY=sav-inception-self WANDB_PROJECT=cosmos-lab WANDB_MODE=online
export WANDB_NAME="yuzh-ppo-${SLURM_JOB_ID}" WANDB_RUN_GROUP=pi05-ppo-smoke
export RAY_TMPDIR="/tmp/yuzh-ray-${SLURM_JOB_ID}"
export B1K_SIM_IPC_DIR="/tmp/yuzh-sim-${SLURM_JOB_ID}"
export B1K_SHM_DIR="/tmp/yuzh-shm-${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$B1K_SIM_IPC_DIR" "$B1K_SHM_DIR"
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/pi05-ppo-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_DIR/output"
export WANDB_DIR="$OUTPUT_DIR"
cp "$COSMOS_LAB_ROOT/runs/r5-ppo-preflight/protocol.json" "$OUTPUT_DIR/protocol.json"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
df -h /dev/shm > "$OUTPUT_DIR/shared-memory.txt"
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun python -c 'import torch, vllm, vllm_omni, ray; assert torch.cuda.device_count() == 4; print(torch.__version__, vllm.__version__, ray.__version__); print(torch.ones(1, device="cuda"))'
srun python -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config examples/cosmos_lab/pi05_ppo_smoke.py --validate-manifest
srun python -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config examples/cosmos_lab/pi05_ppo_smoke.py
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
