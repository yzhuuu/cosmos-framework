#!/usr/bin/env bash
#SBATCH --job-name=yuzh-nano-sft
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/train-nano-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-sft
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export COSMOS_TRAINING=1
export COSMOS_INTERNAL=0
export OMP_NUM_THREADS=4
export DATASET_PATH="$COSMOS_LAB_ROOT/data/bridge-study-v1"
export BASE_CHECKPOINT_PATH="$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano-DCP"
export WAN_VAE_PATH="$COSMOS_LAB_ROOT/cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
export IMAGINAIRE_OUTPUT_ROOT="$COSMOS_LAB_ROOT/runs/training"
export WANDB_ENTITY=sav-inception-self
export WANDB_PROJECT=cosmos-lab
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
TRACE="$COSMOS_LAB_ROOT/runs/train-nano-${SLURM_JOB_ID}"
mkdir "$TRACE"
git rev-parse HEAD > "$TRACE/code-commit.txt"
git diff HEAD > "$TRACE/code.patch"
cp examples/cosmos_lab/bridge_nano.toml "$TRACE/recipe.toml"
cp "$DATASET_PATH/manifest.json" "$TRACE/dataset-manifest.json"
uv pip freeze --python "$(command -v python)" > "$TRACE/packages.txt"
nvidia-smi > "$TRACE/nvidia-smi.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 2 > "$TRACE/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
date -u +%FT%TZ > "$TRACE/start.txt"
srun torchrun --standalone --nnodes=1 --nproc-per-node=4 -m cosmos_framework.scripts.train \
 --sft-toml=examples/cosmos_lab/bridge_nano.toml -- \
 trainer.max_iter="${COSMOS_TRAIN_UPDATES:-10}" checkpoint.save_iter="${COSMOS_SAVE_EVERY:-10}"
date -u +%FT%TZ > "$TRACE/end.txt"
