#!/usr/bin/env bash
#SBATCH --job-name=cosmos-dpo-smoke
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/dpo-smoke-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-dpo
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export COSMOS_TRAINING=1 COSMOS_INTERNAL=0 OMP_NUM_THREADS=4
export TMPDIR="$COSMOS_LAB_ROOT/tmp/j${SLURM_JOB_ID}"
mkdir -m 700 -p "$TMPDIR"
export DPO_BASE_CHECKPOINT="$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano-DCP"
export DPO_TOKENIZER_PATH="$COSMOS_LAB_ROOT/cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
export IMAGINAIRE_OUTPUT_ROOT="$COSMOS_LAB_ROOT/runs/training"
export WANDB_ENTITY=sav-inception-self WANDB_PROJECT=cosmos-lab WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
TRACE="$COSMOS_LAB_ROOT/runs/dpo-smoke-${SLURM_JOB_ID}"
mkdir "$TRACE"
git rev-parse HEAD > "$TRACE/code-commit.txt"
git diff HEAD > "$TRACE/code.patch"
uv pip freeze --python "$(command -v python)" > "$TRACE/packages.txt"
nvidia-smi > "$TRACE/nvidia-smi.txt"
date -u +%FT%TZ > "$TRACE/start.txt"
srun torchrun --standalone --nproc-per-node=4 -m cosmos_framework.posttrain.entrypoints.dpo \
 --config cosmos_framework/posttrain/configs/dpo_t2i_public.py -- \
 rl.max_steps=5 rl.save_every=0 rl.seed=2026 \
 job.project=cosmos-lab job.name=dpo-image-smoke-5 job.wandb_mode=online
date -u +%FT%TZ > "$TRACE/end.txt"
