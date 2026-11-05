#!/usr/bin/env bash
#SBATCH --job-name=yuzh-beta-study
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/beta-study-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-i2v-dpo
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export COSMOS_TRAINING=1 COSMOS_INTERNAL=0 OMP_NUM_THREADS=4 WANDB_DISABLE_CODE=true
: "${YUZH_DPO_BETA:?Supply the preregistered beta}"
export DPO_BASE_CHECKPOINT="$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano-DCP"
export DPO_TOKENIZER_PATH="$COSMOS_LAB_ROOT/cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
export IMAGINAIRE_OUTPUT_ROOT="$COSMOS_LAB_ROOT/runs/training"
export TMPDIR="$COSMOS_LAB_ROOT/tmp/j${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
TRACE="$COSMOS_LAB_ROOT/runs/beta-study-${SLURM_JOB_ID}"
mkdir "$TRACE"
git rev-parse HEAD > "$TRACE/code-commit.txt"
git diff HEAD > "$TRACE/code.patch"
uv pip freeze --python "$(command -v python)" > "$TRACE/packages.txt"
nvidia-smi > "$TRACE/nvidia-smi.txt"
cp "$COSMOS_LAB_ROOT/data/bridge-preferences-labeled-v2/train.jsonl" "$TRACE/train.jsonl"
cp "$COSMOS_LAB_ROOT/data/bridge-preference-validation-labeled-v2/val.jsonl" "$TRACE/val.jsonl"
cp "$COSMOS_LAB_ROOT/data/bridge-beta-study-v1/manifest.json" "$TRACE/study-manifest.json"
date -u +%FT%TZ > "$TRACE/start.txt"
srun torchrun --standalone --nproc-per-node=2 -m examples.cosmos_lab.dpo_i2v_run \
 --config examples/cosmos_lab/dpo_beta_config.py -- \
 job.project=cosmos-lab job.name="yuzh-beta-${YUZH_DPO_BETA}-${SLURM_JOB_ID}" job.wandb_mode=online
date -u +%FT%TZ > "$TRACE/end.txt"
