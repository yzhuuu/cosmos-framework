#!/usr/bin/env bash
#SBATCH --job-name=yuzh-i2v-audit
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/i2v-audit-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-i2v-dpo
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export COSMOS_TRAINING=1 COSMOS_INTERNAL=0 OMP_NUM_THREADS=4 WANDB_DISABLE_CODE=true
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/i2v-audit-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun torchrun --standalone --nproc-per-node=1 -m examples.cosmos_lab.audit_i2v \
 "$COSMOS_LAB_ROOT" "$OUTPUT_DIR" \
 "$COSMOS_LAB_ROOT/cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
