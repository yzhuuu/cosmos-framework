#!/usr/bin/env bash
#SBATCH --job-name=cosmos-nano-export
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/export-nano-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-sft
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export COSMOS_TRAINING=1 COSMOS_DEVICE=cpu OMP_NUM_THREADS=16
TRAIN_DIR="$COSMOS_LAB_ROOT/runs/training/cosmos-lab/sft/bridge-nano-200"
CKPT_ITER=$(printf 'iter_%09d' "${COSMOS_EVAL_STEP:-200}")
OUTPUT_DIR="$COSMOS_LAB_ROOT/checkpoints/Bridge-Nano-${COSMOS_EVAL_STEP:-200}-HF"
TRACE="$COSMOS_LAB_ROOT/runs/export-nano-${SLURM_JOB_ID}"
mkdir "$TRACE"
git rev-parse HEAD > "$TRACE/code-commit.txt"
git diff HEAD > "$TRACE/code.patch"
cp "$TRAIN_DIR/config.yaml" "$TRACE/config.yaml"
date -u +%FT%TZ > "$TRACE/start.txt"
srun python -m cosmos_framework.scripts.export_model \
 --checkpoint-path "$TRAIN_DIR/checkpoints/$CKPT_ITER" \
 --config-file "$TRAIN_DIR/config.yaml" --no-use-ema-weights \
 -o "$OUTPUT_DIR"
date -u +%FT%TZ > "$TRACE/end.txt"
