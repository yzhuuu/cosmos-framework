#!/usr/bin/env bash
#SBATCH --job-name=yuzh-dense-export
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/export-super-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-lora
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export COSMOS_TRAINING=1 COSMOS_DEVICE=cpu OMP_NUM_THREADS=16
TRAIN_DIR="$COSMOS_LAB_ROOT/runs/training/cosmos-lab/sft/bridge-super-lora-10"
OUTPUT_DIR="$COSMOS_LAB_ROOT/checkpoints/Bridge-Super-LoRA-10-Dense-HF"
TRACE="$COSMOS_LAB_ROOT/runs/export-super-${SLURM_JOB_ID}"
test ! -e "$OUTPUT_DIR"
mkdir "$TRACE"
git rev-parse HEAD > "$TRACE/code-commit.txt"
git diff HEAD > "$TRACE/code.patch"
cp "$TRAIN_DIR/config.yaml" "$TRACE/config.yaml"
cp examples/cosmos_lab/export_super.sbatch "$TRACE/"
date -u +%FT%TZ > "$TRACE/start.txt"
srun python -m cosmos_framework.scripts.export_model \
 --checkpoint-path "$TRAIN_DIR/checkpoints/iter_000000010" \
 --config-file "$TRAIN_DIR/config.yaml" --no-use-ema-weights --no-vit --merge-lora \
 -o "$OUTPUT_DIR"
date -u +%FT%TZ > "$TRACE/end.txt"
