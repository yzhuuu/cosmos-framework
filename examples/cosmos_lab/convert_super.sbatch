#!/usr/bin/env bash
#SBATCH --job-name=cosmos-super-dcp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/convert-super-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-lora
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export COSMOS_TRAINING=0
export OMP_NUM_THREADS=16
DEST="$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Super-DCP"
test ! -e "$DEST"
srun python -m cosmos_framework.scripts.convert_model_to_dcp \
 --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Super" -o "$DEST"
