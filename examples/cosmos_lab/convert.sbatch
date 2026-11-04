#!/usr/bin/env bash
#SBATCH --job-name=cosmos-nano-dcp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/convert-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-sft
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export COSMOS_TRAINING=0
export OMP_NUM_THREADS=8
DEST="$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano-DCP"
test ! -e "$DEST"
srun python -m cosmos_framework.scripts.convert_model_to_dcp \
 --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano" -o "$DEST"
