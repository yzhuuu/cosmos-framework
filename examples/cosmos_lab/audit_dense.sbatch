#!/usr/bin/env bash
#SBATCH --job-name=yuzh-dense-audit
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/dense-audit-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-lora
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export OMP_NUM_THREADS=16
TRACE="$COSMOS_LAB_ROOT/runs/dense-audit-${SLURM_JOB_ID}"
mkdir "$TRACE"
git rev-parse HEAD > "$TRACE/code-commit.txt"
srun python examples/cosmos_lab/audit_dense_export.py \
 "$COSMOS_LAB_ROOT/runs/training/cosmos-lab/sft/bridge-super-lora-10/checkpoints/iter_000000010" \
 "$COSMOS_LAB_ROOT/checkpoints/Bridge-Super-LoRA-10-Dense-HF" "$TRACE/audit.json"
