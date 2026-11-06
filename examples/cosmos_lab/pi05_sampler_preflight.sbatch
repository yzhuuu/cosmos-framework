#!/usr/bin/env bash
#SBATCH --job-name=yuzh-sampler-check
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/sampler-check-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-preferences
source examples/cosmos_lab/environment.sh
OUT="$COSMOS_LAB_ROOT/runs/pi05-sampler-check-${SLURM_JOB_ID}"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/code-commit.txt"
git diff HEAD > "$OUT/code.patch"
nvidia-smi > "$OUT/nvidia-smi.txt"
date -u +%FT%TZ > "$OUT/start.txt"
srun "$COSMOS_LAB_ROOT/runtime/pi05-ppo/venv/bin/python" -m examples.cosmos_lab.pi05_sampler_preflight \
  "$COSMOS_LAB_ROOT" "$COSMOS_LAB_ROOT/runs/pi05-closed-loop-528" "$OUT/probe"
date -u +%FT%TZ > "$OUT/end.txt"
