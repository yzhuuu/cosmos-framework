#!/usr/bin/env bash
#SBATCH --job-name=yuzh-saved-replay
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/saved-replay-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-preferences
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/runtime/pi05-ppo/venv/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/pi05-saved-replay-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
srun python -m examples.cosmos_lab.pi05_saved_replay "$COSMOS_LAB_ROOT" "$OUTPUT_DIR/probe"
