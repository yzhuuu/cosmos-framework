#!/usr/bin/env bash
#SBATCH --job-name=yuzh-nano-serve
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/serve-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-serve
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export RAY_TMPDIR="/tmp/yan-cosmos-ray-${SLURM_JOB_ID}"
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/nano-serve-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$HOSTNAME" > "$OUTPUT_DIR/host.txt"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
uv pip freeze > "$OUTPUT_DIR/packages.txt"
python -m cosmos_framework.inference.ray.serve \
  --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano" \
  --parallelism-preset=latency --world-size=1 -o "$OUTPUT_DIR/generated" \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails
