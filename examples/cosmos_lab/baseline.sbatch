#!/usr/bin/env bash
#SBATCH --job-name=cosmos-nano-baseline
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/baseline-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework
source examples/cosmos_lab/environment.sh
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/nano-baseline-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
cp "$COSMOS_LAB_ROOT/data/baseline-robot-relocation/requests.jsonl" "$OUTPUT_DIR/requests.jsonl"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze > "$OUTPUT_DIR/packages.txt"
# Sample device memory once per second; this is device usage, not allocator peak.
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$OUTPUT_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano" \
  -i "$OUTPUT_DIR/requests.jsonl" -o "$OUTPUT_DIR/generated" \
  --seed=2026 --resolution=480 --max-num-seqs=1 \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
python examples/cosmos_lab/validate_outputs.py "$OUTPUT_DIR"
