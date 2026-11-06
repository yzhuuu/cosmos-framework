#!/usr/bin/env bash
#SBATCH --job-name=yuzh-bimanual-baseline
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/bimanual-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework
source examples/cosmos_lab/environment.sh
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/bimanual-baseline-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
cp "$COSMOS_LAB_ROOT/data/bimanual-baseline/requests.jsonl" "$OUTPUT_DIR/requests.jsonl"
cp "$COSMOS_LAB_ROOT/data/bimanual-baseline/pairs.json" "$OUTPUT_DIR/pairs.json"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze > "$OUTPUT_DIR/packages.txt"
# Sample device memory once per second; this is device usage, not allocator peak.
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$OUTPUT_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano-Sim-Bimanual" \
  -i "$OUTPUT_DIR/requests.jsonl" -o "$OUTPUT_DIR/generated" \
  --seed=2026 --resolution=480 --image-size=480 --max-num-seqs=1 \
  --model-mode=FORWARD_DYNAMICS --num-steps=4 --kv-cache-inference-size=64 \
  --attention-sink-size=8 --autoregressive=True --video-save-quality=4 \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
python examples/cosmos_lab/validate_outputs.py "$OUTPUT_DIR"
