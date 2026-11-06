#!/usr/bin/env bash
#SBATCH --job-name=yuzh-super-baseline
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/super-baseline-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework
source examples/cosmos_lab/environment.sh
export OMP_NUM_THREADS=4
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/super-baseline-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
cp "$COSMOS_LAB_ROOT/data/baseline-super-robot-relocation/requests.jsonl" "$OUTPUT_DIR/requests.jsonl"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze > "$OUTPUT_DIR/packages.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$OUTPUT_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  --dp-shard-size=2 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 \
  --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Super" \
  -i "$OUTPUT_DIR/requests.jsonl" -o "$OUTPUT_DIR/generated" \
  --seed=2026 --resolution=480 --max-num-seqs=1 \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
python examples/cosmos_lab/validate_outputs.py "$OUTPUT_DIR"
