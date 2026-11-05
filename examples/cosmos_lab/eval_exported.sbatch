#!/usr/bin/env bash
#SBATCH --job-name=yuzh-bridge-hf-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/bridge-hf-eval-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-sft
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export COSMOS_TRAINING=0
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/bridge-hf-${COSMOS_EVAL_STEP:-200}-${COSMOS_EVAL_SPLIT:-val}-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
cp "$COSMOS_LAB_ROOT/data/bridge-study-v1/${COSMOS_EVAL_SPLIT:-val}-requests.jsonl" "$OUTPUT_DIR/requests.jsonl"
cp "$COSMOS_LAB_ROOT/data/bridge-study-v1/manifest.json" "$OUTPUT_DIR/dataset-manifest.json"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun torchrun --standalone --nproc-per-node=2 -m cosmos_framework.scripts.inference \
 --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Bridge-Nano-${COSMOS_EVAL_STEP:-200}-HF" \
 --parallelism-preset=throughput --dp-shard-size=2 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 \
 -i "$OUTPUT_DIR/requests.jsonl" -o "$OUTPUT_DIR/generated" \
 --seed=2026 --resolution=256 --max-num-seqs=1 \
 --no-use-torch-compile --no-diffusion-cache --no-guardrails
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
python examples/cosmos_lab/validate_outputs.py "$OUTPUT_DIR"

python examples/cosmos_lab/evaluate_bridge.py "$OUTPUT_DIR"
