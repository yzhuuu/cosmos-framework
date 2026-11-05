#!/usr/bin/env bash
#SBATCH --job-name=yuzh-bridge-sft-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/bridge-sft-eval-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-sft
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export COSMOS_TRAINING=1
TRAIN_DIR="$COSMOS_LAB_ROOT/runs/training/cosmos-lab/sft/bridge-nano-200"
CKPT_ITER=$(printf 'iter_%09d' "${COSMOS_EVAL_STEP:-10}")
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/bridge-sft-${COSMOS_EVAL_STEP:-10}-${COSMOS_EVAL_SPLIT:-val}-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
cp "$COSMOS_LAB_ROOT/data/bridge-study-v1/${COSMOS_EVAL_SPLIT:-val}-requests.jsonl" "$OUTPUT_DIR/requests.jsonl"
cp "$COSMOS_LAB_ROOT/data/bridge-study-v1/manifest.json" "$OUTPUT_DIR/dataset-manifest.json"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun torchrun --standalone --nproc-per-node=2 -m examples.cosmos_lab.stable_inference \
 --checkpoint-path "$TRAIN_DIR/checkpoints/$CKPT_ITER" \
 --config-file "$TRAIN_DIR/config.yaml" --no-use-ema-weights \
 --parallelism-preset=throughput --dp-shard-size=2 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 \
 -i "$OUTPUT_DIR/requests.jsonl" -o "$OUTPUT_DIR/generated" \
 --seed=2026 --resolution=256 --max-num-seqs=1 \
 --no-use-torch-compile --no-diffusion-cache --no-guardrails
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
python examples/cosmos_lab/validate_outputs.py "$OUTPUT_DIR"

python examples/cosmos_lab/evaluate_bridge.py "$OUTPUT_DIR"
