#!/usr/bin/env bash
#SBATCH --job-name=yuzh-dense-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/dense-eval-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-lora
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export COSMOS_TRAINING=0
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/super-dense-val-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
cp "$COSMOS_LAB_ROOT/data/bridge-super-val-requests.jsonl" "$OUTPUT_DIR/requests.jsonl"
cp "$COSMOS_LAB_ROOT/data/bridge-study-v1/manifest.json" "$OUTPUT_DIR/dataset-manifest.json"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun torchrun --standalone --nproc-per-node=2 -m examples.cosmos_lab.stable_inference \
 --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Bridge-Super-LoRA-10-Dense-HF" \
 --parallelism-preset=throughput --dp-shard-size=2 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 \
 -i "$OUTPUT_DIR/requests.jsonl" -o "$OUTPUT_DIR/generated" \
 --seed=2026 --resolution=256 --max-num-seqs=1 \
 --no-use-torch-compile --no-diffusion-cache --no-guardrails
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
python examples/cosmos_lab/validate_outputs.py "$OUTPUT_DIR"

python examples/cosmos_lab/evaluate_bridge.py "$OUTPUT_DIR"
