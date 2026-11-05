#!/usr/bin/env bash
#SBATCH --job-name=yuzh-dpo-compare
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/dpo-compare-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-i2v-dpo
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export OMP_NUM_THREADS=4
OUT="$COSMOS_LAB_ROOT/runs/dpo-compare-${SLURM_JOB_ID}"
mkdir "$OUT"
cp "$COSMOS_LAB_ROOT/data/bridge-preference-validation-v1/requests.jsonl" "$OUT/requests.jsonl"
git rev-parse HEAD > "$OUT/code-commit.txt"
git diff HEAD > "$OUT/code.patch"
nvidia-smi > "$OUT/nvidia-smi.txt"
uv pip freeze --python "$(command -v python)" > "$OUT/packages.txt"
date -u +%FT%TZ > "$OUT/start.txt"
generate() {
 srun torchrun --standalone --nproc-per-node=2 -m examples.cosmos_lab.dpo_inference \
  --checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano" \
  --parallelism-preset=throughput --dp-shard-size=2 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 \
  -i "$OUT/requests.jsonl" -o "$OUT/$1" --resolution=256 --max-num-seqs=1 \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails
}
unset YUZH_DPO_OVERLAY
generate base
python -m examples.cosmos_lab.review_dpo_generation "$OUT" --baseline-only
export YUZH_DPO_OVERLAY="$COSMOS_LAB_ROOT/runs/training/cosmos-lab/bridge-i2v-dpo/yuzh-i2v-dpo-481/20261105190703/checkpoint_artifacts/iter_000002"
generate dpo
python -m examples.cosmos_lab.review_dpo_generation "$OUT"
date -u +%FT%TZ > "$OUT/end.txt"
