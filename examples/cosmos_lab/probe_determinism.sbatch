#!/usr/bin/env bash
#SBATCH --job-name=yuzh-parity-fixed
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/parity-fixed-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-sft
source examples/cosmos_lab/environment.sh
export PATH="$PWD/.venv/bin:$PATH"
export TMPDIR="$COSMOS_LAB_ROOT/tmp/j${SLURM_JOB_ID}"
mkdir -m 700 -p "$TMPDIR"
TRACE="$COSMOS_LAB_ROOT/runs/parity-fixed-${SLURM_JOB_ID}"
mkdir "$TRACE"
git rev-parse HEAD > "$TRACE/code-commit.txt"
git diff HEAD > "$TRACE/code.patch"
cp examples/cosmos_lab/probe_determinism.py examples/cosmos_lab/probe_determinism.sbatch "$TRACE/"
cp "$COSMOS_LAB_ROOT/data/bridge-study-v1/val-requests.jsonl" "$TRACE/requests.jsonl"
TRAIN_DIR="$COSMOS_LAB_ROOT/runs/training/cosmos-lab/sft/bridge-nano-200"
for mode in dcp-default dcp-fixed-a dcp-fixed-b hf-fixed; do
    export YUZH_PROBE_CUDNN=fixed
    if [[ "$mode" == dcp-default ]]; then export YUZH_PROBE_CUDNN=default; fi
    export YUZH_PROBE_DIR="$TRACE/$mode/probe"
    if [[ "$mode" == dcp-* ]]; then
        export COSMOS_TRAINING=1
        LOAD=(--checkpoint-path "$TRAIN_DIR/checkpoints/iter_000000200" --config-file "$TRAIN_DIR/config.yaml" --no-use-ema-weights)
    else
        export COSMOS_TRAINING=0
        LOAD=(--checkpoint-path "$COSMOS_LAB_ROOT/checkpoints/Bridge-Nano-200-HF")
    fi
    srun torchrun --standalone --nproc-per-node=2 -m examples.cosmos_lab.probe_determinism \
        "${LOAD[@]}" \
        --parallelism-preset=throughput --dp-shard-size=2 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 \
        -i "$TRACE/requests.jsonl" -o "$TRACE/$mode/generated" \
        --seed=2026 --resolution=256 --max-num-seqs=1 \
        --no-use-torch-compile --no-diffusion-cache --no-guardrails
done
