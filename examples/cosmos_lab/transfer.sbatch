#!/usr/bin/env bash
#SBATCH --job-name=yuzh-transfer
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/transfer-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-transfer
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 COSMOS_SMOKE=0
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$COSMOS_LAB_ROOT/tmp/j${SLURM_JOB_ID}"
mkdir -m 700 -p "$TMPDIR"
SKILL="$PWD/.agents/skills/cosmos3-nano-sim-transfer"
MODEL="$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano-Sim-Transfer"
MODALITY="${YUZH_TRANSFER_MODALITY:-depth}"
STUDY="${YUZH_TRANSFER_STUDY:-transfer-study-v1}"
WINDOW="${YUZH_TRANSFER_CACHE_WINDOW:-0}"
SINK="${YUZH_TRANSFER_SINK:-0}"
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/transfer-${MODALITY}-${SLURM_JOB_ID}"
mkdir "$OUTPUT_DIR"
cp "$COSMOS_LAB_ROOT/data/${STUDY}/${MODALITY}-requests.jsonl" "$OUTPUT_DIR/requests.jsonl"
cp "$COSMOS_LAB_ROOT/data/${STUDY}/manifest.json" "$OUTPUT_DIR/dataset-manifest.json"
cp "$COSMOS_LAB_ROOT/runs/preflight/transfer-checkpoint.json" "$OUTPUT_DIR/model-manifest.json"
git rev-parse HEAD > "$OUTPUT_DIR/code-commit.txt"
git diff HEAD > "$OUTPUT_DIR/code.patch"
cp examples/cosmos_lab/transfer.sbatch "$OUTPUT_DIR/"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv pip freeze --python "$(command -v python)" > "$OUTPUT_DIR/packages.txt"
OVERRIDES=(
 model.config.tokenizer.bucket_name=
 model.config.tokenizer.object_store_credential_path_pretrained=
 "model.config.tokenizer.vae_path=$COSMOS_LAB_ROOT/cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
 model.config.vlm_config.tokenizer.config_variant=hf
 "model.config.vlm_config.tokenizer.pretrained_model_name=$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Nano/text_tokenizer"
 model.config.vlm_config.pretrained_weights.enabled=False
 model.config.teacher_forcing_frames_per_chunk=1
 model.config.kv_cache_dtype=null
)
date -u +%FT%TZ > "$OUTPUT_DIR/start.txt"
srun python -m cosmos_framework.scripts.inference \
 -i "$OUTPUT_DIR/requests.jsonl" -o "$OUTPUT_DIR/generated" \
 --checkpoint-path "$MODEL" --config-file "$MODEL/config.json" --credential-path "" \
 --experiment-overrides "${OVERRIDES[@]}" \
 --defaults-file "$SKILL/inference/assets/sample_args.json" \
 --autoregressive=True --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
 --seed 2026 --num-outputs 1 --max-num-seqs 1 \
 --num-steps 4 --guidance 1 --control-guidance 1 \
 --kv-cache-inference-size "$WINDOW" --attention-sink-size "$SINK" \
 --no-use-torch-compile --no-use-cuda-graphs --no-condition-stream-encode \
 --no-diffusion-cache --prompt-upsampling=False --no-guardrails
date -u +%FT%TZ > "$OUTPUT_DIR/end.txt"
python examples/cosmos_lab/validate_outputs.py "$OUTPUT_DIR"
