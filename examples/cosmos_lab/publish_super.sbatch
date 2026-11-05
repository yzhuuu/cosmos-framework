#!/usr/bin/env bash
#SBATCH --job-name=cosmos-lora-publish
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/publish-super-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-lora
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export WANDB_CONSOLE=off
TRAIN_JOB=${COSMOS_LORA_TRAIN_JOB:?}
EVAL_JOB=${COSMOS_LORA_EVAL_JOB:?}
TRAIN_DIR="$COSMOS_LAB_ROOT/runs/training/cosmos-lab/sft/bridge-super-lora-10"
TRACE="$COSMOS_LAB_ROOT/runs/train-super-$TRAIN_JOB"
COMPARISON="$COSMOS_LAB_ROOT/runs/bridge-super-comparison-$EVAL_JOB"
python examples/cosmos_lab/audit_lora.py "$COSMOS_LAB_ROOT/checkpoints/Cosmos3-Super-DCP" \
 "$TRAIN_DIR/checkpoints/iter_000000010" "$TRACE/checkpoint-audit.json"
HELPERS="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/examples/cosmos_lab"
python "$HELPERS/compare_bridge_runs.py" "$COMPARISON" \
 "Super base=$COSMOS_LAB_ROOT/runs/bridge-super-baseline-382" \
 "Super LoRA 10=$COSMOS_LAB_ROOT/runs/bridge-lora-10-val-$EVAL_JOB"
python "$HELPERS/publish_eval.py" "$COMPARISON" --name super-val-base-vs-lora-10
export COSMOS_COMPARE_DIR="$COMPARISON"
python - <<'PY'
from pathlib import Path
import os
root=Path(os.environ['COSMOS_LAB_ROOT'])
comparison=Path(os.environ['COSMOS_COMPARE_DIR'])
url=(comparison/'wandb-url.txt').read_text().strip()
entry=f'\n## 自动后处理完成（job {os.environ["SLURM_JOB_ID"]}）\n\nLoRA checkpoint 审计、4 条匹配视频对照和 W&B 上传均成功。[评测 run]({url})。这验证流程与参数更新，不代表 10 步 LoRA 改善操作成功率。\n'
with (root/'docs/experiments/2026-11-04-t3-super-lora.md').open('a') as f:f.write(entry)
with (root/'docs/progress.md').open('a') as f:f.write(entry)
PY
