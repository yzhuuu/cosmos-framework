#!/usr/bin/env bash
#SBATCH --job-name=yuzh-dense-report
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/dense-report-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-lora
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export WANDB_CONSOLE=off
NATIVE="$COSMOS_LAB_ROOT/runs/bridge-lora-10-val-461"
DENSE="$COSMOS_LAB_ROOT/runs/super-dense-val-${YUZH_DENSE_JOB:?Specify the completed dense evaluation job}"
OUTPUT_DIR="$COSMOS_LAB_ROOT/runs/super-dense-comparison-${SLURM_JOB_ID}"
HELPERS="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/examples/cosmos_lab"
python "$HELPERS/compare_bridge_runs.py" "$OUTPUT_DIR" "Native LoRA 10=$NATIVE" "Dense merge 10=$DENSE"
python examples/cosmos_lab/compare_merge_pixels.py "$NATIVE" "$DENSE" "$OUTPUT_DIR/merge-pixels.json"
git rev-parse HEAD > "$OUTPUT_DIR/report-code-commit.txt"
python "$HELPERS/publish_eval.py" "$OUTPUT_DIR" --name super-lora-native-vs-dense-10 --training-runs https://wandb.ai/sav-inception-self/cosmos-lab/runs/lfr41t44
