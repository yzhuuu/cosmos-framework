#!/usr/bin/env bash
#SBATCH --job-name=yuzh-beta-compare
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/beta-compare-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-i2v-dpo
source examples/cosmos_lab/environment.sh
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework-sft/.venv/bin:$PATH"
export OMP_NUM_THREADS=4
OUT="$COSMOS_LAB_ROOT/runs/beta-compare-${SLURM_JOB_ID}"
mkdir "$OUT"
cp "$COSMOS_LAB_ROOT/data/bridge-preference-validation-v2/requests.jsonl" "$OUT/requests.jsonl"
cp "$COSMOS_LAB_ROOT/data/bridge-beta-study-v1/jobs.json" "$OUT/training-jobs.json"
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
python -m examples.cosmos_lab.review_beta_generation "$OUT" --baseline-only
export YUZH_DPO_EXPECTED_STEP=12
for beta in 5000 500; do
 export YUZH_DPO_BETA="$beta"
 YUZH_DPO_OVERLAY=$(python - <<'PY'
import json, os
from pathlib import Path
lab=Path(os.environ['COSMOS_LAB_ROOT'])
beta=os.environ['YUZH_DPO_BETA']
job=json.loads((lab/'data/bridge-beta-study-v1/jobs.json').read_text())[beta]
paths=list((lab/f'runs/training/cosmos-lab/bridge-i2v-beta-study/yuzh-beta-{beta}-{job}').glob('*/checkpoint_artifacts/iter_000012'))
assert len(paths)==1, paths
print(paths[0])
PY
)
 export YUZH_DPO_OVERLAY
 srun python -m examples.cosmos_lab.verify_dpo_artifact "$YUZH_DPO_OVERLAY" "$OUT/beta${beta}-sha256.json"
 generate "beta${beta}"
done
python -m examples.cosmos_lab.review_beta_generation "$OUT"
date -u +%FT%TZ > "$OUT/end.txt"
