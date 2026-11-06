#!/usr/bin/env bash
#SBATCH --job-name=yuzh-prefix-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/patient-unit/home/yan/cosmos-lab/runs/preflight/policy-probe-%j.log
set -euo pipefail
cd /mnt/patient-unit/home/yan/cosmos-lab/repos/cosmos-framework-preferences
source examples/cosmos_lab/environment.sh
: "${YUZH_PREFIX_RUN:?}" "${YUZH_PREFIX_PARENT:?}" "${YUZH_POLICY_PROTOCOL:?}"
export COSMOS_TRAINING=1
export POSTTRAIN_RESOURCE_ROOT="$COSMOS_LAB_ROOT/runtime/pi05-ppo/resources"
export LIBERO_MANIFEST_ROOT="$COSMOS_LAB_ROOT/runtime/pi05-ppo/manifests"
"$COSMOS_LAB_ROOT/runtime/pi05-ppo/venv/bin/python" -m examples.cosmos_lab.pi05_training_acceptance \
  "$YUZH_PREFIX_RUN" --start 2 --end 8 --parent-acceptance "$YUZH_PREFIX_PARENT/acceptance.json"
"$COSMOS_LAB_ROOT/runtime/pi05-ppo/venv/bin/python" -m examples.cosmos_lab.pi05_prefix_acceptance \
  "$YUZH_PREFIX_RUN" "$COSMOS_LAB_ROOT/runs/pi05-ppo-526"
exec bash examples/cosmos_lab/pi05_closed_loop.sbatch
