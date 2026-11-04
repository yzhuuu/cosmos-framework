#!/usr/bin/env bash
# Source this from a shell on the same shared filesystem as the lab.
export COSMOS_LAB_ROOT=${COSMOS_LAB_ROOT:-/mnt/patient-unit/home/yan/cosmos-lab}
export COSMOS_TRAINING=0
export PYTHONUNBUFFERED=1
export HF_HOME="$COSMOS_LAB_ROOT/cache/huggingface"
export UV_CACHE_DIR="$COSMOS_LAB_ROOT/cache/uv"
export TORCHINDUCTOR_CACHE_DIR="$COSMOS_LAB_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$COSMOS_LAB_ROOT/cache/triton"
export PATH="$COSMOS_LAB_ROOT/repos/cosmos-framework/.venv/bin:$COSMOS_LAB_ROOT/runtime/ffmpeg/usr/bin:$PATH"
# Preserve any platform-provided CUDA compatibility paths.
export LD_LIBRARY_PATH="$COSMOS_LAB_ROOT/runtime/ffmpeg/usr/lib/x86_64-linux-gnu:$COSMOS_LAB_ROOT/runtime/ffmpeg/usr/lib/x86_64-linux-gnu/blas:$COSMOS_LAB_ROOT/runtime/ffmpeg/usr/lib/x86_64-linux-gnu/lapack${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
umask 077
