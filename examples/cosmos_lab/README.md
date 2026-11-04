# Educational Nano baseline

This experiment prepares versioned public samples and runs a matched text-to-video / image-to-video robot relocation example. It checks the inference path, not held-out task performance. The task text and initial image come from `inputs/omni/i2v.json`: the right robot hand transfers a red sphere from the top shelf to the bottom shelf while the left hand stays still.

## Environment

Validated on Ubuntu 22.04, Python 3.13.5, one H200, driver 595.91.07, and the repository's locked CUDA 13.0 dependency group. Keep this environment separate from other model projects. From the repository root:

```bash
export COSMOS_LAB_ROOT=/mnt/patient-unit/home/yan/cosmos-lab
chmod 700 "$COSMOS_LAB_ROOT"
export UV_CACHE_DIR="$COSMOS_LAB_ROOT/cache/uv"
uvx --from uv==0.11.29 uv sync --locked --no-default-groups --group cu130 --extra guardrail
source examples/cosmos_lab/environment.sh
```

TorchCodec requires FFmpeg shared libraries. On this host, Ubuntu's FFmpeg 4.4.2 and its missing dependencies were downloaded with `apt-get download` and extracted using `dpkg-deb -x` into `$COSMOS_LAB_ROOT/runtime/ffmpeg`; no system packages were installed. The exact package plan and download log are in `runs/preflight/ffmpeg-apt-plan.txt` and `ffmpeg-download.log`. The environment helper prepends that library directory and its BLAS/LAPACK subdirectories while preserving the platform's existing `LD_LIBRARY_PATH`. On another host, use its supported FFmpeg installation or repeat dependency resolution for that host; do not copy this host's missing-package list blindly.

## Prepare and validate

```bash
python examples/cosmos_lab/prepare_assets.py --lab-root "$COSMOS_LAB_ROOT"
python examples/cosmos_lab/validate_samples.py
python examples/cosmos_lab/prepare_assets.py --lab-root "$COSMOS_LAB_ROOT" --nano-weights
python examples/cosmos_lab/prepare_baseline.py
# Run this check inside a GPU allocation, after sourcing environment.sh:
python examples/cosmos_lab/preflight.py
```

The downloader pins Hugging Face revisions and records file sizes and SHA-256 hashes. Only Nano base weights are downloaded; Sim variants contribute small example assets and action metadata. The sample validator fully decodes videos, checks frame timing and official Bimanual checksums, and validates actions through the native Bimanual contract. Transfer controls with mismatched spatial dimensions are explicitly excluded in the receipt instead of silently treated as aligned.

## Run

```bash
sbatch examples/cosmos_lab/baseline.sbatch
```

The Slurm script targets this lab's `all` partition with one H200, 8 CPUs, 192 GB host RAM and a one-hour limit. Adjust absolute workspace paths and scheduler settings for another host. Every job uses a fresh `runs/nano-baseline-JOB_ID/` directory.

Both requests use seed 2026, 193 frames at 24 FPS, 480-class resolution, 35 denoising steps, guidance 6 and shift 10. Prompt upsampling and sound are disabled. Compilation, diffusion cache and guardrails are disabled for this fixed offline robot experiment. There is no public endpoint. These are inference settings, not training updates.

The run records the commit, tracked working-tree patch, installed packages, original requests, resolved native setup/sample arguments, start/end times, and once-per-second device memory/utilization samples. `validate_outputs.py` fully decodes the generated videos, checks count/timing, computes adjacent-frame differences and creates contact sheets. Device memory samples are not exact allocator peaks. Pixel changes and valid video files do not establish task success: inspect the actual transfer, sphere persistence, stationary left arm and end state.

No training, Super inference, serving, simulator rollout or benchmark generalization claim is part of this experiment. Study progress and manual observations live in the separate lab `docs/` folder.
