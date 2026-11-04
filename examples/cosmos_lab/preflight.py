"""Check CUDA computation and actual TorchCodec decoding in an allocation."""

import json
import os
import platform
import subprocess
from pathlib import Path

import cv2
import torch

from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec


def main() -> None:
    """Write a receipt only after both CUDA and media checks succeed."""
    root = Path(os.environ["COSMOS_LAB_ROOT"])
    torch.cuda.init()
    value = torch.ones(4, device="cuda").add_(1)
    torch.cuda.synchronize()
    assert value.tolist() == [2.0] * 4
    codec = load_torchcodec()
    video = root / "data/Cosmos3-Nano-Sim-Bimanual/assets/training/agibot/gt/agibot_001_gt.mp4"
    decoder = codec.decoders.VideoDecoder(str(video), device="cpu")
    frame = decoder[0]
    assert frame.ndim == 3 and frame.shape[0] == 3
    receipt = {
        "status": "passed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "opencv": cv2.__version__,
        "torchcodec": codec.__version__,
        "video": str(video),
        "decoded_first_frame_shape": list(frame.shape),
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    }
    path = root / "runs/preflight" / f"runtime-{receipt['job_id']}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
