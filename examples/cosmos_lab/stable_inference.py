# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native inference CLI with fixed cuDNN choices for controlled lab evaluation.

Applies after checkpoint loaders set their defaults and covers both VAE encoding
and decoding. This does not promise determinism across hardware or software versions.
"""

import functools
import json
import os

import torch

from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.inference import main

original_generate = OmniInference.generate


@functools.wraps(original_generate)
def generate(self, *args, **kwargs):
    previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        if int(os.environ.get("RANK", "0")) == 0:
            output = self.setup_args.output_dir
            output.mkdir(parents=True, exist_ok=True)
            (output / "runtime-numerics.json").write_text(
                json.dumps(
                    {
                        "cudnn_benchmark": False,
                        "cudnn_deterministic": True,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "scope": "Fixed cuDNN choices for generation; no cross-platform guarantee.",
                    },
                    indent=2,
                )
                + "\n"
            )
        return original_generate(self, *args, **kwargs)
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous


OmniInference.generate = generate

if __name__ == "__main__":
    main()
