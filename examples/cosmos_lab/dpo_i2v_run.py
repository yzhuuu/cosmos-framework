"""Native DPO entrypoint with fixed cuDNN choices and explicit experiment metadata."""

import torch

from cosmos_framework.posttrain.entrypoints.dpo import main

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
