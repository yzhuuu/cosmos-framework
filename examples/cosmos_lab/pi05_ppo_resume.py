"""Resume the accepted step-2 checkpoint for six more updates; fixed recipe."""

import os
from pathlib import Path

from examples.cosmos_lab.pi05_ppo_smoke import make_config as smoke_config


def make_config():
    config = smoke_config()
    source = Path(os.environ["YUZH_PPO_RESUME"])
    if source.name != "iter_000002" or not (source / "trainer.pt").is_file():
        raise ValueError("R7 requires an explicit accepted step-2 native checkpoint")
    config.rl.resume = str(source.resolve())
    config.rl.max_steps = 8
    config.rl.save_every = 2
    config.job.group = "pi05-ppo-resume"
    return config
