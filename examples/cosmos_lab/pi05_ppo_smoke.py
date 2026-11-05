"""Two-update native Joint PPO educational probe; shared-pool diagnostics, not generalization."""

import os

from cosmos_framework.posttrain.configs.pi05_libero_spatial_public_4gpu import make_config as public_config


def make_config():
    config = public_config()
    config.rl.max_steps = 2
    config.rl.save_every = 1
    config.rl.keep_optim_last = 2
    config.rl.validation.enabled = False
    config.rl.verify_behavior_logprobs = True
    # Rollout calls are scalar. Probe513 finds strict scalar/B16 head replay drift.
    # Keep global batch384; accumulate96 scalar microbatches per learner rank.
    config.rl.microbatch = 1
    config.job.project = "cosmos-lab"
    config.job.group = "pi05-ppo-smoke"
    config.job.name = "yuzh-ppo-" + os.environ.get("SLURM_JOB_ID", "preflight")
    config.job.wandb_mode = "online"
    return config
