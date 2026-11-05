"""Matched twelve-update arms with only the DPO beta coefficient changed."""

import os

from examples.cosmos_lab.dpo_i2v_config import make_config as base_config


def make_config():
    config = base_config()
    lab = os.environ.get("COSMOS_LAB_ROOT", "/mnt/patient-unit/home/yan/cosmos-lab")
    beta = int(os.environ["YUZH_DPO_BETA"])
    if beta not in {500, 5000}:
        raise ValueError("Only the two preregistered beta values are supported")
    config.job.group = "bridge-i2v-beta-study"
    config.job.name = f"yuzh-beta-{beta}"
    config.rl.beta = float(beta)
    config.rl.max_steps = 12
    config.rl.save_every = 12
    config.rl.validation.every_steps = 4
    config.rl.data.factory_kwargs["jsonl_path"] = f"{lab}/data/bridge-preferences-labeled-v2/train.jsonl"
    config.rl.validation.data.factory_kwargs["jsonl_path"] = (
        f"{lab}/data/bridge-preference-validation-labeled-v2/val.jsonl"
    )
    return config
