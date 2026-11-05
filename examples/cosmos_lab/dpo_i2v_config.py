"""Two-update real-video preference experiment with episode-disjoint validation."""

import os

from cosmos_framework.posttrain.config.rl import DPOValidationConfig, PreferenceEnvConfig
from cosmos_framework.posttrain.configs.dpo_t2i_public import build_config


def make_config():
    config = build_config()
    lab = os.environ.get("COSMOS_LAB_ROOT", "/mnt/patient-unit/home/yan/cosmos-lab")
    config.policy_adapter = "examples.cosmos_lab.i2v_preferences.BridgeI2VFlowPolicy"
    config.grid.ranks = 2
    config.model.config.parallelism.data_parallel_shard_degree = 2
    config.model.config.state_t = 24
    config.job.group = "bridge-i2v-dpo"
    config.job.name = "yuzh-i2v-dpo-2"
    config.rl.max_steps = 2
    config.rl.seed = 2026
    config.rl.save_every = 2
    config.rl.checkpoint_components = ["model"]
    config.rl.data = PreferenceEnvConfig(
        factory="examples.cosmos_lab.i2v_preferences.BridgeI2VPreferences",
        factory_kwargs={
            "jsonl_path": f"{lab}/data/bridge-preferences-labeled-v1/train.jsonl",
            "bridge_manifest_path": f"{lab}/data/bridge-study-v1/manifest.json",
        },
        media="video",
        shuffle=False,
        cfg_dropout_rate=0.0,
    )
    config.rl.validation = DPOValidationConfig(
        enabled=True,
        before_training=True,
        every_steps=0,
        seed=901,
        data=PreferenceEnvConfig(
            factory="examples.cosmos_lab.i2v_preferences.BridgeI2VPreferences",
            factory_kwargs={
                "jsonl_path": f"{lab}/data/bridge-preferences-labeled-v1/val.jsonl",
                "bridge_manifest_path": f"{lab}/data/bridge-study-v1/manifest.json",
                "purpose": "validation",
            },
            media="video",
            shuffle=False,
            cfg_dropout_rate=0.0,
        ),
    )
    return config
