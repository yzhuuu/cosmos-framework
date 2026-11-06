"""R10: one-factor first-reach credit ablation, native2+6 update schedule."""

import os
from pathlib import Path

from cosmos_framework.posttrain.embodied.sims.libero.adapter import LiberoEnvironment
from cosmos_framework.posttrain.embodied.sims.libero.protocol import PI05_SPATIAL_EXECUTION_PROFILE
from cosmos_framework.posttrain.schema import EnvironmentFeedback
from cosmos_framework.utils.lazy_config import LazyCall as L
from examples.cosmos_lab.pi05_ppo_smoke import make_config as smoke_config
from examples.cosmos_lab.pi05_reward_candidates import FIRST_REACH_PREFIX_REVISION, first_reach_prefix_reward


class FirstReachPrefixReward:
    revision = FIRST_REACH_PREFIX_REVISION

    def __call__(self, feedback: EnvironmentFeedback, *, terminated: bool, truncated: bool) -> float:
        return first_reach_prefix_reward(feedback)

    def identity(self) -> dict:
        return {"suite": "libero_spatial", "revision": self.revision, "aggregation": "through-first-success-inclusive"}


class PrefixEnvironment(LiberoEnvironment):
    """Same simulator/seeds/cases; bind changed decision reward to resume identity."""

    def identity(self) -> dict:
        if self.config.execution_profile != PI05_SPATIAL_EXECUTION_PROFILE:
            raise ValueError("Prefix ablation requires the Pi05 fixed episode profile")
        return {**super().identity(), "decision_reward_override": FirstReachPrefixReward().identity()}


def make_config():
    config = smoke_config()
    config.environment._target_ = "examples.cosmos_lab.pi05_prefix_experiment.PrefixEnvironment"
    config.reward.transform = L(FirstReachPrefixReward)()
    config.job.group = "pi05-prefix"
    source = os.environ.get("YUZH_PREFIX_RESUME")
    if source:
        import torch

        path = Path(source)
        if path.name != "iter_000002":
            raise ValueError("R10 resume must start from its own step2 checkpoint")
        trainer = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
        if (
            trainer["objective_identity"]["model_source"].get("decision_reward_override")
            != FirstReachPrefixReward().identity()
        ):
            raise ValueError("Cannot resume native or other reward checkpoint as the prefix ablation")
        config.rl.resume = str(path.resolve())
        config.rl.max_steps = 8
        config.rl.save_every = 2
    return config
