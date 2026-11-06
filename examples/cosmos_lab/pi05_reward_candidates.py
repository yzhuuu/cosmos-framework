"""Reward arithmetic for the separately versioned R10 first-reach ablation."""

from cosmos_framework.posttrain.schema import EnvironmentFeedback

FIRST_REACH_PREFIX_REVISION = "yuzh-first-reach-prefix-v1"


def first_reach_prefix_reward(feedback: EnvironmentFeedback) -> float:
    """Sum through the first successful control in a chunk, inclusive.

    Requires the fixed-episode raw per-control facts. This pure arithmetic helper
    is wired only by the explicit R10 config, which binds its revision into
    checkpoint objective identity and keeps the native reference recipe separate.
    """
    if feedback.control_terminated is None:
        raise ValueError("First-reach prefix requires raw per-control success flags")
    stop = next((index + 1 for index, flag in enumerate(feedback.control_terminated) if flag), feedback.executed_steps)
    return sum(feedback.control_rewards[:stop])
