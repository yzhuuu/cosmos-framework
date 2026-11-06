"""Audit actual PPO target coverage and prove chunk/mask semantics without training."""

import argparse
import collections
import json
from pathlib import Path

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY, JointPPOStats
from cosmos_framework.posttrain.algorithm.joint_ppo.targets import fixed_episode_targets
from cosmos_framework.posttrain.schema import EnvironmentFeedback, ObsBundle, SampleSource, Trajectory, Transition
from examples.cosmos_lab.pi05_reward_candidates import first_reach_prefix_reward


def semantic_probe(flags):
    """Native target function on explicit toy controls; values0 are illustrative only."""
    samples, previous = [], 0
    for index in range(2):
        chunk = flags[index * 5 : (index + 1) * 5]
        rewards = []
        for success in chunk:
            rewards.append(float(success - previous))
            previous = success
        truncated = tuple(index * 5 + offset == 9 for offset in range(5))
        sample_id = f"toy:{index}"
        transition = Transition(
            obs=ObsBundle(),
            source=SampleSource(sample_id, "toy", "fragment"),
            step_idx=index,
            done=any(chunk) or any(truncated),
            truncated=any(truncated),
            feedback=EnvironmentFeedback(tuple(rewards), 5, tuple(map(bool, chunk)), truncated, index == 1),
            collection_end=index == 1,
            reward={"final": sum(rewards)},
            algo_extra={STATS_KEY: JointPPOStats(0.0, 0.0)},
        )
        samples.append(Trajectory([transition], id=sample_id, session="toy"))
    targets = fixed_episode_targets(samples, gamma=0.99, gae_lambda=0.95)
    return {
        "success_flags": flags,
        "chunk_rewards": [s.final_reward for s in samples],
        "untrained_prefix_candidate_rewards": [first_reach_prefix_reward(s.transitions[0].feedback) for s in samples],
        "loss_masks": [t.loss_mask for t in targets],
        "illustrative_returns_zero_old_values": [t.returns for t in targets],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    episodes, sequences = [], []
    for run in args.runs:
        paths = list((run / "output").rglob("windows.jsonl"))
        assert len(paths) == 1
        sequence = []
        for line in paths[0].read_text().splitlines():
            window = json.loads(line)
            groups = collections.defaultdict(list)
            for row in window["target_diagnostics"]:
                groups[row["sample_id"].rsplit(":", 1)[0]].append(row)
            assert len(groups) == 8
            for episode in window["episodes"]:
                rows = sorted(groups[episode["episode_id"]], key=lambda r: int(r["sample_id"].rsplit(":", 1)[1]))
                assert len(rows) == 48 and sum(r["executed_steps"] for r in rows) == 240
                assert episode["success"] == any(r["terminated"] for r in rows)
                first_done = next((i for i, r in enumerate(rows) if r["terminated"]), 47)
                assert [r["loss_mask"] for r in rows] == [i <= first_done for i in range(48)]
                active = [r for r in rows if r["loss_mask"]]
                masked = [r for r in rows if not r["loss_mask"]]
                reward_sum = sum(r["reward"] for r in rows)
                assert reward_sum in [0, 1]  # Relative success rewards telescope from reset previous_reward0.
                episodes.append(
                    {
                        "run": run.name,
                        "step": window["optimizer_step"],
                        "episode_id": episode["episode_id"],
                        "task": episode["task_id"],
                        "state": episode["init_state_id"],
                        "success_any": episode["success"],
                        "reported_return": episode["return"],
                        "physical_reward_sum": reward_sum,
                        "active_reward_sum": sum(r["reward"] for r in active),
                        "active_decisions": len(active),
                        "masked_decisions": len(masked),
                        "negative_decisions_active": sum(r["reward"] < 0 for r in active),
                        "negative_decisions_masked": sum(r["reward"] < 0 for r in masked),
                    }
                )
                sequence.append([episode["task_id"], episode["init_state_id"]])
        sequences.append(sequence)
    probes = {
        "reach_and_hold": semantic_probe([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "reach_then_lose_same_chunk": semantic_probe([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        "reach_then_lose_later_chunk": semantic_probe([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        "never_reach": semantic_probe([0] * 10),
    }
    assert probes["reach_then_lose_same_chunk"]["illustrative_returns_zero_old_values"] == [0, 0]
    assert probes["reach_then_lose_same_chunk"]["untrained_prefix_candidate_rewards"] == [1, 0]
    assert probes["reach_then_lose_later_chunk"]["chunk_rewards"] == [1, -1]
    assert probes["reach_then_lose_later_chunk"]["loss_masks"] == [True, False]
    summary = {
        "episodes": len(episodes),
        "decisions": len(episodes) * 48,
        "any_success_episodes": sum(r["success_any"] for r in episodes),
        "physical_reward_sum_one_episodes": sum(r["physical_reward_sum"] == 1 for r in episodes),
        "successful_with_zero_active_reward": sum(r["success_any"] and r["active_reward_sum"] == 0 for r in episodes),
        "net_negative_active_decisions": sum(r["negative_decisions_active"] for r in episodes),
        "net_negative_masked_decisions": sum(r["negative_decisions_masked"] for r in episodes),
        "masked_decisions": sum(r["masked_decisions"] for r in episodes),
        "resumed_first16_cases_repeat_initial16": sequences[0] == sequences[1][: len(sequences[0])],
        "scope": "Actual saved target diagnostics; no new training or causal performance attribution. Toy native-target examples use zero value placeholders, not real training returns.",
    }
    (args.output / "audit.json").write_text(
        json.dumps({"summary": summary, "episodes": episodes, "native_semantic_probes": probes}, indent=2) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
