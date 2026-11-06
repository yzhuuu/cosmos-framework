"""Verify R10's sole objective change, realized case sequence and first-reach credit."""

import argparse
import collections
import copy
import json
from pathlib import Path

import torch

from examples.cosmos_lab.pi05_prefix_experiment import FirstReachPrefixReward


def run_output(run):
    paths = list((run / "output").rglob("windows.jsonl"))
    assert len(paths) == 1
    return paths[0].parent, [json.loads(line) for line in paths[0].read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    acceptance = json.loads((args.candidate / "acceptance.json").read_text())
    assert acceptance["passed"]
    candidate_output, candidate = run_output(args.candidate)
    reference_output, reference = run_output(args.reference)
    assert [w["optimizer_step"] for w in candidate] == [w["optimizer_step"] for w in reference]
    step = candidate[-1]["optimizer_step"]
    identities = []
    for output in [candidate_output, reference_output]:
        trainer = torch.load(
            output / "checkpoints" / f"iter_{step:06d}" / "trainer.pt", map_location="cpu", weights_only=False
        )
        identities.append(copy.deepcopy(trainer["objective_identity"]))
    revision = identities[0]["model_source"].pop("decision_reward_override")
    assert revision == FirstReachPrefixReward().identity()
    assert identities[0] == identities[1], "Objective identity differs beyond the intended reward override"
    episodes = []
    case_sequences = []
    for after, before in zip(candidate, reference, strict=True):
        assert after["learning_rates"] == before["learning_rates"]
        assert after["frozen_vlm_digest"] == before["frozen_vlm_digest"]

        # episode_id suffix is collector rank and local episode index; session UUID changes.
        def keyed_cases(window):
            return {e["episode_id"].split(":", 1)[1]: [e["task_id"], e["init_state_id"]] for e in window["episodes"]}

        cases = keyed_cases(after)
        assert len(cases) == 8 and cases == keyed_cases(before), "Realized per-rank cases differ"
        case_sequences.append({"optimizer_step": after["optimizer_step"], "cases_by_rank_episode": cases})
        groups = collections.defaultdict(list)
        for row in after["target_diagnostics"]:
            groups[row["sample_id"].rsplit(":", 1)[0]].append(row)
        for episode in after["episodes"]:
            rows = sorted(groups[episode["episode_id"]], key=lambda r: int(r["sample_id"].rsplit(":", 1)[1]))
            assert len(rows) == 48 and sum(r["executed_steps"] for r in rows) == 240
            assert episode["success"] == any(r["terminated"] for r in rows)
            first = next((i for i, r in enumerate(rows) if r["terminated"]), 47)
            assert [r["loss_mask"] for r in rows] == [i <= first for i in range(48)]
            active_reward = sum(r["reward"] for r in rows if r["loss_mask"])
            assert active_reward == int(episode["success"]), "First-reach credit missing or excessive"
            assert all(r["reward"] >= 0 for r in rows if r["loss_mask"])
            episodes.append(
                {
                    "step": after["optimizer_step"],
                    "episode_id": episode["episode_id"],
                    "task": episode["task_id"],
                    "state": episode["init_state_id"],
                    "success_any": episode["success"],
                    "active_reward": active_reward,
                    "masked_decisions": sum(not r["loss_mask"] for r in rows),
                }
            )
    receipt = {
        "passed": True,
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "reward_override": revision,
        "only_objective_identity_difference_is_reward_override": True,
        "realized_cases_match_per_window_rank_episode": True,
        "episodes_checked": len(episodes),
        "success_any": sum(e["success_any"] for e in episodes),
        "successful_with_zero_active_reward": 0,
        "case_sequences": case_sequences,
        "episodes": episodes,
        "scope": "Mechanism acceptance, not a quality claim. Transformed reward sums do not measure final physical success.",
    }
    (args.candidate / "prefix-acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["case_sequences", "episodes"]}))


if __name__ == "__main__":
    main()
