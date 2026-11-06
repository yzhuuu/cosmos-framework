"""Attach completed PPO mechanism receipts to its existing native W&B run."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    assert (args.run / "end.txt").is_file(), "Training launcher has not completed"
    receipt = json.loads((args.run / "acceptance.json").read_text())
    assert receipt["passed"]
    outputs = list((args.run / "output").rglob("config_resolved.yaml"))
    assert len(outputs) == 1
    output = outputs[0].parent
    summary = {
        "acceptance/mechanism_passed": True,
        "acceptance/quality_improvement_claimed": False,
        "acceptance/optimizer_step": receipt["summary"]["optimizer_step"],
        "acceptance/episodes_new": len(receipt["new_training_cases"]),
        "acceptance/control_steps_new": receipt["executed_control_steps_new"],
        "acceptance/simulator_rng_restored": False,
    }
    prefix = args.run / "prefix-acceptance.json"
    if prefix.exists():
        report = json.loads(prefix.read_text())
        assert report["passed"]
        summary.update(
            {
                "acceptance/reward_revision": report["reward_override"]["revision"],
                "acceptance/case_sequence_matched": report["realized_cases_match_per_window_rank_episode"],
                "acceptance/successful_zero_active_reward": report["successful_with_zero_active_reward"],
            }
        )
    paths = [
        args.run / n
        for n in [
            "acceptance.json",
            "protocol.json",
            "code-commit.txt",
            "code.patch",
            "packages.txt",
            "start.txt",
            "end.txt",
        ]
    ]
    paths += [output / "config_resolved.yaml", output / "windows.jsonl"]
    paths += sorted((args.run / "audit").glob("*.json"))
    if prefix.exists():
        paths.append(prefix)
    with wandb.init(
        entity="sav-inception-self", project="cosmos-lab", id=output.name, resume="must", dir=str(args.run)
    ) as run:
        run.summary.update(summary)
        artifact = wandb.Artifact("yuzh-ppo-acceptance-" + args.run.name.rsplit("-", 1)[-1], type="training-audit")
        for path in paths:
            artifact.add_file(str(path), name=str(path.relative_to(args.run)))
        run.log_artifact(artifact).wait()
        url = run.url
    (args.run / "acceptance-wandb-url.txt").write_text(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
