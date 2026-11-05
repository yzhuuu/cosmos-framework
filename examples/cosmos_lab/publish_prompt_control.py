"""Publish all text intervention results, including imperfect stationary behavior."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    review = args.run / "review"
    manifest = json.loads((review / "manifest.json").read_text())
    annotations = json.loads((review / "annotations.json").read_text())
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        group="prompt-control",
        name="yuzh-text-control-" + args.run.name.rsplit("-", 1)[-1],
        dir=str(review),
        config=json.loads((args.run / "candidate-manifest.json").read_text()),
    ) as run:
        table = wandb.Table(
            columns=["episode", "seed", "action_prompt", "stationary_prompt", "comparison", "agent_observation"]
        )
        metrics = wandb.Table(columns=["episode", "seed", "condition", "adjacent_frame_mae", "first_frame_mae"])
        for row in manifest["pairs"]:
            note = annotations["pairs"][row["pair_id"]]
            video = wandb.Video(
                row["comparison"],
                format="mp4",
                caption="Left: action; right: stationary. Same model, image and seed. Different text; no training.",
            )
            run.log({"text_control/" + row["pair_id"]: video})
            table.add_data(
                row["episode"], row["seed"], row["action_goal"], row["stationary_goal"], video, json.dumps(note)
            )
            for condition in ["action", "stationary"]:
                metrics.add_data(
                    row["episode"],
                    row["seed"],
                    condition,
                    row["metrics"][condition + "_mean_adjacent_mae"],
                    row["metrics"][condition + "_mean_first_frame_mae"],
                )
        run.log({"review/text_control": table, "diagnostic/motion": metrics})
        run.summary.update(
            {
                "conclusion": annotations["conclusion"],
                "exact_action_repeats": len(manifest["pairs"]),
                "reviewed_frames": sum(2 * p["frames"] for p in manifest["pairs"]),
                "training_launched": False,
            }
        )
        artifact = wandb.Artifact("yuzh-text-control-" + args.run.name.rsplit("-", 1)[-1], type="evaluation")
        for relative in [
            "candidate-manifest.json",
            "requests.jsonl",
            "output-validation.json",
            "code-commit.txt",
            "code.patch",
            "review/manifest.json",
            "review/annotations.json",
        ]:
            artifact.add_file(str(args.run / relative), name=relative)
        run.log_artifact(artifact).wait()
        url = run.url
    (review / "wandb-url.txt").write_text(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
