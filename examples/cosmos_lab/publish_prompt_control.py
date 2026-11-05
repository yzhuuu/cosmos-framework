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
    frozen = json.loads((args.run / "candidate-manifest.json").read_text())
    kind = frozen.get("study_kind", "text-control")
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        group=kind,
        name="yuzh-" + kind + "-" + args.run.name.rsplit("-", 1)[-1],
        dir=str(review),
        config=frozen,
    ) as run:
        table = wandb.Table(
            columns=[
                "episode",
                "seed",
                "guidance",
                "action_prompt",
                "stationary_prompt",
                "comparison",
                "agent_observation",
            ]
        )
        metrics = wandb.Table(
            columns=["episode", "seed", "guidance", "condition", "adjacent_frame_mae", "first_frame_mae"]
        )
        for row in manifest["pairs"]:
            note = annotations["pairs"][row["pair_id"]]
            video = wandb.Video(
                row["comparison"],
                format="mp4",
                caption=f"Left: action; right: stationary. Same model, image, seed. Guidance {row.get('guidance', 6):g}. No training.",
            )
            run.log({"text_control/" + row["pair_id"]: video})
            table.add_data(
                row["episode"],
                row["seed"],
                row.get("guidance", 6),
                row["action_goal"],
                row["stationary_goal"],
                video,
                json.dumps(note),
            )
            for condition in ["action", "stationary"]:
                metrics.add_data(
                    row["episode"],
                    row["seed"],
                    row.get("guidance", 6),
                    condition,
                    row["metrics"][condition + "_mean_adjacent_mae"],
                    row["metrics"][condition + "_mean_first_frame_mae"],
                )
        run.log({"review/text_control": table, "diagnostic/motion": metrics})
        run.summary.update(
            {
                "conclusion": annotations["conclusion"],
                "exact_action_repeats": sum(p["exact_action_repeat"] for p in manifest["pairs"]),
                "exact_stationary_repeats": sum(p.get("exact_stationary_repeat") is True for p in manifest["pairs"]),
                "reviewed_frames": sum(2 * p["frames"] for p in manifest["pairs"]),
                "training_launched": False,
            }
        )
        artifact = wandb.Artifact("yuzh-" + kind + "-" + args.run.name.rsplit("-", 1)[-1], type="evaluation")
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
