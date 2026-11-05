"""Attach audited I2V DPO provenance and labeled data videos to the completed run."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    lab = Path("/mnt/patient-unit/home/yan/cosmos-lab")
    assert (args.trace / "end.txt").is_file(), "Training process has not completed successfully"
    audit = json.loads((args.trace / "checkpoint-audit.json").read_text())
    run_id = args.run_dir.name
    if not (len(run_id) == 14 and run_id.isdigit()):
        run_id = (args.run_dir / "wandb_run_id.txt").read_text().strip()
    api_run = wandb.Api().run(f"sav-inception-self/cosmos-lab/{run_id}")
    history = list(api_run.scan_history())
    (args.trace / "wandb-history.json").write_text(json.dumps(history, indent=2) + "\n")
    assert max(r.get("optim/optimizer_step", 0) or 0 for r in history) == 2
    val = [r for r in history if r.get("val/pair_count") is not None]
    assert len(val) == 2 and all(r["val/pair_count"] == 1 for r in val)
    assert audit["iteration"] == 2
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        id=run_id,
        resume="must",
        dir=str(args.trace),
        settings=wandb.Settings(disable_code=True),
    ) as run:
        run.summary.update(
            {
                "audit/model_only_checkpoint_verified": True,
                "data/train_agent_pairs": 2,
                "data/val_agent_pairs": 1,
                "data/val_abstentions": 1,
                "audit/generated_quality_evaluated": False,
            }
        )
        artifact = wandb.Artifact(args.trace.name + "-evidence", type="evaluation")
        for name in ["checkpoint-audit.json", "wandb-history.json", "code-commit.txt", "train.jsonl", "val.jsonl"]:
            artifact.add_file(str(args.trace / name), name=name)
        artifact.add_file(str(args.run_dir / "config_resolved.yaml"), name="config_resolved.yaml")
        for job in [476, 479]:
            artifact.add_file(
                str(lab / f"runs/preference-candidates-{job}/review/temporal-review.json"), name=f"review-{job}.json"
            )
        run.log_artifact(artifact)
        media = {}
        for split, job in [("train", 476), ("val", 479)]:
            for row in map(json.loads, (args.trace / f"{split}.jsonl").read_text().splitlines()):
                video = lab / f"runs/preference-candidates-{job}/review/{row['pair_id']}.mp4"
                media[f"data/{split}/{row['pair_id']}"] = wandb.Video(
                    str(video),
                    format="mp4",
                    caption=f"Source | A | B. Frozen pre-training data, agent prefers {row['annotation']['decision'].upper()}; not a post-training generation.",
                )
        video = lab / "runs/preference-candidates-479/review/episode_018766_clip000.mp4"
        media["data/excluded/episode_018766_clip000"] = wandb.Video(
            str(video),
            format="mp4",
            caption="Source | A | B. Agent abstained; excluded from both training and validation loss.",
        )
        run.log(media)
        (args.trace / "wandb-url.txt").write_text(run.url + "\n")
    print(api_run.url)


if __name__ == "__main__":
    main()
