"""Publish complete fixed-input DPO generation comparisons and review limitations."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    assert (args.run / "end.txt").is_file()
    comparison = json.loads((args.run / "comparison.json").read_text())
    review = json.loads((args.run / "review/annotations.json").read_text())
    overlays = [json.loads((args.run / f"dpo/overlay-rank{r}.json").read_text()) for r in range(2)]
    assert len(comparison["records"]) == 4
    assert all(row["baseline_repeat_pixel_exact"] for row in comparison["records"])
    assert all(row["verified_local_values"] for row in overlays)
    changed = sum(t["changed_local_elements"] for r in overlays for t in r["tensors"] if t["trainable"])
    frozen_changed = sum(t["changed_local_elements"] for r in overlays for t in r["tensors"] if not t["trainable"])
    assert frozen_changed == 0
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        name=args.run.name.replace("dpo-compare", "yuzh-dpo-compare"),
        group="bridge-i2v-dpo-evaluation",
        dir=str(args.run),
        config=dict(
            training_run="20261105190703",
            optimizer_steps=2,
            val_episodes=2,
            seeds=[2026, 2027],
            num_steps=35,
            guidance=6,
            shift=10,
            review_scope=review["coverage"],
        ),
        settings=wandb.Settings(disable_code=True),
    ) as run:
        run.summary.update(
            {
                "audit/baseline_repeats_exact": 4,
                "audit/changed_trainable_elements": changed,
                "audit/changed_frozen_elements": frozen_changed,
                "review/conclusion": review["conclusion"],
            }
        )
        artifact = wandb.Artifact(args.run.name + "-evidence", type="evaluation")
        for path in [
            "comparison.json",
            "baseline-repeat.json",
            "requests.jsonl",
            "code-commit.txt",
            "dpo/overlay-rank0.json",
            "dpo/overlay-rank1.json",
            "review/annotations.json",
        ]:
            artifact.add_file(str(args.run / path), name=path)
        run.log_artifact(artifact)
        for row in comparison["records"]:
            name = row["name"]
            run.log(
                {
                    f"comparison/{name}": wandb.Video(
                        row["comparison"],
                        format="mp4",
                        caption=f"Left: original Nano; right: DPO step 2. Same input, seed {row['seed']}.",
                    ),
                    f"diagnostic/{name}/pixel_mae": row["mean_pixel_abs_change"],
                    f"diagnostic/{name}/differing_frames": row["differing_frames"],
                }
            )
        url = run.url
    (args.run / "wandb-url.txt").write_text(url + "\n")
    training = wandb.Api().run("sav-inception-self/cosmos-lab/20261105190703")
    training.summary["evaluation/generation_run_url"] = url
    training.summary.update()
    print(url)


if __name__ == "__main__":
    main()
