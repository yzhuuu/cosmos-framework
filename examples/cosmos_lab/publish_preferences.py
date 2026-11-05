"""Publish candidate review media and explicitly separate screening from labels."""

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
    candidate = json.loads((args.run / "candidate-manifest.json").read_text())
    screening = json.loads((review / "screening.json").read_text())
    table = wandb.Table(columns=["episode", "split", "decision", "training_eligible", "comparison", "screening_notes"])
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        group="preference-candidates",
        name=f"yuzh-preference-review-{args.run.name.rsplit('-', 1)[-1]}",
        dir=str(review),
        config={
            "selection": candidate["selection"],
            "seeds": candidate["seeds"],
            "checkpoint": candidate["checkpoint"],
            "numerical_policy": manifest["numerical_policy"],
            "scope": "Train-only I2V candidates. Agent sampled-frame screening, not final human preferences.",
            "bridge_manifest_sha256": candidate["bridge_manifest_sha256"],
        },
    ) as run:
        for pair in manifest["pairs"]:
            note = screening["pairs"][pair["pair_id"]]
            media = wandb.Video(pair["comparison"], format="mp4", caption="Source | A | B; no winner assigned")
            table.add_data(pair["pair_id"], "train", note["decision"], False, media, json.dumps(note))
            run.log({f"candidates/{pair['pair_id']}": media})
        run.log({"review/candidates": table})
        run.summary["data/candidate_pairs"] = len(manifest["pairs"])
        run.summary["data/training_eligible_pairs"] = 0
        artifact = wandb.Artifact(f"yuzh-preference-review-{args.run.name.rsplit('-', 1)[-1]}", type="dataset")
        for path in [
            args.run / "candidate-manifest.json",
            args.run / "requests.jsonl",
            args.run / "output-validation.json",
            review / "manifest.json",
            review / "screening.json",
        ]:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        url = run.url
    (review / "wandb-url.txt").write_text(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
