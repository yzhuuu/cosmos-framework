"""Publish fixed-input comparison videos and diagnostic scores to a separate W&B run."""

import argparse
import json
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--training-runs", nargs="*", default=[])
    args = parser.parse_args()
    manifest = json.loads((args.comparison / "manifest.json").read_text())
    runs = [(label, Path(path)) for label, path in (item.split("=", 1) for item in manifest["runs"])]
    scores = {}
    for label, path in runs:
        rows = json.loads((path / "paired-evaluation.json").read_text())["results"]
        scores[label] = {r["name"]: r for r in rows}
    labels = [label for label, _ in runs]
    table = wandb.Table(
        columns=["episode", "split", "static_PSNR_dB"] + [f"{label}_PSNR_dB" for label in labels] + ["comparison"]
    )
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        group="bridge-fixed-evaluation",
        name=args.name,
        dir=str(args.comparison),
        config={
            "training_runs": args.training_runs,
            "source_runs": manifest["runs"],
            "scope": "Fixed-seed I2V video comparison. PSNR is a diagnostic, not robot task success or human preference.",
            "video_columns": manifest["columns"],
        },
    ) as run:
        for output in manifest["outputs"]:
            name = output["name"]
            first = scores[labels[0]][name]
            video = wandb.Video(output["path"], format="mp4", caption=" | ".join(manifest["columns"]))
            table.add_data(
                name,
                first["split"],
                first["static_psnr_db"],
                *[scores[label][name]["prediction_psnr_db"] for label in labels],
                video,
            )
            run.log({f"comparison/{name}": video})
        run.log({"diagnostics/paired_videos": table})
        for label in labels:
            values = [row["prediction_psnr_db"] for row in scores[label].values()]
            run.summary[f"diagnostics/{label}_mean_PSNR_dB"] = sum(values) / len(values)
        url = run.url
    (args.comparison / "wandb-url.txt").write_text(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
