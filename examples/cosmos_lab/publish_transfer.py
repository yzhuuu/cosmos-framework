# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Publish completed fixed-control appearance comparisons to the existing study project."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.runs:
        assert (path / "output-validation.json").is_file()
        assert (path / "comparison.mp4").is_file()
    run = wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        job_type="evaluation",
        name="transfer-fixed-control-black-red",
        config={
            "checkpoint_revision": "42da1b57840cec0fd2b642546b72cb95fdc331af",
            "seed": 2026,
            "steps": 4,
            "frames": 121,
            "fps": 30,
            "scope": "Official example2 fixed-control appearance comparison; qualitative, no success-rate claim.",
            "inputs": [str(path.resolve()) for path in args.runs],
        },
    )
    table = wandb.Table(columns=["modality", "name", "prompt", "video"])
    try:
        for path in args.runs:
            rows = [json.loads(x) for x in (path / "requests.jsonl").read_text().splitlines()]
            modality = "depth" if "depth" in rows[0] else "seg"
            run.log(
                {
                    f"comparison/{modality}": wandb.Video(str(path / "comparison.mp4"), format="mp4"),
                    f"contact_sheet/{modality}": wandb.Image(str(path / "comparison.jpg")),
                }
            )
            for row in rows:
                video = path / "generated" / row["name"] / "vision.mp4"
                table.add_data(modality, row["name"], row["prompt"], wandb.Video(str(video), format="mp4"))
        run.log({"samples": table})
        url = run.url
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    for path in args.runs:
        (path / "wandb-url.txt").write_text(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
