"""Prepare pinned, paired public AgiBot examples for an integration study."""

import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(os.environ["COSMOS_LAB_ROOT"])
    assets = root / "data/Cosmos3-Nano-Sim-Bimanual"
    checkpoint = root / "checkpoints/Cosmos3-Nano-Sim-Bimanual"
    meta = json.loads((checkpoint / "checkpoint.json").read_text())["cosmos3_nano_sim_bimanual"]
    assert meta["conditioning"]["schema_version"] == 5
    assert meta["conditioning"]["input_contract"]["action_dim"] == 59
    assert meta["fixed_step_sampler_config"]["t_list"] == [1.0, 0.9375, 0.8333333333333334, 0.625]
    source = assets / "examples/training/sim_bimanual_agibot_examples.jsonl"
    directory = root / "data/bimanual-baseline"
    directory.mkdir(parents=True, exist_ok=True)
    requests, pairs = [], []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        gt = assets / row.pop("ground_truth_path")
        row.pop("purpose")
        row["image_size"] = 480
        paths = {"gt": gt}
        for key in ("vision_path", "action_path"):
            paths[key] = assets / row[key]
            row[key] = str(paths[key])
        assert all(p.is_file() for p in paths.values())
        requests.append(row)
        pairs.append(
            {"name": row["name"], "files": {k: {"path": str(p), "sha256": digest(p)} for k, p in paths.items()}}
        )
    (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (directory / "pairs.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": digest(source),
                "revision": "a9e3adf6235f56f0082be7a96e497348c286c472",
                "scope": "Three public paired examples; integration study, not held-out evaluation",
                "gt_alignment": "Crop bottom 8 pixels of 640x360 GT to native 640x352 output; no time resampling",
                "pairs": pairs,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(requests)} paired requests at {directory}")


if __name__ == "__main__":
    main()
