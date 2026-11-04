"""Copy the recorded Nano requests, changing only Super's recommended sampler shift."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nano_run", type=Path)
    args = parser.parse_args()
    source = args.nano_run.resolve() / "requests.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert {row["model_mode"] for row in rows} == {"text2video", "image2video"}
    for row in rows:
        row["shift"] = 5.0
        if row.get("vision_path"):
            assert Path(row["vision_path"]).is_absolute() and Path(row["vision_path"]).is_file()
    directory = Path(os.environ["COSMOS_LAB_ROOT"]) / "data/baseline-super-robot-relocation"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "requests.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target": str(target),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "request_changes": {"shift": {"nano": 10.0, "super": 5.0}},
        "scope": "matched inputs with model-recommended sampling; not a model-only ablation",
    }
    (directory / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
