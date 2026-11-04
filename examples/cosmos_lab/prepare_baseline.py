"""Build matched T2V/I2V requests from the repository's robot relocation example."""

import hashlib
import json
import os
from pathlib import Path

import requests


def main() -> None:
    """Save explicit sampling settings and retain the pinned source image provenance."""
    root = Path(os.environ["COSMOS_LAB_ROOT"])
    repo = Path(__file__).resolve().parents[2]
    source = repo / "inputs/omni/i2v.json"
    row = json.loads(source.read_text())
    directory = root / "data/baseline-robot-relocation"
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "initial.jpg"
    response = requests.get(row["vision_path"], timeout=120)
    response.raise_for_status()
    image_path.write_bytes(response.content)
    prompt = json.loads(row["prompt"])
    prompt.update({"resolution": {"W": 832, "H": 480}, "duration": "0:08", "fps": 24})
    shared = {
        "prompt": json.dumps(prompt, ensure_ascii=False),
        "seed": 2026,
        "resolution": "480",
        "aspect_ratio": "16,9",
        "fps": 24,
        "num_frames": 193,
        "num_steps": 35,
        "guidance": 6.0,
        "shift": 10.0,
        "enable_sound": False,
        "prompt_upsampling": False,
    }
    rows = [
        {**shared, "name": "robot_relocation_t2v", "model_mode": "text2video"},
        {**shared, "name": "robot_relocation_i2v", "model_mode": "image2video", "vision_path": str(image_path)},
    ]
    manifest = directory / "requests.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in rows))
    receipt = {
        "source_request": str(source),
        "source_image": row["vision_path"],
        "image_sha256": hashlib.sha256(response.content).hexdigest(),
        "request_manifest": str(manifest),
        "request_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "purpose": "integration baseline, not held-out generalization evaluation",
        "prompt_changes": "resolution, duration and FPS metadata only; original robot action description retained",
    }
    (directory / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
