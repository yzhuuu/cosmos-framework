"""Freeze train-only, same-condition I2V candidate pairs before generation."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import av


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def condition_digest(request):
    condition = {k: v for k, v in request.items() if k not in {"name", "seed", "vision_path"}}
    condition["image_sha256"] = sha256(request["vision_path"])
    return hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()


def validate_pair(left, right, source):
    if source["split"] != "train":
        raise ValueError("Preference training candidates must come from train episodes")
    if left["name"] == right["name"] or left["seed"] == right["seed"]:
        raise ValueError("Expected distinct candidate names and seeds")
    if condition_digest(left) != condition_digest(right):
        raise ValueError("Candidate conditions differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    study = args.lab / "data/bridge-study-v1"
    manifest = json.loads((study / "manifest.json").read_text())
    rows = {
        row["uuid"]: row for row in map(json.loads, (study / "train/video_dataset_file.jsonl").read_text().splitlines())
    }
    selected = [row for row in manifest["records"] if row["split"] == "train"][:4]
    held_out = {r["episode"] for r in manifest["records"] if r["split"] != "train"}
    if any(row["episode"] in held_out for row in selected):
        raise ValueError("Episode leakage")
    args.destination.mkdir(parents=True, exist_ok=False)
    requests, pairs = [], []
    for source in selected:
        video = Path(source["video"])
        if sha256(video) != source["sha256"]:
            raise ValueError(f"Source video changed: {video}")
        image = args.destination / f"{source['uuid']}-first.png"
        with av.open(str(video)) as container:
            next(container.decode(video=0)).to_image().save(image)
        caption = rows[source["uuid"]]["t2w_windows"][0]["caption_json"]
        count = (source["frames"] - 1) // 4 * 4 + 1
        candidates = []
        for label, seed in [("a", 2026), ("b", 2027)]:
            request = {
                "name": f"{source['uuid']}-{label}",
                "prompt": json.dumps(caption, ensure_ascii=False),
                "vision_path": str(image.resolve()),
                "seed": seed,
                "resolution": "256",
                "aspect_ratio": "1,1",
                "num_frames": count,
                "fps": source["fps"],
                "num_steps": 35,
                "guidance": 6.0,
                "shift": 10.0,
                "enable_sound": False,
                "prompt_upsampling": False,
            }
            candidates.append(request)
        validate_pair(*candidates, source)
        requests.extend(candidates)
        pairs.append(
            {
                "pair_id": source["uuid"],
                "source": source,
                "candidates": [r["name"] for r in candidates],
                "condition_sha256": condition_digest(candidates[0]),
                "image_sha256": sha256(image),
                "actions": caption.get("actions", []),
                "decision": "unreviewed",
                "training_eligible": False,
            }
        )
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "First four train records in the already frozen Bridge manifest; no output-based selection",
        "bridge_manifest_sha256": sha256(study / "manifest.json"),
        "checkpoint": str(args.lab / "checkpoints/Cosmos3-Nano"),
        "numerical_policy": "fixed-cudnn-v1",
        "seeds": [2026, 2027],
        "scope": "I2V preference candidates, not labeled training data. Native 5 FPS timing retained.",
        "pairs": pairs,
    }
    (args.destination / "manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (args.destination / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    print(json.dumps({"pairs": len(pairs), "requests": len(requests), "destination": str(args.destination)}))


if __name__ == "__main__":
    main()
