"""Freeze two validation episodes before generating independent candidates."""

import json
from datetime import datetime, timezone
from pathlib import Path

import av

from examples.cosmos_lab.prepare_preferences import condition_digest, sha256, validate_pair


def main():
    lab = Path("/mnt/patient-unit/home/yan/cosmos-lab")
    study = lab / "data/bridge-study-v1"
    source = json.loads((study / "manifest.json").read_text())
    rows = {r["name"]: r for r in map(json.loads, (study / "val-requests.jsonl").read_text().splitlines())}
    selected = [r for r in source["records"] if r["split"] == "val"][:2]
    output = lab / "data/bridge-preference-validation-v1"
    output.mkdir(exist_ok=False)
    requests, pairs = [], []
    for record in selected:
        if sha256(record["video"]) != record["sha256"]:
            raise ValueError("Source changed")
        image = output / f"{record['uuid']}.png"
        with av.open(record["video"]) as container:
            next(container.decode(video=0)).to_image().save(image)
        candidates = [
            dict(rows[record["uuid"]], name=f"{record['uuid']}-{label}", seed=seed, vision_path=str(image))
            for label, seed in [("a", 2026), ("b", 2027)]
        ]
        validate_pair(*candidates, record, expected_split="val")
        requests.extend(candidates)
        pairs.append(
            dict(
                pair_id=record["uuid"],
                source=record,
                candidates=[r["name"] for r in candidates],
                condition_sha256=condition_digest(candidates[0]),
                image_sha256=sha256(image),
                actions=json.loads(candidates[0]["prompt"]).get("actions", []),
                decision="unreviewed",
                training_eligible=False,
            )
        )
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        split="val",
        selection="First two frozen Bridge val episodes, selected before generation; never train",
        bridge_manifest_sha256=sha256(study / "manifest.json"),
        pairs=pairs,
        seeds=[2026, 2027],
        checkpoint=str(lab / "checkpoints/Cosmos3-Nano"),
        numerical_policy="fixed-cudnn-v1",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))


if __name__ == "__main__":
    main()
