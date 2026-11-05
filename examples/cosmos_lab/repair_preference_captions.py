"""Version source-audited captions before regenerating both candidate sides."""

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from examples.cosmos_lab.prepare_preferences import condition_digest, sha256, validate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    audit = json.loads((args.source / "source-caption-audit.json").read_text())
    requests = {r["name"]: r for r in map(json.loads, (args.source / "requests.jsonl").read_text().splitlines())}
    args.destination.mkdir(exist_ok=False)
    updated, revisions = [], []
    for pair in manifest["pairs"]:
        note = audit["pairs"][pair["pair_id"]]
        if sha256(pair["source"]["video"]) != pair["source"]["sha256"]:
            raise ValueError("Source changed since caption audit")
        sides = [copy.deepcopy(requests[name]) for name in pair["candidates"]]
        original = sides[0]["prompt"]
        caption = {
            "temporal_caption": note["observed"],
            "actions": [{"description": note["observed"]}],
            "cinematography": {"camera_motion": "Static"},
            "style_medium": "Live-action video",
        }
        image = args.destination / Path(sides[0]["vision_path"]).name
        shutil.copyfile(sides[0]["vision_path"], image)
        for side in sides:
            side["prompt"] = json.dumps(caption, ensure_ascii=False)
            side["vision_path"] = str(image.resolve())
        validate_pair(*sides, pair["source"])
        if sha256(image) != pair["image_sha256"]:
            raise ValueError("Condition image changed")
        revisions.append(
            dict(
                pair_id=pair["pair_id"],
                original_prompt=original,
                revised_prompt=sides[0]["prompt"],
                reason=note,
                reviewer_type=audit["reviewer_type"],
                original_condition_sha256=pair["condition_sha256"],
                revised_condition_sha256=condition_digest(sides[0]),
            )
        )
        pair.update(
            condition_sha256=condition_digest(sides[0]),
            actions=caption["actions"],
            review_goal=note["observed"],
            review_questions=[
                "Does the intended object move after visible tool/gripper contact?",
                "Is the instructed action sequence and final placement visible?",
                "Do tools or objects duplicate, merge or disappear outside occlusion?",
            ],
            decision="unreviewed",
            training_eligible=False,
        )
        updated.extend(sides)
    manifest.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifest=str((args.source / "manifest.json").resolve()),
        parent_manifest_sha256=sha256(args.source / "manifest.json"),
        source_caption_audit_sha256=sha256(args.source / "source-caption-audit.json"),
        caption_policy="Agent source-audited concise actions; new conditions, both sides must be regenerated",
    )
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.destination / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in updated))
    (args.destination / "caption-revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")
    shutil.copyfile(args.source / "source-caption-audit.json", args.destination / "source-caption-audit.json")
    print(json.dumps(dict(pairs=len(manifest["pairs"]), destination=str(args.destination))))


if __name__ == "__main__":
    main()
