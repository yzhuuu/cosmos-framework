"""Export accepted, fully reviewed candidates without changing their conditions."""

import argparse
import json
from pathlib import Path

from examples.cosmos_lab.prepare_preferences import condition_digest, sha256, validate_effective, validate_pair


def annotation_eligible(annotation, purpose):
    if purpose not in {"train", "validation"}:
        raise ValueError("Unknown annotation purpose")
    if purpose == "validation" and annotation.get("training_eligible", False):
        raise ValueError("Validation cannot accept training-eligible annotations")
    if purpose == "train" and annotation.get("validation_eligible", False):
        raise ValueError("Training cannot accept validation annotations")
    field = "training_eligible" if purpose == "train" else "validation_eligible"
    if not annotation.get(field, False):
        return False
    if (
        annotation["decision"] not in {"a", "b"}
        or annotation["confidence"] not in {"medium", "high"}
        or not annotation["all_frames_reviewed"]
        or not annotation["evidence"]
        or annotation["reviewer_type"] != "agent"
    ):
        raise ValueError("Insufficient annotation evidence")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--purpose", choices=["train", "validation"], default="train")
    args = parser.parse_args()
    frozen = json.loads((args.run / "candidate-manifest.json").read_text())
    split = "train" if args.purpose == "train" else "val"
    if frozen.get("split", "train") != split:
        raise ValueError("Requested purpose does not match frozen candidate split")
    requests = {r["name"]: r for r in map(json.loads, (args.run / "requests.jsonl").read_text().splitlines())}
    review = json.loads((args.run / "review/screening.json").read_text())
    media = {row["pair_id"]: row for row in json.loads((args.run / "review/manifest.json").read_text())["pairs"]}
    records = []
    for pair in frozen["pairs"]:
        annotation = review["pairs"][pair["pair_id"]]
        if not annotation_eligible(annotation, args.purpose):
            continue
        sides = [requests[name] for name in pair["candidates"]]
        validate_pair(*sides, pair["source"], expected_split=split)
        if condition_digest(sides[0]) != pair["condition_sha256"]:
            raise ValueError("Condition changed since generation freeze")
        if sha256(pair["source"]["video"]) != pair["source"]["sha256"]:
            raise ValueError("Source changed")
        for side in sides:
            actual = json.loads((args.run / "generated" / side["name"] / "sample_args.json").read_text())
            validate_effective(side, actual)
        winner = 0 if annotation["decision"] == "a" else 1
        row = dict(
            pair_id=pair["pair_id"],
            episode=pair["source"]["episode"],
            source_split=pair["source"]["split"],
            source_sha256=pair["source"]["sha256"],
            condition_image=sides[0]["vision_path"],
            condition_image_sha256=sha256(sides[0]["vision_path"]),
            condition_sha256=pair["condition_sha256"],
            prompt=sides[0]["prompt"],
            frames=sides[0]["num_frames"],
            fps=sides[0]["fps"],
            annotation=annotation,
        )
        for key, index in [("winner", winner), ("loser", 1 - winner)]:
            path = args.run / "generated" / sides[index]["name"] / "vision.mp4"
            row[key + "_video"] = str(path.resolve())
            row[key + "_sha256"] = sha256(path)
            if row[key + "_sha256"] != media[pair["pair_id"]]["video_sha256"][index + 1]:
                raise ValueError("Candidate changed since review media was built")
        if row["winner_sha256"] == row["loser_sha256"]:
            raise ValueError("Identical candidate videos")
        records.append(row)
    if not records:
        raise ValueError("No accepted pairs; do not create a training file")
    prior = list(map(json.loads, args.previous.read_text().splitlines())) if args.previous else []
    for row in prior:
        if row["source_split"] != split or not annotation_eligible(row["annotation"], args.purpose):
            raise ValueError("Previous data has wrong split or ineligible annotations")
    combined = prior + records
    if len({row["episode"] for row in combined}) != len(combined):
        raise ValueError("Duplicate source episode")
    args.destination.mkdir(exist_ok=False)
    for name, rows in [(f"new-{split}.jsonl", records), (f"{split}.jsonl", combined)]:
        (args.destination / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = dict(
        new_pairs=len(records),
        combined_pairs=len(combined),
        purpose=args.purpose,
        source_run=str(args.run.resolve()),
        review_sha256=sha256(args.run / "review/screening.json"),
        candidate_manifest_sha256=sha256(args.run / "candidate-manifest.json"),
        previous_path=str(args.previous) if args.previous else None,
        previous_sha256=sha256(args.previous) if args.previous else None,
        scope="Agent learning labels; not independently validated quality or robot success labels",
    )
    (args.destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
