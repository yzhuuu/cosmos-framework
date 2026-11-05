"""Freeze text-only interventions on all O9 validation inputs and both seeds."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from examples.cosmos_lab.prepare_preferences import sha256

STATIONARY = {
    "episode_032918_clip000": "The robot arm stays stationary above the stove for the entire video. The red rectangular object stays at the front right of the stove. The metal pot stays on the right burner. All objects keep their original positions and appearances. The camera stays fixed.",
    "episode_005206_clip000": "The robot arm stays stationary above the stove for the entire video. The blue object stays at the front left of the stove. The metal pot stays on the right burner. All objects keep their original positions and appearances. The camera stays fixed.",
    "episode_044188_clip000": "The robot arm and gripper stay stationary at the microwave handle for the entire video. The microwave door stays fully closed. All objects keep their original positions and appearances. The camera stays fixed.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    old_manifest = json.loads((args.previous_run / "candidate-manifest.json").read_text())
    if old_manifest["split"] != "val" or {p["pair_id"] for p in old_manifest["pairs"]} != set(STATIONARY):
        raise ValueError("Expected the complete O9 validation cohort, including abstentions")
    old = list(map(json.loads, (args.previous_run / "requests.jsonl").read_text().splitlines()))
    args.destination.mkdir(exist_ok=False)
    rows, groups = [], []
    for request in old:
        episode = request["name"].rsplit("-", 1)[0]
        action, still = copy.deepcopy(request), copy.deepcopy(request)
        action["name"] += "-action"
        still["name"] += "-stationary"
        caption = json.loads(still["prompt"])
        caption["temporal_caption"] = STATIONARY[episode]
        caption["actions"] = [{"description": STATIONARY[episode]}]
        still["prompt"] = json.dumps(caption, ensure_ascii=False)
        if {k for k in action if action[k] != still[k]} != {"name", "prompt"}:
            raise ValueError("Intervention changed more than prompt/name")
        path = args.previous_run / "generated" / request["name"] / "vision.mp4"
        groups.append(
            dict(
                pair_id=request["name"],
                episode=episode,
                seed=request["seed"],
                action=action["name"],
                stationary=still["name"],
                previous_action_video=str(path.resolve()),
                previous_action_sha256=sha256(path),
                image_sha256=sha256(request["vision_path"]),
                action_goal=json.loads(action["prompt"])["temporal_caption"],
                stationary_goal=STATIONARY[episode],
            )
        )
        rows.extend([action, still])
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        split="val",
        checkpoint=old_manifest["checkpoint"],
        numerical_policy="fixed-cudnn-v1",
        seeds=[2026, 2027],
        previous_manifest_sha256=sha256(args.previous_run / "candidate-manifest.json"),
        intervention="Only prompt changes within each seed/image pair; action repeats must match O9 decoded pixels",
        groups=groups,
        budget=dict(gpus=2, walltime_minutes=30, videos=12),
        scope="Prompt adherence diagnostic, not DPO labels, a model comparison or robot success benchmark",
    )
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.destination / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(dict(groups=len(groups), videos=len(rows))))


if __name__ == "__main__":
    main()
