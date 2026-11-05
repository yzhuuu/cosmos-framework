"""Freeze a bounded CFG diagnostic on the O10 blue-object stationary failure."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from examples.cosmos_lab.prepare_preferences import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prior = json.loads((args.previous_run / "candidate-manifest.json").read_text())
    requests = {r["name"]: r for r in map(json.loads, (args.previous_run / "requests.jsonl").read_text().splitlines())}
    selected = [g for g in prior["groups"] if g["episode"] == "episode_005206_clip000"]
    if len(selected) != 2 or {g["seed"] for g in selected} != {2026, 2027}:
        raise ValueError("Expected both O10 seeds")
    args.destination.mkdir(exist_ok=False)
    rows, groups = [], []
    for original in selected:
        for guidance in [3.0, 6.0, 7.0]:
            group = copy.deepcopy(original)
            group["pair_id"] += f"-g{guidance:g}"
            group["guidance"] = guidance
            group["require_exact_repeat"] = guidance == 6.0
            for condition in ["action", "stationary"]:
                request = copy.deepcopy(requests[original[condition]])
                request["name"] += f"-g{guidance:g}"
                request["guidance"] = guidance
                if any(
                    request[k] != requests[original[condition]][k] for k in request if k not in {"name", "guidance"}
                ):
                    raise ValueError("Unexpected intervention")
                path = args.previous_run / "generated" / original[condition] / "vision.mp4"
                group[f"previous_{condition}_video"] = str(path.resolve())
                group[f"previous_{condition}_sha256"] = sha256(path)
                group[condition] = request["name"]
                rows.append(request)
            groups.append(group)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        study_kind="guidance-control",
        split="val",
        checkpoint=prior["checkpoint"],
        numerical_policy=prior["numerical_policy"],
        seeds=prior["seeds"],
        selection="Both seeds of the O10 blue-object stationary failure; diagnostic selection after O10, not a fresh held-out performance estimate",
        intervention="Within level only text changes; across levels only guidance 3/6/7 changes. All four g6 outputs must exactly repeat O10.",
        groups=groups,
        budget=dict(gpus=2, walltime_minutes=30, videos=12),
        scope="Exploratory CFG diagnosis, not training or evidence of global optimum",
    )
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.destination / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(dict(groups=len(groups), videos=len(rows))))


if __name__ == "__main__":
    main()
