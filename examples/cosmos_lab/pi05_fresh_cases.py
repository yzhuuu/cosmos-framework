"""Freeze 50 new cases excluded from realized PPO training and the original diagnostic."""

import argparse
import copy
import datetime
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", type=Path)
    parser.add_argument("training_run", type=Path)
    args = parser.parse_args()
    acceptance_path = args.training_run / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance["passed"] and acceptance["summary"]["optimizer_step"] == 8
    used = set(map(tuple, acceptance["training_cases"]))
    original = {(task, state) for task in range(10) for state in [1, 2]}
    rng = random.Random(20261106)
    selected = []
    for task in range(10):
        eligible = [state for state in range(50) if (task, state) not in used | original]
        selected.extend((task, state) for state in sorted(rng.sample(eligible, 5)))
    assert len(set(selected)) == 50 and not (set(selected) & (used | original))
    assert set(Counter(task for task, _ in selected).values()) == {5}
    folder = args.lab / "runs/r8-fresh-preflight"
    folder.mkdir(exist_ok=False)
    identity = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sampling_seed": 20261106,
        "python": platform.python_version(),
        "algorithm": "Per task0..9 sample5 sorted eligible state0..49; exclude observed PPO cases and original state1/2; sort selected per task",
        "training_acceptance_sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
        "excluded_training": sorted(used),
        "excluded_previous_eval": sorted(original),
        "selected": selected,
        "scope": "No direct exposure in this PPO run; same shared500 and task set, pretrained SFT exposure unknown",
    }
    (folder / "selection.json").write_text(json.dumps(identity, indent=2) + "\n")
    protocol = json.loads((args.lab / "runs/r6-pi05-preflight/baseline-v2-protocol.json").read_text())
    protocol.update(
        created_utc=identity["created_utc"], scope=identity["scope"], selection=str(folder / "selection.json")
    )
    protocol["trials"] = [
        {"task_id": task, "init_state_id": state, "case_id": f"task-{task}-state{state}", "mode": mode}
        for task, state in selected
        for mode in ["zero", "policy"]
    ]
    repeat = dict(protocol["trials"][1], mode="policy_repeat")
    protocol["trials"].append(repeat)
    protocol["repeat_case"] = repeat["case_id"]
    protocol["checkpoint"] = None
    (folder / "baseline-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    post = copy.deepcopy(protocol)
    summaries = list((args.training_run / "output").rglob("summary-step-8.json"))
    assert len(summaries) == 1
    checkpoint = summaries[0].parent / "checkpoints/iter_000008"
    assert (checkpoint / "trainer.pt").is_file()
    post["checkpoint"] = str(checkpoint.resolve())
    (folder / "step8-protocol.json").write_text(json.dumps(post, indent=2) + "\n")
    print(json.dumps({"selected": selected, "trials_per_model": len(post["trials"]), "direct_ppo_overlap": 0}))


if __name__ == "__main__":
    main()
