"""Audit completed native PPO windows, full checkpoints and optional resume attestations."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--parent-acceptance", type=Path)
    args = parser.parse_args()
    outputs = list((args.run / "output").rglob(f"summary-step-{args.end}.json"))
    assert len(outputs) == 1
    output = outputs[0].parent
    summary = json.loads(outputs[0].read_text())
    assert summary["optimizer_step"] == summary["publication_version"] == args.end
    assert summary["stop_reason"] is None
    windows = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in windows] == list(range(args.start + 1, args.end + 1))
    frozen = windows[0]["frozen_vlm_digest"]
    cases, metrics = [], []
    for row in windows:
        assert row["behavior_verified"] and row["max_logprob_delta"] <= 1e-5
        assert row["publication_version"] == row["optimizer_step"]
        assert row["metrics"]["frozen_vlm_verified"] == 1 and row["frozen_vlm_digest"] == frozen
        assert all(math.isfinite(x) for x in row["metrics"].values() if isinstance(x, (float, int)))
        assert all(x > 0 and math.isfinite(x) for x in row["parameter_update_l2"].values())
        assert len(row["episodes"]) == 8
        cases.extend([[r["task_id"], r["init_state_id"]] for r in row["episodes"]])
        metrics.append(
            {
                key: row[key]
                for key in [
                    "optimizer_step",
                    "publication_version",
                    "max_logprob_delta",
                    "metrics",
                    "parameter_update_l2",
                    "seconds",
                ]
            }
        )
    checkpoint = output / "checkpoints" / f"iter_{args.end:06d}"
    trainer = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False)
    assert trainer["optimizer_step"] == trainer["publication_version"] == args.end
    assert trainer["nonfinite_skips_total"] == trainer["nonfinite_skips_consecutive"] == 0
    assert trainer["objective_identity"]["model"]["vision_sdpa"] == "flash-efficient-math-no-cudnn-v1"
    data = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)["collectors"]
    assert set(data) == {str(i) for i in range(4)}
    assert all(row["next_window"] == args.end for row in data.values())
    counts = {
        name: len(FileSystemReader(checkpoint / name).read_metadata().state_dict_metadata)
        for name in ["model", "optim"]
    }
    assert counts["model"] == 216 and counts["optim"] > counts["model"]
    resumed = []
    if args.start:
        assert summary["restored_from"] and Path(summary["restored_from"]).name == f"iter_{args.start:06d}"
        for rank in range(4):
            report = json.loads((args.run / "audit" / f"resume-rank{rank}.json").read_text())
            assert report["rng_exact"]
            assert report["progress"] == {"optimizer_step": args.start, "publication_version": args.start}
            assert report["optimizer_steps"] == [float(args.start)]
            collector = json.loads((args.run / "audit" / f"collector-resume-{rank}.json").read_text())
            assert collector["saved"] == collector["restored"]
            assert collector["restored"]["next_window"] == args.start
            resumed.append({"learner": report, "collector": collector})
    prior = json.loads(args.parent_acceptance.read_text()) if args.parent_acceptance else None
    if prior:
        assert prior["passed"] and prior["frozen_vlm_digest"] == frozen
    all_cases = (prior["training_cases"] if prior else []) + cases
    evaluation = {(task, state) for task in range(10) for state in [1, 2]}
    receipt = {
        "passed": True,
        "scope": "Mechanism and saved-state acceptance; fixed-case quality is reported separately",
        "summary": summary,
        "windows": metrics,
        "checkpoint_counts": counts,
        "collector_next_windows": {key: row["next_window"] for key, row in data.items()},
        "resume_attestations": resumed,
        "frozen_vlm_digest": frozen,
        "new_training_cases": cases,
        "training_cases": all_cases,
        "direct_overlap_with_fixed_eval": sorted(set(map(tuple, all_cases)) & evaluation),
        "executed_control_steps_new": sum(r["steps"] for w in windows for r in w["episodes"]),
    }
    (args.run / "acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in [
                    "passed",
                    "checkpoint_counts",
                    "collector_next_windows",
                    "direct_overlap_with_fixed_eval",
                    "executed_control_steps_new",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
