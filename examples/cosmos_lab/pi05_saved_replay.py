"""Replay persisted Omni records in a clean learner process across SDPA backend settings."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.root
    records_path = root / "runs/pi05-omni-logprob-518/probe/sampling-records.pt"
    records = torch.load(records_path, map_location="cpu", weights_only=False)
    policy = Pi05TrainPolicy(
        Pi05ModelConfig(
            str(root / "checkpoints/pi05-libero"),
            str(root / "runtime/libero/tokenizer.model"),
            str(root / "checkpoints/pi05-libero/physical-intelligence/libero/norm_stats.json"),
        )
    )
    original = torch.backends.cuda.cudnn_sdp_enabled()
    rows = []
    for enabled in [True, False]:
        torch.backends.cuda.enable_cudnn_sdp(enabled)
        for index, record in enumerate(records):
            step = record.transitions[0]
            old = float(step.algo_extra[STATS_KEY].old_logprob)
            new = float(policy.score_ppo_decisions([step]).logprobs[0])
            assert torch.backends.cuda.cudnn_sdp_enabled() == enabled, "Model leaked a global backend change"
            rows.append({"cudnn_sdp": enabled, "index": index, "old": old, "new": new, "delta": abs(new - old)})
            print(json.dumps(rows[-1]), flush=True)
    torch.backends.cuda.enable_cudnn_sdp(original)
    report = {
        "records": str(records_path),
        "original_cudnn_sdp": original,
        "scope": "Real serialized Omni trace replay, no optimizer, unsharded learner",
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
