"""Persist real Omni sampling records and compare unsharded learner replay before training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY
from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, push_stream_to_peer
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.control.loop.weight_sync import confirm_weight_version
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
from cosmos_framework.posttrain.model.pi05.serving.body import Pi05PolicyBody
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer


class DeviceCopyMover(CopyMover):
    def fetch(self, handle):
        return {name: value.to(device="cuda") for name, value in super().fetch(handle).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.root
    config = Pi05ModelConfig(
        str(root / "checkpoints/pi05-libero"),
        str(root / "runtime/libero/tokenizer.model"),
        str(root / "checkpoints/pi05-libero/physical-intelligence/libero/norm_stats.json"),
    )
    policy = Pi05TrainPolicy(config)
    producer = JointPPOTrainer(policy, JointPPOConfig())
    backend = WeightSync(CopyMover())
    body = Pi05PolicyBody(config, output=str(args.output), verify_weight_values=True)
    rows, records = [], []
    try:
        body.bind_weight_backend(WeightSync(DeviceCopyMover()))
        push_stream_to_peer(producer, body, backend)
        confirm_weight_version(body, producer, 0)
        paths = sorted((root / "runs/pi05-closed-loop-507/probe/task-0-state1-policy").glob("observation-*.npz"))[:8]
        for index, path in enumerate(paths):
            with np.load(path) as data:
                obs = {key: data[key] for key in ["image", "wrist_image", "state"]}
                obs["prompt"] = str(data["prompt"].item())
            source = {
                "sample_id": f"probe-{index}",
                "episode_id": f"episode-{index}",
                "fragment_id": "window-0",
                "step_idx": 0,
                "session": "omni-probe",
            }
            decision = body.sample_decisions([obs], sources=[source], seed=42 + index, weight_version=0)[0]
            records.append(decision)
            torch.save(records, args.output / "sampling-records.pt")
            step = decision.transitions[0]
            old = float(step.algo_extra[STATS_KEY].old_logprob)
            with torch.no_grad():
                scalar = float(policy.score_ppo_decision(step).logprobs[0])
                packed = float(policy.score_ppo_decisions([step]).logprobs[0])
            rows.append(
                {
                    "index": index,
                    "old": old,
                    "scalar": scalar,
                    "packed_b1": packed,
                    "scalar_delta": abs(scalar - old),
                    "packed_delta": abs(packed - old),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        report = {
            "scope": "Omni original-weight sampling, CPU-copy weight publication verified, unsharded scalar and packed-B1 replay; no optimizer update",
            "pipeline": body.pipeline_info,
            "rows": rows,
            "all_pass_native_tolerance": all(
                max(r["scalar_delta"], r["packed_delta"]) <= policy.ppo_logprob_tolerance for r in rows
            ),
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        body.close()
        backend.mover.close()


if __name__ == "__main__":
    main()
