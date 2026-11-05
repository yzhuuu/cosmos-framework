"""Separate inference-mode and FSDP scalar replay differences without any optimizer update."""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
from cosmos_framework.posttrain.model.pi05.records import map_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    args.output.mkdir(parents=True, exist_ok=True)
    config = Pi05ModelConfig(
        str(args.root / "checkpoints/pi05-libero"),
        str(args.root / "runtime/libero/tokenizer.model"),
        str(args.root / "checkpoints/pi05-libero/physical-intelligence/libero/norm_stats.json"),
    )
    source = args.root / "runs/pi05-closed-loop-507/probe/task-0-state1-policy"
    observations = []
    for step in [0, 40]:
        with np.load(source / f"observation-{step:03d}.npz") as data:
            obs = {key: data[key] for key in ["image", "wrist_image", "state"]}
            obs["prompt"] = str(data["prompt"].item())
            observations.append(obs)
    plain = Pi05TrainPolicy(config)
    rows, traces, old, references = [], [], [], []
    for index, obs in enumerate(observations):
        with torch.inference_mode():
            condition = plain.ppo_condition([obs])
            sample = plain.ppo_sample(condition, seed=42 + index)
        trace = map_record(sample.trace, lambda tensor: tensor.clone())
        before = sample.scores.logprobs.clone()
        with torch.no_grad():
            condition = plain.ppo_condition([obs])
            scalar = plain.ppo_score(condition, trace).logprobs
        rows.append(
            {"case": index, "phase": "plain_inference_to_nograd", "delta": float((scalar - before).abs().max())}
        )
        traces.append(trace)
        old.append(before)
        references.append(map_record(condition, lambda tensor: tensor.cpu().clone()))
    del plain, sample, condition, scalar
    gc.collect()
    torch.cuda.empty_cache()
    dimensions = ParallelDims(world_size=dist.get_world_size(), dp_shard=dist.get_world_size())
    dimensions.build_device_mesh("cuda")
    policy = Pi05TrainPolicy(config, parallel_dims=dimensions)
    for index, obs in enumerate(observations):
        for grad in [False, True]:
            with torch.set_grad_enabled(grad):
                condition = policy.ppo_condition([obs])
                scores = policy.ppo_score(condition, traces[index])
            reference = references[index]
            prefix_delta = max(
                float((a.detach().cpu().float() - b.float()).abs().max())
                for p, q in zip(condition.prefix_cache, reference.prefix_cache, strict=True)
                for a, b in zip(p, q, strict=True)
            )
            rows.append(
                {
                    "case": index,
                    "phase": "fsdp_grad" if grad else "fsdp_nograd",
                    "delta": float((scores.logprobs.detach() - old[index]).abs().max()),
                    "prefix_delta": prefix_delta,
                }
            )
            del condition, scores
    report = {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "torch": torch.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "rows": rows,
        "scope": "Original weights, scalar real observations, no optimizer. Compare inference/no_grad/grad and unsharded/FSDP.",
    }
    (args.output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
