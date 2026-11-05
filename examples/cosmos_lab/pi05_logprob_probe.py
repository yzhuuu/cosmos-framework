"""Isolate unchanged-weight scalar versus batched Pi05 replay on real saved observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOCondition, Pi05PPOTrace


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.root
    model = Pi05TrainPolicy(
        Pi05ModelConfig(
            str(root / "checkpoints/pi05-libero"),
            str(root / "runtime/libero/tokenizer.model"),
            str(root / "checkpoints/pi05-libero/physical-intelligence/libero/norm_stats.json"),
        )
    )
    paths = sorted((root / "runs/pi05-closed-loop-507/probe/task-0-state1-policy").glob("observation-*.npz"))[:16]
    assert len(paths) == 16
    observations, conditions, samples = [], [], []
    for index, path in enumerate(paths):
        with np.load(path) as data:
            obs = {key: data[key] for key in ["image", "wrist_image", "state"]}
            obs["prompt"] = str(data["prompt"].item())
        condition = model.ppo_condition([obs])
        sample = model.ppo_sample(condition, seed=42 + index)
        observations.append(obs)
        conditions.append(condition)
        samples.append(sample)
    old = torch.cat([sample.scores.logprobs for sample in samples])
    scalar = torch.cat([model.ppo_score(c, s.trace).logprobs for c, s in zip(conditions, samples, strict=True)])
    rows = []
    for size in [1, 2, 4, 16]:
        for source in ["saved_scalar_prefix", "recomputed_batched_prefix"]:
            logprobs, prefix_deltas = [], []
            for start in range(0, 16, size):
                parts = conditions[start : start + size]
                cached = Pi05PPOCondition(
                    tuple(
                        tuple(torch.cat([c.prefix_cache[layer][kv] for c in parts]) for kv in range(2))
                        for layer in range(len(parts[0].prefix_cache))
                    ),
                    torch.cat([c.prefix_mask for c in parts]),
                    torch.cat([c.value_features for c in parts]),
                )
                condition = (
                    cached
                    if source == "saved_scalar_prefix"
                    else model.ppo_condition(observations[start : start + size])
                )
                prefix_deltas.append(
                    max(
                        float((a.float() - b.float()).abs().max())
                        for cp, rp in zip(cached.prefix_cache, condition.prefix_cache, strict=True)
                        for a, b in zip(cp, rp, strict=True)
                    )
                )
                traces = [sample.trace for sample in samples[start : start + size]]
                trace = Pi05PPOTrace(
                    **{
                        key: torch.cat([getattr(t, key) for t in traces])
                        for key in ["latent_before", "latent_after", "denoise_index"]
                    }
                )
                logprobs.append(model.ppo_score(condition, trace).logprobs)
            scored = torch.cat(logprobs)
            rows.append(
                {
                    "batch_size": size,
                    "prefix": source,
                    "max_logprob_delta": float((scored - old).abs().max()),
                    "clip_fraction": float(((scored - old).exp().sub(1).abs() > 0.2).float().mean()),
                    "prefix_cache_max_abs": max(prefix_deltas),
                    "logprobs": scored.tolist(),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    report = {
        "scope": "No optimizer, no FSDP, no Omni, unchanged loaded base; 16 real saved observations. Diagnostic only, not a PPO training acceptance.",
        "torch": torch.__version__,
        "identity": model.ppo_identity(),
        "observation_paths": list(map(str, paths)),
        "scalar_repeat_max_delta": float((old - scalar).abs().max()),
        "old_logprobs": old.tolist(),
        "denoise_indices": [int(s.trace.denoise_index.item()) for s in samples],
        "batches": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
