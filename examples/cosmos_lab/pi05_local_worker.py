"""Local file/stdio transport for native Pi05 ODE or audited selected-flow SDE."""

import argparse
import contextlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sampler", choices=["ode", "selected-flow-sde"], default="ode")
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        import torch

        from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
        from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy

        config = Pi05ModelConfig(
            checkpoint_path=str(args.root / "checkpoints/pi05-libero"),
            tokenizer_path=str(args.root / "runtime/libero/tokenizer.model"),
            norm_stats_path=str(args.root / "checkpoints/pi05-libero/physical-intelligence/libero/norm_stats.json"),
            action_chunk=5,
            denoise_steps=3,
        )
        policy = Pi05TrainPolicy(config, device="cuda")
        if args.checkpoint:
            from examples.cosmos_lab.pi05_checkpoint import restore_heads

            audit = restore_heads(policy, args.checkpoint)
            (args.output / "checkpoint-reload.json").write_text(json.dumps(audit, indent=2) + "\n")
        policy.model.eval()
        policy.net.eval()
        identity = {
            "config": asdict(config),
            "identity": policy.ppo_identity(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "sampling_mode": args.sampler,
            "transport": "local stdio/files; native policy arithmetic, no Omni engine",
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        }
        (args.output / "policy-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        observation = Path(request["observation"]).resolve()
        action_path = Path(request["actions"]).resolve()
        if not observation.is_relative_to(args.output.resolve()) or not action_path.is_relative_to(
            args.output.resolve()
        ):
            raise ValueError("IPC paths must remain inside this private run")
        with np.load(observation, allow_pickle=False) as data:
            obs = {key: data[key] for key in ["image", "wrist_image", "state"]}
            obs["prompt"] = str(data["prompt"].item())
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            condition = policy.ppo_condition([obs])
            if args.sampler == "ode":
                latent = policy.ppo_ode_actions(condition, seed=request["seed"])
            else:
                sampled = policy.ppo_sample(condition, seed=request["seed"])
                replay = policy.ppo_score(condition, sampled.trace)
                delta = float((sampled.scores.logprobs - replay.logprobs).abs().max())
                assert np.isfinite(delta) and delta <= 1e-5, "Native SDE behavior replay mismatch"
                assert torch.isfinite(sampled.scores.logprobs).all() and torch.isfinite(sampled.scores.values).all()
                latent = sampled.normalized_actions
                trace_path = action_path.with_suffix(".trace.npz")
                np.savez_compressed(
                    trace_path,
                    latent_before=sampled.trace.latent_before.cpu().numpy(),
                    latent_after=sampled.trace.latent_after.cpu().numpy(),
                    denoise_index=sampled.trace.denoise_index.cpu().numpy(),
                    normalized_actions=latent.cpu().numpy(),
                )
                audit = {
                    "observation": str(observation.relative_to(args.output.resolve())),
                    "trace": str(trace_path.relative_to(args.output.resolve())),
                    "seed": request["seed"],
                    "denoise_index": int(sampled.trace.denoise_index.item()),
                    "logprob": float(sampled.scores.logprobs.item()),
                    "value": float(sampled.scores.values.item()),
                    "logprob_replay_delta": delta,
                    "sampler_identity": sampled.trace.sampler_identity,
                }
                with (args.output / "sampling-audit.jsonl").open("a") as audit_stream:
                    audit_stream.write(json.dumps(audit) + "\n")
            actions = policy.processor.decode(latent).cpu().numpy()[0]
        assert actions.shape == (5, 7) and actions.dtype == np.float64 and np.isfinite(actions).all()
        np.save(action_path, actions)
        print(json.dumps({"actions": str(action_path), "shape": list(actions.shape)}), flush=True)


if __name__ == "__main__":
    main()
