"""Replay saved ODE decisions and verify native selected-flow SDE sampling/scoring."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.reference / "probe/report.json").read_text())
    assert report["protocol"].get("checkpoint") is None
    config = Pi05ModelConfig(
        checkpoint_path=str(args.root / "checkpoints/pi05-libero"),
        tokenizer_path=str(args.root / "runtime/libero/tokenizer.model"),
        norm_stats_path=str(args.root / "checkpoints/pi05-libero/physical-intelligence/libero/norm_stats.json"),
        action_chunk=5,
        denoise_steps=3,
    )
    policy = Pi05TrainPolicy(config, device="cuda")
    policy.model.eval()
    policy.net.eval()
    rows = []
    with torch.inference_mode():
        for record in report["trials"]:
            if record["mode"] != "policy":
                continue
            folder = args.reference / "probe" / record["name"]
            with np.load(folder / "observation-000.npz") as data:
                obs = {key: data[key] for key in ["image", "wrist_image", "state"]}
                obs["prompt"] = str(data["prompt"].item())
            condition = policy.ppo_condition([obs])
            ode = policy.processor.decode(policy.ppo_ode_actions(condition, seed=2026)).cpu().numpy()[0]
            expected = np.load(folder / "actions-000.npy")
            ode_error = float(np.max(np.abs(ode - expected)))
            sampled = policy.ppo_sample(condition, seed=2026)
            replay = policy.ppo_score(condition, sampled.trace)
            repeated = policy.ppo_sample(condition, seed=2026)
            delta = float((sampled.scores.logprobs - replay.logprobs).abs().max())
            actions = policy.processor.decode(sampled.normalized_actions).cpu().numpy()[0]
            exact_repeat = torch.equal(sampled.normalized_actions, repeated.normalized_actions)
            row = {
                "case": record["case_id"],
                "ode_saved_actions_max_error": ode_error,
                "sde_logprob_replay_delta": delta,
                "sde_repeat_exact": exact_repeat,
                "selected_denoise_index": int(sampled.trace.denoise_index.item()),
                "sde_actions_finite": bool(np.isfinite(actions).all()),
                "ode_sde_action_rmse": float(np.sqrt(np.mean((actions - ode) ** 2))),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "scope": "50 saved initial observations, not complete trajectories or a quality evaluation. Native SDE sample/score/repeat plus cross-node ODE action replay.",
        "config": asdict(config),
        "identity": policy.ppo_identity(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "cases": rows,
        "passed": len(rows) == 50
        and all(
            r["ode_saved_actions_max_error"] == 0
            and r["sde_logprob_replay_delta"] <= 1e-5
            and r["sde_repeat_exact"]
            and r["sde_actions_finite"]
            for r in rows
        ),
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"], "Full preflight receipt saved, do not launch SDE evaluation"


if __name__ == "__main__":
    main()
