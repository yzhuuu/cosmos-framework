"""Audit real Wan VAE conditioning, native packing and masked-loss gradients.

This does not run the diffusion transformer or update any model weights.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb

from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface
from cosmos_framework.posttrain.data.packer.preference import unpack_preferences
from cosmos_framework.posttrain.model.cosmos3 import _Cosmos3FlowCondition, _per_sample_mse_fp32
from cosmos_framework.posttrain.roles.env.preference import DatasetPreferenceEnv
from examples.cosmos_lab.i2v_preferences import BridgeI2VFlowPolicy, BridgeI2VPreferences, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("vae", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = BridgeI2VPreferences(
        args.lab / "data/bridge-preferences-labeled-v1/train.jsonl", args.lab / "data/bridge-study-v1/manifest.json"
    )
    env = DatasetPreferenceEnv(dataset, rank=0, world_size=1, shuffle=False)
    tokenizer = Wan2pt2VAEInterface(vae_path=str(args.vae))
    policy = object.__new__(BridgeI2VFlowPolicy)
    receipts = []
    for row in dataset.rows:
        winner, loser = unpack_preferences(env.get_samples(1))
        latents, conditions = [], []
        for batch in [winner, loser]:
            pixels = batch["video"][0][0].unsqueeze(0).to("cuda").float() / 127.5 - 1
            latent = tokenizer.encode(pixels).float()
            if latent.shape[2] != (row["frames"] - 1) // 4 + 1:
                raise ValueError("VAE changed expected latent duration")
            latents.append(latent)
            conditions.append(
                _Cosmos3FlowCondition(
                    gen_data_clean=SimpleNamespace(x0_tokens_vision=[latent], fps_vision=batch["conditioning_fps"][0]),
                    sequence_plans=batch["sequence_plan"],
                    text_tokens=[[1, 2, 3]],
                    is_image_batch=False,
                    resolutions=["256"],
                )
            )
        policy.assert_same_condition_context(*conditions)
        single_image = winner["video"][0][0][:, :1].unsqueeze(0).cuda().float() / 127.5 - 1
        image_latent = tokenizer.encode(single_image).float()
        first_error = float((latents[0][:, :, :1] - image_latent).abs().max())
        if first_error != 0:
            raise ValueError(f"Full video first latent differs from standalone condition: {first_error}")
        builder = PackedSequenceBuilder()
        builder.begin_sample(0)
        builder.pack_vision_tokens(
            latents[0],
            condition_frame_indexes_vision=[0],
            input_timestep=0.5,
            latent_patch_size=2,
            vision_fps=row["fps"],
            enable_fps_modulation=True,
            base_fps=24.0,
            temporal_compression_factor=4,
            vision_temporal_positions=None,
            temporal_position_period=None,
        )
        mask = builder.vision.condition_mask[0]
        if mask[0].item() != 1 or torch.count_nonzero(mask[1:]).item() != 0:
            raise ValueError("Native packer did not mask exactly the first latent frame")
        target = torch.zeros_like(latents[0][0])
        prediction = torch.ones_like(target, requires_grad=True)
        loss = _per_sample_mse_fp32([prediction], [target], [mask]).sum()
        loss.backward()
        condition_grad = float(prediction.grad[:, :1].abs().max())
        future_grad = float(prediction.grad[:, 1:].abs().sum())
        if condition_grad != 0 or future_grad <= 0 or not torch.isfinite(loss):
            raise ValueError("Masked loss gradient contract failed")
        bad = latents[1].clone()
        bad[:, :, :1] += 0.125
        conditions[1].gen_data_clean.x0_tokens_vision = [bad]
        try:
            policy.assert_same_condition_context(*conditions)
        except ValueError as exc:
            if "conditioning latents differ" not in str(exc):
                raise
        else:
            raise ValueError("Mismatched condition was accepted")
        receipts.append(
            {
                "pair_id": row["pair_id"],
                "latent_shape": list(latents[0].shape),
                "first_latent_max_abs_difference": float((latents[0][:, :, :1] - latents[1][:, :, :1]).abs().max()),
                "standalone_image_latent_max_abs_difference": first_error,
                "future_latents_different": not torch.equal(latents[0][:, :, 1:], latents[1][:, :, 1:]),
                "condition_gradient_max": condition_grad,
                "future_gradient_abs_sum": future_grad,
                "native_masked_mse": float(loss.detach()),
                "mismatched_condition_rejected": True,
            }
        )
    report = {
        "status": "passed",
        "pairs": receipts,
        "dataset_fingerprint": dataset.cursor_fingerprint(),
        "vae_sha256": digest(args.vae),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "scope": "Real VAE encoding + native env/collation/packing/masked-MSE backward. No transformer forward, optimizer step or quality claim. Text token IDs in condition guard are placeholders; prompts checked at data boundary.",
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        name=args.output.name,
        group="i2v-preference-audit",
        dir=str(args.output),
        config={"scope": report["scope"]},
    ) as run:
        run.summary["audit/passed"] = True
        run.summary["data/agent_labeled_pairs"] = len(receipts)
        artifact = wandb.Artifact(args.output.name, type="evaluation")
        artifact.add_file(str(args.output / "audit.json"))
        artifact.add_file(str(args.lab / "runs/preference-candidates-476/review/temporal-review.json"))
        artifact.add_file(str(args.lab / "data/bridge-preferences-labeled-v1/train.jsonl"))
        run.log_artifact(artifact)
        for row in dataset.rows:
            video = args.lab / "runs/preference-candidates-476/review" / f"{row['pair_id']}.mp4"
            run.log(
                {
                    f"preferences/{row['pair_id']}": wandb.Video(
                        str(video),
                        format="mp4",
                        caption=f"Source | A | B. Agent preference: {row['annotation']['decision'].upper()}",
                    )
                }
            )
        url = run.url
    (args.output / "wandb-url.txt").write_text(url + "\n")
    print(json.dumps(report, indent=2))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
