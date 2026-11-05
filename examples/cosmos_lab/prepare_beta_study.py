"""Freeze matched configuration, data hashes, and actual rank sampling before training."""

import copy
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from examples.cosmos_lab.dpo_beta_config import make_config


def main():
    lab = Path(os.environ["COSMOS_LAB_ROOT"])
    dest = lab / "data/bridge-beta-study-v1"
    dest.mkdir(exist_ok=True)
    if (dest / "manifest.json").exists():
        raise FileExistsError("Study is already frozen")
    os.environ["DPO_BASE_CHECKPOINT"] = str(lab / "checkpoints/Cosmos3-Nano-DCP")
    os.environ["DPO_TOKENIZER_PATH"] = str(
        lab
        / "cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots"
        / "921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
    )
    configs = {}
    for beta in (5000, 500):
        os.environ["YUZH_DPO_BETA"] = str(beta)
        config = make_config()
        flat = OmegaConf.to_container(OmegaConf.create(config.to_dict(), flags={"allow_objects": True}), resolve=True)
        configs[beta] = json.loads(json.dumps(flat, default=str))
        (dest / f"config-beta-{beta}.json").write_text(json.dumps(configs[beta], indent=2) + "\n")
    matched = copy.deepcopy(configs[500])
    matched["rl"]["beta"] = configs[5000]["rl"]["beta"]
    matched["job"]["name"] = configs[5000]["job"]["name"]
    assert matched == configs[5000], "An unintended difference exists between the two arms"
    train = lab / "data/bridge-preferences-labeled-v2/train.jsonl"
    val = lab / "data/bridge-preference-validation-labeled-v2/val.jsonl"
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    schedule = {}
    for rank in range(2):
        cursor = IndexedCursor(rows, shard_index=rank, shard_count=2, shuffle=False, seed=2026)
        schedule[rank] = [rows[cursor.next_index()]["pair_id"] for _ in range(12)]
    manifest = dict(
        betas=[5000, 500],
        max_updates=12,
        learning_rate=1e-6,
        grad_clip_norm=1.0,
        seed=2026,
        validation_seed=901,
        validation_steps=[0, 4, 8, 12],
        config_only_differences=["rl.beta", "job.name"],
        train_path=str(train),
        train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
        val_path=str(val),
        val_sha256=hashlib.sha256(val.read_bytes()).hexdigest(),
        rank_sample_schedule=schedule,
        sampling_note="Native strided cursor: rank 0 alternates rows 0/2; rank 1 repeats row 1. "
        "Presentations are 6/12/6, identical across arms, not uniform epochs.",
        initialization="Original Nano DCP independently for both arms; frozen reference; no resume or LoRA",
        budget=dict(
            training_jobs=2,
            gpus_per_job=2,
            minutes_per_training_job=40,
            model_artifacts_gb=126,
            generation_jobs=1,
            generation_minutes=45,
            generation_outputs=24,
        ),
        evaluation="All four revised val episodes, both seeds, including abstentions. Base and both final "
        "step-12 checkpoints. Require base repeat to equal job 493. Test videos untouched.",
        interpretation="Initial fixed-noise raw errors must match. Compare margin/beta and raw errors; "
        "raw DPO loss is not a common scale across beta. Preclip gradient and finite update checks; "
        "video evidence required for quality. Agent-labeled 3 train / 2 val pairs are not a generalization study.",
    )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
