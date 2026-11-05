"""Audit both completed arms and report beta-normalized, fixed-noise validation."""

import json
import math
from pathlib import Path

import wandb
import yaml

from examples.cosmos_lab.review_dpo_generation import sha


def main():
    lab = Path("/mnt/patient-unit/home/yan/cosmos-lab")
    study = lab / "data/bridge-beta-study-v1"
    jobs = json.loads((study / "jobs.json").read_text())
    manifest = json.loads((study / "manifest.json").read_text())
    arms = {}
    configs = {}
    for beta in (5000, 500):
        job = jobs[str(beta)]
        trace = lab / f"runs/beta-study-{job}"
        assert (trace / "end.txt").is_file()
        assert sha(trace / "train.jsonl") == manifest["train_sha256"]
        assert sha(trace / "val.jsonl") == manifest["val_sha256"]
        directories = list(
            (lab / f"runs/training/cosmos-lab/bridge-i2v-beta-study/yuzh-beta-{beta}-{job}").glob(
                "*/config_resolved.yaml"
            )
        )
        assert len(directories) == 1
        run_dir = directories[0].parent
        config = yaml.safe_load(directories[0].read_text())
        assert config["rl"]["beta"] == beta and config["rl"]["max_steps"] == 12
        assert config["rl"]["optimizer"]["lr"] == 1e-6
        assert config["rl"]["init_from"] == str(lab / "checkpoints/Cosmos3-Nano-DCP")
        configs[beta] = config
        checkpoint = run_dir / "checkpoint_artifacts/iter_000012"
        receipt = json.loads((checkpoint / ".posttrain-checkpoint-artifact.json").read_text())
        assert receipt["iteration"] == 12 and receipt["checkpoint_components"] == ["model"]
        assert (checkpoint / "model.pt").stat().st_size == receipt["components"]["model"]["size"]
        remote = wandb.Api().run(f"sav-inception-self/cosmos-lab/{run_dir.name}")
        assert remote.state == "finished", remote.state
        history = list(remote.scan_history())
        (trace / "wandb-history.json").write_text(json.dumps(history, indent=2) + "\n")
        training = [row for row in history if row.get("optim/optimizer_step") is not None]
        assert [row["optim/optimizer_step"] for row in training] == list(range(1, 13))
        for row in training:
            assert all(
                row[k] == 0
                for k in (
                    "optim/nonfinite_loss",
                    "optim/nonfinite_gradient",
                    "optim/step_skipped",
                    "optim/nonfinite_skips_total",
                    "optim/reference_refreshed",
                )
            )
            assert all(math.isfinite(row[k]) for k in ("train/grad_norm", "train/loss_avg", "train/dpo_reward_margin"))
            assert row["train/learning_rate"] == 1e-6
        validation = [row for row in history if row.get("val/pair_count") is not None]
        assert [row["_step"] for row in validation] == [0, 4, 8, 12]
        assert all(row["val/pair_count"] == 2 for row in validation)
        assert all(
            row["val/dpo_err_ref_w"] == validation[0]["val/dpo_err_ref_w"]
            and row["val/dpo_err_ref_l"] == validation[0]["val/dpo_err_ref_l"]
            for row in validation
        )
        val_rows = []
        for row in validation:
            margin = row["val/dpo_reward_margin"] / beta
            recomputed = -0.5 * (
                (row["val/dpo_err_pol_w"] - row["val/dpo_err_ref_w"])
                - (row["val/dpo_err_pol_l"] - row["val/dpo_err_ref_l"])
            )
            assert abs(margin - recomputed) < 1e-7
            val_rows.append(
                dict(
                    step=row["_step"],
                    raw_loss=row["val/loss_avg"],
                    margin_per_beta=margin,
                    accuracy=row["val/dpo_accuracy"],
                    **{
                        key: row[f"val/{key}"]
                        for key in ("dpo_err_pol_w", "dpo_err_pol_l", "dpo_err_ref_w", "dpo_err_ref_l")
                    },
                )
            )
        norms = [row["train/grad_norm"] for row in training]
        arms[str(beta)] = dict(
            job=job,
            run_id=run_dir.name,
            wandb_url=remote.url,
            checkpoint=str(checkpoint),
            checkpoint_manifest=receipt,
            updates=12,
            nonfinite_skips=0,
            preclip_grad_norm=norms,
            clipped_steps=sum(n > 1 for n in norms),
            validation=val_rows,
        )
        (trace / "wandb-url.txt").write_text(remote.url + "\n")
    assert arms["5000"]["validation"][0] == arms["500"]["validation"][0], "Initial validation mismatch"
    # Compare actual runtime settings while allowing run identity and beta to differ.
    for config in configs.values():
        config["job"]["name"] = "matched"
        config["rl"]["beta"] = 0
        config["rl"]["run_ts"] = "matched"
    assert configs[5000] == configs[500], "Actual runtime settings differ beyond beta and run identity"
    report = dict(
        arms=arms,
        initial_validation_exact=True,
        actual_config_matched=True,
        scope="Small agent-labeled fixed validation; normalized margin is not task success. "
        "Checkpoint hash and actual tensor reload are separately checked during generation.",
    )
    (study / "training-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
