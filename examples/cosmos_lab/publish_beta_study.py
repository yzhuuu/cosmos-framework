"""Publish audited training traces and all fixed-final-checkpoint videos with limitations."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    lab = Path("/mnt/patient-unit/home/yan/cosmos-lab")
    study = lab / "data/bridge-beta-study-v1"
    assert (args.run / "end.txt").is_file()
    training = json.loads((study / "training-audit.json").read_text())
    comparison = json.loads((args.run / "comparison.json").read_text())
    review = json.loads((args.run / "review/annotations.json").read_text())
    assert len(comparison["records"]) == 8 and all(r["baseline_repeat_pixel_exact"] for r in comparison["records"])
    assert {r["name"] for r in review["records"]} == {r["name"] for r in comparison["records"]}
    assert training["initial_validation_exact"] and training["actual_config_matched"]
    overlay_summary = {}
    for beta in (5000, 500):
        arm = training["arms"][str(beta)]
        digest = json.loads((args.run / f"beta{beta}-sha256.json").read_text())
        assert digest["manifest_matches"]
        assert digest["sha256"] == arm["checkpoint_manifest"]["components"]["model"]["sha256"]
        changed = 0
        for rank in range(2):
            receipt = json.loads((args.run / f"beta{beta}/overlay-rank{rank}.json").read_text())
            assert receipt["verified_local_values"] and receipt["manifest"]["iteration"] == 12
            assert receipt["checkpoint"] == arm["checkpoint"] and len(receipt["tensors"]) == 1155
            assert not any(t["changed_local_elements"] for t in receipt["tensors"] if not t["trainable"])
            changed += sum(t["changed_local_elements"] for t in receipt["tensors"] if t["trainable"])
        assert changed > 0
        overlay_summary[str(beta)] = dict(
            changed_local_trainable_elements=changed,
            frozen_changes=0,
            tensors_per_rank=1155,
            ranks=2,
            sha256=digest["sha256"],
        )
    (args.run / "overlay-audit.json").write_text(json.dumps(overlay_summary, indent=2) + "\n")
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        name=args.run.name.replace("beta-compare", "yuzh-beta-compare"),
        group="bridge-i2v-beta-study",
        dir=str(args.run),
        config=json.loads((study / "manifest.json").read_text()),
        settings=wandb.Settings(disable_code=True),
    ) as run:
        run.summary.update(
            {
                "audit/base_repeats_pixel_exact": 8,
                "audit/matched_initial_validation": True,
                "audit/frozen_parameter_changes": 0,
                "review/conclusion": review["conclusion"],
                "review/coverage": review["coverage"],
            }
        )
        run.define_metric("validation_step")
        run.define_metric("normalized_validation/*", step_metric="validation_step")
        for i, step in enumerate((0, 4, 8, 12)):
            values = {"validation_step": step}
            for beta in (5000, 500):
                val = training["arms"][str(beta)]["validation"][i]
                values[f"normalized_validation/beta{beta}_margin_per_beta"] = val["margin_per_beta"]
                values[f"normalized_validation/beta{beta}_accuracy"] = val["accuracy"]
            run.log(values)
        table = wandb.Table(columns=["name", "seed", "goal", "observation", "evidence_frames", "limitations"])
        annotations = {r["name"]: r for r in review["records"]}
        for row in comparison["records"]:
            note = annotations[row["name"]]
            table.add_data(
                row["name"],
                row["seed"],
                row["goal"],
                note["observation"],
                json.dumps(note["evidence_frames"]),
                note["limitations"],
            )
            run.log(
                {
                    f"comparison/{row['name']}": wandb.Video(
                        row["comparison"],
                        format="mp4",
                        caption=f"Left original Nano; center beta 5000; right beta 500. Both step 12; same seed {row['seed']}. Goal: {row['goal']}",
                    ),
                    **{
                        f"pixel_diagnostic/{row['name']}/{arm}_mae": row["arms"][arm]["mean_pixel_abs_change"]
                        for arm in ("beta5000", "beta500")
                    },
                }
            )
        run.log({"review/evidence": table})
        artifact = wandb.Artifact(args.run.name + "-evidence", type="evaluation")
        for name in (
            "manifest.json",
            "jobs.json",
            "config-beta-5000.json",
            "config-beta-500.json",
            "training-audit.json",
        ):
            artifact.add_file(str(study / name), name="study/" + name)
        for name in (
            "comparison.json",
            "baseline-repeat.json",
            "requests.jsonl",
            "code-commit.txt",
            "code.patch",
            "beta5000-sha256.json",
            "beta500-sha256.json",
            "overlay-audit.json",
            "review/annotations.json",
        ):
            artifact.add_file(str(args.run / name), name=name)
        artifact.add_file(str(lab / "docs/experiments/2026-11-05-o8-beta-study.md"), name="study/experiment.md")
        for beta in (5000, 500):
            arm = training["arms"][str(beta)]
            run.summary[f"training/beta{beta}_url"] = arm["wandb_url"]
            run.summary[f"training/beta{beta}_clipped_steps"] = arm["clipped_steps"]
            trace = lab / f"runs/beta-study-{arm['job']}"
            for name in ("wandb-history.json", "train.jsonl", "val.jsonl", "code-commit.txt", "code.patch"):
                artifact.add_file(str(trace / name), name=f"training/beta{beta}/{name}")
            config = Path(arm["checkpoint"]).parent.parent / "config_resolved.yaml"
            artifact.add_file(str(config), name=f"training/beta{beta}/config_resolved.yaml")
            for rank in range(2):
                name = f"beta{beta}/overlay-rank{rank}.json"
                artifact.add_file(str(args.run / name), name=name)
        run.log_artifact(artifact).wait()
        url = run.url
    (args.run / "wandb-url.txt").write_text(url + "\n")
    for arm in training["arms"].values():
        remote = wandb.Api().run(f"sav-inception-self/cosmos-lab/{arm['run_id']}")
        remote.summary["evaluation/generation_comparison_url"] = url
        remote.summary["audit/finite_updates"] = 12
        remote.summary.update()
    print(url)


if __name__ == "__main__":
    main()
