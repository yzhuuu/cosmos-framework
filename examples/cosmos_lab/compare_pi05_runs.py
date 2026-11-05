"""Publish all fixed-case Pi05 before/after videos and native task outcomes."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont


def frames(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == 20
        decoded = list(container.decode(stream))
        times = [float(frame.pts * frame.time_base) for frame in decoded]
        assert len(decoded) == 241 and np.allclose(np.diff(times), 0.05)
        return np.stack([frame.to_ndarray(format="rgb24") for frame in decoded])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("updated", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = [args.baseline, args.updated]
    reports = [json.loads((run / "probe/report.json").read_text()) for run in runs]
    protocols = [report["protocol"] for report in reports]
    for key in ["revision", "seed", "sampling", "max_steps", "reset_steps", "suite", "trials", "policy_python"]:
        assert protocols[0][key] == protocols[1][key], key
    assert protocols[0].get("checkpoint") is None and protocols[1]["checkpoint"]
    assert Path(protocols[1]["checkpoint"]).name == "iter_000002", "This comparison is frozen to PPO step 2"
    identities = [json.loads((run / "probe/policy-identity.json").read_text()) for run in runs]
    for key in ["config", "identity", "torch", "device", "sampling_mode"]:
        assert identities[0][key] == identities[1][key], key
    records = [{row["case_id"]: row for row in report["trials"] if row["mode"] == "policy"} for report in reports]
    assert records[0].keys() == records[1].keys() and len(records[0]) == 20
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    rows = []
    for case in sorted(records[0]):
        pair = [record[case] for record in records]
        folders = [run / "probe" / row["name"] for run, row in zip(runs, pair, strict=True)]
        with np.load(folders[0] / "trajectory.npz") as a, np.load(folders[1] / "trajectory.npz") as b:
            assert np.array_equal(a["initial_sim_state"], b["initial_sim_state"]), case
            action_rmse = float(np.sqrt(np.mean((a["actions"] - b["actions"]) ** 2)))
            state_rmse = float(np.sqrt(np.mean((a["states"] - b["states"]) ** 2)))
            for data, row in zip([a, b], pair, strict=True):
                assert np.isfinite(data["actions"]).all() and np.isfinite(data["states"]).all()
                assert bool(data["success"].any()) == row["success_any"]
                assert bool(data["success"][-1]) == row["success_final"]
        arrays = [frames(folder / "views.mp4") for folder in folders]
        assert arrays[0].shape == arrays[1].shape == (241, 256, 512, 3)
        row = {
            "case": case,
            "instruction": pair[0]["language"],
            "base_any": pair[0]["success_any"],
            "after_any": pair[1]["success_any"],
            "base_final": pair[0]["success_final"],
            "after_final": pair[1]["success_final"],
            "action_rmse": action_rmse,
            "state_rmse": state_rmse,
            "decoded_pixels_equal": bool(np.array_equal(*arrays)),
        }
        assert pair[0]["language"] == pair[1]["language"]
        rows.append(row)
        with av.open(str(args.output / f"{case}.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=20)
            stream.width, stream.height, stream.pix_fmt = 1024, 312, "yuv420p"
            for step in range(241):
                image = Image.new("RGB", (1024, 312), "white")
                draw = ImageDraw.Draw(image)
                for index, label in enumerate(["SFT baseline", "PPO step 2"]):
                    image.paste(Image.fromarray(arrays[index][step]), (index * 512, 28))
                    draw.text(
                        (index * 512 + 4, 4), f"{label} | agent + wrist | {step / 20:.2f}s", font=font, fill="black"
                    )
                draw.text(
                    (4, 290),
                    f"{case}; same initial physics / prompt / seed; shared training pool",
                    font=font,
                    fill="black",
                )
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    summary = {
        "cases": len(rows),
        "frames_validated": 20 * 241 * 2,
        "initial_physics_identical": True,
        "matched_policy_runtime": True,
        "base_any": sum(row["base_any"] for row in rows),
        "after_any": sum(row["after_any"] for row in rows),
        "base_final": sum(row["base_final"] for row in rows),
        "after_final": sum(row["after_final"] for row in rows),
        "scope": "Fixed shared-pool diagnostic, not independent generalization; no human video score claimed.",
        "baseline": str(args.baseline),
        "updated": str(args.updated),
    }
    (args.output / "comparison.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n")
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        group="pi05-ppo-eval",
        name="yuzh-ppo-compare-" + args.updated.name.rsplit("-", 1)[-1],
        dir=str(args.output),
        config={"protocols": protocols},
    ) as run:
        columns = list(rows[0])
        table = wandb.Table(columns=columns + ["comparison"])
        for row in rows:
            table.add_data(
                *[row[key] for key in columns],
                wandb.Video(
                    str(args.output / (row["case"] + ".mp4")),
                    format="mp4",
                    caption="Left: SFT baseline. Right: PPO step 2. Each half: agent + wrist.",
                ),
            )
        run.log({"comparison/all_cases": table})
        run.summary.update(summary)
        artifact = wandb.Artifact("yuzh-pi05-ppo-comparison-" + args.updated.name.rsplit("-", 1)[-1], type="evaluation")
        artifact.add_file(str(args.output / "comparison.json"))
        artifact.add_file(str(args.updated / "probe/checkpoint-reload.json"))
        run.log_artifact(artifact).wait()
        url = run.url
    (args.output / "wandb-url.txt").write_text(url + "\n")
    print(json.dumps(summary))
    print(url)


if __name__ == "__main__":
    main()
