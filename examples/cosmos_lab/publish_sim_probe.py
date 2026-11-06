"""Validate and publish bounded simulator probes with explicit control/policy labels."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    probe = args.run / "probe"
    report = json.loads((probe / "report.json").read_text())
    policy = "repeat_task0" in report
    kind = "policy-probe" if policy else "sim-probe"
    review = args.run / "review"
    review.mkdir(exist_ok=False)
    pixels = {}
    initial_states = {}
    validations = []
    for row in report["trials"]:
        path = probe / row["name"] / "views.mp4" if policy else probe / (row["name"] + ".mp4")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            decoded = list(container.decode(stream))
            times = np.asarray([float(frame.pts * frame.time_base) for frame in decoded])
            assert len(decoded) == row["frames"] and float(stream.average_rate) == 20
            assert np.allclose(np.diff(times), 0.05)
            frames = np.stack([frame.to_ndarray(format="rgb24") for frame in decoded])
            assert frames.shape == (row["frames"], 256, 512, 3)
            pixels[row["name"]] = frames
        npz = probe / row["name"] / "trajectory.npz" if policy else probe / (row["name"] + ".npz")
        with np.load(npz, allow_pickle=False) as data:
            initial_states[row["name"]] = data["initial_sim_state"].copy()
            assert np.isfinite(data["states"]).all() and np.isfinite(data["actions"]).all()
            assert data["states"].shape == (row["frames"], 8)
            assert data["actions"].shape == (row["frames"] - 1, 7)
            assert bool(data["success"].any()) == row["success_any"]
            assert bool(data["success"][-1]) == row["success_final"]
        validations.append(
            {"name": row["name"], "frames": len(frames), "decode_timestamps_state_actions_success": "passed"}
        )
    modes = ["zero", "policy"] if policy else ["zero", "positive_x", "negative_x"]
    checkpoint = report["protocol"].get("checkpoint")
    policy_label = report["protocol"].get("policy_label") or (
        "PPO " + Path(checkpoint).name if checkpoint else "Pretrained policy"
    )
    labels = ["Zero control", policy_label] if policy else ["Zero control", "+X control", "-X control"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for case_id in sorted({r.get("case_id", f"task-{r['task_id']}") for r in report["trials"]}):
        arrays = [pixels[f"{case_id}-{m}"] for m in modes]
        assert all(np.array_equal(initial_states[f"{case_id}-{m}"], initial_states[f"{case_id}-zero"]) for m in modes)
        width = 256 * len(modes)
        with av.open(str(review / f"{case_id}.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=20)
            stream.width, stream.height, stream.pix_fmt = width, 312, "yuv420p"
            for step in range(len(arrays[0])):
                image = Image.new("RGB", (width, 312), "white")
                draw = ImageDraw.Draw(image)
                for j, (array, label) in enumerate(zip(arrays, labels, strict=True)):
                    image.paste(Image.fromarray(array[step, :, :256]), (j * 256, 28))
                    draw.text((j * 256 + 4, 4), f"{label} | {step / 20:.2f}s", font=font, fill="black")
                draw.text((4, 290), f"{case_id}, same initial state; fixed ODE evaluation.", font=font, fill="black")
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    repeats = {}
    for name in pixels:
        if name.endswith("_repeat"):
            repeats[name] = bool(np.array_equal(pixels[name.removesuffix("_repeat")], pixels[name]))
    audit = {
        "videos": validations,
        "pixel_exact_repeats": repeats,
        "paired_initial_physics_identical": True,
        "visual_review_scope": "Decoded all frames and checked trajectory records. This publisher performs no visual review or human video assessment.",
    }
    (review / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    with wandb.init(
        entity="sav-inception-self",
        project="cosmos-lab",
        group=kind,
        name="yuzh-" + kind + "-" + args.run.name.rsplit("-", 1)[-1],
        config=report["protocol"],
        dir=str(review),
    ) as run:
        table = wandb.Table(columns=["case", "condition", "instruction", "success_any", "success_final", "video"])
        for row in report["trials"]:
            path = probe / row["name"] / "views.mp4" if policy else probe / (row["name"] + ".mp4")
            video = wandb.Video(str(path), format="mp4", caption="Left: agent view; right: wrist view. " + row["name"])
            table.add_data(
                row.get("case_id", f"task-{row['task_id']}"),
                row.get("mode", row.get("condition")),
                row["language"],
                row["success_any"],
                row["success_final"],
                video,
            )
        run.log({"trials/all": table})
        for path in sorted(review.glob("task-*.mp4")):
            run.log({"comparison/" + path.stem: wandb.Video(str(path), format="mp4", caption=" | ".join(labels))})
        run.summary.update(
            {
                "scope": report["scope"],
                "training_launched": False,
                "evaluated_training_checkpoint": checkpoint,
                "validated_frames": sum(x["frames"] for x in validations),
                "pixel_exact_repeats": repeats,
                "checks": report.get("checks", report.get("repeat_task0")),
            }
        )
        artifact = wandb.Artifact("yuzh-" + kind + "-" + args.run.name.rsplit("-", 1)[-1], type="evaluation")
        for path in (
            sorted(args.run.glob("*.txt"))
            + sorted(args.run.glob("*.json"))
            + [probe / "report.json", review / "validation.json"]
        ):
            artifact.add_file(str(path), name=str(path.relative_to(args.run)))
        artifact.add_file(str(args.run / "code.patch"), name="code.patch")
        for path in sorted(probe.rglob("*.npz")):
            if path.name == "trajectory.npz" or path.parent == probe:
                artifact.add_file(str(path), name=str(path.relative_to(args.run)))
        identity = probe / "policy-identity.json"
        if identity.exists():
            artifact.add_file(str(identity), name="policy-identity.json")
        reload_audit = probe / "checkpoint-reload.json"
        if reload_audit.exists():
            artifact.add_file(str(reload_audit), name="checkpoint-reload.json")
        run.log_artifact(artifact).wait()
        url = run.url
    (review / "wandb-url.txt").write_text(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
