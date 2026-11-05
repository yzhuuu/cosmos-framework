"""Verify exact action repeats and render action/stationary comparisons."""

import argparse
import json
import textwrap
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.cosmos_lab.prepare_preferences import sha256, validate_effective
from examples.cosmos_lab.review_preferences import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    frozen = json.loads((args.run / "candidate-manifest.json").read_text())
    requests = {r["name"]: r for r in map(json.loads, (args.run / "requests.jsonl").read_text().splitlines())}
    policy = json.loads((args.run / "generated/runtime-numerics.json").read_text())
    if policy["cudnn_benchmark"] is not False or policy["cudnn_deterministic"] is not True:
        raise ValueError("Numerical policy changed")
    output = args.run / "review"
    output.mkdir(exist_ok=False)
    pairs = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for group in frozen["groups"]:
        sides = [requests[group[k]] for k in ["action", "stationary"]]
        if {k for k in sides[0] if sides[0][k] != sides[1][k]} != {"name", "prompt"}:
            raise ValueError("Unexpected intervention difference")
        if sha256(sides[0]["vision_path"]) != group["image_sha256"]:
            raise ValueError("Condition image changed")
        if sha256(group["previous_action_video"]) != group["previous_action_sha256"]:
            raise ValueError("Prior action video changed")
        paths, frames = [], []
        for request in sides:
            base = args.run / "generated" / request["name"]
            validate_effective(request, json.loads((base / "sample_args.json").read_text()))
            path = base / "vision.mp4"
            paths.append(path)
            frames.append(np.stack(decode(path)))
        prior = np.stack(decode(group["previous_action_video"]))
        action_repeat = np.array_equal(prior, frames[0])
        required_repeat = group.get("require_exact_repeat", True)
        if required_repeat and not action_repeat:
            raise ValueError("Action repeat is not pixel exact; do not attribute differences solely to text")
        stationary_repeat = None
        if "previous_stationary_video" in group:
            previous_still = group["previous_stationary_video"]
            if sha256(previous_still) != group["previous_stationary_sha256"]:
                raise ValueError("Prior stationary video changed")
            stationary_repeat = np.array_equal(np.stack(decode(previous_still)), frames[1])
            if required_repeat and not stationary_repeat:
                raise ValueError("Stationary repeat is not pixel exact")
        if frames[0].shape != frames[1].shape or len(frames[0]) != sides[0]["num_frames"]:
            raise ValueError("Frame shapes/count mismatch")
        if not np.array_equal(frames[0][0], frames[1][0]):
            raise ValueError("Generated first frames differ")
        captions = [
            f"GUIDANCE {sides[0]['guidance']:g} | SEED {group['seed']}",
            "ACTION: " + group["action_goal"],
            "STATIONARY: " + group["stationary_goal"],
        ]
        lines = [line for c in captions for line in textwrap.wrap(c, width=64)]
        height = 286 + 18 * len(lines)
        video = output / f"{group['pair_id']}.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=int(sides[0]["fps"]))
            stream.width, stream.height, stream.pix_fmt = 512, height, "yuv420p"
            for i in range(len(frames[0])):
                canvas = Image.new("RGB", (512, height), "white")
                draw = ImageDraw.Draw(canvas)
                for j, label in enumerate(["Action prompt", "Stationary prompt"]):
                    canvas.paste(Image.fromarray(frames[j][i]), (j * 256, 30))
                    draw.text((j * 256 + 4, 5), f"{label} | {i / sides[0]['fps']:.1f}s", font=font, fill="black")
                for n, line in enumerate(lines):
                    draw.text((4, 286 + n * 18), line, font=font, fill="black")
                for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        metrics = {}
        for label, pixels in zip(["action", "stationary"], frames, strict=True):
            f = pixels.astype(np.float32)
            metrics[label + "_mean_adjacent_mae"] = float(np.abs(f[1:] - f[:-1]).mean())
            metrics[label + "_mean_first_frame_mae"] = float(np.abs(f[1:] - f[:1]).mean())
            metrics[label + "_last_first_frame_mae"] = float(np.abs(f[-1] - f[0]).mean())
        pair = dict(
            group,
            comparison=str(video),
            videos=[group["previous_action_video"], *map(str, paths)],
            video_sha256=[group["previous_action_sha256"], *map(sha256, paths)],
            frames=len(frames[0]),
            fps=sides[0]["fps"],
            exact_action_repeat=bool(action_repeat),
            exact_stationary_repeat=None if stationary_repeat is None else bool(stationary_repeat),
            first_frames_identical=True,
            metrics=metrics,
            training_eligible=False,
        )
        pairs.append(pair)
    report = dict(
        pairs=pairs,
        numerical_policy=policy,
        scope="Within each seed: same original Nano/image/settings, only text changes. A=action, B=stationary. Pixel motion diagnostics are not task-success scores.",
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(exact_repeats=len(pairs), comparisons=len(pairs))))


if __name__ == "__main__":
    main()
