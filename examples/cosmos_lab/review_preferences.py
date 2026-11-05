"""Build synchronized source/A/B review media without assigning preference labels."""

import argparse
import html
import json
import textwrap
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from examples.cosmos_lab.prepare_preferences import condition_digest, sha256, validate_effective, validate_pair


def decode(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "candidate-manifest.json").read_text())
    requests = {r["name"]: r for r in map(json.loads, (args.run / "requests.jsonl").read_text().splitlines())}
    policy = json.loads((args.run / "generated/runtime-numerics.json").read_text())
    if policy.get("cudnn_benchmark") is not False or policy.get("cudnn_deterministic") is not True:
        raise ValueError("Expected fixed cuDNN policy")
    output = args.run / "review"
    output.mkdir(exist_ok=False)
    summaries, sections = [], []
    for pair in manifest["pairs"]:
        left, right = [requests[name] for name in pair["candidates"]]
        validate_pair(left, right, pair["source"], expected_split=manifest.get("split", "train"))
        if condition_digest(left) != pair["condition_sha256"]:
            raise ValueError("Conditions changed since freeze")
        if sha256(pair["source"]["video"]) != pair["source"]["sha256"]:
            raise ValueError("Source changed since freeze")
        paths = [Path(pair["source"]["video"])]
        for request in [left, right]:
            effective = json.loads((args.run / "generated" / request["name"] / "sample_args.json").read_text())
            validate_effective(request, effective)
            candidates = list((args.run / "generated" / request["name"]).rglob("vision.mp4"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one generated video: {candidates}")
            paths.append(candidates[0])
        frames = [decode(path) for path in paths]
        count = left["num_frames"]
        if len(frames[1]) != count or len(frames[2]) != count or len(frames[0]) < count:
            raise ValueError("Unexpected frame counts")
        same_first = np.array_equal(frames[1][0], frames[2][0])
        if not same_first:
            raise ValueError("Candidate first frames differ; review conditioning before pairing")
        video = output / f"{pair['pair_id']}.mp4"
        labels = ["Source (caption audit)", "Candidate A", "Candidate B"]
        goal_lines = textwrap.wrap(pair.get("review_goal", ""), width=110)
        height = 286 + 18 * len(goal_lines)
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=int(left["fps"]))
            stream.width, stream.height, stream.pix_fmt = 768, height, "yuv420p"
            for index in range(count):
                canvas = Image.new("RGB", (768, height), "white")
                draw = ImageDraw.Draw(canvas)
                for col in range(3):
                    canvas.paste(Image.fromarray(frames[col][index]), (col * 256, 30))
                    draw.text((col * 256 + 5, 8), f"{labels[col]} | {index / left['fps']:.1f}s", fill="black")
                for line_index, line in enumerate(goal_lines):
                    draw.text((5, 288 + line_index * 18), line, fill="black")
                for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        indices = np.linspace(0, count - 1, 9, dtype=int).tolist()
        sheet = Image.new("RGB", (256 * 9, 286 * 3), "white")
        draw = ImageDraw.Draw(sheet)
        for row in range(3):
            for column, index in enumerate(indices):
                x, y = column * 256, row * 286
                sheet.paste(Image.fromarray(frames[row][index]), (x, y + 30))
                draw.text((x + 3, y + 8), f"{labels[row]} {index / left['fps']:.1f}s", fill="black")
        sheet_path = output / f"{pair['pair_id']}.jpg"
        sheet.save(sheet_path)
        summaries.append(
            {
                "pair_id": pair["pair_id"],
                "comparison": str(video),
                "contact_sheet": str(sheet_path),
                "videos": [str(p) for p in paths],
                "video_sha256": [sha256(p) for p in paths],
                "condition_sha256": pair["condition_sha256"],
                "first_frames_identical": same_first,
                "sampled_frame_indices": indices,
                "frames": count,
                "fps": left["fps"],
                "decision": "unreviewed",
                "training_eligible": False,
                "review_goal": pair.get("review_goal", ""),
                "review_questions": pair.get("review_questions", []),
            }
        )
        actions = html.escape(json.dumps(pair["actions"], indent=2))
        prompt = html.escape(left["prompt"])
        sections.append(
            f"<section><h2>{pair['pair_id']}</h2><p>Decision: unreviewed; consult manifest for split</p>"
            f'<video controls preload="metadata" src="{video.name}" width="100%"></video>'
            f"<p>{html.escape(pair.get('review_goal', ''))}</p><pre>{actions}</pre>"
            f"<details><summary>Exact full prompt</summary><pre>{prompt}</pre></details>"
            f'<a href="{sheet_path.name}">Timestamped contact sheet</a></section>'
        )
    (output / "manifest.json").write_text(json.dumps({"pairs": summaries, "numerical_policy": policy}, indent=2) + "\n")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Candidate review</title>'
        "<style>body{max-width:1100px;margin:40px auto;font:17px sans-serif}"
        "pre{white-space:pre-wrap}section{margin:50px 0}</style>"
        "<h1>I2V candidate review</h1><p>Source | A | B. No winner labels assigned. "
        "Source footage is a scene reference, not a required pixel trajectory.</p>" + "".join(sections)
    )
    print(output)


if __name__ == "__main__":
    main()
