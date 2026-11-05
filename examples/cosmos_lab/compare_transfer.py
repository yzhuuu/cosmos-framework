# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build a synchronized source/control/appearance comparison at native FPS."""

import argparse
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw


def frames(path):
    with av.open(str(path)) as container:
        return [frame.to_image() for frame in container.decode(video=0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--modality", choices=["depth", "seg"], default="depth")
    args = parser.parse_args()
    rows = [json.loads(x) for x in (args.run / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    control = Path(rows[0][args.modality]["control_path"])
    source = control.with_name("example2.mp4")
    paths = [source, control] + [args.run / "generated" / row["name"] / "vision.mp4" for row in rows]
    videos = [frames(path) for path in paths]
    assert all(len(video) == 121 for video in videos)
    labels = ["Source RGB (resized for display)", args.modality + " control", "Black dress prompt", "Red dress prompt"]
    width, height, label_height = 416, 240, 24
    panels = []
    for i in range(121):
        canvas = Image.new("RGB", (4 * width, height + label_height), "white")
        draw = ImageDraw.Draw(canvas)
        for col, video in enumerate(videos):
            resample = Image.Resampling.NEAREST if col == 1 and args.modality == "seg" else Image.Resampling.BICUBIC
            canvas.paste(video[i].resize((width, height), resample), (col * width, label_height))
            draw.text((col * width + 4, 4), labels[col] + f" | {i / 30:.1f}s", fill="black")
        panels.append(canvas)
    output = args.run / "comparison.mp4"
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = panels[0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for panel in panels:
            for packet in stream.encode(av.VideoFrame.from_image(panel)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert len(frames(output)) == 121
    sheet = Image.new("RGB", (4 * width, 5 * (height + label_height)), "white")
    for row, index in enumerate((0, 30, 60, 90, 120)):
        sheet.paste(panels[index], (0, row * (height + label_height)))
    sheet.save(args.run / "comparison.jpg")
    (args.run / "comparison.json").write_text(
        json.dumps(
            {
                "inputs": [str(p.resolve()) for p in paths],
                "frames": 121,
                "fps": 30,
                "display": "Source/control resized for panel display; generated files remain native resolution.",
                "scope": "Qualitative appearance and structure comparison; no task-success metric.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
