# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stream synchronized source/control/appearance panels at native FPS."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--modality", choices=["depth", "seg"], default="depth")
    args = parser.parse_args()
    rows = [json.loads(x) for x in (args.run / "requests.jsonl").read_text().splitlines()]
    manifest = json.loads((args.run / "dataset-manifest.json").read_text())
    assert len(rows) == 2
    control = Path(rows[0][args.modality]["control_path"])
    assert rows[0][args.modality] == rows[1][args.modality]
    source = Path(manifest.get("source_rgb", str(control.with_name("example2.mp4"))))
    paths = [source, control] + [args.run / "generated" / row["name"] / "vision.mp4" for row in rows]
    count = min(rows[0]["num_frames"], rows[0].get("max_frames", rows[0]["num_frames"]))
    fps = rows[0]["fps"]
    labels = ["Source RGB", args.modality + " control"] + manifest.get("labels", ["Black dress", "Red dress"])
    width = 320 if rows[0]["aspect_ratio"] == "1,1" else 416
    height, label_height = (320 if width == 320 else 240), 24
    selected = set(np.linspace(0, count - 1, 9, dtype=int).tolist())
    sheet = Image.new("RGB", (4 * width, len(selected) * (height + label_height)), "white")
    output = args.run / "comparison.mp4"
    tile = 0
    with ExitStack() as stack:
        containers = [stack.enter_context(av.open(str(path))) for path in paths]
        assert all(float(c.streams.video[0].average_rate) == fps for c in containers)
        decoders = [iter(c.decode(video=0)) for c in containers]
        target = stack.enter_context(av.open(str(output), "w"))
        stream = target.add_stream("libx264", rate=int(fps))
        stream.width, stream.height = 4 * width, height + label_height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for index in range(count):
            canvas = Image.new("RGB", (4 * width, height + label_height), "white")
            draw = ImageDraw.Draw(canvas)
            for col, decoder in enumerate(decoders):
                frame = next(decoder)
                assert abs(float(frame.pts * frame.time_base) - index / fps) < 1e-4
                resample = Image.Resampling.NEAREST if col == 1 and args.modality == "seg" else Image.Resampling.BICUBIC
                canvas.paste(frame.to_image().resize((width, height), resample), (col * width, label_height))
                draw.text((col * width + 4, 4), labels[col] + f" | {index / fps:.2f}s", fill="black")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                target.mux(packet)
            if index in selected:
                sheet.paste(canvas, (0, tile * (height + label_height)))
                tile += 1
        assert all(next(decoder, None) is None for decoder in decoders)
        for packet in stream.encode():
            target.mux(packet)
    with av.open(str(output)) as container:
        assert float(container.streams.video[0].average_rate) == fps
        assert sum(1 for _ in container.decode(video=0)) == count
    sheet.save(args.run / "comparison.jpg")
    (args.run / "comparison.json").write_text(
        json.dumps(
            {
                "inputs": [str(p.resolve()) for p in paths],
                "labels": labels,
                "frames": count,
                "fps": fps,
                "display": "Source/control resized for panel display; generated files remain native resolution.",
                "scope": "Qualitative appearance and structure comparison; no task-success metric.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
