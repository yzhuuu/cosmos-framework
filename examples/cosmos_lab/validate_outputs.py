"""Fully decode baseline outputs and save contact sheets and basic motion diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    requests = [json.loads(line) for line in (args.run_dir / "requests.jsonl").read_text().splitlines()]
    records = []
    for row in requests:
        videos = list((args.run_dir / "generated" / row["name"]).rglob("vision.mp4"))
        assert len(videos) == 1, videos
        path = videos[0]
        previous = None
        differences, timestamps, geometry = [], [], set()
        samples = {}
        chosen = set(np.linspace(0, row["num_frames"] - 1, 9, dtype=int).tolist())
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            for index, frame in enumerate(container.decode(stream)):
                assert frame.pts is not None and frame.time_base is not None
                timestamps.append(float(frame.pts * frame.time_base))
                geometry.add((frame.width, frame.height))
                rgb = frame.to_ndarray(format="rgb24")
                if previous is not None:
                    differences.append(float(np.abs(rgb.astype(np.float32) - previous).mean()))
                previous = rgb.astype(np.float32)
                if index in chosen:
                    samples[index] = Image.fromarray(rgb).resize((416, 240))
        assert len(timestamps) == row["num_frames"], (path, len(timestamps))
        assert fps == row["fps"] and len(geometry) == 1, path
        assert all(abs(t - timestamps[0] - i / fps) < 1e-4 for i, t in enumerate(timestamps)), path
        sheet = Image.new("RGB", (416 * 3, 268 * 3), "white")
        draw = ImageDraw.Draw(sheet)
        for tile, (index, frame) in enumerate(samples.items()):
            x, y = tile % 3 * 416, tile // 3 * 268
            sheet.paste(frame, (x, y))
            draw.text((x + 8, y + 244), f"Frame {index}, {index / fps:.2f}s", fill="black")
        sheet.save(args.run_dir / f"{row['name']}-contact-sheet.jpg")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(
            {
                "path": str(path),
                "sha256": digest,
                "frames": len(timestamps),
                "fps": fps,
                "geometry": list(geometry)[0],
                "mean_adjacent_frame_abs_difference_0_255": float(np.mean(differences)),
                "max_adjacent_frame_abs_difference_0_255": max(differences),
            }
        )
    receipt = {
        "status": "passed",
        "outputs": records,
        "scope": "complete decode, expected frame count and timing; pixel differences do not measure task success",
    }
    (args.run_dir / "output-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
