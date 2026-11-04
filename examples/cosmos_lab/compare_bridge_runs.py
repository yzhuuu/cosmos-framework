"""Build synchronized GT/static/model comparison videos from fixed Bridge runs."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from evaluate_bridge import frames
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", help="Label=/absolute/run/directory")
    args = parser.parse_args()
    runs = [(label, Path(path)) for label, path in (item.split("=", 1) for item in args.runs)]
    requests = [(run / "requests.jsonl").read_text() for _, run in runs]
    assert all(text == requests[0] for text in requests), "Comparison requires identical requests, including seeds"
    rows = [json.loads(line) for line in requests[0].splitlines()]
    dataset = json.loads((runs[0][1] / "dataset-manifest.json").read_text())
    records = {row["uuid"]: row for row in dataset["records"]}
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = []
    for row in rows:
        count = row["num_frames"]
        gt = frames(Path(records[row["name"]]["video"]))[:count]
        static = np.asarray(Image.open(row["vision_path"]).convert("RGB"))
        columns = [("GT", gt), ("Static first frame", [static] * count)]
        for label, run in runs:
            paths = list((run / "generated" / row["name"]).rglob("vision.mp4"))
            assert len(paths) == 1
            column = frames(paths[0])
            assert len(column) == count
            columns.append((label, column))
        width = 256 * len(columns)
        indices = set(np.linspace(0, count - 1, 5, dtype=int).tolist())
        sheet = Image.new("RGB", (width, 280 * len(indices)), "white")
        sheet_index = 0
        video = args.output / f"{row['name']}.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=5)
            stream.width, stream.height, stream.pix_fmt = width, 280, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            for index in range(count):
                canvas = Image.new("RGB", (width, 280), "white")
                draw = ImageDraw.Draw(canvas)
                for col, (label, images) in enumerate(columns):
                    canvas.paste(Image.fromarray(images[index].astype(np.uint8)), (col * 256, 24))
                    draw.text((col * 256 + 4, 4), f"{label} | {index / 5:.1f}s", fill="black")
                if index in indices:
                    sheet.paste(canvas, (0, 280 * sheet_index))
                    sheet_index += 1
                frame = av.VideoFrame.from_image(canvas)
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(video)) as container:
            decoded = list(container.decode(video=0))
            assert len(decoded) == count
            assert all((f.width, f.height) == (width, 280) for f in decoded)
            assert float(container.streams.video[0].average_rate) == 5
        sheet.save(args.output / f"{row['name']}.jpg")
        outputs.append({"name": row["name"], "path": str(video), "frames": count, "fps": 5})
    receipt = {
        "runs": args.runs,
        "columns": ["GT", "Static first frame"] + [name for name, _ in runs],
        "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Created and decoded {len(outputs)} comparison videos in {args.output}")


if __name__ == "__main__":
    main()
