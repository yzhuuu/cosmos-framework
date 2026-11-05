"""Render every A/B decoded frame chronologically, preserving the review coverage."""

import argparse
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    out = args.review / "temporal"
    out.mkdir(exist_ok=False)
    for pair in json.loads((args.review / "manifest.json").read_text())["pairs"]:
        frames = []
        for path in pair["videos"][1:]:
            with av.open(path) as container:
                frames.append([frame.to_image() for frame in container.decode(video=0)])
        assert len(frames[0]) == len(frames[1]) == pair["frames"]
        for start in range(0, len(frames[0]), 20):
            sheet = Image.new("RGB", (1024, 760), "white")
            draw = ImageDraw.Draw(sheet)
            for j, i in enumerate(range(start, min(start + 20, len(frames[0])))):
                x, y = j % 4 * 256, j // 4 * 152
                draw.text((x + 2, y + 2), f"{i} / {i / pair['fps']:.1f}s: A | B", fill="black")
                for k in range(2):
                    sheet.paste(frames[k][i].resize((128, 128)), (x + 128 * k, y + 22))
            sheet.save(out / f"{pair['pair_id']}-{start:03d}.png")


if __name__ == "__main__":
    main()
