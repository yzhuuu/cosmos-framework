"""Render source footage for caption auditing, preserving every source frame."""

import argparse
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

from examples.cosmos_lab.prepare_preferences import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    output = args.study / "source-audit"
    output.mkdir(exist_ok=False)
    for pair in json.loads((args.study / "manifest.json").read_text())["pairs"]:
        source = pair["source"]
        if sha256(source["video"]) != source["sha256"]:
            raise ValueError("Source video changed")
        with av.open(source["video"]) as container:
            frames = [f.to_image() for f in container.decode(video=0)]
        if len(frames) != source["frames"]:
            raise ValueError("Source frame count differs")
        for start in range(0, len(frames), 40):
            sheet = Image.new("RGB", (1024, 760), "white")
            draw = ImageDraw.Draw(sheet)
            for j, index in enumerate(range(start, min(start + 40, len(frames)))):
                x, y = j % 8 * 128, j // 8 * 152
                sheet.paste(frames[index].resize((128, 128)), (x, y + 24))
                draw.text((x + 2, y + 4), f"{index} / {index / source['fps']:.1f}s", fill="black")
            sheet.save(output / f"{pair['pair_id']}-{start:03d}.png")
        sheet = Image.new("RGB", (768, 840), "white")
        draw = ImageDraw.Draw(sheet)
        for j in range(9):
            index = round(j * (len(frames) - 1) / 8)
            x, y = j % 3 * 256, j // 3 * 280
            sheet.paste(frames[index], (x, y + 24))
            draw.text((x + 2, y + 4), f"Source {index / source['fps']:.1f}s", fill="black")
        sheet.save(output / f"{pair['pair_id']}-native.png")


if __name__ == "__main__":
    main()
