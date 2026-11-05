"""Validate all fixed inputs and build synchronized Base / beta5000 / beta500 evidence."""

import argparse
import html
import json
import textwrap
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.cosmos_lab.review_dpo_generation import decode, effective, sha

ARMS = ("base", "beta5000", "beta500")
LABELS = ("Original Nano", "DPO beta=5000, step 12", "DPO beta=500, step 12")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    lab = Path("/mnt/patient-unit/home/yan/cosmos-lab")
    requests = [json.loads(line) for line in (args.run / "requests.jsonl").read_text().splitlines()]
    assert len(requests) == 8
    frozen = lab / "data/bridge-preference-validation-v2"
    assert sha(args.run / "requests.jsonl") == sha(frozen / "requests.jsonl")
    pairs = json.loads((frozen / "manifest.json").read_text())["pairs"]
    goals = {name: pair["review_goal"] for pair in pairs for name in pair["candidates"]}
    for pair in pairs:
        request = next(row for row in requests if row["name"] == pair["candidates"][0])
        assert sha(Path(request["vision_path"])) == pair["image_sha256"]
    output = args.run / "review"
    if not args.baseline_only:
        output.mkdir(exist_ok=False)
        (output / "temporal").mkdir()
    records, sections = [], []
    header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    goal_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for req in requests:
        name = req["name"]
        old = lab / "runs/preference-candidates-493/generated"
        effective(old, req)
        effective(args.run / "base", req)
        base_path = args.run / "base" / name / "vision.mp4"
        base = decode(base_path, req)
        assert np.array_equal(base, decode(old / name / "vision.mp4", req)), f"Baseline drift: {name}"
        row = dict(
            name=name,
            seed=req["seed"],
            frames=len(base),
            fps=req["fps"],
            baseline_repeat_pixel_exact=True,
            baseline_sha256=sha(base_path),
            goal=goals[name],
        )
        if not args.baseline_only:
            videos = [base]
            row["arms"] = {}
            for arm in ARMS[1:]:
                effective(args.run / arm, req)
                path = args.run / arm / name / "vision.mp4"
                pixels = decode(path, req)
                assert np.array_equal(base[0], pixels[0]), f"Condition frame differs: {name}/{arm}"
                delta = np.abs(base.astype(np.float32) - pixels.astype(np.float32))
                row["arms"][arm] = dict(
                    sha256=sha(path),
                    mean_pixel_abs_change=float(delta.mean()),
                    differing_frames=int(np.any(base != pixels, axis=(1, 2, 3)).sum()),
                )
                videos.append(pixels)
            lines = textwrap.wrap(goals[name], width=83)
            height = 290 + 22 * len(lines)
            path = output / f"{name}.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=int(req["fps"]))
                stream.width, stream.height, stream.pix_fmt = 768, height, "yuv420p"
                for index in range(len(base)):
                    canvas = Image.new("RGB", (768, height), "white")
                    draw = ImageDraw.Draw(canvas)
                    for col, (label, pixels) in enumerate(zip(LABELS, videos, strict=True)):
                        draw.text((col * 256 + 4, 7), label, font=header, fill="black")
                        canvas.paste(Image.fromarray(pixels[index]), (col * 256, 30))
                    for j, line in enumerate(lines):
                        draw.text((4, 288 + 22 * j), line, font=goal_font, fill="black")
                    draw.text(
                        (680, 270),
                        f"{index / req['fps']:.1f}s",
                        font=header,
                        fill="white",
                        stroke_width=1,
                        stroke_fill="black",
                    )
                    for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(str(path)) as container:
                assert sum(1 for _ in container.decode(video=0)) == len(base)
            row["comparison"] = str(path)
            for start in range(0, len(base), 20):
                sheet = Image.new("RGB", (1536, 760), "white")
                draw = ImageDraw.Draw(sheet)
                for j, index in enumerate(range(start, min(start + 20, len(base)))):
                    x, y = j % 4 * 384, j // 4 * 152
                    draw.text((x + 2, y + 2), f"{index} / {index / req['fps']:.1f}s Base | 5000 | 500", fill="black")
                    for col, pixels in enumerate(videos):
                        sheet.paste(Image.fromarray(pixels[index]).resize((128, 128)), (x + col * 128, y + 22))
                sheet.save(output / "temporal" / f"{name}-{start:03d}.png")
            indices = np.linspace(0, len(base) - 1, 6, dtype=int).tolist()
            detail = Image.new("RGB", (1536, 840), "white")
            draw = ImageDraw.Draw(detail)
            for r, (label, pixels) in enumerate(zip(LABELS, videos, strict=True)):
                for c, index in enumerate(indices):
                    draw.text((c * 256 + 2, r * 280 + 3), f"{label} | {index / req['fps']:.1f}s", fill="black")
                    detail.paste(Image.fromarray(pixels[index]), (c * 256, r * 280 + 24))
            detail.save(output / f"{name}-detail.png")
            row["native_detail_frames"] = indices
            sections.append(
                f"<h2>{html.escape(name)} / seed {req['seed']}</h2><p>{html.escape(goals[name])}</p>"
                f'<video controls preload="metadata" src="{name}.mp4" width="100%"></video>'
            )
        records.append(row)
    payload = dict(records=records, scope="Matched fixed final checkpoints; pixel changes are not task scores.")
    (args.run / ("baseline-repeat.json" if args.baseline_only else "comparison.json")).write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    if not args.baseline_only:
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Yuzh beta study</title>'
            '<main style="max-width:1100px;margin:30px auto;font-family:sans-serif">'
            "<h1>Original Nano / beta 5000 / beta 500</h1><p>Both trained models: 12 updates. "
            "Each row uses the same text, image and seed. No automatic quality score.</p>"
            + "\n".join(sections)
            + "</main>"
        )
    print(f"Validated {len(records)} complete comparisons; baseline_only={args.baseline_only}")


if __name__ == "__main__":
    main()
