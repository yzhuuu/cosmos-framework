"""Check matched inference settings and preserve every before/after comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def decode(path, request):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert float(stream.average_rate) == request["fps"]
        assert len(frames) == request["num_frames"]
        assert all(abs(float(f.time) - i / request["fps"]) < 1e-4 for i, f in enumerate(frames))
        pixels = np.stack([f.to_ndarray(format="rgb24") for f in frames])
        assert pixels.shape[1:] == (256, 256, 3)
        return pixels


def effective(root, request):
    actual = json.loads((root / request["name"] / "sample_args.json").read_text())
    for key in ("seed", "prompt", "num_frames", "fps", "num_steps", "guidance", "shift", "resolution", "aspect_ratio"):
        if actual[key] != request[key]:
            raise ValueError(f"Runtime {key} changed: {request['name']}")
    if Path(actual["vision_path"]).resolve() != Path(request["vision_path"]).resolve():
        raise ValueError("Condition image path changed")
    policy = json.loads((root / "runtime-numerics.json").read_text())
    assert not policy["cudnn_benchmark"] and policy["cudnn_deterministic"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    lab = Path("/mnt/patient-unit/home/yan/cosmos-lab")
    old = lab / "runs/preference-candidates-479/generated"
    requests = list(map(json.loads, (args.run / "requests.jsonl").read_text().splitlines()))
    assert len(requests) == 4
    records = []
    review = args.run / "review"
    if not args.baseline_only:
        review.mkdir(exist_ok=False)
    for req in requests:
        name = req["name"]
        effective(args.run / "base", req)
        base_path = args.run / "base" / name / "vision.mp4"
        base, reference = decode(base_path, req), decode(old / name / "vision.mp4", req)
        if not np.array_equal(base, reference):
            raise ValueError(f"Baseline repeat is not pixel-exact: {name}")
        row = dict(name=name, seed=req["seed"], frames=len(base), fps=req["fps"], baseline_repeat_pixel_exact=True)
        if not args.baseline_only:
            effective(args.run / "dpo", req)
            dpo_path = args.run / "dpo" / name / "vision.mp4"
            dpo = decode(dpo_path, req)
            assert np.array_equal(base[0], dpo[0]), "Condition-frame output changed"
            delta = np.abs(base.astype(np.float32) - dpo.astype(np.float32))
            row.update(
                baseline_sha256=sha(base_path),
                dpo_sha256=sha(dpo_path),
                mean_pixel_abs_change=float(delta.mean()),
                max_pixel_abs_change=float(delta.max()),
                differing_frames=int(np.any(base != dpo, axis=(1, 2, 3)).sum()),
                comparison=str(review / f"{name}.mp4"),
            )
            with av.open(str(review / f"{name}.mp4"), "w") as container:
                stream = container.add_stream("libx264", rate=int(req["fps"]))
                stream.width, stream.height, stream.pix_fmt = 512, 286, "yuv420p"
                for i in range(len(base)):
                    canvas = Image.new("RGB", (512, 286), "white")
                    draw = ImageDraw.Draw(canvas)
                    for col, (label, video) in enumerate([("Base", base), ("DPO step 2", dpo)]):
                        draw.text((col * 256 + 4, 8), f"{label} | {i / req['fps']:.1f}s", fill="black")
                        canvas.paste(Image.fromarray(video[i]), (col * 256, 30))
                    for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            for start in range(0, len(base), 20):
                sheet = Image.new("RGB", (1024, 760), "white")
                draw = ImageDraw.Draw(sheet)
                for j, i in enumerate(range(start, min(start + 20, len(base)))):
                    x, y = j % 4 * 256, j // 4 * 152
                    draw.text((x + 2, y + 2), f"{i} / {i / req['fps']:.1f}s Base | DPO", fill="black")
                    for col, video in enumerate([base, dpo]):
                        sheet.paste(Image.fromarray(video[i]).resize((128, 128)), (x + col * 128, y + 22))
                sheet.save(review / f"{name}-{start:03d}.png")
        records.append(row)
    filename = "baseline-repeat.json" if args.baseline_only else "comparison.json"
    (args.run / filename).write_text(
        json.dumps(
            dict(records=records, scope="Pixel differences are diagnostics, not task success or preference scores."),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
