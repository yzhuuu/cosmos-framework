"""Score fixed Bridge I2V outputs against GT and static conditioning controls."""

import argparse
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        assert float(container.streams.video[0].average_rate) == 5
        output = []
        for index, frame in enumerate(container.decode(video=0)):
            assert (frame.width, frame.height) == (256, 256)
            assert abs(float(frame.pts * frame.time_base) - index / 5) < 1e-4
            output.append(frame.to_ndarray(format="rgb24").astype(np.float32))
        return output


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.square(a - b).mean())
    return 10 * math.log10(255**2 / mse) if mse else 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    dataset = json.loads((run / "dataset-manifest.json").read_text())
    records = {r["uuid"]: r for r in dataset["records"]}
    rows = [json.loads(s) for s in (run / "requests.jsonl").read_text().splitlines()]
    results = []
    for row in rows:
        record = records[row["name"]]
        gt = frames(Path(record["video"]))
        paths = list((run / "generated" / row["name"]).rglob("vision.mp4"))
        assert len(paths) == 1
        pred = frames(paths[0])
        assert len(pred) == row["num_frames"] and 0 <= len(gt) - len(pred) < 4
        static = np.asarray(Image.open(row["vision_path"]).convert("RGB")).astype(np.float32)
        assert static.shape == pred[0].shape
        scores = [psnr(p, g) for p, g in zip(pred[1:], gt[1 : len(pred)], strict=True)]
        control = [psnr(static, g) for g in gt[1 : len(pred)]]
        indices = np.linspace(0, len(pred) - 1, 5, dtype=int)
        sheet = Image.new("RGB", (768, 280 * len(indices)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, index in enumerate(indices):
            for col, (label, value) in enumerate(
                zip(("GT", "Prediction", "Static"), (gt[index], pred[index], static), strict=True)
            ):
                sheet.paste(Image.fromarray(value.astype(np.uint8)), (col * 256, j * 280))
                draw.text((col * 256 + 4, j * 280 + 258), f"{label} | {index / 5:.1f}s", fill="black")
        sheet.save(run / f"{row['name']}-paired.jpg")
        results.append(
            {
                "name": row["name"],
                "split": record["split"],
                "source_frames": len(gt),
                "predicted_frames": len(pred),
                "excluded_conditioning_frames": 1,
                "gt_tail_not_scored": len(gt) - len(pred),
                "prediction_psnr_db": float(np.mean(scores)),
                "static_psnr_db": float(np.mean(control)),
                "motion_mean_abs_difference": float(
                    np.mean([np.abs(a - b).mean() for a, b in zip(pred[1:], pred[:-1], strict=True)])
                ),
                "prediction_per_frame_psnr_db": scores,
            }
        )
    receipt = {
        "scope": "I2V diagnostic, not action success. Native 4k+1 duration may omit up to 3 GT tail frames; no resampling. Infinite PSNR capped at 100 dB.",
        "results": results,
    }
    (run / "paired-evaluation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if "per_frame" not in k} for r in results], indent=2))


if __name__ == "__main__":
    main()
