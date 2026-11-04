"""Compare complete paired rollouts to cropped GT and a repeated initial image."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    pairs = json.loads((args.run_dir / "pairs.json").read_text())["pairs"]
    results = []
    for pair in pairs:
        name = pair["name"]
        gt = Path(pair["files"]["gt"]["path"])
        image = Path(pair["files"]["vision_path"]["path"])
        predictions = list((args.run_dir / "generated" / name).rglob("vision.mp4"))
        assert len(predictions) == 1
        prediction = predictions[0]
        static = np.asarray(Image.open(image).convert("RGB"))[:352, :640].astype(np.float32)
        rows, psnr_pred, psnr_static = [], [], []
        chosen = {0, 30, 150, 300, 450, 600, 900}
        with av.open(str(gt)) as gc, av.open(str(prediction)) as pc:
            gs, ps = gc.streams.video[0], pc.streams.video[0]
            assert float(gs.average_rate) == float(ps.average_rate) == 30
            for index, (gf, pf) in enumerate(zip(gc.decode(gs), pc.decode(ps), strict=True)):
                assert (gf.width, gf.height) == (640, 360)
                assert (pf.width, pf.height) == (640, 352)
                assert abs(float(gf.pts * gf.time_base) - index / 30) < 1e-4
                assert abs(float(pf.pts * pf.time_base) - index / 30) < 1e-4
                truth = gf.to_ndarray(format="rgb24")[:352].astype(np.float32)
                pred = pf.to_ndarray(format="rgb24").astype(np.float32)
                if index:
                    for value, destination in ((pred, psnr_pred), (static, psnr_static)):
                        mse = float(np.square(value - truth).mean())
                        # JSON-safe null denotes a mathematically infinite PSNR.
                        destination.append(10 * math.log10(255**2 / mse) if mse else None)
                if index in chosen:
                    rows.append(
                        (index, [Image.fromarray(x.astype(np.uint8)).resize((320, 176)) for x in (truth, pred, static)])
                    )
        assert index == 900
        sheet = Image.new("RGB", (960, len(rows) * 200), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (index, images) in enumerate(rows):
            for col, (label, picture) in enumerate(
                zip(("GT", "Prediction", "Static initial image"), images, strict=True)
            ):
                sheet.paste(picture, (col * 320, j * 200))
                draw.text((col * 320 + 4, j * 200 + 178), f"{label} | {index / 30:.1f}s", fill="black")
        sheet.save(args.run_dir / f"{name}-paired.jpg")
        output = args.run_dir / f"{name}-comparison.mp4"
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        filters = []
        for i, label in enumerate(("Ground truth", "Bimanual prediction", "Repeated initial image")):
            filters.append(
                f"[{i}:v]crop=640:352:0:0,pad=640:380:0:28:color=white,drawtext=fontfile={font}:text='{label}':x=8:y=5:fontsize=18:fontcolor=black[v{i}]"
            )
        filters.append("[v0][v1][v2]hstack=inputs=3[v]")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-threads",
                "1",
                "-i",
                str(gt),
                "-i",
                str(prediction),
                "-loop",
                "1",
                "-framerate",
                "30",
                "-i",
                str(image),
                "-filter_complex_threads",
                "1",
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[v]",
                "-frames:v",
                "901",
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        with av.open(str(output)) as container:
            assert sum(1 for _ in container.decode(video=0)) == 901
        assert all(v is not None for v in psnr_pred + psnr_static)
        results.append(
            {
                "name": name,
                "frames_scored": [1, 900],
                "prediction_mean_frame_psnr_db": float(np.mean(psnr_pred)),
                "static_mean_frame_psnr_db": float(np.mean(psnr_static)),
                "prediction_per_frame_psnr_db": psnr_pred,
                "static_per_frame_psnr_db": psnr_static,
                "comparison": str(output),
            }
        )
    (args.run_dir / "paired-evaluation.json").write_text(
        json.dumps(
            {
                "scope": "Public paired integration samples. Not held-out. Pixel PSNR is not task success.",
                "geometry": "GT crop [0:352,0:640]; no resampling; RGB decoded uint8, framewise PSNR averaged in dB",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps([{k: v for k, v in r.items() if "per_frame" not in k} for r in results], indent=2))


if __name__ == "__main__":
    main()
