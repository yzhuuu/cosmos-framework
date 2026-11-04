"""Create a synchronized, labelled 2x2 viewing copy of Nano/Super T2V/I2V results."""

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nano_run", type=Path)
    parser.add_argument("super_run", type=Path)
    args = parser.parse_args()
    source_files, labels = [], []
    for name, directory in (("Nano", args.nano_run), ("Super", args.super_run)):
        receipt = json.loads((directory / "output-validation.json").read_text())
        assert receipt["status"] == "passed"
        for mode in ("t2v", "i2v"):
            video = directory / "generated" / f"robot_relocation_{mode}" / "vision.mp4"
            assert video.is_file(), video
            record = next(row for row in receipt["outputs"] if Path(row["path"]).resolve() == video.resolve())
            assert record["frames"] == 193 and record["fps"] == 24
            source_files.append(video)
            labels.append(f"{name} {mode.upper()} - shift {10 if name == 'Nano' else 5}")
    target = args.super_run / "nano-super-comparison.mp4"
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    assert font.is_file(), "Install DejaVuSans or adjust this host's font path"
    filters = [
        f"[{i}:v]scale=624:360,setsar=1,pad=624:390:0:30:color=white,"
        f"drawtext=fontfile={font}:text='{label}':fontcolor=black:fontsize=18:x=8:y=5[v{i}]"
        for i, label in enumerate(labels)
    ]
    filters.extend(
        ["[v0][v1]hstack=inputs=2[top]", "[v2][v3]hstack=inputs=2[bottom]", "[top][bottom]vstack=inputs=2[out]"]
    )
    command = ["ffmpeg", "-nostdin", "-n", "-hide_banner", "-loglevel", "error", "-filter_complex_threads", "2"]
    for video in source_files:
        command.extend(["-i", str(video)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    subprocess.run(command, check=True, timeout=120)
    (args.super_run / "comparison-provenance.json").write_text(
        json.dumps(
            {
                "sources": [str(p.resolve()) for p in source_files],
                "labels": labels,
                "output": str(target.resolve()),
                "scope": "scaled viewing copy, synchronized by playback time; retain originals for metrics",
            },
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
