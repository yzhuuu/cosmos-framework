"""Measure full decoded native-LoRA vs dense-export differences without an inferred pass threshold."""

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert (args.native / "requests.jsonl").read_bytes() == (args.dense / "requests.jsonl").read_bytes()
    assert (args.native / "generated/runtime-numerics.json").read_bytes() == (
        args.dense / "generated/runtime-numerics.json"
    ).read_bytes()
    checks = []
    for row in [json.loads(x) for x in (args.native / "requests.jsonl").read_text().splitlines()]:
        paths = [p / "generated" / row["name"] / "vision.mp4" for p in (args.native, args.dense)]
        values = []
        for p in paths:
            with av.open(str(p)) as container:
                assert float(container.streams.video[0].average_rate) == row["fps"]
                value = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)]).astype(np.float32)
                assert len(value) == row["num_frames"]
                values.append(value)
        x, y = values
        assert x.shape == y.shape
        delta = x - y
        per_frame = np.abs(delta).mean(axis=(1, 2, 3))
        checks.append(
            {
                "sample": row["name"],
                "frames": len(x),
                "byte_equal": hashlib.sha256(paths[0].read_bytes()).digest()
                == hashlib.sha256(paths[1].read_bytes()).digest(),
                "mae_0_255": float(np.abs(delta).mean()),
                "rmse_0_255": float(np.sqrt(np.square(delta).mean())),
                "max_abs": float(np.abs(delta).max()),
                "first_frame_mae": float(per_frame[0]),
                "generated_frames_mae": float(per_frame[1:].mean()),
                "per_frame_mae": per_frame.tolist(),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "checks": checks,
                "scope": "Full decoded paired videos, same requests/seed/numerical policy. No automatic perceptual-equivalence verdict.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps([{k: v for k, v in x.items() if k != "per_frame_mae"} for x in checks], indent=2))


if __name__ == "__main__":
    main()
