# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compare recorded network state, inputs, VAE encodes, noise and output pixels."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import av
import numpy as np
import torch


def video(path):
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    modes = sorted(p for p in args.run.iterdir() if (p / "probe").is_dir())
    results = []
    for a, b in itertools.combinations(modes, 2):
        result = {"a": a.name, "b": b.name, "ranks": [], "videos": []}
        for rank in (0, 1):
            rows = [
                [json.loads(x) for x in (p / "probe" / f"rank-{rank}.jsonl").read_text().splitlines()] for p in (a, b)
            ]
            row = {"rank": rank}
            for event in ("parameter", "buffer"):
                states = [{x["name"]: x for x in seq if x["event"] == event} for seq in rows]
                assert states[0].keys() == states[1].keys()
                row[event] = {
                    "count": len(states[0]),
                    "different": [k for k in states[0] if states[0][k] != states[1][k]],
                }
            for event in ("sample_input", "initial_noise", "sample_output", "vae_encode"):
                selected = [[x for x in seq if x["event"] == event] for seq in rows]
                assert len(selected[0]) == len(selected[1])
                if event == "sample_input":
                    row["sample_input_equal"] = [
                        x["data"] == y["data"] and x["options"] == y["options"] for x, y in zip(*selected)
                    ]
                else:
                    row[event + "_equal"] = [x == y for x, y in zip(*selected)]
            row["encode_differences"] = []
            for i, p in enumerate(sorted((a / "probe").glob(f"encode-rank-{rank}-*.pt"))):
                q = b / "probe" / p.name
                if not q.exists():
                    continue
                x = torch.load(p, map_location="cpu", weights_only=True).float()
                y = torch.load(q, map_location="cpu", weights_only=True).float()
                delta = x - y
                row["encode_differences"].append(
                    {"index": i, "mae": float(delta.abs().mean()), "max_abs": float(delta.abs().max())}
                )
            result["ranks"].append(row)
        for p in sorted((a / "generated").glob("*/vision.mp4")):
            q = b / "generated" / p.parent.name / "vision.mp4"
            x, y = video(p), video(q)
            assert x.shape == y.shape
            delta = x - y
            result["videos"].append(
                {
                    "sample": p.parent.name,
                    "byte_equal": hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(q.read_bytes()).digest(),
                    "pixel_mae_0_255": float(np.abs(delta).mean()),
                    "pixel_rmse_0_255": float(np.sqrt((delta**2).mean())),
                }
            )
        results.append(result)
    (args.run / "comparison-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["a"], result["b"], [(v["sample"], round(v["pixel_mae_0_255"], 6)) for v in result["videos"]])


if __name__ == "__main__":
    main()
