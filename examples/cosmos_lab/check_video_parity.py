# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Check complete decoded videos from plain-entrypoint repeat/export runs."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import av
import numpy as np


def decode(path):
    with av.open(str(path)) as container:
        rate = float(container.streams.video[0].average_rate)
        values = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    return rate, values.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    modes = ["dcp-a", "dcp-b", "hf-a"]
    requests = [json.loads(x) for x in (args.run / "requests.jsonl").read_text().splitlines()]
    checks = []
    for row in requests:
        samples = {mode: args.run / mode / "generated" / row["name"] for mode in modes}
        configurations = [json.loads((p / "sample_args.json").read_text()) for p in samples.values()]
        for config in configurations:
            config.pop("output_dir")
        assert all(c == configurations[0] for c in configurations)
        videos = {mode: decode(p / "vision.mp4") for mode, p in samples.items()}
        for rate, pixels in videos.values():
            assert rate == row["fps"]
            assert len(pixels) == row["num_frames"]
        for a, b in itertools.combinations(modes, 2):
            x, y = videos[a][1], videos[b][1]
            assert x.shape == y.shape
            delta = x - y
            hashes = [hashlib.sha256((samples[m] / "vision.mp4").read_bytes()).hexdigest() for m in (a, b)]
            checks.append(
                {
                    "sample": row["name"],
                    "a": a,
                    "b": b,
                    "byte_equal": hashes[0] == hashes[1],
                    "pixel_mae_0_255": float(np.abs(delta).mean()),
                    "pixel_max_abs": float(np.abs(delta).max()),
                }
            )
    passed = all(row["pixel_max_abs"] == 0 for row in checks)
    receipt = {
        "passed": passed,
        "scope": "Same checkpoint, four fixed requests, same two-H200 runtime; no cross-platform guarantee.",
        "checks": checks,
    }
    (args.run / "parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert passed, "Decoded video mismatch; see parity.json"
    print(f"Passed {len(checks)} full-video comparisons; byte-identical: {sum(row['byte_equal'] for row in checks)}")


if __name__ == "__main__":
    main()
