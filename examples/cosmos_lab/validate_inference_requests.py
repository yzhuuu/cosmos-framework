"""Validate frozen requests using the native inference schema before GPU submission."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cosmos_framework.inference.args import OmniSampleOverrides
from examples.cosmos_lab.prepare_preferences import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    args = parser.parse_args()
    samples = OmniSampleOverrides.from_files([args.requests])
    if len({sample.name for sample in samples}) != len(samples):
        raise ValueError("Duplicate output names")
    raw = list(map(json.loads, args.requests.read_text().splitlines()))
    for row in raw:
        if "seed" not in row or not Path(row["vision_path"]).is_file():
            raise ValueError("Require fixed seed and a local condition image")
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        request_sha256=sha256(args.requests),
        native_schema="OmniSampleOverrides.from_files",
        validated_samples=len(samples),
        guidance_values=sorted({r["guidance"] for r in raw}),
        status="passed",
        scope="Native request schema and local inputs; does not load model or guarantee runtime success",
    )
    args.requests.with_suffix(".preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
