"""Stream-verify the artifact and warm the compute-node file cache before mmap loading."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.checkpoint / ".posttrain-checkpoint-artifact.json").read_text())
    expected = manifest["components"]["model"]
    started = time.monotonic()
    digest = hashlib.sha256()
    count = 0
    with (args.checkpoint / "model.pt").open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
            count += len(block)
    if count != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("Model artifact content does not match committed manifest")
    report = dict(bytes=count, sha256=digest.hexdigest(), manifest_matches=True, seconds=time.monotonic() - started)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
