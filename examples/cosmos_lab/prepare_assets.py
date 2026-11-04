"""Download pinned educational samples and optionally the Nano checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REVISIONS = {
    "nvidia/Cosmos3-Super": "f543c56225b2e04d0ad141e29655be3a45d9c455",
    "nvidia/Cosmos3-Nano": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
    "nvidia/Cosmos3-Nano-Sim-Bimanual": "a9e3adf6235f56f0082be7a96e497348c286c472",
    "nvidia/Cosmos3-Nano-Sim-Transfer": "42da1b57840cec0fd2b642546b72cb95fdc331af",
}


def main() -> None:
    """Materialize assets and record hashes without downloading simulator weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, required=True)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--nano-weights", action="store_true")
    weights.add_argument("--super-weights", action="store_true")
    args = parser.parse_args()
    root = args.lab_root.resolve()
    selections = {
        "nvidia/Cosmos3-Nano-Sim-Bimanual": [
            "assets/training/*",
            "examples/training/*",
            "config.json",
            "transformer/config.json",
        ],
        "nvidia/Cosmos3-Nano-Sim-Transfer": ["assets/sample_data/*"],
    }
    if args.nano_weights:
        selections = {"nvidia/Cosmos3-Nano": None}
    elif args.super_weights:
        selections = {"nvidia/Cosmos3-Super": None}
    receipts = []
    for repo_id, patterns in selections.items():
        revision = REVISIONS[repo_id]
        category = "checkpoints" if args.nano_weights or args.super_weights else "data"
        destination = root / category / repo_id.split("/")[-1]
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=patterns,
            local_dir=destination,
            max_workers=4,
        )
        records = []
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or ".cache" in path.relative_to(destination).parts:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            records.append(
                {"path": str(path.relative_to(destination)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
            )
        receipts.append({"repo_id": repo_id, "revision": revision, "directory": str(destination), "files": records})
        print(f"Prepared {repo_id}@{revision}: {len(records)} files", flush=True)
    receipt_name = (
        "super-download.json"
        if args.super_weights
        else ("nano-download.json" if args.nano_weights else "sample-downloads.json")
    )
    output = root / "runs" / "preflight" / receipt_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
