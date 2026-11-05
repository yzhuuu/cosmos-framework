# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare a fixed-control, one-attribute Transfer comparison from official assets."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    skill = Path(__file__).resolve().parents[2] / ".agents/skills/cosmos3-nano-sim-transfer"
    prepare = runpy.run_path(str(skill / "scripts/prepare_hf_sample.py"))["prepare"]
    prepare("example2", args.source, args.output / "native")
    rows = []
    for modality in ("depth", "seg"):
        base = json.loads((args.output / "native/requests" / f"{modality}.json").read_text())
        assert "black dress" in base["prompt"]
        for color in ("black", "red"):
            row = {**base, "name": f"example2_{modality}_{color}", "seed": 2026}
            row["prompt"] = base["prompt"].replace("black dress", f"{color} dress")
            rows.append(row)
    args.output.mkdir(exist_ok=True)
    for modality in ("depth", "seg"):
        path = args.output / f"{modality}-requests.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows if f"_{modality}_" in row["name"]))
    files = []
    for p in sorted(args.source.glob("example2*")):
        files.append({"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {
        "revision": "42da1b57840cec0fd2b642546b72cb95fdc331af",
        "scope": "Official example2; fixed control/seed, change only black dress to red dress.",
        "files": files,
        "requests": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared four requests; first run uses the two depth requests.")


if __name__ == "__main__":
    main()
