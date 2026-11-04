"""Decode every sample frame and check hashes, action masks, and control alignment."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import av
from PIL import Image

from cosmos_framework.inference.bimanual import load_bimanual_action_contract


def decode_video(path: Path) -> tuple[dict, list]:
    """Return decoded geometry and exact normalized timestamps, rejecting bad media."""
    timestamps = []
    geometry = set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            assert frame.pts is not None and frame.time_base is not None, path
            timestamps.append(frame.pts * frame.time_base)
            geometry.add((frame.width, frame.height))
    assert timestamps and len(geometry) == 1, path
    assert all(b > a for a, b in zip(timestamps, timestamps[1:])), path
    normalized = [time - timestamps[0] for time in timestamps]
    assert all(abs(float(time) - index / fps) < 1e-4 for index, time in enumerate(normalized)), path
    width, height = geometry.pop()
    return {"path": str(path), "frames": len(timestamps), "fps": fps, "width": width, "height": height}, normalized


def main() -> None:
    """Validate released assets without asserting model quality or dataset generalization."""
    root = Path(os.environ["COSMOS_LAB_ROOT"])
    bimanual = root / "data/Cosmos3-Nano-Sim-Bimanual"
    checksum_count = 0
    for line in (bimanual / "examples/training/SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        with (bimanual / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
        checksum_count += 1
    contract = load_bimanual_action_contract(bimanual)
    assert contract is not None
    rows = [
        json.loads(line)
        for line in (bimanual / "examples/training/sim_bimanual_agibot_examples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    robots = []
    for row in rows:
        payload = json.loads((bimanual / row["action_path"]).read_text())
        actions, domain_ids = contract.validate(
            payload, action_count=row["action_chunk_size"], domain_name=row["domain_name"]
        )
        media, _ = decode_video(bimanual / row["ground_truth_path"])
        assert media["frames"] == row["num_frames"] == len(actions) + 1, row["name"]
        assert media["fps"] == row["fps"], row["name"]
        with Image.open(bimanual / row["vision_path"]) as image:
            assert image.size == (media["width"], media["height"]), row["name"]
        robots.append(
            {
                "name": row["name"],
                **media,
                "actions": len(actions),
                "action_dim": contract.action_dim,
                "domain_counts": dict(Counter(domain_ids)),
            }
        )
    transfer = []
    excluded_controls = []
    for folder in sorted((root / "data/Cosmos3-Nano-Sim-Transfer/assets/sample_data").iterdir()):
        base, clock = decode_video(folder / f"{folder.name}.mp4")
        controls = []
        for path in sorted(folder.glob(f"{folder.name}_*.mp4")):
            info, times = decode_video(path)
            assert times == clock, path
            assert all(info[key] == base[key] for key in ("frames", "fps")), path
            if any(info[key] != base[key] for key in ("width", "height")):
                excluded_controls.append(
                    {**info, "reason": "geometry differs from source; exclude until resize/crop contract is verified"}
                )
                continue
            controls.append(info)
        transfer.append({"source": base, "controls": controls})
    receipt = {
        "status": "passed_with_exclusions" if excluded_controls else "passed",
        "official_checksums_verified": checksum_count,
        "bimanual": robots,
        "transfer": transfer,
        "excluded_controls": excluded_controls,
        "scope": "asset integrity, full decode, frame alignment, native action contract; not model quality",
    }
    (root / "runs/preflight/sample-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
