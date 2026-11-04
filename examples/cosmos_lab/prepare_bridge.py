"""Freeze a small official Bridge subset with disjoint episodes and media checks."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
from huggingface_hub import hf_hub_download

REVISION = "40d018ac1c1a2a4b9734f17fdb21f3d933c49a01"
REPO = "nvidia/BridgeData2-Subset-Synthetic-Captions"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(name: str) -> str:
    return hashlib.sha256(f"cosmos-lab-2026:{name}".encode()).hexdigest()


def main() -> None:
    root = Path(os.environ["COSMOS_LAB_ROOT"])
    source = root / "data/bridge-source"
    target = root / "data/bridge-study-v1"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists():
        raise FileExistsError("Frozen dataset exists; use its recorded inputs")
    info = json.loads((root / "runs/preflight/bridge-assets.json").read_text())
    assert info["revision"] == REVISION
    paths = {r["path"] for r in info["files"]}
    train_manifest = source / "sft_dataset_bridge/train/video_dataset_file.jsonl"
    train_rows = [json.loads(s) for s in train_manifest.read_text().splitlines()]
    # Select one clip per episode; never place related clips in another split.
    groups = {}
    for row in sorted(train_rows, key=lambda r: r["uuid"]):
        groups.setdefault(row["uuid"].split("_clip")[0], row)
    training = [groups[k] for k in sorted(groups, key=rank)[:32]]
    validation_ids = sorted({Path(p).stem for p in paths if "/val/videos/" in p}, key=rank)
    selected_val, seen = [], set(groups)
    for name in validation_ids:
        episode = name.split("_clip")[0]
        if episode not in seen:
            selected_val.append(name)
            seen.add(episode)
        if len(selected_val) == 12:
            break
    assert len(selected_val) == 12
    selections = {"train": [r["uuid"] for r in training], "val": selected_val[:4], "test": selected_val[4:]}
    required = [f"sft_dataset_bridge/train/{r['vision_path']}" for r in training]
    for name in selected_val:
        required.extend(
            f"sft_dataset_bridge/val/{kind}/{name}{extension}"
            for kind, extension in [
                ("videos", ".mp4"),
                ("images", ".jpg"),
                ("inference_prompt_i2v", ".json"),
            ]
        )
    assert set(required) <= paths

    def download(path: str) -> None:
        hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION, local_dir=source)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, required))
    records, requests = [], {"val": [], "test": []}
    all_hashes = set()
    for split, names in selections.items():
        origin = "train" if split == "train" else "val"
        for name in names:
            video = source / f"sft_dataset_bridge/{origin}/videos/{name}.mp4"
            digest = sha(video)
            assert digest not in all_hashes, "duplicate media across selected samples"
            all_hashes.add(digest)
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                stamps, shapes = [], set()
                for frame in container.decode(stream):
                    stamps.append(float(frame.pts * frame.time_base))
                    shapes.add((frame.width, frame.height))
            assert shapes == {(256, 256)} and fps == 5 and len(stamps) > 40
            assert all(abs(t - stamps[0] - i / fps) < 1e-4 for i, t in enumerate(stamps))
            records.append(
                {
                    "uuid": name,
                    "episode": name.split("_clip")[0],
                    "split": split,
                    "source_split": origin,
                    "video": str(video),
                    "sha256": digest,
                    "frames": len(stamps),
                    "fps": fps,
                    "geometry": [256, 256],
                }
            )
            if split != "train":
                original = source / f"sft_dataset_bridge/val/inference_prompt_i2v/{name}.json"
                row = json.loads(original.read_text())
                # Preserve official caption and duration; resolve its image locally.
                row.update(
                    name=name,
                    vision_path=str(source / f"sft_dataset_bridge/val/images/{name}.jpg"),
                    seed=2026,
                    num_steps=35,
                    guidance=6.0,
                    shift=10.0,
                    enable_sound=False,
                    prompt_upsampling=False,
                )
                assert row["num_frames"] <= len(stamps), (name, row["num_frames"], len(stamps))
                assert row["fps"] == 5
                requests[split].append(row)
                records[-1]["prompt_source"] = str(original)
                records[-1]["prompt_source_sha256"] = sha(original)
    directory = target / "train"
    directory.mkdir(exist_ok=True)
    for row in training:
        row["vision_path"] = str(source / "sft_dataset_bridge/train" / row["vision_path"])
    (directory / "video_dataset_file.jsonl").write_text("".join(json.dumps(r) + "\n" for r in training))
    for split, rows in requests.items():
        (target / f"{split}-requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "selection": "sha256(cosmos-lab-2026:episode), one clip per episode",
        "split_counts": {k: len(v) for k, v in selections.items()},
        "records": records,
        "train_source_sha256": sha(train_manifest),
        "limits": "Synthetic captions are unverified. Official val is subdivided into local val/test. Not proven absent from model pretraining.",
        "protocol": "Fixed I2V seed 2026, official full prompt. Test sealed from training; use val for iteration. Compare first-frame repetition. PSNR is diagnostic, not task success.",
    }
    manifest["artifacts"] = {str(p.relative_to(target)): sha(p) for p in target.rglob("*.jsonl")}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["split_counts"]))


if __name__ == "__main__":
    main()
