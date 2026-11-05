"""Freeze and download new official-val episodes before any source/output review."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import av
from huggingface_hub import hf_hub_download

from examples.cosmos_lab.prepare_bridge import REPO, REVISION, rank
from examples.cosmos_lab.prepare_preferences import condition_digest, sha256, validate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError("Bound source audit to 1–16 new episodes")
    original_path = args.lab / "data/bridge-study-v1/manifest.json"
    original = json.loads(original_path.read_text())
    inventory = json.loads((args.lab / "runs/preflight/bridge-assets.json").read_text())
    if inventory["revision"] != REVISION or original["revision"] != REVISION:
        raise ValueError("Dataset revision mismatch")
    source_root = args.lab / "data/bridge-source"
    train_rows = map(
        json.loads, (source_root / "sft_dataset_bridge/train/video_dataset_file.jsonl").read_text().splitlines()
    )
    excluded = {r["uuid"].split("_clip")[0] for r in train_rows}
    excluded.update(r["episode"] for r in original["records"])
    groups = {}
    for item in sorted(inventory["files"], key=lambda r: r["path"]):
        if "/val/videos/" in item["path"] and item["path"].endswith(".mp4"):
            name = Path(item["path"]).stem
            episode = name.split("_clip")[0]
            if episode not in excluded:
                groups.setdefault(episode, name)
    selected = [groups[e] for e in sorted(groups, key=rank)[: args.count]]
    if len(selected) != args.count:
        raise ValueError("Not enough disjoint official validation episodes")
    args.destination.mkdir(parents=True, exist_ok=False)
    selection = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": REPO,
        "revision": REVISION,
        "rule": "One lexically first clip per new official val episode; sha256(cosmos-lab-2026:episode) order",
        "selected": selected,
        "excluded_episodes": sorted(excluded),
        "original_manifest_sha256": sha256(original_path),
        "scope": "New local development validation; not a sealed final test or proven absent from pretraining",
    }
    # This is deliberately persisted BEFORE fetching or inspecting source footage.
    (args.destination / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    pairs, requests = [], []
    seen_hashes = {r["sha256"] for r in original["records"]}
    for name in selected:
        paths = {}
        for kind, extension in [("videos", ".mp4"), ("inference_prompt_i2v", ".json")]:
            relative = f"sft_dataset_bridge/val/{kind}/{name}{extension}"
            paths[kind] = Path(
                hf_hub_download(REPO, relative, repo_type="dataset", revision=REVISION, local_dir=source_root)
            )
        video = paths["videos"]
        digest = sha256(video)
        if digest in seen_hashes:
            raise ValueError("Duplicate source media")
        seen_hashes.add(digest)
        with av.open(str(video)) as container:
            fps = float(container.streams.video[0].average_rate)
            frames = list(container.decode(video=0))
        stamps = [float(f.pts * f.time_base) for f in frames]
        if fps != 5 or len(frames) <= 40 or {(f.width, f.height) for f in frames} != {(256, 256)}:
            raise ValueError("Unexpected source format")
        if any(abs(t - stamps[0] - i / fps) > 1e-4 for i, t in enumerate(stamps)):
            raise ValueError("Nonuniform source timestamps")
        image = args.destination / f"{name}-first.png"
        frames[0].to_image().save(image)
        source = dict(
            uuid=name,
            episode=name.split("_clip")[0],
            split="val",
            source_split="val",
            video=str(video),
            sha256=digest,
            frames=len(frames),
            fps=fps,
            geometry=[256, 256],
            prompt_source=str(paths["inference_prompt_i2v"]),
            prompt_source_sha256=sha256(paths["inference_prompt_i2v"]),
        )
        official = json.loads(paths["inference_prompt_i2v"].read_text())
        prompt = official["prompt"]
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt)
        caption = json.loads(prompt)
        sides = [
            dict(
                name=f"{name}-{label}",
                prompt=prompt,
                vision_path=str(image.resolve()),
                seed=seed,
                resolution="256",
                aspect_ratio="1,1",
                num_frames=(len(frames) - 1) // 4 * 4 + 1,
                fps=fps,
                num_steps=35,
                guidance=6.0,
                shift=10.0,
                enable_sound=False,
                prompt_upsampling=False,
            )
            for label, seed in [("a", 2026), ("b", 2027)]
        ]
        validate_pair(*sides, source, expected_split="val")
        requests.extend(sides)
        pairs.append(
            dict(
                pair_id=name,
                source=source,
                candidates=[s["name"] for s in sides],
                condition_sha256=condition_digest(sides[0]),
                image_sha256=sha256(image),
                actions=caption.get("actions", []),
                decision="unreviewed",
                training_eligible=False,
            )
        )
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        split="val",
        selection=selection["rule"],
        selection_sha256=sha256(args.destination / "selection.json"),
        bridge_manifest_sha256=sha256(original_path),
        checkpoint=str(args.lab / "checkpoints/Cosmos3-Nano"),
        numerical_policy="fixed-cudnn-v1",
        seeds=[2026, 2027],
        scope=selection["scope"],
        pairs=pairs,
    )
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.destination / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    print(json.dumps(dict(source_episodes=len(pairs), destination=str(args.destination))))


if __name__ == "__main__":
    main()
