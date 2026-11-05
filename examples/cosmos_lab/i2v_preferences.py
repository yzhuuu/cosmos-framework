"""Strict single-view Bridge I2V preference data for the native dataset factory."""

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.posttrain.model.cosmos3 import Cosmos3FlowPolicy


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class BridgeI2VPreferences:
    """No resampling/caption dropout; exact full clips and an explicit shared first frame.

    Only reviewed training episodes are accepted. Membership aliases cover source
    episodes as well as generated media, so re-generating seeds cannot hide leakage.
    """

    def __init__(self, jsonl_path, bridge_manifest_path):
        self.path = Path(jsonl_path)
        self.bridge_path = Path(bridge_manifest_path)
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        source = {r["uuid"]: r for r in json.loads(self.bridge_path.read_text())["records"]}
        if not self.rows:
            raise ValueError("No accepted preferences")
        if len({r["episode"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate episode")
        for row in self.rows:
            original = source[row["pair_id"]]
            if original["split"] != "train" or row["source_split"] != "train":
                raise ValueError("Held-out episode cannot enter training")
            if row["episode"] != original["episode"] or row["source_sha256"] != original["sha256"]:
                raise ValueError("Source identity differs from frozen Bridge manifest")
            annotation = row["annotation"]
            if annotation["decision"] not in {"a", "b"} or not annotation["training_eligible"]:
                raise ValueError("Unaccepted preference")
            if annotation["confidence"] not in {"medium", "high"} or not annotation["all_frames_reviewed"]:
                raise ValueError("Insufficient annotation evidence")
            if not annotation["evidence"] or annotation["reviewer_type"] != "agent":
                raise ValueError("Expected documented agent annotation")
            if row["winner_video"] == row["loser_video"] or row["winner_sha256"] == row["loser_sha256"]:
                raise ValueError("Identical candidates")
            if row["frames"] < 5 or row["frames"] % 4 != 1 or row["fps"] <= 0:
                raise ValueError("Expected genuine timing and 4n+1 frames")
            for side in ("winner", "loser"):
                if digest(row[f"{side}_video"]) != row[f"{side}_sha256"]:
                    raise ValueError("Candidate media hash mismatch")
            if digest(row["condition_image"]) != row["condition_image_sha256"]:
                raise ValueError("Condition image hash mismatch")
        self._fingerprint = digest(self.path) + ":" + digest(self.bridge_path)

    def __len__(self):
        return len(self.rows)

    def cursor_fingerprint(self):
        return "bridge-i2v-fullclip-v1:" + self._fingerprint

    def membership_ids(self):
        return tuple(
            sorted(
                {
                    value
                    for row in self.rows
                    for value in (
                        "bridge-episode:" + row["episode"],
                        "source-sha256:" + row["source_sha256"],
                        "condition-sha256:" + row["condition_image_sha256"],
                        "media-sha256:" + row["winner_sha256"],
                        "media-sha256:" + row["loser_sha256"],
                    )
                }
            )
        )

    def __getitem__(self, index):
        row = self.rows[index]
        # Recheck mutable local paths at use time, not just construction.
        if digest(row["condition_image"]) != row["condition_image_sha256"]:
            raise ValueError("Condition changed after manifest validation")
        image = torch.from_numpy(np.array(Image.open(row["condition_image"]).convert("RGB"))).permute(2, 0, 1)
        sides = []
        for side in ("winner", "loser"):
            path = row[f"{side}_video"]
            if digest(path) != row[f"{side}_sha256"]:
                raise ValueError("Candidate changed after manifest validation")
            with av.open(path) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                frames = list(container.decode(stream))
                times = [float(frame.time) for frame in frames]
                pixels = torch.from_numpy(np.stack([f.to_ndarray(format="rgb24") for f in frames])).permute(3, 0, 1, 2)
            if len(frames) != row["frames"] or fps != row["fps"]:
                raise ValueError("Frame count/FPS mismatch; temporal resampling is forbidden")
            if any(abs(t - times[0] - i / fps) > 1e-4 for i, t in enumerate(times)):
                raise ValueError("Irregular timestamps")
            if pixels.shape[2:] != image.shape[1:] or tuple(image.shape[1:]) != (256, 256):
                raise ValueError("Expected matching 256-square condition and outputs")
            # Lossy MP4 reconstruction of frame 0 is not the authoritative input I.
            pixels[:, 0] = image
            sides.append(
                {
                    "video": [pixels.contiguous()],
                    "ai_caption": row["prompt"],
                    "image_size": torch.tensor([256, 256, 256, 256], dtype=torch.float),
                    "conditioning_fps": torch.tensor([fps], dtype=torch.bfloat16),
                    "sequence_plan": SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0]),
                }
            )
        return row["prompt"], *sides


class BridgeI2VFlowPolicy(Cosmos3FlowPolicy):
    """Narrow single-view adapter: indices alone do not prove shared conditioning."""

    def assert_same_condition_context(self, winner, loser):
        super().assert_same_condition_context(winner, loser)
        plans = winner.sequence_plans
        left = winner.gen_data_clean.x0_tokens_vision
        right = loser.gen_data_clean.x0_tokens_vision
        if len(left) != len(right) or len(left) != len(plans):
            raise ValueError("I2V requires one vision item per sample")
        for a, b, plan in zip(left, right, plans, strict=True):
            if list(plan.condition_frame_indexes_vision) != [0]:
                raise ValueError("I2V requires exactly first latent frame as condition")
            if a.ndim != 5 or a.shape != b.shape or a.shape[0] != 1:
                raise ValueError("Unexpected I2V latent shape")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Nonfinite I2V latents")
            if not torch.equal(a[:, :, :1], b[:, :, :1]):
                raise ValueError("I2V conditioning latents differ")
