"""Contract regressions for shared conditioning and label/split admission."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.posttrain.config.rl import PreferenceEnvConfig
from cosmos_framework.posttrain.model.cosmos3 import _Cosmos3FlowCondition
from cosmos_framework.posttrain.roles.env.preference import build_preference_env
from examples.cosmos_lab.i2v_preferences import BridgeI2VFlowPolicy, BridgeI2VPreferences

LAB = Path("/mnt/patient-unit/home/yan/cosmos-lab")
MANIFEST = LAB / "data/bridge-study-v1/manifest.json"
PAIRS = LAB / "data/bridge-preferences-labeled-v1/train.jsonl"


def changed_manifest(tmp_path, mutate):
    rows = [json.loads(line) for line in PAIRS.read_text().splitlines()]
    mutate(rows)
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_real_pairs_keep_authoritative_image_and_plan():
    dataset = BridgeI2VPreferences(PAIRS, MANIFEST)
    assert len(dataset) == 2
    for i, row in enumerate(dataset.rows):
        prompt, winner, loser = dataset[i]
        assert prompt == winner["ai_caption"] == loser["ai_caption"] == row["prompt"]
        assert winner["video"][0].shape == (3, row["frames"], 256, 256)
        assert torch.equal(winner["video"][0][:, 0], loser["video"][0][:, 0])
        assert not torch.equal(winner["video"][0][:, 1:], loser["video"][0][:, 1:])
        assert winner["sequence_plan"].condition_frame_indexes_vision == [0]
        assert loser["sequence_plan"].condition_frame_indexes_vision == [0]


@pytest.mark.parametrize(
    "field,value",
    [("decision", "abstain"), ("training_eligible", False), ("all_frames_reviewed", False), ("confidence", "low")],
)
def test_reject_unaccepted_labels(tmp_path, field, value):
    path = changed_manifest(tmp_path, lambda rows: rows[0]["annotation"].update({field: value}))
    with pytest.raises(ValueError):
        BridgeI2VPreferences(path, MANIFEST)


def test_frozen_manifest_prevents_fake_train_split(tmp_path):
    held_out = next(r for r in json.loads(MANIFEST.read_text())["records"] if r["split"] == "val")
    path = changed_manifest(tmp_path, lambda rows: rows[0].update(pair_id=held_out["uuid"]))
    with pytest.raises(ValueError, match="Held-out"):
        BridgeI2VPreferences(path, MANIFEST)


def test_changed_condition_hash_rejected(tmp_path):
    path = changed_manifest(tmp_path, lambda rows: rows[0].update(condition_image_sha256="0" * 64))
    with pytest.raises(ValueError, match="Condition image hash"):
        BridgeI2VPreferences(path, MANIFEST)


def test_membership_catches_same_episode_with_different_manifest(tmp_path):
    dataset = BridgeI2VPreferences(PAIRS, MANIFEST)
    path = changed_manifest(tmp_path, lambda rows: rows.reverse())
    other = BridgeI2VPreferences(path, MANIFEST)
    assert dataset.cursor_fingerprint() != other.cursor_fingerprint()
    assert set(dataset.membership_ids()) == set(other.membership_ids())


def test_native_factory_rejects_renamed_reordered_validation_overlap(tmp_path):
    other = changed_manifest(tmp_path, lambda rows: rows.reverse())

    def build(path):
        config = PreferenceEnvConfig(
            factory="examples.cosmos_lab.i2v_preferences.BridgeI2VPreferences",
            factory_kwargs={"jsonl_path": str(path), "bridge_manifest_path": str(MANIFEST)},
            shuffle=False,
        )
        return build_preference_env(config, seed=2026, include_source_identity=True)

    train, validation = build(PAIRS), build(other)
    with pytest.raises(ValueError, match="overlap semantically"):
        train.reject_preference_source_overlap(validation.preference_source_identity())


def condition(latent, frames=None):
    return _Cosmos3FlowCondition(
        gen_data_clean=SimpleNamespace(x0_tokens_vision=[latent], fps_vision=torch.tensor([5.0])),
        sequence_plans=[
            SequencePlan(
                has_text=True, has_vision=True, condition_frame_indexes_vision=[0] if frames is None else frames
            )
        ],
        text_tokens=[[1, 2, 3]],
        is_image_batch=False,
        resolutions=["256"],
    )


def test_latent_guard_allows_future_but_rejects_condition_difference():
    policy = object.__new__(BridgeI2VFlowPolicy)
    a = torch.zeros(1, 4, 3, 2, 2)
    b = a.clone()
    b[:, :, 1:] = 1
    policy.assert_same_condition_context(condition(a), condition(b))
    b[:, :, 0] = 1
    with pytest.raises(ValueError, match="conditioning latents differ"):
        policy.assert_same_condition_context(condition(a), condition(b))


def test_no_silent_t2v_fallback():
    policy = object.__new__(BridgeI2VFlowPolicy)
    a = torch.zeros(1, 4, 3, 2, 2)
    with pytest.raises(ValueError, match="exactly first latent frame"):
        policy.assert_same_condition_context(condition(a, []), condition(a, []))
