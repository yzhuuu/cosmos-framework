"""Reject invalid comparisons before any candidate is labeled for training."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare_preferences", Path(__file__).with_name("prepare_preferences.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def pair(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"condition-image")
    left = {"name": "a", "seed": 2026, "prompt": "Move left", "vision_path": str(image), "fps": 5}
    return left, dict(left, name="b", seed=2027), {"split": "train"}


def test_same_condition_seeds_allowed(pair):
    module.validate_pair(*pair)


@pytest.mark.parametrize("field,value", [("prompt", "Move right"), ("fps", 24), ("seed", 2026), ("name", "a")])
def test_mismatch_rejected(pair, field, value):
    left, right, source = copy.deepcopy(pair)
    right[field] = value
    with pytest.raises(ValueError):
        module.validate_pair(left, right, source)


@pytest.mark.parametrize("split", ["val", "test"])
def test_held_out_rejected(pair, split):
    left, right, _ = pair
    with pytest.raises(ValueError):
        module.validate_pair(left, right, {"split": split})


def test_different_image_rejected(pair, tmp_path):
    left, right, source = pair
    image = tmp_path / "other.png"
    image.write_bytes(b"different-condition")
    right["vision_path"] = str(image)
    with pytest.raises(ValueError):
        module.validate_pair(left, right, source)


def test_global_seed_override_rejected():
    request = dict(name="b", seed=2027, prompt="move", num_frames=93, fps=5, num_steps=35, guidance=6, shift=10)
    module.validate_effective(request, dict(request))
    with pytest.raises(ValueError, match="Effective seed"):
        module.validate_effective(request, dict(request, seed=2026))
