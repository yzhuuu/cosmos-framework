"""Overlay must update real module storage and reject silent partial loads."""

import pytest
import torch

from examples.cosmos_lab.dpo_inference import apply_overlay


def test_copy_updates_forward_and_preserves_explicitly_missing_head():
    net = torch.nn.Linear(3, 2)
    before = net.bias.detach().clone()
    source = {"weight": torch.full_like(net.weight, 2.0)}
    receipt = apply_overlay(net, source, {"bias"})
    assert torch.equal(net.weight, source["weight"])
    assert torch.equal(net.bias, before)
    assert torch.allclose(net(torch.ones(1, 3)), 6 + before)
    assert receipt["verified_local_values"]


@pytest.mark.parametrize("kind", ["missing", "extra", "shape", "nonfinite"])
def test_bad_overlay_is_rejected(kind):
    net = torch.nn.Linear(3, 2)
    source = {k: v.clone() for k, v in net.state_dict().items()}
    if kind == "missing":
        del source["weight"]
    elif kind == "extra":
        source["extra"] = torch.zeros(1)
    elif kind == "shape":
        source["weight"] = torch.zeros(2, 2)
    else:
        source["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        apply_overlay(net, source, set())
