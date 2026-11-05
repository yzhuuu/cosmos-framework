"""Strictly restore a native trainable-head checkpoint onto its original frozen Pi05 base."""

from pathlib import Path


def restore_heads(policy, checkpoint: Path):
    import torch
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.posttrain.checkpoint.io import is_complete_iter_dir

    if not is_complete_iter_dir(str(checkpoint)):
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    before = {key: value.detach().cpu().clone() for key, value in policy.net.state_dict().items()}
    # Native DCP stores the flat model state, unlike consolidated model.pt['net'].
    state = {key: value.clone() for key, value in before.items()}
    reader = dcp.FileSystemReader(str(checkpoint / "model"))
    metadata = reader.read_metadata().state_dict_metadata
    expected = set(before)
    if set(metadata) != expected:
        raise ValueError(
            f"Checkpoint keys differ: missing={expected - set(metadata)}, extra={set(metadata) - expected}"
        )
    for key, value in before.items():
        descriptor = metadata[key]
        if tuple(descriptor.size) != tuple(value.shape) or descriptor.properties.dtype != value.dtype:
            raise ValueError(f"Shape/dtype mismatch for {key}")
    dcp.load(state, storage_reader=reader)
    rows = []
    for key, old in before.items():
        new = state[key]
        if not torch.isfinite(new).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
        delta = new.float() - old.float()
        rows.append(
            {
                "name": key,
                "dtype": str(new.dtype),
                "numel": new.numel(),
                "changed": int(torch.count_nonzero(new != old)),
                "delta_l2": float(delta.double().square().sum().sqrt()),
                "delta_max": float(delta.abs().max()),
                "group": "value" if key.startswith("value_head.") else "actor",
            }
        )
    frozen_before = policy.ppo_frozen_state()
    policy.net.load_state_dict(state, strict=True)
    for key, value in policy.net.state_dict().items():
        if not torch.equal(value.cpu(), state[key]):
            raise ValueError(f"Reload mismatch: {key}")
    if policy.ppo_frozen_state() != frozen_before:
        raise ValueError("Frozen VLM changed while restoring trainable heads")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "exact_reload": True,
        "frozen_vlm_unchanged_during_reload": True,
        "tensors": rows,
    }
