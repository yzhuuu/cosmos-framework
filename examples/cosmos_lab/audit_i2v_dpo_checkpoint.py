"""Verify a model-only DPO artifact and sample updated/frozen weights against its base."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.posttrain.checkpoint.artifact import read_local_artifact_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read_local_artifact_manifest(str(args.checkpoint))
    assert manifest["iteration"] == 2 and manifest["checkpoint_components"] == ["model"]
    payload = torch.load(args.checkpoint / "model.pt", map_location="cpu", mmap=True, weights_only=True)
    assert set(payload) == {"net"}
    state = payload["net"]
    reader = dcp.FileSystemReader(args.base / "model")
    metadata = reader.read_metadata().state_dict_metadata
    train_keys = ("moe_gen", "time_embedder", "vae2llm", "llm2vae")
    eligible = sorted(k for k in state if "net." + k in metadata and state[k].is_floating_point())
    updated = []
    for marker in train_keys:
        candidates = [k for k in eligible if marker in k]
        # Keep this an explicit, inexpensive sample, not a full tensor-inventory audit.
        updated.extend(sorted(candidates, key=lambda k: (state[k].numel(), k))[:2])
    frozen = [k for k in eligible if not any(marker in k for marker in train_keys)]
    frozen = sorted(frozen, key=lambda k: (state[k].numel(), k))[:8]
    assert len(updated) == 8 and len(frozen) == 8
    base = {
        "net." + k: torch.empty(metadata["net." + k].size, dtype=metadata["net." + k].properties.dtype)
        for k in updated + frozen
    }
    dcp.load(base, storage_reader=reader)
    rows = []
    for k in updated + frozen:
        value, original = state[k], base["net." + k]
        assert value.shape == original.shape and torch.isfinite(value).all()
        delta = (value.float() - original.float()).abs()
        rows.append(
            dict(
                name=k,
                trainable=k in updated,
                elements=math.prod(value.shape),
                changed_elements=int(delta.count_nonzero()),
                max_abs_change=float(delta.max()),
            )
        )
    assert all(row["changed_elements"] == 0 for row in rows if not row["trainable"])
    assert any(row["changed_elements"] > 0 for row in rows if row["trainable"])
    receipt = dict(
        checkpoint=str(args.checkpoint),
        iteration=manifest["iteration"],
        components=manifest["checkpoint_components"],
        total_state_tensors=len(state),
        file_bytes=(args.checkpoint / "model.pt").stat().st_size,
        sampled_tensors=rows,
        scope="8 selected trainable and 8 selected frozen tensors only. Model-only load, not optimizer resume or generation-quality validation.",
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
