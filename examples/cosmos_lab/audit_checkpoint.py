"""Read selected DCP tensors and optimizer/scheduler state without loading the model."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def load(path: Path, keys: list[str]) -> dict:
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata().state_dict_metadata
    state = {
        k: torch.empty(metadata[k].size, dtype=metadata[k].properties.dtype) if hasattr(metadata[k], "size") else None
        for k in keys
    }
    dcp.load(state, storage_reader=reader)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bm = dcp.FileSystemReader(args.base / "model").read_metadata().state_dict_metadata
    cm = dcp.FileSystemReader(args.checkpoint / "model").read_metadata().state_dict_metadata
    selected = [k for k, v in cm.items() if k.startswith("net.") and "moe_gen" in k and math.prod(v.size) <= 4096][:6]
    frozen = [
        k
        for k, v in cm.items()
        if k.startswith("net.language_model.") and "moe_gen" not in k and k in bm and math.prod(v.size) <= 4096
    ][:6]
    assert len(selected) == len(frozen) == 6
    keys = selected + frozen
    before, after = load(args.base / "model", keys), load(args.checkpoint / "model", keys)
    changes = [
        {
            "name": k,
            "group": "selected" if k in selected else "frozen",
            "elements": after[k].numel(),
            "max_absolute_delta": float((before[k].float() - after[k].float()).abs().max()),
            "changed_elements": int((before[k].float() != after[k].float()).sum()),
        }
        for k in keys
    ]
    assert any(r["changed_elements"] for r in changes if r["group"] == "selected")
    assert all(r["changed_elements"] == 0 for r in changes if r["group"] == "frozen")
    state = {}
    for component in ("scheduler", "trainer", "optim"):
        meta = dcp.FileSystemReader(args.checkpoint / component).read_metadata().state_dict_metadata
        names = [k for k in meta if k.endswith(("last_epoch", "_step_count", "iteration", ".step"))][:5]
        state[component] = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in load(args.checkpoint / component, names).items()
        }
    receipt = {
        "scope": "Sampled 6 selected and 6 frozen tensors, not exhaustive equality of all frozen weights.",
        "tensors": changes,
        "state": state,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
