"""Audit trained LoRA B tensors and a sample of frozen base weights in DCP."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def load(path: Path, keys: list[str]) -> dict:
    reader = dcp.FileSystemReader(path)
    meta = reader.read_metadata().state_dict_metadata
    state = {
        k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype) if hasattr(meta[k], "size") else None for k in keys
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
    adapters = [k for k in cm if ".lora_B." in k]
    assert adapters and all(k not in bm for k in adapters)
    values = load(args.checkpoint / "model", adapters)
    adapter_rows = []
    for name, value in values.items():
        assert torch.isfinite(value).all()
        adapter_rows.append({"name": name, "elements": value.numel(), "nonzero": int(value.count_nonzero())})
    assert all(r["nonzero"] for r in adapter_rows)
    small = [k for k, v in cm.items() if k.startswith("net.") and k in bm and math.prod(v.size) <= 4096][:6]
    projections = [k for k in cm if k in bm and k.endswith("q_proj_moe_gen.weight")][:2]
    frozen = small + projections
    assert len(small) == 6 and len(projections) == 2
    before = load(args.base / "model", frozen)
    after = load(args.checkpoint / "model", frozen)
    assert all(torch.equal(before[k], after[k]) for k in frozen)
    optimizer_meta = dcp.FileSystemReader(args.checkpoint / "optim").read_metadata().state_dict_metadata
    parameter_states = [k for k in optimizer_meta if k.startswith("state.")]
    assert parameter_states and all("lora_" in k for k in parameter_states)
    trainer = load(args.checkpoint / "trainer", ["iteration"])
    receipt = {
        "scope": "All LoRA B tensors checked finite/nonzero against documented zero initialization; sampled 8 frozen tensors only.",
        "adapters": adapter_rows,
        "frozen_tensors_bitwise_equal": frozen,
        "optimizer_only_lora": True,
        "optimizer_state_entries": len(parameter_states),
        "iteration": trainer["iteration"],
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "adapters"}, indent=2))
    print(f"Checked {len(adapter_rows)} nonzero finite LoRA B tensors")


if __name__ == "__main__":
    main()
