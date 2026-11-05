"""Independently verify every merged projection against native DCP tensors."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.export / "config.json").read_text())["model"]["config"]
    assert config["lora_enabled"] is False
    scale = config["lora_alpha"] / config["lora_rank"]
    index = json.loads((args.export / "model.safetensors.index.json").read_text())["weight_map"]
    assert not any(".lora_A." in k or ".lora_B." in k for k in index)
    reader = dcp.FileSystemReader(args.checkpoint / "model")
    metadata = reader.read_metadata().state_dict_metadata
    prefixes = sorted(k.removesuffix(".lora_A.weight") for k in metadata if k.endswith(".lora_A.weight"))
    assert len(prefixes) == 256
    keys = [p + suffix for p in prefixes for suffix in (".weight", ".lora_A.weight", ".lora_B.weight")]
    frozen = [
        k
        for k, v in metadata.items()
        if k.startswith("net.")
        and k.endswith(".weight")
        and k.removesuffix(".weight") not in prefixes
        and v.size.numel() <= 8192
    ][:8]
    assert len(frozen) == 8
    keys += frozen
    state = {k: torch.empty(metadata[k].size, dtype=metadata[k].properties.dtype) for k in keys}
    dcp.load(state, storage_reader=reader)
    checks = []
    for prefix in prefixes:
        name = "model." + prefix + ".weight"
        with safe_open(args.export / index[name], framework="pt") as handle:
            actual = handle.get_tensor(name)
        # Native export loads the DCP into the configured inference dtype first.
        dtype = actual.dtype
        base, a, b = [state[prefix + suffix].to(dtype) for suffix in (".weight", ".lora_A.weight", ".lora_B.weight")]
        expected = (base.float() + scale * (b.float() @ a.float())).to(dtype)
        checks.append(
            {
                "name": name,
                "dtype": str(dtype),
                "equal": torch.equal(actual, expected),
                "changed_elements_vs_base": int((actual != base).sum()),
                "max_formula_error": float((actual.float() - expected.float()).abs().max()),
            }
        )
    frozen_checks = []
    for key in frozen:
        name = "model." + key
        with safe_open(args.export / index[name], framework="pt") as handle:
            actual = handle.get_tensor(name)
        frozen_checks.append({"name": name, "equal": torch.equal(actual, state[key].to(actual.dtype))})
    passed = all(r["equal"] and r["changed_elements_vs_base"] > 0 for r in checks) and all(
        r["equal"] for r in frozen_checks
    )
    args.output.write_text(
        json.dumps(
            {
                "passed": passed,
                "merged_modules": len(checks),
                "adapter_keys": 0,
                "scope": "All 256 merged projections; 8 frozen tensors sampled. FP32 merge after native inference-dtype load, not direct FP32 master-weight merge.",
                "checks": checks,
                "frozen": frozen_checks,
            },
            indent=2,
        )
        + "\n"
    )
    assert passed, "Dense export audit failed; see receipt"
    print(f"Passed all {len(checks)} merged projections and {len(frozen_checks)} frozen samples")


if __name__ == "__main__":
    main()
