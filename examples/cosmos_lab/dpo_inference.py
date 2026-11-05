"""Load a strict model-only DPO overlay into the original Nano inference runtime."""

import functools
import json
import os
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

MISSING_HEADS = {
    "llm2sound.weight",
    "llm2sound.bias",
    "sound2llm.weight",
    "sound2llm.bias",
    "sound_modality_embed",
    "action2llm.bias.weight",
    "action2llm.fc.weight",
    "action_modality_embed",
    "llm2action.bias.weight",
    "llm2action.fc.weight",
}
TRAIN_KEYS = ("moe_gen", "time_embedder", "vae2llm", "llm2vae")


def apply_overlay(net, source, allowed_missing):
    """Copy one global tensor at a time into actual local parameter/buffer storage."""
    target = net.state_dict()
    missing, unexpected = set(target) - set(source), set(source) - set(target)
    if missing != set(allowed_missing) or unexpected:
        raise ValueError(f"Overlay inventory mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, value in source.items():
        if target[name].shape != value.shape:
            raise ValueError(f"Overlay shape mismatch: {name}")
    preserved = {k: (target[k].to_local() if isinstance(target[k], DTensor) else target[k]).clone() for k in missing}
    rows = []
    with torch.no_grad():
        for index, name in enumerate(sorted(source), 1):
            dest = target[name]
            local = dest.to_local() if isinstance(dest, DTensor) else dest
            full = source[name].to(device=local.device, dtype=local.dtype)
            if not torch.isfinite(full).all():
                raise ValueError(f"Nonfinite checkpoint tensor: {name}")
            expected = (
                distribute_tensor(full, dest.device_mesh, dest.placements, src_data_rank=None).to_local()
                if isinstance(dest, DTensor)
                else full
            )
            changed = int((local != expected).count_nonzero())
            local.copy_(expected)
            if not torch.equal(local, expected):
                raise ValueError(f"Overlay copy failed: {name}")
            rows.append(
                dict(
                    name=name,
                    dtype=str(local.dtype),
                    checkpoint_dtype=str(source[name].dtype),
                    changed_local_elements=changed,
                    local_elements=local.numel(),
                    trainable=any(key in name for key in TRAIN_KEYS),
                )
            )
            del full, expected
            if index % 200 == 0 and int(os.environ.get("RANK", "0")) == 0:
                print(f"Verified overlay tensors: {index}/{len(source)}", flush=True)
    after = net.state_dict()
    for name, before in preserved.items():
        value = after[name].to_local() if isinstance(after[name], DTensor) else after[name]
        if not torch.equal(before, value):
            raise ValueError(f"Untrained head changed: {name}")
    return dict(tensors=rows, preserved_heads=sorted(missing), verified_local_values=True)


def main():
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.scripts.inference import main as native_main

    original = OmniInference.generate

    @functools.wraps(original)
    def generate(self, *args, **kwargs):
        output = self.setup_args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        overlay = os.environ.get("YUZH_DPO_OVERLAY")
        if overlay and not getattr(self, "_yuzh_overlay_loaded", False):
            checkpoint = Path(overlay)
            manifest = json.loads((checkpoint / ".posttrain-checkpoint-artifact.json").read_text())
            if manifest["iteration"] != 2 or manifest["checkpoint_components"] != ["model"]:
                raise ValueError("Expected step 2 model-only artifact")
            if (checkpoint / "model.pt").stat().st_size != manifest["components"]["model"]["size"]:
                raise ValueError("Checkpoint size changed")
            payload = torch.load(checkpoint / "model.pt", map_location="cpu", mmap=True, weights_only=True)
            if set(payload) != {"net"}:
                raise ValueError("Unexpected model-only payload")
            receipt = apply_overlay(self.model.net, payload["net"], MISSING_HEADS)
            receipt.update(checkpoint=str(checkpoint), manifest=manifest, rank=rank)
            if any(r["changed_local_elements"] for r in receipt["tensors"] if not r["trainable"]):
                raise ValueError("Frozen inference parameters differ from DPO checkpoint")
            (output / f"overlay-rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
            del payload
            self._yuzh_overlay_loaded = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if rank == 0:
            (output / "runtime-numerics.json").write_text(
                json.dumps(
                    dict(
                        cudnn_benchmark=False,
                        cudnn_deterministic=True,
                        deterministic_algorithms=False,
                        scope="Fixed cuDNN choices, same native inference as baseline; no cross-platform guarantee.",
                    ),
                    indent=2,
                )
                + "\n"
            )
        return original(self, *args, **kwargs)

    OmniInference.generate = generate
    native_main()


if __name__ == "__main__":
    main()
