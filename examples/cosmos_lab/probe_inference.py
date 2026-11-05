# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Observe native inference inputs/weights/noise without changing sampling settings."""

import functools
import hashlib
import json
import os
from pathlib import Path

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.scripts.inference import main
from cosmos_framework.utils import misc

TRACE = Path(os.environ["YUZH_PROBE_DIR"])
TRACE.mkdir(parents=True, exist_ok=True)
RANK = int(os.environ.get("RANK", "0"))
RECEIPT = TRACE / f"rank-{RANK}.jsonl"


def digest(tensor):
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    value = tensor.detach().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def summarize(value):
    if isinstance(value, torch.Tensor):
        return digest(value)
    if isinstance(value, dict):
        return {str(k): summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def emit(event, **values):
    with RECEIPT.open("a") as stream:
        stream.write(json.dumps({"event": event, "rank": RANK, **values}, sort_keys=True) + "\n")


def snapshot(model):
    emit(
        "runtime",
        training=model.training,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
    for name, param in model.net.named_parameters():
        emit("parameter", name=name, **digest(param))
    for name, buffer in model.named_buffers():
        emit("buffer", name=name, **digest(buffer))


original_generate = OmniMoTModel.generate_samples_from_batch
seen = set()


@functools.wraps(original_generate)
def generate(self, data_batch, *args, **kwargs):
    if id(self) not in seen:
        snapshot(self)
        seen.add(id(self))
    emit(
        "sample_input",
        data=summarize(data_batch),
        options=summarize(kwargs),
        cpu_rng=digest(torch.get_rng_state()),
        cuda_rng=digest(torch.cuda.get_rng_state()),
    )
    output = original_generate(self, data_batch, *args, **kwargs)
    emit("sample_output", data=summarize(output))
    return output


original_rand = misc.arch_invariant_rand


@functools.wraps(original_rand)
def rand(*args, **kwargs):
    output = original_rand(*args, **kwargs)
    emit("initial_noise", args=summarize(args), kwargs=summarize(kwargs), result=digest(output))
    return output


OmniMoTModel.generate_samples_from_batch = generate
misc.arch_invariant_rand = rand

if __name__ == "__main__":
    main()
