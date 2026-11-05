"""Native Joint PPO with failure-record capture and per-update frozen-VLM attestations."""

import json
import os
from pathlib import Path
from types import MethodType

from cosmos_framework.posttrain.entrypoints import joint_ppo

_NATIVE_SETUP = joint_ppo.setup_policy


def audited_on_policy(self, samples):
    try:
        return self._lab_on_policy(samples)
    except Exception as error:
        import torch
        import torch.distributed as dist

        folder = Path(os.environ["YUZH_PPO_AUDIT_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"replay-failure-step{self.optimizer_step}-rank{dist.get_rank()}"
        torch.save(samples, folder / f"{stem}.pt")
        (folder / f"{stem}.json").write_text(
            json.dumps(
                {
                    "error": str(error),
                    "model_identity": self.model.ppo_identity(),
                    "recipe_identity": self.recipe.identity(),
                    "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "samples": len(samples),
                },
                indent=2,
            )
            + "\n"
        )
        raise


def audited_train_batch(self, batch_id):
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    before = run_rank_converged(self.model.ppo_frozen_state, "lab frozen VLM before update")
    result = self._lab_train_batch(batch_id)
    after = run_rank_converged(self.model.ppo_frozen_state, "lab frozen VLM after update")

    def verify():
        if before != after or before != self._frozen_model_state:
            raise RuntimeError("Frozen VLM changed during PPO training")

    run_rank_converged(verify, "lab frozen VLM consistency")
    result["metrics"]["frozen_vlm_verified"] = 1
    result["frozen_vlm_digest"] = after
    return result


def audited_setup_policy(*args, **kwargs):
    bodies = _NATIVE_SETUP(*args, **kwargs)
    trainer = bodies["policy"]
    trainer._lab_on_policy = trainer._on_policy
    trainer._on_policy = MethodType(audited_on_policy, trainer)
    trainer._lab_train_batch = trainer.train_batch
    trainer.train_batch = MethodType(audited_train_batch, trainer)
    return bodies


if __name__ == "__main__":
    # Give Ray an importable callback; worker imports capture its native setup before patching.
    from examples.cosmos_lab.pi05_ppo_audited import audited_setup_policy as setup

    joint_ppo.setup_policy = setup
    joint_ppo.main()
