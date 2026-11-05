"""Native Joint PPO with failure-record capture and per-update frozen-VLM attestations."""

import json
import os
from pathlib import Path
from types import MethodType

from cosmos_framework.posttrain.entrypoints import joint_ppo

_NATIVE_SETUP = joint_ppo.setup_policy
_NATIVE_EXPERIENCE = joint_ppo.setup_experience


def attest_resume(trainer, resume):
    import torch
    import torch.distributed as dist

    from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest
    from cosmos_framework.posttrain.checkpoint import collect_rng_state
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    saved = torch.load(Path(resume) / "trainer.pt", map_location="cpu", weights_only=False)
    live_rng = collect_rng_state()

    def verify():
        assert trainer.progress() == {
            "optimizer_step": saved["optimizer_step"],
            "publication_version": saved["publication_version"],
        }
        assert canonical_digest(live_rng) == canonical_digest(saved["rng"]), "Resume RNG mismatch"
        steps, moment_l1 = [], {"exp_avg": 0.0, "exp_avg_sq": 0.0}
        for state in trainer.opt.state.values():
            step = state["step"]
            steps.append(float(step.item()) if hasattr(step, "item") else float(step))
            for key in moment_l1:
                value = state[key]
                value = value.to_local() if hasattr(value, "to_local") else value
                assert torch.isfinite(value).all(), "Nonfinite restored moment"
                moment_l1[key] += float(value.double().abs().sum().item())
        assert steps and set(steps) == {float(saved["optimizer_step"])}
        assert all(value > 0 for value in moment_l1.values())
        return {
            "checkpoint": str(resume),
            "progress": trainer.progress(),
            "rng_exact": True,
            "optimizer_state_entries": len(steps),
            "optimizer_steps": sorted(set(steps)),
            "local_moment_l1": moment_l1,
            "scope": "Native strict restore plus live counters/RNG/moment audit; not uninterrupted-trajectory equivalence",
        }

    report = run_rank_converged(verify, "lab resume state verification")
    folder = Path(os.environ["YUZH_PPO_AUDIT_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"resume-rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n")


def audited_setup_experience(
    config, output, resume, recipe, width, members=("sim", "collector", "rollout", "rm", "replay")
):
    bodies = _NATIVE_EXPERIENCE(config, output, resume, recipe, width, members)
    if resume and "collector" in bodies:
        import torch

        from cosmos_framework.posttrain.config.build import REPLICA_INDEX_ENV

        replica = os.environ[REPLICA_INDEX_ENV]
        saved = torch.load(Path(resume) / "data.pt", map_location="cpu", weights_only=False)["collectors"][replica]
        live = bodies["collector"].state_dict()
        assert live["next_window"] == saved["next_window"]
        report = {"checkpoint": str(resume), "saved": saved, "restored": live}
        folder = Path(os.environ["YUZH_PPO_AUDIT_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"collector-resume-{replica}.json").write_text(json.dumps(report, indent=2) + "\n")
    return bodies


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
    resume = kwargs.get("resume", args[2] if len(args) > 2 else None)
    if resume:
        attest_resume(trainer, resume)
    trainer._lab_on_policy = trainer._on_policy
    trainer._on_policy = MethodType(audited_on_policy, trainer)
    trainer._lab_train_batch = trainer.train_batch
    trainer.train_batch = MethodType(audited_train_batch, trainer)
    return bodies


if __name__ == "__main__":
    # Give Ray an importable callback; worker imports capture its native setup before patching.
    from examples.cosmos_lab.pi05_ppo_audited import audited_setup_experience as experience
    from examples.cosmos_lab.pi05_ppo_audited import audited_setup_policy as setup

    joint_ppo.setup_policy = setup
    joint_ppo.setup_experience = experience
    joint_ppo.main()
