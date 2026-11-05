"""Small, fixed simulator action probe. No learned policy or task-success benchmark."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession

    suite = benchmark.get_benchmark_dict()[protocol["suite"]]()
    records = []
    for task_id in protocol["task_ids"]:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        initial = suite.get_task_init_states(task_id)[protocol["init_state_id"]]
        for condition in protocol["conditions"]:
            name = f"task-{task_id}-{condition}"
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=0,
                control_freq=20,
                horizon=100000,
                ignore_done=True,
            )
            states, actions, success, frames = [], [], [], []
            try:
                env.seed(protocol["seed"])
                env.reset()
                obs = env.set_init_state(initial)
                zero = np.zeros(7, dtype=np.float32)
                zero[-1] = -1
                for _ in range(protocol["settle_steps"]):
                    obs, _, _, _ = env.step(zero)
                start_state = env.get_sim_state().copy()
                for step in range(protocol["control_steps"] + 1):
                    view = LiberoChunkSession._observation(obs, task.language)
                    assert view["state"].shape == (8,) and np.isfinite(view["state"]).all()
                    assert view["image"].shape == view["wrist_image"].shape == (256, 256, 3)
                    assert view["image"].dtype == view["wrist_image"].dtype == np.uint8
                    states.append(view["state"].copy())
                    frames.append(np.concatenate([view["image"], view["wrist_image"]], axis=1))
                    success.append(bool(env.check_success()))
                    if step == protocol["control_steps"]:
                        break
                    action = zero.copy()
                    if 10 <= step < 40:
                        if condition.startswith("positive_x"):
                            action[0] = 0.2
                        elif condition == "negative_x":
                            action[0] = -0.2
                    actions.append(action.copy())
                    obs, _, _, _ = env.step(action)
                end_state = env.get_sim_state().copy()
                controller = env.env.robots[0].controller
                contract = {
                    "type": type(controller).__name__,
                    "control_freq": 20,
                    "input_min": np.asarray(controller.input_min).tolist(),
                    "input_max": np.asarray(controller.input_max).tolist(),
                    "output_min": np.asarray(controller.output_min).tolist(),
                    "output_max": np.asarray(controller.output_max).tolist(),
                }
            finally:
                env.close()
            states = np.stack(states)
            np.savez_compressed(
                args.output / (name + ".npz"),
                states=states,
                actions=np.stack(actions),
                success=success,
                initial_sim_state=start_state,
                final_sim_state=end_state,
            )
            process = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "512x256",
                    "-r",
                    "20",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(args.output / (name + ".mp4")),
                ],
                input=np.stack(frames).tobytes(),
                check=True,
                capture_output=True,
            )
            del process
            sheet = Image.new("RGB", (512 * 3, 256 * 2), "white")
            for j, index in enumerate([0, 10, 25, 40, 60, 80]):
                sheet.paste(Image.fromarray(frames[index]), ((j % 3) * 512, (j // 3) * 256))
            sheet.save(args.output / (name + ".png"))
            record = {
                "name": name,
                "task_id": task_id,
                "condition": condition,
                "language": task.language,
                "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
                "initial_sim_state_sha256": hashlib.sha256(start_state.tobytes()).hexdigest(),
                "frames": len(frames),
                "actions": len(actions),
                "fps": 20,
                "eef_displacement": (states[-1, :3] - states[0, :3]).tolist(),
                "success_any": any(success),
                "success_final": success[-1],
                "controller": contract,
            }
            records.append(record)
            (args.output / "partial.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    checks = []
    for task_id in protocol["task_ids"]:
        arrays = {c: np.load(args.output / f"task-{task_id}-{c}.npz") for c in protocol["conditions"]}
        zero = arrays["zero"]["states"][-1, 0]
        checks.append(
            {
                "task_id": task_id,
                "initial_states_identical": all(
                    np.array_equal(v["initial_sim_state"], arrays["zero"]["initial_sim_state"]) for v in arrays.values()
                ),
                "repeat_max_state_error": float(
                    np.abs(arrays["positive_x"]["states"] - arrays["positive_x_repeat"]["states"]).max()
                ),
                "positive_x_relative_zero": float(arrays["positive_x"]["states"][-1, 0] - zero),
                "negative_x_relative_zero": float(arrays["negative_x"]["states"][-1, 0] - zero),
            }
        )
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "trials": records,
                "checks": checks,
                "scope": "Scripted controls verify simulator causality/repeatability; no trained policy or benchmark score.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
