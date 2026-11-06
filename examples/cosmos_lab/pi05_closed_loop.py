"""Bounded local two-interpreter LIBERO policy probe, using native Pi05 ODE inference."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    stderr = (args.output / "policy-worker.log").open("w")
    worker = subprocess.Popen(
        [
            str(args.root / protocol.get("policy_python", "repos/cosmos-framework-sft/.venv/bin/python")),
            "-m",
            "examples.cosmos_lab.pi05_local_worker",
            str(args.root),
            str(args.output),
            "--sampler",
            protocol.get("sampler_mode", "ode"),
            *(["--checkpoint", protocol["checkpoint"]] if protocol.get("checkpoint") else []),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
        bufsize=1,
    )
    records = []
    try:
        ready = worker.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            raise RuntimeError("Policy worker failed to start; see policy-worker.log")
        for trial in protocol["trials"]:
            task_id, mode = trial["task_id"], trial["mode"]
            case_id = trial.get("case_id", f"task-{task_id}")
            name = f"{case_id}-{mode}"
            initial_id = trial.get("init_state_id", protocol["init_state_id"])
            folder = args.output / name
            folder.mkdir()
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=0,
                control_freq=20,
                horizon=100000,
                ignore_done=True,
            )
            states, actions, frames, successes, latencies, clips = [], [], [], [], [], []
            start = time.monotonic()
            try:
                env.seed(protocol["seed"])
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[initial_id])
                zero = np.zeros(7, dtype=np.float64)
                zero[-1] = -1
                for _ in range(15):
                    obs, _, _, _ = env.step(zero)
                initial_sim_state = env.get_sim_state().copy()
                queue = []
                for step in range(protocol["max_steps"] + 1):
                    view = LiberoChunkSession._observation(obs, task.language)
                    assert view["state"].shape == (8,) and np.isfinite(view["state"]).all()
                    states.append(view["state"].copy())
                    frames.append(np.concatenate([view["image"], view["wrist_image"]], axis=1))
                    successes.append(bool(env.check_success()))
                    if step == protocol["max_steps"]:
                        break
                    if mode == "zero":
                        action = zero.copy()
                    else:
                        if not queue:
                            observation = folder / f"observation-{step:03d}.npz"
                            action_path = folder / f"actions-{step:03d}.npy"
                            np.savez_compressed(
                                observation, **{k: view[k] for k in ["image", "wrist_image", "state", "prompt"]}
                            )
                            request = {
                                "observation": str(observation),
                                "actions": str(action_path),
                                "seed": protocol.get("policy_seed", protocol["seed"]) + step // 5,
                            }
                            before = time.monotonic()
                            worker.stdin.write(json.dumps(request) + "\n")
                            worker.stdin.flush()
                            response = worker.stdout.readline()
                            if not response or json.loads(response).get("actions") != str(action_path.resolve()):
                                raise RuntimeError("Policy action reply missing or mismatched")
                            chunk = np.load(action_path, allow_pickle=False)
                            assert chunk.shape == (5, 7) and np.isfinite(chunk).all()
                            latencies.append(time.monotonic() - before)
                            queue = list(chunk)
                        action = queue.pop(0)
                    actions.append(action.copy())
                    # Native OSC clips pose inputs itself. Preserve raw actions and report clipping.
                    clips.append(bool((np.abs(action[:6]) > 1).any()))
                    obs, _, _, _ = env.step(action)
                final_sim_state = env.get_sim_state().copy()
            finally:
                env.close()
            np.savez_compressed(
                folder / "trajectory.npz",
                states=states,
                actions=actions,
                success=successes,
                initial_sim_state=initial_sim_state,
                final_sim_state=final_sim_state,
            )
            subprocess.run(
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
                    str(folder / "views.mp4"),
                ],
                input=np.stack(frames).tobytes(),
                check=True,
                capture_output=True,
            )
            sheet = Image.new("RGB", (1536, 768), "white")
            for j, index in enumerate(np.linspace(0, protocol["max_steps"], 9).astype(int)):
                sheet.paste(Image.fromarray(frames[index]), ((j % 3) * 512, (j // 3) * 256))
            sheet.save(folder / "contact.png")
            record = {
                **trial,
                "name": name,
                "language": task.language,
                "frames": len(frames),
                "steps": len(actions),
                "fps": 20,
                "success_any": any(successes),
                "success_final": successes[-1],
                "first_success_step": next((i for i, x in enumerate(successes) if x), None),
                "pose_clipped_steps": sum(clips),
                "latency_s": latencies,
                "wall_seconds": time.monotonic() - start,
            }
            records.append(record)
            (args.output / "partial.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        repeat_case = protocol.get("repeat_case", "task-0")
        a = np.load(args.output / f"{repeat_case}-policy/trajectory.npz")
        b = np.load(args.output / f"{repeat_case}-policy_repeat/trajectory.npz")
        repeat = {
            key + "_max_error": float(np.abs(a[key] - b[key]).max())
            for key in ["states", "actions", "initial_sim_state", "final_sim_state"]
        }
        (args.output / "report.json").write_text(
            json.dumps(
                {
                    "protocol": protocol,
                    "trials": records,
                    "repeat_task0": repeat,
                    "scope": protocol["scope"],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        stderr.close()


if __name__ == "__main__":
    main()
