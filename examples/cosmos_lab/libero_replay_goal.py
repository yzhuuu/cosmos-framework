"""Replay recorded controls without policy inference; audit live versus forwarded terminal predicates."""

import argparse
import json
from pathlib import Path

import numpy as np


def goal(env):
    source, target = [env.env.object_states_dict[name] for name in ["akita_black_bowl_1", "plate_1"]]
    a, b = source.get_geom_state()["pos"], target.get_geom_state()["pos"]
    return {
        "success": bool(env.check_success()),
        "xy_distance_m": float(np.linalg.norm(a[:2] - b[:2])),
        "contact": bool(target.check_contact(source)),
        "source_at_or_above_target": bool(a[2] >= b[2]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--case", default="task-3-state2")
    args = parser.parse_args()
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.utils import get_libero_path

    report = json.loads((args.run / "probe/report.json").read_text())
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    row = next(row for row in report["trials"] if row["mode"] == "policy" and row["case_id"] == args.case)
    task = suite.get_task(row["task_id"])
    env = ControlEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        camera_names=[],
        control_freq=20,
        horizon=100000,
        ignore_done=True,
    )
    try:
        env.seed(report["protocol"]["seed"])
        env.reset()
        env.set_init_state(suite.get_task_init_states(row["task_id"])[row["init_state_id"]])
        zero = np.zeros(7, dtype=np.float64)
        zero[-1] = -1
        for _ in range(15):
            env.step(zero)
        with np.load(args.run / "probe" / row["name"] / "trajectory.npz") as data:
            initial_error = float(np.max(np.abs(env.get_sim_state() - data["initial_sim_state"])))
            live = [goal(env)]
            for action in data["actions"]:
                env.step(action)
                live.append(goal(env))
            final_error = float(np.max(np.abs(env.get_sim_state() - data["final_sim_state"])))
            success_equal = bool(np.array_equal([r["success"] for r in live], data["success"]))
        before = env.get_sim_state().copy()
        env.sim.forward()
        forwarded = goal(env)
        state_change = float(np.max(np.abs(env.get_sim_state() - before)))
        output = {
            "case": row["name"],
            "initial_state_max_error": initial_error,
            "final_state_max_error": final_error,
            "all_recorded_success_equal": success_equal,
            "live_terminal": live[-1],
            "after_forward": forwarded,
            "forward_physics_state_max_change": state_change,
            "live_last_10": live[-10:],
        }
        (args.run / f"goal-replay-{args.case}.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output))
    finally:
        env.close()


if __name__ == "__main__":
    main()
