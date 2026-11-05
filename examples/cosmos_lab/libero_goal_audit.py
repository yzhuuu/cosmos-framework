"""Reconstruct saved terminal physics to explain native LIBERO success predicates, without rendering."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.utils import get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    report = json.loads((args.run / "probe/report.json").read_text())
    results = []
    for row in report["trials"]:
        if row["mode"] != "policy":
            continue
        task = suite.get_task(row["task_id"])
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = ControlEnv(
            bddl_file_name=str(bddl),
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=False,
            camera_names=[],
        )
        try:
            env.reset()
            with np.load(args.run / "probe" / row["name"] / "trajectory.npz") as data:
                env.set_state(data["final_sim_state"])
            env.env.sim.forward()
            actual = bool(env.check_success())
            predicates = []
            for goal in env.env.parsed_problem["goal_state"]:
                item = {"goal": goal, "value": bool(env.env._eval_predicate(goal))}
                if goal[0].lower() == "on":
                    source, target = [env.env.object_states_dict[x] for x in goal[1:]]
                    if source.object_state_type == target.object_state_type == "object":
                        a, b = source.get_geom_state()["pos"], target.get_geom_state()["pos"]
                        item.update(
                            source_xyz=a.tolist(),
                            target_xyz=b.tolist(),
                            xy_distance_m=float(np.linalg.norm(a[:2] - b[:2])),
                            source_at_or_above_target=bool(a[2] >= b[2]),
                            contact=bool(target.check_contact(source)),
                            xy_threshold_m=0.03,
                        )
                predicates.append(item)
            results.append(
                {
                    "case": row["name"],
                    "recorded_success": row["success_final"],
                    "reconstructed_success": actual,
                    "matches_recorded": actual == row["success_final"],
                    "predicates": predicates,
                }
            )
            print(json.dumps(results[-1]), flush=True)
        finally:
            env.close()
    (args.run / "terminal-goal-audit.json").write_text(json.dumps(results, indent=2) + "\n")
    assert all(row["matches_recorded"] for row in results), "Reconstructed terminal success differs; full audit saved"


if __name__ == "__main__":
    main()
