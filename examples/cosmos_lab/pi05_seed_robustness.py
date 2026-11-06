"""Compare two evaluation seeds without pretending they are independent training runs."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.cosmos_lab.pi05_ablation_report import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed0", type=Path, nargs=3, required=True, metavar=("SFT", "NATIVE", "PREFIX"))
    parser.add_argument("--seed1", type=Path, nargs=3, required=True, metavar=("SFT", "NATIVE", "PREFIX"))
    args = parser.parse_args()
    methods = ["SFT baseline", "Native PPO8", "Prefix PPO8"]
    metrics = ["any", "final", "last_1s_all", "last_2s_all"]
    paths = [args.seed0, args.seed1]
    all_loaded = [[load(path) for path in group] for group in paths]
    reference_report, reference_cases = all_loaded[0][0]
    seeds, counts, cases_by_seed = [], [], []
    for group in all_loaded:
        policy_seeds = {report["protocol"].get("policy_seed", report["protocol"]["seed"]) for report, _ in group}
        assert len(policy_seeds) == 1
        seeds.append(policy_seeds.pop())
        for report, cases in group:
            for key in [
                "seed",
                "revision",
                "max_steps",
                "reset_steps",
                "suite",
                "trials",
                "policy_python",
                "vision_sdpa",
                "control_hz",
            ]:
                assert report["protocol"][key] == reference_report["protocol"][key], key
            assert cases.keys() == reference_cases.keys()
            assert all(
                np.array_equal(row["initial_state"], reference_cases[case]["initial_state"])
                for case, row in cases.items()
            )
        cases_by_seed.append([cases for _, cases in group])
        counts.append(
            [{metric: sum(row[metric] for row in cases.values()) for metric in metrics} for _, cases in group]
        )
    assert seeds == [2026, 3026]
    for index in range(3):
        assert all_loaded[0][index][0]["protocol"].get("checkpoint") == all_loaded[1][index][0]["protocol"].get(
            "checkpoint"
        )
    per_method = {}
    for index, method in enumerate(methods):
        a, b = cases_by_seed[0][index], cases_by_seed[1][index]
        per_method[method] = {}
        for metric in metrics:
            per_method[method][metric] = {
                "success_both": [case for case in sorted(a) if a[case][metric] and b[case][metric]],
                "only_2026": [case for case in sorted(a) if a[case][metric] and not b[case][metric]],
                "only_3026": [case for case in sorted(a) if not a[case][metric] and b[case][metric]],
                "neither": [case for case in sorted(a) if not a[case][metric] and not b[case][metric]],
            }
    effects = {}
    for metric in metrics:
        rows = []
        for case in sorted(reference_cases):
            deltas = [int(c[2][case][metric]) - int(c[1][case][metric]) for c in cases_by_seed]
            rows.append({"case": case, "delta_2026": deltas[0], "delta_3026": deltas[1]})
        effects[metric] = {
            "gained_both": [r["case"] for r in rows if r["delta_2026"] == r["delta_3026"] == 1],
            "lost_both": [r["case"] for r in rows if r["delta_2026"] == r["delta_3026"] == -1],
            "opposite_sign": [r["case"] for r in rows if r["delta_2026"] * r["delta_3026"] == -1],
            "cases": rows,
        }
    receipt = {
        "scope": "Same50 reused validation cases, two policy noise seeds, single training seed. Not100 independent initial states or independent training replications. R10 seed2026 primary result retained.",
        "cross_seed_initial_physics_identical_all300_policy_trials": True,
        "environment_seed": reference_report["protocol"]["seed"],
        "counts_by_seed": {
            str(seed): dict(zip(methods, group, strict=True)) for seed, group in zip(seeds, counts, strict=True)
        },
        "per_method_seed_sensitivity": per_method,
        "prefix_minus_native_case_effects": effects,
        "runs": [[str(p) for p in group] for group in paths],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "seed-robustness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, seed, group in zip(axes, seeds, counts, strict=True):
        x, width = np.arange(4), 0.25
        for index, (method, count, color) in enumerate(
            zip(methods, group, ["#277da1", "#33994c", "#b35ac8"], strict=True)
        ):
            bars = ax.bar(x + (index - 1) * width, [count[m] for m in metrics], width=width, label=method, color=color)
            ax.bar_label(bars, padding=2, fontsize=9)
        ax.set(
            xticks=x,
            xticklabels=["Any reach", "Final", "All last1s", "All last2s"],
            ylim=(0, 55),
            title=f"Policy seed {seed}",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Successful cases / 50")
    axes[1].legend(loc="upper right")
    fig.suptitle("R11: same cases and models, different policy noise")
    fig.text(
        0.02,
        0.015,
        "Descriptive reused-validation results. Two policy seeds do not replicate the training experiment.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"seed-robustness.{suffix}", dpi=180)
    print(json.dumps(receipt["counts_by_seed"]))


if __name__ == "__main__":
    main()
