"""Compare ODE and selected-flow SDE at fixed checkpoints, cases and seed."""

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
    parser.add_argument("--ode", type=Path, nargs=3, required=True)
    parser.add_argument("--sde", type=Path, nargs=3, required=True)
    args = parser.parse_args()
    methods = ["SFT baseline", "Native PPO8", "Prefix PPO8"]
    modes = ["ode", "selected-flow-sde"]
    metrics = ["any", "final", "last_1s_all", "last_2s_all"]
    paths = [args.ode, args.sde]
    loaded = [[load(path) for path in group] for group in paths]
    reference, reference_cases = loaded[0][0]
    counts, audits = {}, []
    for mode, group, group_paths in zip(modes, loaded, paths, strict=True):
        counts[mode] = {}
        for method, (report, cases), path in zip(methods, group, group_paths, strict=True):
            assert report["protocol"].get("sampler_mode", "ode") == mode
            assert report["protocol"].get("policy_seed", report["protocol"]["seed"]) == 2026
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
                assert report["protocol"][key] == reference["protocol"][key], key
            assert cases.keys() == reference_cases.keys()
            assert all(
                np.array_equal(row["initial_state"], reference_cases[case]["initial_state"])
                for case, row in cases.items()
            )
            counts[mode][method] = {metric: sum(row[metric] for row in cases.values()) for metric in metrics}
            if mode == "selected-flow-sde":
                audit = json.loads((path / "sampling-acceptance.json").read_text())
                assert audit["passed"]
                audits.append(audit)
    transitions = {}
    for index, method in enumerate(methods):
        ode_report, ode_cases = loaded[0][index]
        sde_report, sde_cases = loaded[1][index]
        assert ode_report["protocol"].get("checkpoint") == sde_report["protocol"].get("checkpoint")
        transitions[method] = {}
        for metric in metrics:
            gained = [case for case in sorted(ode_cases) if sde_cases[case][metric] and not ode_cases[case][metric]]
            lost = [case for case in sorted(ode_cases) if not sde_cases[case][metric] and ode_cases[case][metric]]
            transitions[method][metric] = {
                "gained_in_sde": gained,
                "lost_in_sde": lost,
                "delta": len(gained) - len(lost),
            }
    result = {
        "scope": "Diagnostic evaluation sampler comparison, not new training or a replacement for R10 ODE primary results. Same50 reused validation cases; one training seed, pretraining exposure unknown.",
        "counts": counts,
        "ode_to_sde_case_changes": transitions,
        "all300_policy_initial_physics_identical": True,
        "policy_seed": 2026,
        "sde_audits": audits,
        "runs": [[str(path) for path in group] for group in paths],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sampler-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mode, title in zip(
        axes, modes, ["ODE (existing primary)", "Training sampler: selected-flow SDE"], strict=True
    ):
        x, width = np.arange(4), 0.25
        for index, (method, color) in enumerate(zip(methods, ["#277da1", "#33994c", "#b35ac8"], strict=True)):
            bars = ax.bar(
                x + (index - 1) * width,
                [counts[mode][method][m] for m in metrics],
                width=width,
                label=method,
                color=color,
            )
            ax.bar_label(bars, padding=2, fontsize=9)
        ax.set(xticks=x, xticklabels=["Any reach", "Final", "All last1s", "All last2s"], ylim=(0, 55), title=title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Successful cases / 50")
    axes[1].legend(loc="upper right")
    fig.suptitle("R12: same models / cases / seed, different evaluation sampler")
    fig.text(
        0.02,
        0.015,
        "Per-decision native SDE behavior replay required. Reused validation; no independent training replication.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"sampler-comparison.{suffix}", dpi=180)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
