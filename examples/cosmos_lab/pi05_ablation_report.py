"""Report every fixed validation case across SFT, native PPO and prefix PPO."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    report = json.loads((run / "probe/report.json").read_text())
    assert report["protocol"]["control_hz"] == 20
    cases = {}
    for row in report["trials"]:
        if row["mode"] != "policy":
            continue
        with np.load(run / "probe" / row["name"] / "trajectory.npz") as data:
            flags = data["success"].astype(bool)
            assert len(flags) == 241
            assert bool(flags.any()) == row["success_any"] and bool(flags[-1]) == row["success_final"]
            cases[row["case_id"]] = {
                "task": row["task_id"],
                "any": bool(flags.any()),
                "final": bool(flags[-1]),
                "last_1s_all": bool(flags[-21:].all()),
                "last_2s_all": bool(flags[-41:].all()),
                "first_success_control": int(np.flatnonzero(flags)[0]) if flags.any() else None,
                "initial_state": data["initial_sim_state"].copy(),
            }
    assert len(cases) == 50
    return report, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = [args.baseline, args.native, args.prefix]
    loaded = [load(path) for path in paths]
    reports, case_sets = zip(*loaded, strict=True)
    for key in [
        "revision",
        "seed",
        "sampling",
        "max_steps",
        "reset_steps",
        "suite",
        "trials",
        "policy_python",
        "vision_sdpa",
    ]:
        assert all(r["protocol"][key] == reports[0]["protocol"][key] for r in reports), key
    assert all(c.keys() == case_sets[0].keys() for c in case_sets)
    for case in case_sets[0]:
        assert all(np.array_equal(c[case]["initial_state"], case_sets[0][case]["initial_state"]) for c in case_sets)
        for cases in case_sets:
            del cases[case]["initial_state"]
    names = ["SFT baseline", "Native PPO step8", "Prefix PPO step8"]
    metrics = ["any", "final", "last_1s_all", "last_2s_all"]
    counts = [{metric: sum(row[metric] for row in cases.values()) for metric in metrics} for cases in case_sets]
    paired = {}
    for index, name in [(0, "versus_sft"), (1, "versus_native_ppo8")]:
        paired[name] = {}
        for metric in metrics:
            before, after = case_sets[index], case_sets[2]
            gained = [c for c in sorted(after) if after[c][metric] and not before[c][metric]]
            lost = [c for c in sorted(after) if not after[c][metric] and before[c][metric]]
            paired[name][metric] = {"gained": gained, "lost": lost, "delta_cases": len(gained) - len(lost)}
    result = {
        "scope": "Reused 50-case validation, one training seed, same tasks/shared500; unknown pretrained exposure. Any is primary; final and sampled persistence are secondary. No human video score or generalization claim.",
        "counts": dict(zip(names, counts, strict=True)),
        "paired": paired,
        "runs": dict(zip(names, map(str, paths), strict=True)),
        "cases": dict(zip(names, case_sets, strict=True)),
        "persistence_definition": "All21/41 native success checks in the last1/2seconds at20Hz; not continuous-contact proof between controls.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "cases.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "method", "task", *metrics, "first_success_control"])
        writer.writeheader()
        for name, cases in zip(names, case_sets, strict=True):
            for case in sorted(cases):
                writer.writerow({"case": case, "method": name, **cases[case]})
    colors = ["#277da1", "#33994c", "#b35ac8"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x, width = np.arange(4), 0.25
    for index, (name, count, color) in enumerate(zip(names, counts, colors, strict=True)):
        bars = ax.bar(x + (index - 1) * width, [count[m] for m in metrics], width=width, color=color, label=name)
        ax.bar_label(bars, padding=2)
    ax.set(
        xticks=x,
        xticklabels=["Any success\n(primary)", "Final success", "All last1s", "All last2s"],
        ylabel="Successful cases / 50",
        ylim=(0, 55),
        title="R10: first-reach credit ablation on the same 50 validation cases",
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.02,
        0.015,
        "Same evaluation initial states / prompts / seed. Reused validation, single trained seed; no generalization claim.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"outcomes.{suffix}", dpi=180)
    print(json.dumps({"counts": result["counts"], "paired": paired}))


if __name__ == "__main__":
    main()
