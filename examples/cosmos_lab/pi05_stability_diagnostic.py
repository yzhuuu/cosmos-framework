"""Post-hoc native success persistence; never replace primary any/final scores."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", nargs=3, action="append", metavar=("GROUP", "STEP", "PATH"), required=True)
    args = parser.parse_args()
    rows = []
    for group, step, path in args.run:
        run = Path(path)
        report = json.loads((run / "probe/report.json").read_text())
        assert report["protocol"]["control_hz"] == 20
        cases = []
        for record in report["trials"]:
            if record["mode"] != "policy":
                continue
            with np.load(run / "probe" / record["name"] / "trajectory.npz") as data:
                success = data["success"].astype(bool)
                assert len(success) == 241
                cases.append(
                    {
                        "case": record["case_id"],
                        "any": bool(success.any()),
                        "final": bool(success[-1]),
                        "last_1s_all": bool(success[-21:].all()),
                        "last_2s_all": bool(success[-41:].all()),
                    }
                )
        rows.append(
            {
                "group": group,
                "step": int(step),
                "run": str(run.resolve()),
                "n": len(cases),
                "counts": {
                    key: sum(row[key] for row in cases) for key in ["any", "final", "last_1s_all", "last_2s_all"]
                },
                "cases": cases,
            }
        )
    result = {
        "scope": "Post-hoc descriptive diagnostic, primary native any/final retained. Last1/2s = all21/41 native checks at20Hz, not continuous between-step contact proof.",
        "results": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stability-diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    groups = list(dict.fromkeys(row["group"] for row in rows))
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.8), sharey=True, squeeze=False)
    for ax, group in zip(axes[0], groups, strict=True):
        for row in rows:
            if row["group"] != group:
                continue
            label = "SFT baseline" if row["step"] == 0 else f"PPO step {row['step']}"
            ax.plot(
                range(4),
                [100 * value / row["n"] for value in row["counts"].values()],
                marker="o",
                label=label,
                color={0: "#277da1", 2: "#e76f51", 8: "#33994c"}.get(row["step"]),
            )
        ax.set(
            xticks=range(4),
            xticklabels=["Any reach", "Final check", "All last 1s", "All last 2s"],
            ylim=(0, 100),
            title=group,
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left")
        ax.tick_params(axis="x", rotation=12)
    axes[0][0].set_ylabel("Cases meeting criterion (%)")
    fig.suptitle("Reaching success and maintaining success are different objectives")
    fig.text(
        0.02,
        0.015,
        "Post-hoc diagnostic; native predicates retained; single evaluation seed; shared500, pretraining exposure unknown.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"stability-diagnostic.{suffix}", dpi=170)
    print(json.dumps([{k: v for k, v in row.items() if k != "cases"} for row in rows]))


if __name__ == "__main__":
    main()
