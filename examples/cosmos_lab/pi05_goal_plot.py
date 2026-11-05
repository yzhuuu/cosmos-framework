"""Plot task-linked terminal geometry and native success; retain replay exceptions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(run):
    audit = json.loads((run / "terminal-goal-audit.json").read_text())
    rows = {}
    for row in audit:
        case = row["case"].removesuffix("-policy")
        predicate = row["predicates"][0]
        replay_path = run / f"goal-replay-{case}.json"
        source = "reconstructed"
        if replay_path.exists():
            replay = json.loads(replay_path.read_text())
            assert replay["final_state_max_error"] == 0 and replay["all_recorded_success_equal"]
            predicate = replay["live_terminal"]
            source = "exact action replay, live predicate"
        rows[case] = {
            "distance_cm": 100 * predicate["xy_distance_m"],
            "contact": predicate["contact"],
            "final_success": row["recorded_success"],
            "source": source,
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("updated", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", default="PPO step 2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sets = [read(args.baseline), read(args.updated)]
    assert sets[0].keys() == sets[1].keys()
    cases = sorted(sets[0])
    fig, ax = plt.subplots(figsize=(10, 9))
    for i, case in enumerate(cases):
        a, b = [data[case] for data in sets]
        ax.plot([a["distance_cm"], b["distance_cm"]], [i, i], color="#c6cbd1", zorder=1)
        for index, row in enumerate([a, b]):
            ax.scatter(
                row["distance_cm"],
                i + (index - 0.5) * 0.16,
                color=["#277da1", "#e76f51"][index],
                marker="o" if row["final_success"] else "x",
                s=45,
                zorder=2,
            )
        if not b["contact"]:
            ax.annotate(
                "No contact",
                (b["distance_cm"], i),
                xytext=(-85 if b["distance_cm"] > 8 else 15, -14),
                textcoords="offset points",
                fontsize=9,
                color="#9c2f1f",
            )
    ax.axvline(3, color="#666", linestyle="--", label="3 cm center-distance threshold")
    ax.scatter([], [], color="#277da1", label="SFT baseline")
    ax.scatter([], [], color="#e76f51", label=args.label)
    ax.set_yticks(range(len(cases)), [case.replace("task-", "Task ").replace("-state", " / state ") for case in cases])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Bowl-to-plate horizontal center distance at end (cm; lower is closer)")
    ax.set_title(
        "Intended task: place the black bowl on the plate\nNative final success also requires contact and height",
        pad=15,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.text(
        0.02,
        0.035,
        "Circle = native final success; cross = failure. Same 20 initial states, instructions and seeds.\n"
        "Geometry reconstructed from saved physics; any replay exception uses its verified live predicate.\n"
        "Shared training pool; small diagnostic, not a held-out benchmark or quality claim.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"terminal-goals.{suffix}", dpi=170)
    ax.set_xlim(0, 8)
    ax.set_title(
        "Detail: terminal center distance 0–8 cm\nLarge misses outside this range remain in the full plot", pad=15
    )
    fig.savefig(args.output / "terminal-goals-zoom.png", dpi=170)
    (args.output / "terminal-goals.json").write_text(
        json.dumps({"baseline": sets[0], "updated": sets[1]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
