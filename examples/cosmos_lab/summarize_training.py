"""Export W&B histories and plot the trajectory retained by the final Nano checkpoint."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", default="sav-inception-self/cosmos-lab/l4d2jvzl")
    parser.add_argument("--recovery", default="sav-inception-self/cosmos-lab/igizxobc")
    parser.add_argument("--resume-update", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    histories = {}
    for label, path in (("parent", args.parent), ("recovery", args.recovery)):
        run = api.run(path)
        rows = list(run.scan_history(page_size=250))
        histories[label] = [
            r
            for r in rows
            if isinstance(r.get("iteration"), (int, float)) and isinstance(r.get("train/loss"), (int, float))
        ]
        (args.output / f"{label}-history.json").write_text(json.dumps(rows, indent=2) + "\n")
    valid = [r for r in histories["parent"] if r.get("iteration", -1) <= args.resume_update]
    valid += [r for r in histories["recovery"] if r.get("iteration", -1) > args.resume_update]
    valid = sorted((r for r in valid if "train/loss" in r), key=lambda row: row["iteration"])
    assert len({r["iteration"] for r in valid}) == len(valid)
    assert all(np.isfinite(r["train/loss"]) for r in valid)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ("train/loss", "clip_grad_norm/video/global"),
        ("Flow training loss (individual noisy batches)", "Logged gradient norm"),
        strict=True,
    ):
        values = [(r["iteration"], r[key]) for r in valid if key in r]
        ax.plot(*zip(*values, strict=True), linewidth=1, color="#2463a8", label="Retained trajectory")
        discarded = [
            (r["iteration"], r[key])
            for r in histories["parent"]
            if r.get("iteration", -1) > args.resume_update and key in r
        ]
        if discarded:
            ax.plot(
                *zip(*discarded, strict=True),
                linewidth=1,
                alpha=0.4,
                color="#999999",
                label="Discarded after interruption",
            )
        ax.axvline(args.resume_update, color="#c05822", linestyle="--", label="Restored checkpoint")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Completed optimizer updates (W&B iteration)")
    fig.savefig(args.output / "training-trajectory.png", dpi=160)
    plt.close(fig)
    loss = [r["train/loss"] for r in valid]
    receipt = {
        "scope": "Training diagnostics only; no quality/generalization conclusion. Failed-run updates after the restored checkpoint are excluded from the retained path.",
        "parent": args.parent,
        "recovery": args.recovery,
        "logged_loss_rows": len(loss),
        "iteration_range": [valid[0]["iteration"], valid[-1]["iteration"]],
        "all_loss_finite": True,
        "first_10_median": float(np.median(loss[:10])),
        "last_10_median": float(np.median(loss[-10:])),
    }
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
