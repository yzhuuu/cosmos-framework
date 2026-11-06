"""Audit every native SDE decision and saved trace in a completed evaluation."""

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    assert (args.run / "end.txt").exists()
    probe = args.run / "probe"
    report = json.loads((probe / "report.json").read_text())
    assert report["protocol"]["sampler_mode"] == "selected-flow-sde"
    identity = json.loads((probe / "policy-identity.json").read_text())
    assert identity["sampling_mode"] == "selected-flow-sde"
    expected = {
        f"{r['name']}/observation-{step:03d}.npz"
        for r in report["trials"]
        if r["mode"] != "zero"
        for step in range(0, 240, 5)
    }
    rows = [json.loads(line) for line in (probe / "sampling-audit.jsonl").read_text().splitlines()]
    assert len(rows) == len(expected) == 2448
    assert {row["observation"] for row in rows} == expected
    indices = collections.Counter()
    for row in rows:
        assert row["sampler_identity"] == "pi05-openpi-selected-flow-sde-v1"
        assert all(math.isfinite(row[key]) for key in ["logprob", "value", "logprob_replay_delta"])
        assert row["logprob_replay_delta"] <= 1e-5
        step = int(Path(row["observation"]).stem.rsplit("-", 1)[1])
        assert row["seed"] == report["protocol"]["policy_seed"] + step // 5
        path = (probe / row["trace"]).resolve()
        assert path.is_relative_to(probe.resolve())
        with np.load(path) as trace:
            for name in ["latent_before", "latent_after", "normalized_actions"]:
                assert trace[name].shape == (1, 10, 32) and trace[name].dtype == np.float32
                assert np.isfinite(trace[name]).all()
            assert trace["denoise_index"].shape == (1,)
            assert int(trace["denoise_index"][0]) == row["denoise_index"]
        indices[row["denoise_index"]] += 1
    assert set(indices) == {0, 1, 2}
    result = {
        "passed": True,
        "decisions": len(rows),
        "selected_indices": dict(sorted(indices.items())),
        "max_logprob_replay_delta": max(row["logprob_replay_delta"] for row in rows),
        "sampler_identity": "pi05-openpi-selected-flow-sde-v1",
        "scope": "Every saved SDE decision/trace and behavior replay checked; no training or quality claim.",
    }
    (args.run / "sampling-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
