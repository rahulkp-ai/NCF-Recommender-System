"""
research/evaluation/compare.py

Loads training logs from both models and produces the comparison
data structures consumed by the dashboard and thesis plots.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

LOG_DIR  = Path("research/evaluation")
PROC_DIR = Path("research/datasets/processed")


def load_logs() -> tuple[dict, dict]:
    """Load scratch and PyTorch training logs. Raise clearly if missing."""
    s_path = LOG_DIR / "scratch_ncf_log.json"
    p_path = LOG_DIR / "pytorch_ncf_log.json"

    if not s_path.exists():
        raise FileNotFoundError(
            f"Scratch log not found at {s_path}. Run Phase 3 training first."
        )
    if not p_path.exists():
        raise FileNotFoundError(
            f"PyTorch log not found at {p_path}. Run Phase 4 training first."
        )

    with open(s_path) as f: scratch = json.load(f)
    with open(p_path) as f: pytorch = json.load(f)
    return scratch, pytorch


def build_comparison_table() -> dict:
    """
    Build the master comparison table for the dashboard and thesis.

    Returns a dict ready for JSON serialisation.
    """
    scratch, pytorch = load_logs()

    def epoch_series(log, key):
        return [e[key] for e in log["epochs"]]

    def epoch_series_nonnull(log, key):
        return [(e["epoch"], e[key]) for e in log["epochs"] if e[key] is not None]

    s_loss  = epoch_series(scratch, "loss")
    p_loss  = epoch_series(pytorch, "loss")
    s_times = epoch_series(scratch, "time_sec")
    p_times = epoch_series(pytorch, "time_sec")

    s_hit_pairs = epoch_series_nonnull(scratch, "hit_at_10")
    p_hit_pairs = epoch_series_nonnull(pytorch, "hit_at_10")

    # Per-epoch speedup ratio
    speedup_per_epoch = [
        round(s / p, 2) if p > 0 else None
        for s, p in zip(s_times, p_times)
    ]

    # Convergence: first epoch where loss < threshold
    def first_below(losses, threshold=0.35):
        for i, l in enumerate(losses):
            if l < threshold:
                return i + 1
        return None

    return {
        "epochs":         list(range(1, len(s_loss) + 1)),
        "scratch": {
            "loss":            [round(x, 6) for x in s_loss],
            "hit_at_10":       s_hit_pairs,
            "time_per_epoch":  [round(x, 1) for x in s_times],
            "total_time":      scratch.get("total_time_sec"),
            "best_hit":        scratch.get("best_hit_at_10"),
            "config":          scratch.get("config", {}),
            "converge_epoch":  first_below(s_loss),
        },
        "pytorch": {
            "loss":            [round(x, 6) for x in p_loss],
            "hit_at_10":       p_hit_pairs,
            "time_per_epoch":  [round(x, 1) for x in p_times],
            "total_time":      pytorch.get("total_time_sec"),
            "best_hit":        pytorch.get("best_hit_at_10"),
            "config":          pytorch.get("config", {}),
            "converge_epoch":  first_below(p_loss),
        },
        "comparison": {
            "speedup_per_epoch":     speedup_per_epoch,
            "avg_speedup":           round(float(np.mean(
                [x for x in speedup_per_epoch if x])
            ), 2),
            "total_speedup":         round(
                scratch.get("total_time_sec", 1) /
                max(pytorch.get("total_time_sec", 1), 1), 2
            ),
            "final_loss_delta":      round(
                abs(s_loss[-1] - p_loss[-1]), 6
            ),
            "best_hit_delta":        round(
                abs(scratch.get("best_hit_at_10", 0) -
                    pytorch.get("best_hit_at_10", 0)), 4
            ),
        },
    }


def save_comparison(outpath: str = "research/evaluation/comparison.json"):
    data = build_comparison_table()
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Comparison saved to {outpath}")

    c = data["comparison"]
    print(f"\n{'='*50}")
    print(f"  Average speedup      : {c['avg_speedup']}×")
    print(f"  Total time speedup   : {c['total_speedup']}×")
    print(f"  Final loss delta     : {c['final_loss_delta']:.6f}")
    print(f"  Best Hit@10 delta    : {c['best_hit_delta']:.4f}")
    print(f"{'='*50}")
    return data


if __name__ == "__main__":
    save_comparison()