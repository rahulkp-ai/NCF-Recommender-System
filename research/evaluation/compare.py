"""
research/evaluation/compare.py

Aggregates log files from NumPy scratch NCF and PyTorch NCF into a unified
comparison payload containing all metrics: Hit@10, NDCG@10, Precision@10, and Recall@10.
"""

import json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR

def extract_metric_series(epochs_data: list, metric_key: str) -> list:
    """Extracts epoch-value pairs for a specific metric key."""
    series = []
    for ep in epochs_data:
        if ep.get(metric_key) is not None:
            series.append([ep["epoch"], ep[metric_key]])
    return series

def run_comparison():
    scratch_path = LOG_DIR / "scratch_ncf_log.json"
    pytorch_path = LOG_DIR / "pytorch_ncf_log.json"

    if not scratch_path.exists() or not pytorch_path.exists():
        raise FileNotFoundError(
            "Missing log files. Ensure both models have been trained and saved "
            "logs to evaluation/scratch_ncf_log.json and evaluation/pytorch_ncf_log.json"
        )

    with open(scratch_path) as f:
        scratch_log = json.load(f)

    with open(pytorch_path) as f:
        pytorch_log = json.load(f)

    # Process Scratch metrics
    scratch_epochs = [e["epoch"] for e in scratch_log["epochs"]]
    scratch_loss   = [e["loss"] for e in scratch_log["epochs"]]
    scratch_time   = [e["time_sec"] for e in scratch_log["epochs"]]

    # Process PyTorch metrics
    pytorch_epochs = [e["epoch"] for e in pytorch_log["epochs"]]
    pytorch_loss   = [e["loss"] for e in pytorch_log["epochs"]]
    pytorch_time   = [e["time_sec"] for e in pytorch_log["epochs"]]

    # Calculate per-epoch speedup (Scratch Time / PyTorch Time)
    speedups = [
        round(s / p, 2) if p > 0 else 1.0 
        for s, p in zip(scratch_time, pytorch_time)
    ]
    avg_speedup   = round(float(np.mean(speedups)), 2)
    total_speedup = round(scratch_log["total_time_sec"] / pytorch_log["total_time_sec"], 2)

    # Calculate key metric deltas
    final_loss_delta = round(pytorch_loss[-1] - scratch_loss[-1], 6)
    best_hit_scratch = scratch_log.get("best_hit_at_10", 0.0)
    best_hit_pytorch = pytorch_log.get("best_hit_at_10", 0.0)
    best_hit_delta   = round(best_hit_pytorch - best_hit_scratch, 4)

    payload = {
        "epochs": scratch_epochs,
        "scratch": {
            "loss":            scratch_loss,
            "hit_at_10":       extract_metric_series(scratch_log["epochs"], "hit_at_10"),
            "ndcg_at_10":      extract_metric_series(scratch_log["epochs"], "ndcg_at_10"),
            "precision_at_10": extract_metric_series(scratch_log["epochs"], "precision_at_10"),
            "recall_at_10":    extract_metric_series(scratch_log["epochs"], "recall_at_10"),
            "time_per_epoch":  scratch_time,
            "total_time":      scratch_log["total_time_sec"],
            "best_hit":        best_hit_scratch,
            "best_ndcg":       scratch_log.get("best_ndcg_at_10", 0.0),
            "best_precision":  scratch_log.get("best_precision_at_10", 0.0),
            "best_recall":     scratch_log.get("best_recall_at_10", 0.0),
            "config":          scratch_log.get("config", {})
        },
        "pytorch": {
            "loss":            pytorch_loss,
            "hit_at_10":       extract_metric_series(pytorch_log["epochs"], "hit_at_10"),
            "ndcg_at_10":      extract_metric_series(pytorch_log["epochs"], "ndcg_at_10"),
            "precision_at_10": extract_metric_series(pytorch_log["epochs"], "precision_at_10"),
            "recall_at_10":    extract_metric_series(pytorch_log["epochs"], "recall_at_10"),
            "time_per_epoch":  pytorch_time,
            "total_time":      pytorch_log["total_time_sec"],
            "best_hit":        best_hit_pytorch,
            "best_ndcg":       pytorch_log.get("best_ndcg_at_10", 0.0),
            "best_precision":  pytorch_log.get("best_precision_at_10", 0.0),
            "best_recall":     pytorch_log.get("best_recall_at_10", 0.0),
            "config":          pytorch_log.get("config", {})
        },
        "comparison": {
            "speedup_per_epoch": speedups,
            "avg_speedup":       avg_speedup,
            "total_speedup":     total_speedup,
            "final_loss_delta":  final_loss_delta,
            "best_hit_delta":    best_hit_delta
        }
    }

    out_file = LOG_DIR / "comparison.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Comparison saved to {out_file}\n")
    print("=" * 50)
    print(f"  Average speedup      : {avg_speedup}×")
    print(f"  Total time speedup   : {total_speedup}×")
    print(f"  Final loss delta     : {final_loss_delta}")
    print(f"  Best Hit@10 delta    : {best_hit_delta}")
    print("=" * 50)

if __name__ == "__main__":
    run_comparison()