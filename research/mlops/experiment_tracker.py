"""
mlops/experiment_tracker.py

Lightweight experiment tracking — logs every training run to
mlops/experiments.jsonl (one JSON object per line).

Usage:
    tracker = ExperimentTracker()
    run_id  = tracker.start_run("scratch_ncf", config)
    tracker.log_epoch(run_id, epoch=1, loss=0.45, hit=0.12)
    tracker.finish_run(run_id, best_hit=0.31)

Then query with:
    tracker.summary_table()
"""

import json
import time
import uuid
from pathlib import Path
from datetime import datetime

LOG_FILE = Path(__file__).resolve().parent / "experiments.jsonl"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class ExperimentTracker:

    def start_run(self, name: str, config: dict) -> str:
        run_id = str(uuid.uuid4())[:8]
        record = {
            "run_id":    run_id,
            "name":      name,
            "config":    config,
            "started":   datetime.utcnow().isoformat(),
            "epochs":    [],
            "status":    "running",
        }
        self._append(record)
        print(f"Experiment {run_id} ({name}) started")
        return run_id

    def log_epoch(self, run_id: str, **metrics):
        """Append per-epoch metrics to the run record."""
        record = {"run_id": run_id, "type": "epoch", **metrics,
                  "ts": time.time()}
        self._append(record)

    def finish_run(self, run_id: str, **final_metrics):
        record = {"run_id": run_id, "type": "finish",
                  "finished": datetime.utcnow().isoformat(),
                  **final_metrics}
        self._append(record)
        print(f"Experiment {run_id} finished: {final_metrics}")

    def summary_table(self):
        """Print a summary of all completed runs."""
        runs = {}
        for line in LOG_FILE.read_text().splitlines():
            rec = json.loads(line)
            rid = rec["run_id"]
            if rid not in runs:
                runs[rid] = rec
            if rec.get("type") == "finish":
                runs[rid].update(rec)

        print(f"\n{'Run ID':<10} {'Name':<20} {'Best Hit@10':<14} {'Epochs':<8} {'Finished'}")
        print("-" * 70)
        for r in runs.values():
            if r.get("type") != "epoch":
                print(f"{r['run_id']:<10} {r['name']:<20} "
                      f"{r.get('best_hit_at_10', '—')!s:<14} "
                      f"{r.get('n_epochs', '—')!s:<8} "
                      f"{r.get('finished', '—')[:19]}")

    def _append(self, record: dict):
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")