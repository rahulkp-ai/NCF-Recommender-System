"""
production/tests/unit/test_metrics.py — verify metric implementations.

Moved from tests/test_metrics.py in Phase 4. sys.path insert updated from
parents[1] to parents[3] for the new depth (same fix class as
test_api.py). NOTE: the import below (research.evaluation.metrics) is a
pre-existing production -> research import that Phase 1's boundary check
missed — flagged in the Phase 4 report as a correction to ADR 0001,
left as-is here pending a decision on where these metric functions
should live (see docs/decisions/, to be added).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from research.evaluation.metrics import hit_at_k, ndcg_at_k


def test_hit_at_k():
    ranked = [5, 2, 8, 1, 3, 7, 9, 4, 6, 10]
    assert hit_at_k(ranked, 5, k=1) == 1.0  # rank 1 → hit at K=1
    assert hit_at_k(ranked, 5, k=10) == 1.0  # rank 1 → hit at K=10
    assert hit_at_k(ranked, 6, k=5) == 0.0  # rank 9 → miss at K=5
    assert hit_at_k(ranked, 6, k=9) == 1.0  # rank 9 → hit at K=9


def test_ndcg_at_k():
    ranked = [5, 2, 8, 1, 3]
    assert ndcg_at_k(ranked, 5, k=5) == pytest.approx(1.0)  # rank 1
    assert ndcg_at_k(ranked, 8, k=5) == pytest.approx(1 / np.log2(4))  # rank 3
    assert ndcg_at_k(ranked, 9, k=5) == 0.0  # not in top-5


def test_random_baseline():
    # Random model: positive item equally likely at any position (1-100)
    # Expected Hit@10 ≈ 10/100 = 0.10
    np.random.seed(0)
    hits = [hit_at_k(list(np.random.permutation(100)), 0, k=10) for _ in range(2000)]
    assert 0.08 < np.mean(hits) < 0.12, "Random baseline should be ~0.10"


import pytest

if __name__ == "__main__":
    test_hit_at_k()
    test_ndcg_at_k()
    test_random_baseline()
    print("All metric tests passed.")
