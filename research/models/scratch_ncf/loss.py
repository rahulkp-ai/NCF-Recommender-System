"""
research/models/scratch_ncf/loss.py

Binary Cross-Entropy loss with numerically stable implementation.

BCE(y, ŷ) = -[y·log(ŷ) + (1−y)·log(1−ŷ)]

The combined sigmoid+BCE gradient is:  dL/d(logit) = ŷ − y
This is the most important simplification in the whole model.
"""

import numpy as np


class BCELoss:
    """
    Binary Cross-Entropy loss.

    We operate on the raw logit (pre-sigmoid output) for numerical stability.
    Internally applies sigmoid, computes loss, and stores the clean gradient.
    """

    EPS = 1e-12   # prevent log(0)

    def __init__(self):
        self._y_hat: float | None = None
        self._y:     float | None = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, logit: float, y: float) -> float:
        """
        Parameters
        ----------
        logit : float — raw pre-sigmoid output of the network
        y     : float — ground truth label, 0 or 1

        Returns
        -------
        loss  : float — scalar BCE loss for this sample
        """
        # Apply sigmoid here for numerical stability
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        self._y_hat = y_hat
        self._y     = y

        loss = -(y * np.log(y_hat + self.EPS)
                 + (1.0 - y) * np.log(1.0 - y_hat + self.EPS))
        return float(loss)

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self) -> float:
        """
        Gradient of BCE loss w.r.t. the LOGIT (pre-sigmoid value).

        Derivation:
          dL/d(logit) = dL/dŷ · dŷ/d(logit)
                      = [-(y/ŷ) + (1-y)/(1-ŷ)] · ŷ·(1-ŷ)
                      = ŷ - y      ← beautiful simplification

        This is why we compute BCE on the logit, not on ŷ directly.
        """
        return self._y_hat - self._y    # scalar

    def zero_grad(self):
        self._y_hat = None
        self._y     = None