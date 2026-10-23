"""
models/scratch_ncf/loss.py

Binary Cross-Entropy loss with numerically stable implementation

BCE(y, yhat) = -[y * log(yhat) + (1 - y) * log(1 - yhat)]

The combined sigmoid+BCE gradient is dL / d(logit) = yhat - y
This is the most important simplification in the whole model.

"""

import numpy as np

class BCELoss:
    """
    Binary Cross-Entropy loss

    We operate on the raw logit (pre-sigmoid output) for numerical stability.
    Internally applies sigmoid, compute loss, and stores the clean gradient.
    """

    EPS = 1e-12

    def __init__(self):
        self._y_hat: float | None = None
        self._y: float | None = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, logit: float, y:float) -> float:
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
        self._y = y

        # Compute numerically stable BCE loss
        loss = -(y * np.log(y_hat + self.EPS)
                + (1.0 - y) * np.log(1.0 - y_hat + self.EPS))
        return loss
    
    def backward(self) -> float:
        """
        Gradient of BCE loss w.r.t. the LOGIT (pre-sigmoid value).

        Derivation:
          dL/d(logit) = dL/dŷ · dŷ/d(logit)
                      = [-(y/ŷ) + (1-y)/(1-ŷ)] · ŷ·(1-ŷ)
                      = ŷ - y      ← beautiful simplification

        This is why we compute BCE on the logit, not on ŷ directly.
        """
        if self._y_hat is None or self._y is None:
            raise ValueError("forward() must be called before backward()")
        
        return self._y_hat - self._y        # scalar
    
    def zero_grad(self):
        """
        Reset stored values for the next forward pass.
        """
        self._y_hat = None
        self._y = None