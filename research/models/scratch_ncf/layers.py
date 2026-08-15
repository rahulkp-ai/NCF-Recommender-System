"""
research/models/scratch_ncf/layers.py

Linear layer and activation functions with full forward/backward support.

Each layer stores its inputs during forward pass — these are needed
to compute weight gradients during backward pass.
"""

import numpy as np

class LinearLayer:
    """
    Fully-connected layer: z = W @ x + b

    Parameters
    ----------
    in_features  : int — input dimension
    out_features : int — output dimension
    seed         : int — for reproducible initialisation

    Shapes
    ------
    W  : (out_features, in_features)
    b  : (out_features,)
    dW : (out_features, in_features)  — gradient of L w.r.t. W
    db : (out_features,)              — gradient of L w.r.t. b
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)

        # He initialisation: std = sqrt(2 / in_features)
        std = np.sqrt(2.0 / in_features)
        self.W  = rng.normal(0.0, std, (out_features, in_features))
        self.b  = np.zeros(out_features)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cached input for backward pass
        self._x: np.ndarray | None = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        z = W @ x + b
        """
        self._x = x.copy()                  # cache for backward
        return self.W @ x + self.b          # shape (out_features,)

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Given upstream gradient delta = dL/dz, compute:
          dL/dW = delta ⊗ x^T   (outer product)
          dL/db = delta
          dL/dx = W^T @ delta   (passed to previous layer)
        """
        assert self._x is not None, "Call forward() before backward()"
        
        # Ensure delta is at least 1D (handles scalar inputs from output layer)
        delta = np.atleast_1d(delta)

        # Accumulate parameter gradients
        # np.outer handles (out,) and (in,) to create (out, in) matrix
        grad_w = np.outer(delta, self._x)
        
        if self.dW.shape != grad_w.shape:
            raise ValueError(
                f"Shape mismatch: dW is {self.dW.shape}, grad is {grad_w.shape}. "
                f"Check layer dimensions or backward loop order."
            )

        self.dW += grad_w
        self.db += delta

        # Gradient to pass to previous layer
        dx = self.W.T @ delta                 # (in_features,)
        return dx

    def zero_grad(self) -> None:
        self.dW[:] = 0.0
        self.db[:] = 0.0
        self._x = None

    def __repr__(self):
        return (f"LinearLayer(in={self.W.shape[1]}, "
                f"out={self.W.shape[0]})")


# ── Activation functions ──────────────────────────────────────────────────────

class ReLU:
    """
    ReLU(x) = max(0, x)
    Derivative: 1 if x > 0, else 0
    """

    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0).astype(float)
        return x * self._mask

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Pass gradient only through active neurons."""
        assert self._mask is not None
        return delta * self._mask

    def zero_grad(self):
        self._mask = None


class Sigmoid:
    """
    σ(x) = 1 / (1 + exp(-x))
    """

    def __init__(self):
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable clip
        x_clipped = np.clip(x, -500, 500)
        out = 1.0 / (1.0 + np.exp(-x_clipped))
        self._out = out
        return out

    def backward(self, delta: np.ndarray) -> np.ndarray:
        s = self._out
        assert s is not None
        return delta * s * (1.0 - s)

    def zero_grad(self):
        self._out = None