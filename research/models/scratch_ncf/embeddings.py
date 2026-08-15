"""
research/models/scratch_ncf/embeddings.py

Embedding matrices P (users) and Q (items).

An embedding is a lookup table: given an integer ID, return a dense vector.
In NumPy this is just matrix indexing: P[user_id, :].

Thesis note: This is equivalent to a one-hot encoded input multiplied by a
weight matrix, but the lookup avoids the massive sparse matrix multiply.
"""

import numpy as np


class EmbeddingLayer:
    """
    Learnable embedding matrix of shape (num_embeddings, embedding_dim).

    Attributes
    ----------
    W       : ndarray (n, k) — the embedding matrix
    dW      : ndarray (n, k) — accumulated gradients (sparse updates)
    n, k    : int — vocabulary size and embedding dimension
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,seed: int = 42):
        rng = np.random.default_rng(seed)

        # He (Kaiming) initialisation for ReLU networks:
        #   std = sqrt(2 / fan_in)
        # For embeddings fan_in = embedding_dim
        std = np.sqrt(2.0 / embedding_dim)
        self.W  = rng.normal(0.0, std, (num_embeddings, embedding_dim))
        self.dW = np.zeros_like(self.W)

        self.n = num_embeddings
        self.k = embedding_dim

        # Cache for backprop — which row did we look up?
        self._last_idx: int | None = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, idx: int) -> np.ndarray:
        """
        Look up embedding vector for index idx.

        Parameters
        ----------
        idx : int — the user or item integer ID

        Returns
        -------
        vec : ndarray shape (k,) — the embedding vector (a VIEW of W)
        """
        assert 0 <= idx < self.n, f"Index {idx} out of range [0, {self.n})"
        self._last_idx = idx
        return self.W[idx]                   # shape (k,)

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self, grad: np.ndarray) -> None:
        """
        Accumulate gradient into dW[last_idx].

        Embedding gradients are SPARSE: only the looked-up row receives a
        gradient. All other rows have zero gradient for this sample.

        Parameters
        ----------
        grad : ndarray shape (k,) — gradient of loss w.r.t. embedding vector
        """
        assert self._last_idx is not None, "Call forward() before backward()"
        self.dW[self._last_idx] += grad      # accumulate (for mini-batches)

    def zero_grad(self) -> None:
        """Reset accumulated gradients to zero before each batch."""
        self.dW[:] = 0.0
        self._last_idx = None

    def __repr__(self):
        return (f"EmbeddingLayer(num_embeddings={self.n}, "
                f"embedding_dim={self.k})")