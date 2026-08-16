"""
research/models/scratch_ncf/forward.py

NCF model: wires embeddings + linear layers into a complete forward pass.
Also provides the full backward pass through the same wiring in reverse.
"""

import numpy as np

from .embeddings import EmbeddingLayer
from .layers import LinearLayer, ReLU


class ScratchNCF:
    """
    Neural Collaborative Filtering implemented from scratch with NumPy.

    Architecture:
    User/Item IDs -> Embeddings -> Concatenate -> MLP (Linear+ReLU) -> Output Logit
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        hidden: list[int] | None = None,
        seed: int = 42,
    ):
        if hidden is None:
            hidden = [64, 32]
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.hidden = hidden

        # 1. Embeddings
        self.user_emb = EmbeddingLayer(n_users, emb_dim, seed=seed)
        self.item_emb = EmbeddingLayer(n_items, emb_dim, seed=seed + 1)

        # 2. MLP layers logic: 2k -> h1 -> h2 -> 1
        layer_sizes = [2 * emb_dim] + hidden + [1]

        self.linears = []
        for i in range(len(layer_sizes) - 1):
            self.linears.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1], seed=seed + i + 2))

        # ReLUs are only for hidden layers (not the final output layer)
        self.relus = [ReLU() for _ in hidden]

        # Caches for backward pass
        self._p_u: np.ndarray | None = None
        self._q_i: np.ndarray | None = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, user_id: int, item_id: int) -> float:
        """
        Run one forward pass for a single (user, item) pair.
        """
        # 1. Embedding lookup
        self._p_u = self.user_emb.forward(user_id)  # (k,)
        self._q_i = self.item_emb.forward(item_id)  # (k,)

        # 2. Concatenate
        x = np.concatenate([self._p_u, self._q_i])  # (2k,)

        # 3. MLP forward
        for idx, linear in enumerate(self.linears):
            x = linear.forward(x)  # Linear: Wx + b
            if idx < len(self.relus):  # Apply ReLU if not output layer
                x = self.relus[idx].forward(x)

        return float(x[0])  # Return scalar logit

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self, loss_grad: float) -> None:
        """
        Backpropagate loss gradient through the entire network.
        loss_grad: (ŷ − y) scalar
        """
        # Start with the gradient of the loss w.r.t logit
        delta = np.array([loss_grad])  # shape (1,)

        # Propagate backward through Linear and ReLU layers in reverse
        for idx in range(len(self.linears) - 1, -1, -1):
            # If the current linear layer was followed by a ReLU (hidden layers),
            # we must pass delta through ReLU backward first.
            if idx < len(self.relus):
                delta = self.relus[idx].backward(delta)

            # Pass through Linear layer backward
            delta = self.linears[idx].backward(delta)

        # After the loop, delta is the gradient w.r.t the concatenated embeddings (2k,)
        grad_p = delta[: self.emb_dim]  # User embedding gradient
        grad_q = delta[self.emb_dim :]  # Item embedding gradient

        self.user_emb.backward(grad_p)
        self.item_emb.backward(grad_q)

    # ── Utilities ────────────────────────────────────────────────────────────

    def zero_grad(self):
        """Reset all accumulated gradients."""
        self.user_emb.zero_grad()
        self.item_emb.zero_grad()
        for layer in self.linears:
            layer.zero_grad()
        for relu in self.relus:
            relu.zero_grad()

    def predict(self, user_id: int, item_id: int) -> float:
        """Returns probability ŷ ∈ [0,1]."""
        logit = self.forward(user_id, item_id)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))

    def get_all_params(self) -> list[dict]:
        """Return list of {param, grad} dicts for the optimizer."""
        params = []
        params.append({"param": self.user_emb.W, "grad": self.user_emb.dW})
        params.append({"param": self.item_emb.W, "grad": self.item_emb.dW})
        for layer in self.linears:
            params.append({"param": layer.W, "grad": layer.dW})
            params.append({"param": layer.b, "grad": layer.db})
        return params

    def __repr__(self):
        return (
            f"ScratchNCF(users={self.n_users}, items={self.n_items}, "
            f"emb_dim={self.emb_dim}, architecture={self.hidden})"
        )
