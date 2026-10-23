"""
research/models/scratch_ncf/forward.py

NCF model: wires embeddings + linear layers into a complete forward pass.
Also provides the full backward pass through the same wiring in reverse.
"""

import numpy as np
from .embeddings import EmbeddingLayer
from .layers import LinearLayer, ReLU

class Scratch_NCF:
    """
    Neural Collaborative Filtering implemented from scratch with NumPy.

    Architecture
    ------------
    user_id  item_id
       |        |
    P[u,:]   Q[i,:]        ← embedding lookup (k-dim each)
        \     /
       concat z0            ← shape (2k,)
          |
      Linear(2k → h1)
      ReLU
          |
      Linear(h1 → h2)
      ReLU
          |
      Linear(h2 → 1)       ← output logit (sigmoid applied in loss)
    """

    def __init__(
        self,
        n_users:    int,
        n_items:    int,
        emb_dim:    int = 32,
        hidden:     list[int] = [64, 32],
        seed:       int = 42,
    ):
        self.n_users    = n_users
        self.n_items    = n_items
        self.emb_dim    = emb_dim
        self.hidden     = hidden


        # Ebeddings
        self.user_emb   = EmbeddingLayer(n_users, emb_dim, seed=seed)
        self.item_emb   = EmbeddingLayer(n_items, emb_dim, seed=seed + 1)

        # MLP layers - sizes: 2k -> h1 -> h2 -> 1
        layer_sizes =[2 * emb_dim] + hidden + [1]
        self.linears = [
            LinearLayer(layer_sizes[i], layer_sizes[i + 1], seed=seed + i + 2)
            for i in range(len(layer_sizes) - 1)
        ]
        self.relus = [ReLU() for _ in range(len(hidden))]   # one per hidden layer, not output

        # Caches filled during forward, consumed during backward
        self._p_u:  np.ndarray | None = None
        self._q_i:  np.ndarray | None = None
        self._z:    list[np.ndarray] = []       #pre-activation at each layer
        self._a:    list[np.ndarray] = []       #post-activation (output of ReLU) at each layer
        
# ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, user_id : int, item_id: int) -> float:
        """
        Run one forward pass for a single (user, item) pair.

        Returns
        -------
        logit : float — pre-sigmoid score
        """
        
        # 1. Embedding lookp
        self._p_u   = self.user_emb.forward(user_id)    # (k,)
        self._q_i   = self.item_emb.forward(item_id)    # (k,) 

        # 2. Concatenate
        z0 = np.concatenate([self._p_u, self._q_i])     # (2k,)


        # 3. MLP forward
        self._z = []
        self._a = [z0]                                  # a[0] = input to layer 0

        x = z0
        for idx, linear in enumerate(self.linears):
            z = linear.forward(x)                    # pre-activation
            self._z.append(z)
            if idx < len(self.relus):                # hidden layers get ReLU
                x = self.relus[idx].forward(z)
            else:                                    # output layer — no activation
                x = z
            self._a.append(x)

        logit = float(x[0])                          # scalar
        return logit

# ── Backward ─────────────────────────────────────────────────────────────
    
    def backward(self, loss_grad: float) -> None:
        """
        Backpropagate loss gradient through the entire network.

        Parameters
        ----------
        loss_grad : float — dL/d(logit) from BCELoss.backward()
                            This is (ŷ − y), a scalar.
        """

        delta = np.array([loss_grad])

        # Propagate backward through layers in reverse
        for idx in range(len(self.linears) - 1, -1, -1):
            # If this linear layer is a hidden layer (idx < len(self.relus)), 
            # it was followed by a ReLU in forward pass. Push delta through it.
            if idx < len(self.relus):
                delta = self.relus[idx].backward(delta)

            delta = self.linears[idx].backward(delta)

        # FIX: Un-indent this block so it executes AFTER the loop finishes 
        # delta now holds dL/dz0, shape (2k,)
        # split into user and item embedding gradients
        grad_p = delta[:self.emb_dim]           # first k components
        grad_q = delta[self.emb_dim:]           # last k components
        
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
        """Inference mode — returns probability ŷ ∈ [0,1]."""
        logit = self.forward(user_id, item_id)
        return 1.0 / (1.0 + np.exp(-logit))
    
    def get_all_params(self) -> list[dict]:
        """Return list of {param, grad} dicts for the optimizer."""
        params = []
        params.append({"param": self.user_emb.W, "grad": self.user_emb.dW})
        params.append({"param": self.item_emb.W, "grad": self.item_emb.dW})
        for layer in self.linears:
            params.append({"param": layer.W, "grad": layer.dW})
            params.append({"param": layer.b, "grad": layer.db})
        return params