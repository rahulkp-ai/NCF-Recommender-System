"""
models/pytorch_ncf/model.py

NCF implemented in PyTorch — identical architecture to ScratchNCF.

Architecture parity with scratch implementation:
  - nn.Embedding  ↔  EmbeddingLayer
  - nn.Linear     ↔  LinearLayer
  - F.relu        ↔  ReLU.forward()
  - autograd      ↔  manual backward()
  - BCEWithLogitsLoss ↔  BCELoss (numerically equivalent)

MPS (Apple Silicon) acceleration is enabled automatically when available.
"""

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Return MPS device on M1/M2 Mac, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PyTorchNCF(nn.Module):
    """
    Neural Collaborative Filtering — PyTorch implementation.

    Parameters
    ----------
    n_users  : int   — number of users (embedding table rows)
    n_items  : int   — number of items (embedding table rows)
    emb_dim  : int   — embedding dimension k
    hidden   : list  — hidden layer sizes, e.g. [64, 32]
    dropout  : float — dropout rate (0 = disabled, for fair scratch comparison)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        hidden: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden is None:
            hidden = [64, 32]

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        # ── Embedding layers (identical to EmbeddingLayer in scratch) ──
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # ── MLP layers ────────────────────────────────────────────────
        layer_sizes = [2 * emb_dim] + hidden
        mlp_layers = []
        for in_sz, out_sz in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            mlp_layers.append(nn.Linear(in_sz, out_sz))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_layers)

        # ── Output layer (logit — sigmoid applied inside loss) ─────────
        self.output = nn.Linear(hidden[-1], 1)

        # He initialisation — matches scratch implementation
        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialisation for all linear layers and embeddings."""
        nn.init.kaiming_normal_(self.user_emb.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.item_emb.weight, mode="fan_in", nonlinearity="relu")
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.kaiming_normal_(self.output.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.output.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Batch forward pass.

        Parameters
        ----------
        user_ids : LongTensor shape (B,)
        item_ids : LongTensor shape (B,)

        Returns
        -------
        logits   : FloatTensor shape (B,)  — pre-sigmoid scores
        """
        p_u = self.user_emb(user_ids)  # (B, k)
        q_i = self.item_emb(item_ids)  # (B, k)

        z0 = torch.cat([p_u, q_i], dim=1)  # (B, 2k)
        hidden = self.mlp(z0)  # (B, h_last)
        logits = self.output(hidden).squeeze(1)  # (B,)
        return logits

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Inference mode — returns probabilities ŷ ∈ [0,1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(user_ids, item_ids))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"PyTorchNCF(n_users={self.n_users}, n_items={self.n_items}, "
            f"emb_dim={self.emb_dim}, params={self.count_parameters():,})"
        )
