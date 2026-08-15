"""
production/recommenders/ncf/model.py
MLP-NCF matching production_model.pt checkpoint exactly.
  user_emb: [n_users, 32]   item_emb: [n_items, 32]
  mlp: Linear(64→64) ReLU Linear(64→32) ReLU
  output: Linear(32→1)
"""
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchNCF(nn.Module):

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = [64, 32]

        # NO +1 — checkpoint was saved without padding index
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        layers = []
        in_dim = emb_dim * 2
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(hidden[-1], 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(user_ids), self.item_emb(item_ids)], dim=-1)
        return self.output(self.mlp(x)).squeeze(-1)