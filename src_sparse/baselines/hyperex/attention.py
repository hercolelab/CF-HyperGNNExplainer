import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Attention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 16, max_hops: int = 3) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_hops = int(max_hops)
        self.bias = nn.Parameter(torch.tensor([1.0]))
        self.emb_linear_node = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU()
        )
        self.emb_linear_hedge = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU()
        )

        for linear in (self.emb_linear_node[0], self.emb_linear_hedge[0]):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0.01)

    def forward_dense(
        self,
        z_nodes: Tensor,
        h_edges: Tensor,
        H_dense: Tensor,
        hop_distances: Tensor,
    ) -> tuple[Tensor, Tensor]:
        q = self.emb_linear_node(z_nodes)
        k = self.emb_linear_hedge(h_edges)
        source_scale = torch.where(
            hop_distances == 0,
            torch.ones_like(hop_distances, dtype=q.dtype),
            self.bias.to(q.dtype).expand_as(hop_distances.to(q.dtype)),
        ).unsqueeze(1)
        omega = source_scale * (q @ k.t())

        pair_mask = H_dense > 0
        omega_masked = omega.masked_fill(~pair_mask, float("-inf"))
        flat_scores = omega[pair_mask]
        alpha = torch.zeros_like(H_dense, dtype=z_nodes.dtype)
        if flat_scores.numel() > 0:
            alpha[pair_mask] = F.softmax(flat_scores, dim=0)
        return alpha, omega_masked
