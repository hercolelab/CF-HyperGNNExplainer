import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Attention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 16, max_hops: int = 3) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_hops = int(max_hops)
        self.hop_scale = nn.Embedding(self.max_hops + 1, 1)
        self.emb_linear_node = nn.Linear(embed_dim, hidden_dim)
        self.emb_linear_hedge = nn.Linear(embed_dim, hidden_dim)

        nn.init.ones_(self.hop_scale.weight)
        for linear in (self.emb_linear_node, self.emb_linear_hedge):
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
        scale = self.hop_scale(hop_distances.clamp(max=self.max_hops)).to(q.dtype)
        omega = scale * (q @ k.t())

        pair_mask = H_dense > 0
        omega_masked = omega.masked_fill(~pair_mask, float("-inf"))
        alpha = F.softmax(omega_masked, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0) * pair_mask.to(dtype=z_nodes.dtype)
        return alpha, omega_masked
