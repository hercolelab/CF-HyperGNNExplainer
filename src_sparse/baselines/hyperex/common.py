import torch
from torch import Tensor


def build_local_to_global(node_dict: dict[int, int], device: torch.device) -> Tensor:
    local_to_global = torch.empty(len(node_dict), dtype=torch.long, device=device)
    for global_idx, local_idx in node_dict.items():
        local_to_global[local_idx] = global_idx
    return local_to_global


def empty_hypergraph_like(H: Tensor, values_dtype: torch.dtype | None = None) -> Tensor:
    H = H.coalesce()
    dtype = values_dtype or H.dtype
    return torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long, device=H.device),
        torch.zeros(0, dtype=dtype, device=H.device),
        H.size(),
        device=H.device,
        dtype=dtype,
    ).coalesce()


def sparse_incidence_to_dense(comp_H: Tensor) -> Tensor:
    """Binary dense incidence [N, E] from sparse COO."""
    comp_H = comp_H.coalesce()
    n, e = comp_H.shape
    if n == 0 or e == 0:
        return torch.zeros(n, e, dtype=comp_H.dtype, device=comp_H.device)
    dense = torch.zeros(n, e, dtype=comp_H.dtype, device=comp_H.device)
    r, c = comp_H.indices()
    dense[r, c] = 1.0
    return dense


def compute_hyperedge_embeddings_global(full_H: Tensor, global_z: Tensor) -> Tensor:
    full_H = full_H.coalesce()
    _, num_edges = full_H.shape
    indices = full_H.indices()
    device = global_z.device
    dtype = global_z.dtype

    h_sum = torch.zeros(num_edges, global_z.size(1), dtype=dtype, device=device)
    h_sum.index_add_(0, indices[1], global_z[indices[0]])

    deg = torch.zeros(num_edges, dtype=dtype, device=device)
    deg.index_add_(
        0,
        indices[1],
        torch.ones(indices.size(1), dtype=dtype, device=device),
    )
    return h_sum / deg.clamp(min=1.0).unsqueeze(1)


def extract_induced_edge_global_ids(
    full_H: Tensor, node_dict: dict[int, int]
) -> Tensor:
    full_H = full_H.coalesce()
    num_nodes, num_edges = full_H.shape
    device = full_H.device

    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if node_dict:
        global_ids = torch.tensor(
            list(node_dict.keys()), dtype=torch.long, device=device
        )
        visited[global_ids] = True

    indices = full_H.indices()
    node_in_visited = visited[indices[0]].long()

    visited_count = torch.zeros(num_edges, dtype=torch.long, device=device)
    visited_count.scatter_add_(0, indices[1], node_in_visited)

    total_count = torch.zeros(num_edges, dtype=torch.long, device=device)
    total_count.scatter_add_(
        0, indices[1], torch.ones(indices.size(1), dtype=torch.long, device=device)
    )

    included = (visited_count == total_count) & (total_count > 0)
    return torch.nonzero(included, as_tuple=False).view(-1)


def local_class_probabilities(global_probs: Tensor, local_to_global: Tensor) -> Tensor:
    return global_probs[local_to_global]


def normalize_propagation_dense(H: Tensor) -> Tensor:
    d_inv_sqrt = H.sum(dim=1).clamp(min=1).pow(-0.5)
    b_inv = H.sum(dim=0).clamp(min=1).pow(-1.0)

    H_left = d_inv_sqrt.unsqueeze(1) * H * b_inv.unsqueeze(0)
    H_right = d_inv_sqrt.unsqueeze(1) * H
    return H_left @ H_right.t()


def select_topk_indices(score: Tensor, thresh_num: int) -> Tensor:
    if score.ndim != 1:
        raise ValueError("Expected a 1D score tensor.")
    if score.numel() == 0 or thresh_num <= 0:
        return torch.zeros(0, dtype=torch.long, device=score.device)
    k = min(int(thresh_num), score.numel())
    return torch.topk(score, k=k, largest=True).indices


def build_binary_hypergraph(comp_H: Tensor, selected_inds: Tensor) -> Tensor:
    comp_H = comp_H.coalesce()
    if selected_inds.numel() == 0:
        return empty_hypergraph_like(comp_H, values_dtype=comp_H.dtype)

    indices = comp_H.indices()[:, selected_inds]
    values = torch.ones(
        selected_inds.numel(),
        dtype=comp_H.dtype,
        device=comp_H.device,
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        comp_H.size(),
        device=comp_H.device,
        dtype=comp_H.dtype,
    ).coalesce()


def build_explanation_hypergraph_from_alpha(
    comp_H: Tensor,
    alpha: Tensor,
    thresh_num: int,
) -> Tensor:
    comp_H = comp_H.coalesce()
    rows, cols = comp_H.indices()
    if rows.numel() == 0:
        return empty_hypergraph_like(comp_H, values_dtype=comp_H.dtype)

    flat = alpha[rows, cols].detach()
    selected_inds = select_topk_indices(flat, thresh_num)
    return build_binary_hypergraph(comp_H, selected_inds)


def training_soft_weights_from_alpha(alpha: Tensor, H_dense: Tensor) -> Tensor:
    pair_mask = H_dense > 0
    return alpha * pair_mask.to(dtype=alpha.dtype)


def training_dense_weights_from_alpha(
    alpha: Tensor,
    H_dense: Tensor,
    thresh_num: int,
) -> Tensor:
    pair_mask = H_dense > 0
    flat = alpha[pair_mask]
    if flat.numel() == 0:
        return torch.zeros_like(H_dense)

    k = min(int(thresh_num), flat.numel())
    top_idx = torch.topk(flat, k=k, largest=True).indices
    masked_flat = torch.zeros_like(flat)
    masked_flat.scatter_(0, top_idx, flat[top_idx])
    s = masked_flat.sum().clamp(min=1e-12)
    masked_flat = masked_flat / s

    dense_w = torch.zeros_like(H_dense)
    dense_w[pair_mask] = masked_flat
    return dense_w
