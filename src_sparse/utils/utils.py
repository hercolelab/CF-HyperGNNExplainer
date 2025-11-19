import torch
from typing import Tuple, Dict


def normalize_propagation(H: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Compute the normalized propagation matrix of a hypergraph.
    Args:
        H: Hypergraph incidence matrix (torch.sparse_coo_tensor)
    Returns:
        S: Normalized propagation matrix (torch.sparse_coo_tensor)
    """
    assert H.layout == torch.sparse_coo

    H = H.coalesce()

    node_degrees = torch.sparse.sum(H, dim=1).to_dense()  # [num_nodes]
    hyperedge_degrees = torch.sparse.sum(H, dim=0).to_dense()  # [num_hyperedges]

    # D^{-1/2}
    d_inv_sqrt = node_degrees.pow(-0.5)
    d_inv_sqrt = torch.where(
        torch.isinf(d_inv_sqrt),
        torch.zeros_like(d_inv_sqrt),
        d_inv_sqrt,
    )

    # B^{-1}
    b_inv = hyperedge_degrees.pow(-1.0)
    b_inv = torch.where(
        torch.isinf(b_inv),
        torch.zeros_like(b_inv),
        b_inv,
    )

    indices = H.indices()  # [2, nnz]
    values = H.values()  # [nnz]
    row, col = indices[0], indices[1]

    d_row = d_inv_sqrt[row]  # D^{-1/2} on rows
    b_col = b_inv[col]  # B^{-1} on cols

    # Build two scaled versions of H:
    #   H_left  = D^{-1/2} H B^{-1}
    #   H_right = D^{-1/2} H
    values_left = values * d_row * b_col
    values_right = values * d_row  # this will be transposed to generate H^T D^{-1/2}

    H_left = torch.sparse_coo_tensor(
        indices, values_left, H.size(), device=H.device, dtype=H.dtype
    ).coalesce()
    H_right = torch.sparse_coo_tensor(
        indices, values_right, H.size(), device=H.device, dtype=H.dtype
    ).coalesce()

    # S = H_left @ H_right^T = D^{-1/2} H B^{-1} H^T D^{-1/2}
    S = torch.sparse.mm(H_left, H_right.t())

    return S


def bool_to_col(vec_bool, H):
    return vec_bool.to(dtype=H.dtype).view(-1, 1)


# def get_hyper_neighbourhood_fast(
#     node_idx: int,
#     H: torch.sparse_coo_tensor,
#     n_hops: int,
#     features: torch.Tensor,
#     labels: torch.Tensor,
# ) -> Tuple[torch.sparse_coo_tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
#     """
#     Extract k-hop neighborhood subgraph from hypergraph incidence matrix (faster).

#     This uses bipartite propagation (node -> hyperedge -> node) with sparse-dense matmul
#     and avoids materializing HH^T or using torch.isin over large index arrays.

#     Returns: (sub_H, sub_feat, sub_labels, node_dict)
#     """
#     device = features.device

#     if not H.is_sparse:
#         H = H.to_sparse()
#     H = H.to(device).coalesce()

#     N = H.size(0)
#     E = H.size(1)

#     if node_idx < 0 or node_idx >= N:
#         raise IndexError("node_idx out of range")

#     visited = torch.zeros(N, dtype=torch.bool, device=device)
#     current_bool = torch.zeros(N, dtype=torch.bool, device=device)
#     current_bool[node_idx] = True
#     visited[node_idx] = True

#     for _ in range(n_hops):
#         if not current_bool.any():
#             break

#         current_vec = bool_to_col(current_bool, H)  # shape (N,1)
#         e_scores = torch.sparse.mm(H.transpose(0, 1), current_vec)  # dense (E,1)
#         e_mask = e_scores.view(-1) > 0  # which hyperedges are incident

#         if not e_mask.any():
#             break

#         e_vec = bool_to_col(e_mask, H)  # (E,1)
#         node_scores = torch.sparse.mm(H, e_vec)  # dense (N,1)
#         next_bool = node_scores.view(-1) > 0

#         new_nodes = next_bool & (~visited)
#         visited |= next_bool

#         if not new_nodes.any():
#             break

#         current_bool = new_nodes

#     nodes = visited.nonzero(as_tuple=False).view(-1)
#     if nodes.numel() == 0:
#         empty_idx = torch.zeros((2, 0), dtype=torch.long, device=device)
#         empty_vals = torch.zeros((0,), dtype=H.dtype, device=device)
#         return (
#             torch.sparse_coo_tensor(
#                 empty_idx, empty_vals, (0, 0), device=device, dtype=H.dtype
#             ),
#             features[[]],
#             labels[[]],
#             {},
#         )

#     H_indices = H.indices()
#     H_values = H.values()
#     H_rows = H_indices[0]
#     H_cols = H_indices[1]

#     row_mask = visited[H_rows]
#     sub_H_rows_orig = H_rows[row_mask]
#     sub_H_cols_orig = H_cols[row_mask]
#     sub_H_values = H_values[row_mask]

#     node_to_sub_idx = torch.full((N,), -1, dtype=torch.long, device=device)
#     node_to_sub_idx[nodes] = torch.arange(
#         nodes.numel(), device=device, dtype=torch.long
#     )
#     sub_H_rows = node_to_sub_idx[sub_H_rows_orig]

#     if sub_H_cols_orig.numel() > 0:
#         unique_edges = torch.unique(sub_H_cols_orig)
#         edge_to_sub_idx = torch.full((E,), -1, dtype=torch.long, device=device)
#         edge_to_sub_idx[unique_edges] = torch.arange(
#             unique_edges.numel(), device=device, dtype=torch.long
#         )
#         sub_H_cols = edge_to_sub_idx[sub_H_cols_orig]

#         sub_H_indices = torch.stack([sub_H_rows, sub_H_cols], dim=0)
#         sub_H = torch.sparse_coo_tensor(
#             sub_H_indices,
#             sub_H_values,
#             (nodes.numel(), unique_edges.numel()),
#             device=device,
#             dtype=H.dtype,
#         ).coalesce()
#     else:
#         sub_H = torch.sparse_coo_tensor(
#             torch.zeros((2, 0), dtype=torch.long, device=device),
#             torch.zeros(0, dtype=H.dtype, device=device),
#             (nodes.numel(), 0),
#             device=device,
#             dtype=H.dtype,
#         )

#     sub_feat = features[nodes]
#     sub_labels = labels[nodes]

#     node_dict = {int(orig): int(i) for i, orig in enumerate(nodes.cpu().numpy())}

#     return sub_H, sub_feat, sub_labels, node_dict


def get_hyper_neighbourhood_fast_2(
    node_idx: int,
    H: torch.sparse_coo_tensor,
    n_hops: int,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.sparse_coo_tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    Extract k-hop neighborhood subgraph from hypergraph incidence matrix.

    Args:
        node_idx: The target central node index.
        H: Sparse Incidence Matrix (Nodes x Hyperedges).
        n_hops: Number of hops for BFS.
        features: Node feature tensor (N, D).
        labels: Node label tensor (N,).

    Returns:
        sub_H: Sparse incidence matrix of the subgraph (sub_N, sub_E).
        sub_feat: Features of the subgraph nodes.
        sub_labels: Labels of the subgraph nodes.
        node_dict: Mapping {Global Node ID -> Local Node ID}.
    """
    device = features.device

    if H.device != device:
        H = H.to(device)
    if not H.is_sparse:
        H = H.to_sparse()
    if not H.is_coalesced():
        H = H.coalesce()

    N, E = H.shape

    if not (0 <= node_idx < N):
        raise IndexError(f"node_idx {node_idx} is out of bounds for size {N}")

    # BFS Traversal (Node -> Hyperedge -> Node)
    visited = torch.zeros(N, dtype=torch.bool, device=device)
    frontier = torch.zeros(N, dtype=torch.bool, device=device)

    visited[node_idx] = True
    frontier[node_idx] = True

    H_t = H.t()

    for _ in range(n_hops):
        if not frontier.any():
            break

        # Propagate Nodes -> Hyperedges
        node_vec = frontier.to(dtype=H.dtype).view(-1, 1)

        edge_scores = torch.sparse.mm(H_t, node_vec)
        active_edges_mask = edge_scores.view(-1) > 0

        if not active_edges_mask.any():
            break

        # Propagate Hyperedges -> Nodes
        edge_vec = active_edges_mask.to(dtype=H.dtype).view(-1, 1)

        node_scores = torch.sparse.mm(H, edge_vec)
        found_nodes_mask = node_scores.view(-1) > 0

        new_nodes = found_nodes_mask & (~visited)

        if not new_nodes.any():
            break

        visited |= new_nodes
        frontier = new_nodes

    # Subgraph Extraction
    subset_node_indices = visited.nonzero(as_tuple=False).view(-1)
    num_sub_nodes = subset_node_indices.numel()

    if num_sub_nodes == 0:
        return _create_empty_response(device, H.dtype)

    H_indices = H.indices()
    H_vals = H.values()

    row_mask = visited[H_indices[0]]

    raw_sub_rows = H_indices[0][row_mask]
    raw_sub_cols = H_indices[1][row_mask]
    sub_vals = H_vals[row_mask]

    # Remap Nodes (Global ID -> Local ID 0..k)
    global_to_local_node = torch.full((N,), -1, dtype=torch.long, device=device)
    global_to_local_node[subset_node_indices] = torch.arange(
        num_sub_nodes, device=device
    )

    new_sub_rows = global_to_local_node[raw_sub_rows]

    # Remap Hyperedges (Global Edge ID -> Local Edge ID 0..m)
    if raw_sub_cols.numel() > 0:
        unique_edges, new_sub_cols = torch.unique(raw_sub_cols, return_inverse=True)
        num_sub_edges = unique_edges.numel()

        sub_H_indices = torch.stack([new_sub_rows, new_sub_cols], dim=0)

        sub_H = torch.sparse_coo_tensor(
            sub_H_indices,
            sub_vals,
            (num_sub_nodes, num_sub_edges),
            device=device,
            dtype=H.dtype,
        ).coalesce()
    else:
        sub_H = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=H.dtype, device=device),
            (num_sub_nodes, 0),
            device=device,
            dtype=H.dtype,
        )

    # Features & Labels
    sub_feat = features[subset_node_indices]
    sub_labels = labels[subset_node_indices]

    # Create mapping dict (CPU)
    subset_nodes_cpu = subset_node_indices.cpu().numpy()
    node_dict = {
        int(global_id): local_id for local_id, global_id in enumerate(subset_nodes_cpu)
    }

    return sub_H, sub_feat, sub_labels, node_dict


def _create_empty_response(device: torch.device, dtype: torch.dtype):
    """Helper to return empty structures if node is isolated."""
    empty_sparse = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long, device=device),
        torch.zeros(0, dtype=dtype, device=device),
        (0, 0),
        device=device,
        dtype=dtype,
    )
    return (
        empty_sparse,
        torch.tensor([], device=device),
        torch.tensor([], device=device),
        {},
    )
