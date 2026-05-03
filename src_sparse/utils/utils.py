import torch
from torch import Tensor
from typing import Tuple, Dict, Any

def build_incidence_matrix(
    edge_index: Tensor, num_nodes: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Convert CORA's edge_index to a hypergraph incidence matrix (sparse COO format).
    Each node defines a hyperedge containing itself and its first-order neighbors.
    """
    edge_index = edge_index.cpu()
    row, col = edge_index

    adjacency = [set[Any]() for _ in range(num_nodes)]
    for src, dst in zip(row.tolist(), col.tolist()):
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    rows, cols = [], []
    for hyperedge_id in range(num_nodes):
        hyper_nodes = set(adjacency[hyperedge_id])
        hyper_nodes.add(hyperedge_id)
        for node_id in hyper_nodes:
            rows.append(node_id)
            cols.append(hyperedge_id)

    row_idx = torch.tensor(rows, dtype=torch.long)
    col_idx = torch.tensor(cols, dtype=torch.long)
    indices = torch.stack([row_idx, col_idx], dim=0)
    values = torch.ones(len(rows), dtype=torch.float32)

    H = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), dtype=torch.float32, device=device
    ).coalesce()

    return H

def graph_to_hypergraph(
    edge_index: Tensor, num_nodes: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Convert edge_index to a hypergraph incidence matrix (sparse COO).

    Each undirected edge is one hyperedge on its two endpoints (one node if self-loop).
    Duplicate orientations in edge_index are merged. Shape is (num_nodes, num_edges).
    """
    edge_index = edge_index.cpu()
    row, col = edge_index[0], edge_index[1]
    if row.numel() == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            (num_nodes, 0),
            dtype=torch.float32,
            device=device,
        ).coalesce()

    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    pairs = torch.stack([u, v], dim=1)
    unique_pairs = torch.unique(pairs, dim=0)
    num_edges = unique_pairs.shape[0]

    ue, ve = unique_pairs[:, 0], unique_pairs[:, 1]
    e = torch.arange(num_edges, dtype=torch.long)
    row_dup = torch.cat([ue, ve])
    col_dup = torch.cat([e, e])
    stacked = torch.stack([row_dup, col_dup], dim=1)
    unique_stacked = torch.unique(stacked, dim=0)
    row_idx = unique_stacked[:, 0].contiguous()
    col_idx = unique_stacked[:, 1].contiguous()

    indices = torch.stack([row_idx, col_idx], dim=0)
    values = torch.ones(row_idx.shape[0], dtype=torch.float32)

    H = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_edges), dtype=torch.float32, device=device
    ).coalesce()

    return H


def normalize_propagation(H: torch.Tensor) -> torch.Tensor:
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


def _ensure_sparse_coalesced(H: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move H to device and ensure it is a coalesced sparse COO tensor."""
    if H.device != device:
        H = H.to(device)
    if not H.is_sparse:
        H = H.to_sparse()
    if not H.is_coalesced():
        H = H.coalesce()
    return H


def _expand_frontier(
    frontier: torch.Tensor,
    H: torch.Tensor,
    H_t: torch.Tensor,
    visited: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expand the current node frontier by one hop through hyperedges.

    Returns:
        A pair of boolean masks `(updated_visited, new_frontier)`.
    """
    edge_scores = torch.sparse.mm(H_t, bool_to_col(frontier, H)).view(-1)
    edges_hit = edge_scores > 0
    if not edges_hit.any():
        return visited, torch.zeros_like(frontier)

    node_scores = torch.sparse.mm(H, bool_to_col(edges_hit, H)).view(-1)
    new_nodes = (node_scores > 0) & ~visited
    return visited | new_nodes, new_nodes


def _collect_visited_nodes(
    node_idx: int,
    H: torch.Tensor,
    n_hops: int,
    device: torch.device,
) -> torch.Tensor:
    """Run an n-hop BFS over the hypergraph and return the visited-node mask."""
    num_nodes = H.shape[0]
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    frontier = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    visited[node_idx] = True
    frontier[node_idx] = True

    H_t = H.t().coalesce()
    for _ in range(n_hops):
        if not frontier.any():
            break
        visited, frontier = _expand_frontier(frontier, H, H_t, visited)
        if not frontier.any():
            break

    return visited


def _build_induced_subhypergraph(
    H: torch.Tensor,
    visited: torch.Tensor,
    subset_node_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the induced subhypergraph over the visited nodes.

    A hyperedge is included iff all of its incident nodes are visited.
    Node ids are remapped to local indices and included hyperedge ids are compressed.
    """
    H_indices = H.indices()
    H_vals = H.values()
    num_nodes, num_edges = H.shape
    num_sub_nodes = subset_node_indices.numel()

    node_in_visited = visited[H_indices[0]].to(dtype=torch.long)

    visited_count = torch.zeros(num_edges, dtype=torch.long, device=device)
    visited_count.scatter_add_(0, H_indices[1], node_in_visited)

    total_count = torch.zeros(num_edges, dtype=torch.long, device=device)
    total_count.scatter_add_(
        0, H_indices[1], torch.ones(H_indices.shape[1], dtype=torch.long, device=device)
    )

    included_edges = (visited_count == total_count) & (total_count > 0)
    entry_mask = included_edges[H_indices[1]]

    raw_sub_rows = H_indices[0][entry_mask]
    raw_sub_cols = H_indices[1][entry_mask]
    sub_vals = H_vals[entry_mask]

    global_to_local_node = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    global_to_local_node[subset_node_indices] = torch.arange(
        num_sub_nodes, device=device
    )
    new_sub_rows = global_to_local_node[raw_sub_rows]

    if raw_sub_cols.numel() == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=H.dtype, device=device),
            (num_sub_nodes, 0),
            device=device,
            dtype=H.dtype,
        )

    unique_edges, new_sub_cols = torch.unique(raw_sub_cols, return_inverse=True)
    num_sub_edges = unique_edges.numel()

    return torch.sparse_coo_tensor(
        torch.stack([new_sub_rows, new_sub_cols], dim=0),
        sub_vals,
        (num_sub_nodes, num_sub_edges),
        device=device,
        dtype=H.dtype,
    ).coalesce()


def _subset_features_and_map(
    features: torch.Tensor,
    labels: torch.Tensor,
    subset_node_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    sub_feat = features[subset_node_indices]
    sub_labels = labels[subset_node_indices]
    subset_nodes_cpu = subset_node_indices.cpu().numpy()
    node_dict = {
        int(global_idx): local_idx
        for local_idx, global_idx in enumerate(subset_nodes_cpu)
    }
    return sub_feat, sub_labels, node_dict


def get_hyper_neighbourhood_fast(
    node_idx: int,
    H: torch.Tensor,
    n_hops: int,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:

    device = features.device
    H = _ensure_sparse_coalesced(H, device)
    num_nodes = H.shape[0]

    if not (0 <= node_idx < num_nodes):
        raise IndexError(f"node_idx {node_idx} is out of bounds for size {num_nodes}")

    visited = _collect_visited_nodes(node_idx, H, n_hops, device)
    subset_node_indices = visited.nonzero(as_tuple=False).view(-1)

    sub_H = _build_induced_subhypergraph(H, visited, subset_node_indices, device)
    sub_feat, sub_labels, node_dict = _subset_features_and_map(
        features, labels, subset_node_indices
    )
    return sub_H, sub_feat, sub_labels, node_dict

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