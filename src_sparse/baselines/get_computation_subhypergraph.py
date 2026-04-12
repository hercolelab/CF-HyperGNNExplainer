from typing import Dict, Tuple

import torch
from torch import Tensor


def get_computation_subhypergraph(
    node_idx: int,
    H: Tensor,
    n_hops: int,
    features: Tensor,
    labels: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Dict[int, int]]:
    """
    Extract the exact computation subhypergraph for SHypX.

    Parameters
    ----------
    node_idx : int
        Global index of the target node to explain.
    H : sparse COO Tensor  (N, E)
        Full hypergraph incidence matrix.
    n_hops : int
        Number of hops, which should equal the number of layers in the hyperGNN.
    features : Tensor  (N, D)
        Node feature matrix (on the same device as H).
    labels : Tensor  (N,)
        Node label vector.

    Returns
    -------
    sub_H : sparse COO Tensor  (sub_N, sub_E)
        Incidence matrix of the computation subhypergraph with re-mapped indices.
    sub_feat : Tensor  (sub_N, D)
    sub_labels : Tensor  (sub_N,)
    node_dict : Dict[int, int]
        Maps global node id → local node id within sub_H.
    """
    device = features.device
    H = H.to(device)
    if not H.is_coalesced():
        H = H.coalesce()

    N, E = H.shape

    if not (0 <= node_idx < N):
        raise IndexError(f"node_idx {node_idx} out of range [0, {N})")

    link_rows = H.indices()[0]
    link_cols = H.indices()[1]
    link_vals = H.values()

    visited_nodes = torch.zeros(N, dtype=torch.bool, device=device)
    traversed_edges = torch.zeros(E, dtype=torch.bool, device=device)

    visited_nodes[node_idx] = True
    frontier_nodes = visited_nodes.clone()

    for _ in range(n_hops):
        if not frontier_nodes.any():
            break

        frontier_link_mask = frontier_nodes[link_rows]
        candidate_edges = torch.zeros(E, dtype=torch.bool, device=device)
        candidate_edges[link_cols[frontier_link_mask]] = True
        newly_traversed = candidate_edges & ~traversed_edges

        if not newly_traversed.any():
            break

        traversed_edges |= newly_traversed

        new_edge_link_mask = newly_traversed[link_cols]
        found_nodes = torch.zeros(N, dtype=torch.bool, device=device)
        found_nodes[link_rows[new_edge_link_mask]] = True

        frontier_nodes = found_nodes & ~visited_nodes
        visited_nodes |= found_nodes

    link_mask = traversed_edges[link_cols]

    if not link_mask.any():
        sub_H = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=H.dtype, device=device),
            (1, 0),
            device=device,
            dtype=H.dtype,
        ).coalesce()
        return (
            sub_H,
            features[[node_idx]],
            labels[[node_idx]],
            {node_idx: 0},
        )

    sub_node_global = link_rows[link_mask]
    sub_edge_global = link_cols[link_mask]
    sub_vals = link_vals[link_mask]

    unique_nodes, remapped_rows = torch.unique(sub_node_global, return_inverse=True)
    unique_edges, remapped_cols = torch.unique(sub_edge_global, return_inverse=True)

    num_sub_nodes = unique_nodes.numel()
    num_sub_edges = unique_edges.numel()

    sub_H = torch.sparse_coo_tensor(
        torch.stack([remapped_rows, remapped_cols], dim=0),
        sub_vals,
        (num_sub_nodes, num_sub_edges),
        device=device,
        dtype=H.dtype,
    ).coalesce()

    sub_feat = features[unique_nodes]
    sub_labels = labels[unique_nodes]

    node_dict: Dict[int, int] = {
        int(g): int(l) for l, g in enumerate(unique_nodes.cpu().tolist())
    }

    return sub_H, sub_feat, sub_labels, node_dict
