import torch
from torch import Tensor
from typing import Any


def build_incidence_matrix(edge_index: Tensor, num_nodes: int) -> torch.Tensor:
    """
    Convert CORA's edge_index to a hypergraph incidence matrix.
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
    values = torch.ones(len(rows), dtype=torch.float32)

    H = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    H[row_idx, col_idx] = values
    return H


def normalize_propagation(H: torch.Tensor) -> torch.Tensor:
    D = H.sum(dim=1)
    D = torch.diag(D)
    D_inv_sqrt = D.pow(-0.5)
    D_inv_sqrt = torch.where(
        torch.isinf(D_inv_sqrt),
        torch.zeros_like(D_inv_sqrt),
        D_inv_sqrt,
    )

    B = H.sum(dim=0)
    B = torch.diag(B)
    B_inv = B.pow(-1.0)
    B_inv = torch.where(
        torch.isinf(B_inv),
        torch.zeros_like(B_inv),
        B_inv,
    )

    S = D_inv_sqrt @ H @ B_inv @ H.t() @ D_inv_sqrt

    return S


def get_hyper_neighbourhood(node_idx, H, n_hops, features, labels):
    device = features.device
    H = H.to(device)
    N = H.size(0)

    A = H @ H.t()
    A = A > 0
    A.fill_diagonal_(False)  # drop self-loops

    visited = torch.zeros(N, dtype=torch.bool, device=device)
    current = torch.zeros(N, dtype=torch.bool, device=device)
    current[node_idx] = True
    visited |= current

    for _ in range(n_hops):
        neighbors = A[current].any(dim=0)
        new = neighbors & (~visited)
        visited |= neighbors
        if not new.any():
            break
        current = new

    nodes = visited.nonzero(as_tuple=False).view(-1)
    if nodes.numel() == 0:
        return (
            torch.zeros((0, 0), device=device, dtype=H.dtype),
            features[[]],
            labels[[]],
            {},
        )

    sub_H_rows = H[nodes]
    edge_mask = sub_H_rows.gt(0).any(dim=0)
    if edge_mask.any():
        sub_H = sub_H_rows[:, edge_mask]
    else:
        sub_H = torch.zeros((nodes.numel(), 0), device=device, dtype=H.dtype)

    sub_feat = features[nodes]
    sub_labels = labels[nodes]

    node_dict = {int(orig): int(i) for i, orig in enumerate(nodes.cpu().numpy())}

    return sub_H, sub_feat, sub_labels, node_dict
