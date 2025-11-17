import torch


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


# def normalize_propagation(H: torch.Tensor) -> torch.Tensor:
#     D = H.sum(dim=1)
#     D = torch.diag(D)
#     D_inv_sqrt = D.pow(-0.5)
#     D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

#     B = H.sum(dim=0)
#     B = torch.diag(B)
#     B_inv = B.pow(-1.0)
#     B_inv[torch.isinf(B_inv)] = 0.0

#     S = D_inv_sqrt @ H @ B_inv @ H.t() @ D_inv_sqrt

#     return S


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
