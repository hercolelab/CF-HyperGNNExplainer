import torch


def normalize_propagation(H: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    # H: sparse_coo_tensor of shape [num_nodes, num_hyperedges]
    assert H.layout == torch.sparse_coo

    # Make sure H is coalesced for correct indices/values semantics
    H = H.coalesce()

    # Row and column sums: D = diag(H 1), B = diag(1^T H)
    # Summing over all sparse dims returns a dense tensor for that dim.
    row_sum = torch.sparse.sum(H, dim=1).to_dense()  # shape [num_nodes]
    col_sum = torch.sparse.sum(H, dim=0).to_dense()  # shape [num_hyperedges]

    # D^{-1/2}
    d_inv_sqrt = row_sum.pow(-0.5)
    d_inv_sqrt = torch.where(
        torch.isinf(d_inv_sqrt),
        torch.zeros_like(d_inv_sqrt),
        d_inv_sqrt,
    )

    # B^{-1}
    b_inv = col_sum.pow(-1.0)
    b_inv = torch.where(
        torch.isinf(b_inv),
        torch.zeros_like(b_inv),
        b_inv,
    )

    # Access COO structure
    indices = H.indices()  # [2, nnz]
    values = H.values()  # [nnz]
    row, col = indices[0], indices[1]

    # Gather per-entry scalings for rows/cols
    d_row = d_inv_sqrt[row]  # D^{-1/2} on rows
    b_col = b_inv[col]  # B^{-1} on cols

    # Build two scaled versions of H:
    #   H_left  = D^{-1/2} H B^{-1}
    #   H_right = D^{-1/2} H
    values_left = values * d_row * b_col
    values_right = values * d_row

    H_left = torch.sparse_coo_tensor(
        indices, values_left, H.size(), device=H.device, dtype=H.dtype
    ).coalesce()
    H_right = torch.sparse_coo_tensor(
        indices, values_right, H.size(), device=H.device, dtype=H.dtype
    ).coalesce()

    # S = H_left @ H_right^T = D^{-1/2} H B^{-1} H^T D^{-1/2}
    # torch.sparse.mm supports sparse @ sparse and returns sparse.[web:2]
    S = torch.sparse.mm(H_left, H_right.transpose(0, 1))

    return S


if __name__ == "__main__":
    H = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
        dtype=torch.float32,
        requires_grad=True,
    )

    H = H.to_sparse()
    S = normalize_propagation(H)
    print(S)
    print(S.requires_grad)
