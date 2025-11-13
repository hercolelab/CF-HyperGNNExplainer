import torch


def normalize_propagation(H: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized propagation matrix of a hypergraph.
    Args:
        H: Hypergraph incidence matrix [num_nodes, num_hyperedges]
    Returns:
        S: Normalized propagation matrix [num_nodes, num_nodes]

    The normalized propagation matrix is defined as:
    S := D^{-1/2}HB^{-1}H^{\top}D^{-1/2}

    where:
    - D is the degree matrix of the hypergraph
    - B is the degree matrix of the hyperedges
    - H is the hypergraph incidence matrix
    """

    D = torch.sparse.sum(H, dim=1).to_dense()
    D_exp = D ** (-1 / 2)  # D^{-1/2}
    D_exp[D_exp == float("inf")] = 0
    D_exp = torch.diag(D_exp).to_sparse_coo()

    B = torch.sparse.sum(H, dim=0, dtype=torch.float32).to_dense()
    B_inv = B ** (-1)  # B^{-1}
    B_inv[B_inv == float("inf")] = 0
    B_inv = torch.diag(B_inv).to_sparse_coo()

    # S = D^{-1/2}HB^{-1}H^T D^{-1/2}
    S = D_exp @ H @ B_inv @ H.T @ D_exp

    return S
