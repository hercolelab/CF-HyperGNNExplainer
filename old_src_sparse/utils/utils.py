import torch
from torch_sparse import SparseTensor, matmul


def normalize_propagation(H: SparseTensor) -> SparseTensor:
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

    num_nodes, num_hyperedges = H.sparse_sizes()

    D = H.sum(dim=1).to(torch.float32)
    D_inv_sqrt = D.pow(-0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

    B = H.sum(dim=0).to(torch.float32)
    B_inv = B.pow(-1.0)
    B_inv[torch.isinf(B_inv)] = 0.0

    node_idx = torch.arange(num_nodes, device=D_inv_sqrt.device)
    edge_idx = torch.arange(num_hyperedges, device=B_inv.device)

    D_inv_sqrt = SparseTensor(
        row=node_idx,
        col=node_idx,
        value=D_inv_sqrt,
        sparse_sizes=(num_nodes, num_nodes),
    )

    B_inv = SparseTensor(
        row=edge_idx,
        col=edge_idx,
        value=B_inv,
        sparse_sizes=(num_hyperedges, num_hyperedges),
    )

    S = matmul(D_inv_sqrt, H)
    S = matmul(S, B_inv)
    S = matmul(S, H.t())
    S = matmul(S, D_inv_sqrt)

    return S
