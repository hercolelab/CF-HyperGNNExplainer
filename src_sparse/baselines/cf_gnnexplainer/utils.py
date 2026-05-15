import errno
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph, subgraph, to_dense_adj


@dataclass
class StarExpandedGraph:
    x: Tensor
    y: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    adj: Tensor
    edge_index: Tensor
    num_original_nodes: int
    num_hyperedge_nodes: int

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open(path, w):
    mkdir_p(os.path.dirname(path))
    return open(path, w)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_degree_matrix(adj):
    return torch.diag(torch.sum(adj, dim=0))


def normalize_adj(adj):
    A_tilde = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    D_tilde = get_degree_matrix(A_tilde)
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
    return torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)


def star_expand_hypergraph(
    H: Tensor,
    features: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
) -> StarExpandedGraph:
    H = H.coalesce()
    device = features.device
    num_nodes, num_hyperedges = H.shape
    total_nodes = num_nodes + num_hyperedges

    indices = H.indices().to(device)
    node_ids = indices[0]
    hyperedge_ids = indices[1] + num_nodes

    adj = torch.zeros(
        (total_nodes, total_nodes),
        dtype=features.dtype,
        device=device,
    )
    if node_ids.numel() > 0:
        adj[node_ids, hyperedge_ids] = 1.0
        adj[hyperedge_ids, node_ids] = 1.0

    edge_index = torch.cat(
        (
            torch.stack((node_ids, hyperedge_ids), dim=0),
            torch.stack((hyperedge_ids, node_ids), dim=0),
        ),
        dim=1,
    )

    hyperedge_features = torch.zeros(
        (num_hyperedges, features.size(1)),
        dtype=features.dtype,
        device=device,
    )
    x = torch.cat((features, hyperedge_features), dim=0)
    y = torch.cat(
        (
            labels,
            labels.new_full((num_hyperedges,), -1),
        ),
        dim=0,
    )

    false_hyperedges = torch.zeros(
        num_hyperedges,
        dtype=torch.bool,
        device=device,
    )
    return StarExpandedGraph(
        x=x,
        y=y,
        train_mask=torch.cat((train_mask.bool(), false_hyperedges), dim=0),
        val_mask=torch.cat((val_mask.bool(), false_hyperedges), dim=0),
        test_mask=torch.cat((test_mask.bool(), false_hyperedges), dim=0),
        adj=adj,
        edge_index=edge_index,
        num_original_nodes=num_nodes,
        num_hyperedge_nodes=num_hyperedges,
    )


def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)
    edge_subset_relabel = subgraph(edge_subset[0], edge_index, relabel_nodes=True)
    sub_adj = to_dense_adj(
        edge_subset_relabel[0],
        max_num_nodes=edge_subset[0].numel(),
    ).squeeze(0)
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].detach().cpu().numpy(), new_index))
    return sub_adj, sub_feat, sub_labels, node_dict


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = vector.new_zeros((n_rows, n_rows))
    idx = torch.tril_indices(n_rows, n_rows, device=vector.device)
    matrix[idx[0], idx[1]] = vector
    return torch.tril(matrix) + torch.tril(matrix, -1).t()


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0], device=matrix.device)
    return matrix[idx[0], idx[1]]


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
