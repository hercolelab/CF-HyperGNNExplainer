import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalize_propagation


def sparse_hadamard_product(sparse_tensor, dense_matrix):
    """
    Hadamard (element-wise) product of a sparse COO tensor with a dense matrix.
    Result is a sparse tensor.


    Args:
        sparse_tensor: sparse COO tensor
        dense_matrix: dense tensor of same shape
    Returns:
        sparse COO tensor with values multiplied element-wise
    """
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()  # [2, nnz]
    values = sparse_tensor.values()  # [nnz]

    rows, cols = indices[0], indices[1]
    dense_values = dense_matrix[rows, cols]

    new_values = values * dense_values

    return torch.sparse_coo_tensor(
        indices,
        new_values,
        sparse_tensor.size(),
        device=sparse_tensor.device,
        dtype=sparse_tensor.dtype,
    ).coalesce()


def extract_sparse_row(sparse_tensor, row_idx):
    """
    Extract a specific row from a sparse COO tensor.
    Returns the non-zero values and their column indices for that row.

    Args:
        sparse_tensor: sparse COO tensor [num_rows, num_cols]
        row_idx: index of the row to extract
    Returns:
        values: non-zero values in the row
        col_indices: column indices of non-zero values
    """
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()

    row_mask = indices[0] == row_idx
    col_indices = indices[1][row_mask]
    row_values = values[row_mask]

    return row_values, col_indices


class HypergraphConvPerturb(nn.Module):
    """
    Hypergraph Convolutional Layer
    """

    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels)
        )  # P in the notation

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, S):
        """
        Args:
            x: Node features [num_nodes, in_channels] (dense)
            S: propagation matrix [num_nodes, num_nodes] (sparse)
        """
        out = torch.sparse.mm(S, x @ self.weight)
        if self.bias is not None:
            out += self.bias
        return out


class HGCN_Perturb(nn.Module):
    def __init__(self, nfeat, nhid, nout, nclass, dropout, H, target_node, beta):
        super(HGCN_Perturb, self).__init__()
        self.conv1 = HypergraphConvPerturb(nfeat, nhid)
        self.conv2 = HypergraphConvPerturb(nhid, nhid)
        self.conv3 = HypergraphConvPerturb(nhid, nout)
        self.linear = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

        self.beta = beta
        self.target_node = target_node

        assert H.layout == torch.sparse_coo
        self.H = H.coalesce()
        num_nodes, num_edges = self.H.shape

        self.pi_i_hat = nn.Parameter(
            torch.ones(num_edges, device=H.device), requires_grad=True
        )

        self.pi_i = None
        self.H_tilde = None

    def forward(self, x, sub_H):
        sub_H = sub_H.coalesce()

        indices = sub_H.indices()
        values = sub_H.values().clone()

        target_mask = indices[0] == self.target_node
        target_cols = indices[1][target_mask]

        values[target_mask] = values[target_mask] * F.sigmoid(
            self.pi_i_hat[target_cols]
        )

        H_tilde = torch.sparse_coo_tensor(
            indices, values, sub_H.size(), device=sub_H.device, dtype=sub_H.dtype
        ).coalesce()

        S_tilde = normalize_propagation(H_tilde)

        x1 = F.leaky_relu(self.conv1(x, S_tilde))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.leaky_relu(self.conv2(x1, S_tilde))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.conv3(x2, S_tilde)
        x = self.linear(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)

    def forward_pred(self, x):
        self.pi_i = (F.sigmoid(self.pi_i_hat) >= 0.5).float()

        H_coalesced = self.H.coalesce()
        indices = H_coalesced.indices()
        values = H_coalesced.values().clone()

        target_mask = indices[0] == self.target_node
        target_cols = indices[1][target_mask]

        values[target_mask] = values[target_mask] * self.pi_i[target_cols]

        self.H_tilde = torch.sparse_coo_tensor(
            indices, values, self.H.size(), device=self.H.device, dtype=self.H.dtype
        ).coalesce()

        S_tilde = normalize_propagation(self.H_tilde)

        x1 = F.leaky_relu(self.conv1(x, S_tilde))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.leaky_relu(self.conv2(x1, S_tilde))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.conv3(x2, S_tilde)
        x = self.linear(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1), self.H_tilde

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()

        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        loss_pred = -F.nll_loss(output, y_pred_orig)

        loss_graph_dist = self.pi_i.shape[0] - self.pi_i.sum()

        loss = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss, loss_pred, loss_graph_dist, self.H_tilde
