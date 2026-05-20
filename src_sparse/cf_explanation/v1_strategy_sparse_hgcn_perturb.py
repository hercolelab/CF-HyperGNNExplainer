import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalize_propagation

DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT = 6
PERTURBATION_INIZIALIZATION = 4.0

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
            torch.full((num_edges,), PERTURBATION_INIZIALIZATION, device=H.device), requires_grad=True
        )

        self.pi_i = None
        self.H_tilde = None
        # Flag set when the target's incident masks are essentially all zero
        # meaning there are no more editable interactions for this node.
        self.no_more_edits = False
        self.no_available_edits = (
            False  # Flag for the case where the target has no incident edges at all
        )

    def reset_perturbation(self) -> None:
        with torch.no_grad():
            self.pi_i_hat.fill_(PERTURBATION_INIZIALIZATION)
        self.pi_i = None
        self.H_tilde = None
        self.no_more_edits = False
        self.no_available_edits = False

    def format_target_perturbation_debug(
        self,
        lr_debug: dict[str, object] | None = None,
        lr_active_mask: torch.Tensor | None = None,
        lr_grad: torch.Tensor | None = None,
    ) -> list[str]:
        lr_pi_hat = None
        if lr_debug is not None:
            lr_pi_hat = lr_debug.get("pi_hat")
            lr_active_mask = lr_debug.get("active_mask")  # type: ignore[assignment]
            lr_grad = lr_debug.get("grad")  # type: ignore[assignment]

        _, col_indices = extract_sparse_row(self.H, self.target_node)
        if col_indices.numel() == 0:
            return [
                f"Target-edge debug: node {self.target_node} has no incident hyperedges."
            ]

        col_indices = col_indices.to(self.pi_i_hat.device)
        with torch.no_grad():
            pi_hat_values = self.pi_i_hat[col_indices].detach().cpu()
            soft_values = torch.sigmoid(self.pi_i_hat[col_indices]).detach().cpu()
            hard_values = (soft_values >= 0.5).to(torch.int64)
            grad_values = None
            lr_active_values = None
            lr_grad_values = None
            lr_pi_hat_values = None
            if self.pi_i_hat.grad is not None:
                grad_values = self.pi_i_hat.grad[col_indices].detach().cpu()
            if isinstance(lr_pi_hat, torch.Tensor):
                lr_pi_hat_values = lr_pi_hat.to(device=self.pi_i_hat.device)[
                    col_indices
                ].detach().cpu()
            if isinstance(lr_active_mask, torch.Tensor):
                lr_active_values = lr_active_mask.to(
                    device=self.pi_i_hat.device,
                    dtype=torch.bool,
                )[col_indices].detach().cpu()
            if isinstance(lr_grad, torch.Tensor):
                lr_grad_values = lr_grad.to(device=self.pi_i_hat.device)[
                    col_indices
                ].detach().cpu()

        debug_lines = []
        debug_lines.append(
            "Target-edge debug "
            "(edge_idx, calib_pi_hat, calib_active, calib_grad, "
            "current_pi_hat, current_sigmoid, current_hard, current_grad):"
        )
        for idx, edge_idx in enumerate(col_indices.detach().cpu().tolist()):
            grad_str = "None"
            lr_pi_hat_str = "None"
            lr_active_str = "None"
            lr_grad_str = "None"
            if grad_values is not None:
                grad_str = f"{float(grad_values[idx]):+.6e}"
            if lr_pi_hat_values is not None:
                lr_pi_hat_str = f"{float(lr_pi_hat_values[idx]):+.6f}"
            if lr_active_values is not None:
                lr_is_active = bool(lr_active_values[idx])
                lr_active_str = "1" if lr_is_active else "0"
            if lr_grad_values is not None:
                lr_grad_str = f"{float(lr_grad_values[idx]):+.6e}"
            debug_lines.append(
                f"  edge {edge_idx}: calib_pi_hat={lr_pi_hat_str}, "
                f"calib_active={lr_active_str}, calib_grad={lr_grad_str}, "
                f"current_pi_hat={float(pi_hat_values[idx]):+.6f}, "
                f"current_s={float(soft_values[idx]):.6f}, "
                f"current_hard={int(hard_values[idx])}, current_grad={grad_str}"
            )
        return debug_lines

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
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        loss_pred = -F.nll_loss(output, y_pred_orig)

        # Differentiable L1 distance between the (soft) perturbation and the
        # original incidence values for the target node only. This follows the
        # paper formulation where we regularize the perturbation via L1.
        s = F.sigmoid(self.pi_i_hat)  # soft mask in (0,1) for each hyperedge

        # Extract the original incidence values and the corresponding edge
        # indices for the target node. This is robust to isolated nodes.
        row_vals, col_indices = extract_sparse_row(self.H, self.target_node)
        if col_indices.numel() == 0:
            # No incident hyperedges for the target: no graph distance
            loss_graph_dist = torch.tensor(0.0, device=output.device)
            self.no_available_edits = (
                True  # No edges to edit, so we can stop after this
            )
        else:
            col_indices = col_indices.to(self.pi_i_hat.device)
            row_vals = row_vals.to(self.pi_i_hat.device)
            s_target = s[col_indices]
            # L1 between original incidence values and the soft mask
            # mean is used because different nodes have different numbers of incident edges, and we want a consistent scale for the loss across nodes. This also follows the paper formulation where they use the mean perturbation value in the regularization term.
            loss_graph_dist = torch.mean(torch.abs(row_vals - s_target))
            if getattr(self, "verbose", True):
                print(
                    "s_target values: ",
                    s_target.detach().cpu().numpy().mean(),
                    s_target.detach().cpu().numpy().max(),
                    s_target.detach().cpu().numpy().min(),
                )

            active_upper = torch.sigmoid(
                s_target.new_tensor(DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT)
            )
            active_lower = 1 - active_upper
            if torch.all((s_target < active_lower) | (s_target > active_upper)):
                self.no_more_edits = True
        if getattr(self, "verbose", True):
            print(loss_pred.item(), loss_graph_dist.item())
        # loss = pred_same * loss_pred + self.beta * loss_graph_dist
        loss = loss_pred + self.beta * loss_graph_dist
        return loss, loss_pred, loss_graph_dist, self.H_tilde
