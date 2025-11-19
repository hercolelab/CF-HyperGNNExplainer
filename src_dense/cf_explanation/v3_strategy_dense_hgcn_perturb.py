import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalize_propagation


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
            x: Node features [num_nodes, in_channels]
            S: propagation matrix [num_nodes, num_nodes]
        """
        out = S @ x @ self.weight
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

        self.H = H
        num_nodes, num_edges = self.H.shape

        # The differentiable version of pi_i, the perturbation vector
        # One weight per hyperedge (column)
        self.pi_i_hat = nn.Parameter(
            torch.ones(num_edges, device=H.device), requires_grad=True
        )

        self.pi_i = None
        self.H_tilde = None

    def forward(self, x, sub_H):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            sub_H: Sub-hypergraph incidence matrix [num_nodes, num_edges] (dense)
        """

        weights = F.sigmoid(self.pi_i_hat)
        H_tilde = sub_H * weights.unsqueeze(0)  # Broadcasting over rows

        S_tilde = normalize_propagation(H_tilde)

        x1 = F.leaky_relu(self.conv1(x, S_tilde))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.leaky_relu(self.conv2(x1, S_tilde))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.conv3(x2, S_tilde)
        x = self.linear(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)

    def forward_pred(self, x):
        """
        Non-differentiable version of the forward pass
        """
        self.pi_i = (F.sigmoid(self.pi_i_hat) >= 0.5).float()

        self.H_tilde = self.H * self.pi_i.unsqueeze(0)

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

        weights = 1 - self.pi_i

        loss_graph_dist = torch.sum(self.H * weights.unsqueeze(0))

        cf_H = self.H_tilde

        loss = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss, loss_pred, loss_graph_dist, cf_H


# if __name__ == "__main__":
#     H = torch.tensor(
#         [[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
#         dtype=torch.float32,
#     )

#     model = HGCN_Perturb(3, 2, 2, 5, 0.5, H, 0, 0.1)

#     out = model(torch.randn(6, 3), H)
#     print(out)
