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

        # All these variables are updated using forward and forward_pred methods

        # The differentiable version of pi_i, the perturbation vector
        self.pi_i_hat = nn.Parameter(torch.zeros(num_edges), requires_grad=True)
        self.pi_i = None

        self.Pi_hat = None  # The differentiable version of self.Pi
        self.Pi = None  # Pi^{inc(i)} in the notation
        self.sub_H = None
        self.H_tilde = None  # H' in the notation
        self.S_tilde = None  # S_{v1}(Pi^{inc}) in the notation

    def forward(self, x, sub_H):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            sub_H: Sub-hypergraph incidence matrix [num_nodes, num_edges]
            !!! this is different from the implementation in hgcn.py, which uses directly the propagation matrix S directly
        """
        self.sub_H = sub_H
        num_nodes, num_edges = self.sub_H.shape

        self.Pi_hat = torch.ones(
            num_nodes, num_edges, device=self.sub_H.device, dtype=self.sub_H.dtype
        )
        self.Pi_hat[self.target_node] = self.pi_i_hat

        H_tilde = self.sub_H * F.sigmoid(self.Pi_hat)
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
        Non-differentiable version of the forward pass, used to compute the prediction and the Pi matrix
        Args:
            x: Node features [num_nodes, in_channels]
        """
        num_nodes, num_edges = self.H.shape

        self.Pi = torch.ones(
            num_nodes, num_edges, device=self.H.device, dtype=self.H.dtype
        )
        self.pi_i = (F.sigmoid(self.pi_i_hat) >= 0.5).float()  # thresholded pi_i_hat
        self.Pi[self.target_node] = self.pi_i

        H_tilde = self.H * self.Pi
        S_tilde = normalize_propagation(H_tilde)

        x1 = F.leaky_relu(self.conv1(x, S_tilde))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.leaky_relu(self.conv2(x1, S_tilde))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.conv3(x2, S_tilde)
        x = self.linear(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1), self.Pi

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()

        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        cf_H = self.H * self.Pi
        cf_H.requires_grad = True  # Used in https://github.com/a-lucic/cf-gnnexplainer/blob/main/src/cf_explanation/gcn_perturb.py

        loss_pred = -F.nll_loss(output, y_pred_orig)

        loss_graph_dist = sum(
            abs(cf_H[self.target_node] - self.H[self.target_node])
        )  # Distance is measured considering only the perturbed row

        loss = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss, loss_pred, loss_graph_dist, cf_H
