import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    """
    Hypergraph Convolutional Layer
    """

    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

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
        x is dense
        S is sparse
        """
        out = S @ x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class HGCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(HGCN, self).__init__()
        self.conv1 = HypergraphConv(nfeat, nhid)
        self.conv2 = HypergraphConv(nhid, nhid)
        self.conv3 = HypergraphConv(nhid, nout)
        self.linear = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def forward(self, x, S, return_embeddings: bool = False):
        x1 = F.leaky_relu(self.conv1(x, S))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.leaky_relu(self.conv2(x1, S))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.conv3(x2, S)
        x = self.linear(torch.cat((x1, x2, x3), dim=1))
        if return_embeddings:
            return x
        return F.log_softmax(x, dim=1)

    def loss(self, pred, target):
        return F.nll_loss(pred, target)


# if __name__ == "__main__":
#     from utils import normalize_propagation

#     H = torch.tensor(
#         [[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
#         dtype=torch.float32,
#     )
#     H = H.to_sparse()

#     S = normalize_propagation(H)

#     model = HGCN(3, 2, 2, 5, 0.5)

#     out = model(torch.randn(6, 3), S)
#     print(out.shape)
