from typing import List, Tuple

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .dense_hgcn_perturb import HGCN_Perturb


class CFExplainer:
    """
    Counterfactual explainer for HGCN-based hypergraphs
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sub_H: Tensor,
        sub_feat: Tensor,
        sub_labels: Tensor,
        y_pred_orig: Tensor,
        beta: float,
        target_node_sub_idx: int,
        device: torch.device,
    ):
        """
        Args:
            model: Trained base `HGCN` model
            sub_H: Dense incidence matrix of the local sub-hypergraph
            sub_feat: Node features for the subgraph
            sub_labels: Node labels for the subgraph
            y_pred_orig: Original prediction of the target node
            beta: Trade-off weight between prediction and graph distance losses
            target_node_sub_idx: Index of the target node in the subgraph
            device: Torch device to run the CF model
        """
        super().__init__()

        self.model = model
        self.model.eval()

        self.sub_H = sub_H
        self.sub_feat = sub_feat
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.target_node_sub_idx = int(target_node_sub_idx)
        self.device = device

        nhid = model.conv1.out_channels
        nout = model.conv3.out_channels
        nclass = model.linear.out_features
        dropout = model.dropout

        self.cf_model = HGCN_Perturb(
            nfeat=self.sub_feat.shape[1],
            nhid=nhid,
            nout=nout,
            nclass=nclass,
            dropout=dropout,
            H=self.sub_H,
            target_node=self.target_node_sub_idx,
            beta=self.beta,
        ).to(self.device)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)

        self.node_idx: int = -1
        self.new_idx: int = -1
        self.cf_optimizer: optim.Optimizer | None = None

    def explain(
        self,
        cf_optimizer: str,
        node_idx: int,
        new_idx: int,
        lr: float,
        n_momentum: float,
        num_epochs: int,
    ) -> List[List]:
        """
        Run counterfactual optimization and return the best CF examples

        Args:
            cf_optimizer: One of {"SGD", "Adadelta"}
            node_idx: Index of the target node in the original (full) hypergraph
            new_idx: Index of the target node in the subgraph
            lr: Learning rate
            n_momentum: Momentum for SGD (0.0 disables momentum)
            num_epochs: Number of optimization epochs
        """
        self.node_idx = int(node_idx)
        self.new_idx = int(new_idx)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(
                self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum
            )
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported cf_optimizer '{cf_optimizer}'")

        best_cf_example: List[List] = []
        best_loss = np.inf
        num_cf_examples = 0

        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            new_example, loss_total = self.train(epoch)
            if new_example and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1

        print(f"{num_cf_examples} CF examples for node_idx = {self.node_idx}")
        print(" ")
        return best_cf_example

    def train(self, epoch: int) -> Tuple[List, float]:
        """
        Single training epoch for the counterfactual model
        """
        assert self.cf_optimizer is not None, "Call `explain` before `train`"

        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        output = self.cf_model.forward(self.sub_feat, self.sub_H)

        output_actual, self.Pi = self.cf_model.forward_pred(self.sub_feat)

        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        loss_total, loss_pred, loss_graph_dist, cf_H = self.cf_model.loss(
            output[self.new_idx], self.y_pred_orig, y_pred_new_actual
        )
        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        print(
            "Node idx: {}".format(self.node_idx),
            "New idx: {}".format(self.new_idx),
            "Epoch: {:04d}".format(epoch + 1),
            "loss: {:.4f}".format(loss_total.item()),
            "pred loss: {:.4f}".format(loss_pred.item()),
            "graph loss: {:.4f}".format(loss_graph_dist.item()),
        )
        print(
            "Output: {}\n".format(output[self.new_idx].data),
            "Output nondiff: {}\n".format(output_actual[self.new_idx].data),
            "orig pred: {}, new pred: {}, new pred nondiff: {}".format(
                self.y_pred_orig, y_pred_new, y_pred_new_actual
            ),
        )
        print(" ")

        cf_stats: List = []
        if y_pred_new_actual != self.y_pred_orig:
            cf_stats = [
                int(self.node_idx),
                int(self.new_idx),
                cf_H.detach().cpu().numpy(),
                self.sub_H.detach().cpu().numpy(),
                int(self.y_pred_orig.item()),
                int(y_pred_new.item()),
                int(y_pred_new_actual.item()),
                self.sub_labels[self.new_idx].detach().cpu().numpy(),
                int(self.sub_H.shape[0]),
                float(loss_total.item()),
                float(loss_pred.item()),
                float(loss_graph_dist.item()),
            ]

        return cf_stats, float(loss_total.item())
