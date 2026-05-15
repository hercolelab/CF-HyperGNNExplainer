import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .gcn_perturb import GCNSyntheticPerturb
from .utils import get_degree_matrix, normalize_adj


def _as_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(
        self,
        model,
        sub_adj,
        sub_feat,
        n_hid,
        dropout,
        sub_labels,
        y_pred_orig,
        num_classes,
        beta,
        device,
        log_prob_orig=None,
        quiet=False,
    ):
        super(CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.sub_adj = sub_adj.to(device)
        self.sub_feat = sub_feat.to(device)
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_labels = sub_labels.to(device)
        self.y_pred_orig = torch.as_tensor(y_pred_orig, device=device, dtype=torch.long)
        self.log_prob_orig = (
            None if log_prob_orig is None else log_prob_orig.detach().to(device)
        )
        self.beta = beta
        self.num_classes = num_classes
        self.device = device
        self.quiet = quiet

        self.cf_model = GCNSyntheticPerturb(
            self.sub_feat.shape[1],
            n_hid,
            n_hid,
            self.num_classes,
            self.sub_adj,
            dropout,
            beta,
        ).to(device)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

    def _log(self, *items):
        if not self.quiet:
            print(*items)

    def explain(
        self,
        cf_optimizer,
        node_idx,
        new_idx,
        lr,
        n_momentum,
        num_epochs,
        patience=5,
    ):
        self.node_idx = node_idx
        self.new_idx = int(new_idx)

        self.x = self.sub_feat
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(
                self.cf_model.parameters(),
                lr=lr,
                nesterov=True,
                momentum=n_momentum,
            )
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported cf_optimizer '{cf_optimizer}'")

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        stop_counter = 0
        last_pred = -1
        for epoch in range(num_epochs):
            new_example, loss_total, grad_is_zero, current_pred = self.train(epoch)
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1

            if grad_is_zero and current_pred == last_pred:
                stop_counter += 1
            else:
                stop_counter = 0

            if patience > 0 and stop_counter >= patience:
                self._log(f"\nEarly stopping triggered at epoch {epoch + 1}")
                self._log(
                    f"Reason: Gradient zero and prediction stable for {patience} epochs."
                )
                break

            last_pred = current_pred
        self._log(f"{num_cf_examples} CF examples for node_idx = {self.node_idx}")
        self._log(" ")
        return best_cf_example

    def train(self, epoch):
        self.cf_model.eval()
        self.cf_optimizer.zero_grad()

        output = self.cf_model.forward(self.x, self.A_x)
        output_actual, self.P = self.cf_model.forward_prediction(self.x)

        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
            output[self.new_idx],
            self.y_pred_orig,
            y_pred_new_actual,
        )
        loss_total.backward()
        p_vec_grad = self.cf_model.P_vec.grad
        grad_is_zero = p_vec_grad is None or torch.all(p_vec_grad == 0)
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        self._log(
            "Node idx: {}".format(self.node_idx),
            "New idx: {}".format(self.new_idx),
            "Epoch: {:04d}".format(epoch + 1),
            "loss: {:.4f}".format(loss_total.item()),
            "pred loss: {:.4f}".format(loss_pred.item()),
            "graph loss: {:.4f}".format(loss_graph_dist.item()),
        )
        self._log(
            "Output: {}\n".format(output[self.new_idx].data),
            "Output nondiff: {}\n".format(output_actual[self.new_idx].data),
            "orig pred: {}, new pred: {}, new pred nondiff: {}".format(
                self.y_pred_orig,
                y_pred_new,
                y_pred_new_actual,
            ),
        )
        self._log(" ")

        cf_stats = []
        if y_pred_new_actual != self.y_pred_orig:
            cf_stats = [
                _as_int(self.node_idx),
                self.new_idx,
                cf_adj.detach().cpu(),
                self.sub_adj.detach().cpu(),
                _as_int(self.y_pred_orig),
                _as_int(y_pred_new),
                _as_int(y_pred_new_actual),
                self.sub_labels[self.new_idx].detach().cpu(),
                self.sub_adj.shape[0],
                loss_total.item(),
                loss_pred.item(),
                loss_graph_dist.item(),
            ]
            if self.log_prob_orig is not None:
                removed_adj = torch.clamp(self.sub_adj - cf_adj.detach(), min=0.0)
                with torch.no_grad():
                    removed_output = self.model(self.sub_feat, normalize_adj(removed_adj))
                cf_stats.extend(
                    [
                        self.log_prob_orig.detach().cpu(),
                        output[self.new_idx].detach().cpu(),
                        output_actual[self.new_idx].detach().cpu(),
                        removed_output[self.new_idx].detach().cpu(),
                    ]
                )

        return cf_stats, loss_total.item(), bool(grad_is_zero), _as_int(y_pred_new_actual)
