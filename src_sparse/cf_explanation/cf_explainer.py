import math
from typing import List, Tuple


from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm_


from .v1_strategy_sparse_hgcn_perturb import HGCN_Perturb as HGCN_Perturb_v1
from .v3_strategy_sparse_hgcn_perturb import HGCN_Perturb as HGCN_Perturb_v3


DEFAULT_INCREMENTAL_BETA_MIN = 1e-6
DEFAULT_INCREMENTAL_BETA_FACTOR = 2.0
DEFAULT_INCREMENTAL_BETA_BUDGET = 30
DEFAULT_INCREMENTAL_BETA_REFINEMENT_RATIO = 1.10
DEFAULT_DYNAMIC_LR_EPSILON = 1e-8


class CFExplainer:
    """
    Counterfactual explainer for HGCN-based hypergraphs (sparse version)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sub_H: Tensor,
        sub_feat: Tensor,
        sub_labels: Tensor,
        y_pred_orig: Tensor,
        log_prob_orig: Tensor,
        beta: float,
        target_node_sub_idx: int,
        device: torch.device,
        strategy: str = "v1",
    ):
        """
        Args:
            model: Trained base `HGCN` model
            sub_H: Sparse incidence matrix of the local sub-hypergraph (torch.sparse_coo_tensor)
            sub_feat: Node features for the subgraph
            sub_labels: Node labels for the subgraph
            y_pred_orig: Original prediction of the target node
            beta: Trade-off weight between prediction and graph distance losses
            target_node_sub_idx: Index of the target node in the subgraph
            device: Torch device to run the CF model
            strategy: Explanation strategy ("v1" or "v3")
        """
        super().__init__()

        self.model = model
        self.model.eval()

        # Ensure sub_H is sparse
        assert sub_H.layout == torch.sparse_coo, "sub_H must be a sparse COO tensor"
        self.sub_H = sub_H.coalesce()
        self.sub_feat = sub_feat
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig
        self.log_prob_orig = log_prob_orig
        self.beta = beta
        self.target_node_sub_idx = int(target_node_sub_idx)
        self.device = device

        nhid = model.conv1.out_channels
        nout = model.conv3.out_channels
        nclass = model.linear.out_features
        dropout = model.dropout

        if strategy == "v1":
            StrategyClass = HGCN_Perturb_v1
        elif strategy == "v3":
            StrategyClass = HGCN_Perturb_v3
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.cf_model = StrategyClass(
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

        #for name, param in self.model.named_parameters():
        #    print("orig model requires_grad: ", name, param.requires_grad)
        #for name, param in self.cf_model.named_parameters():
        #    print("cf model requires_grad: ", name, param.requires_grad)

        self.node_idx: int = -1
        self.new_idx: int = -1
        self.cf_optimizer: optim.Optimizer | None = None

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)
        self.cf_model.beta = self.beta

    def compute_dynamic_lr(self, num_epochs: int, epsilon: float = 1e-8) -> float:
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive when computing a dynamic learning rate.")

        original_beta = self.beta
        self.cf_model.reset_perturbation()
        self.set_beta(0.0)
        self.cf_model.eval()
        self.cf_model.zero_grad(set_to_none=True)

        output = self.cf_model.forward(self.sub_feat, self.sub_H)
        target_output = output[self.target_node_sub_idx].unsqueeze(0)
        target_label = self.y_pred_orig.view(1)
        nll = F.nll_loss(target_output, target_label)
        nll.backward()

        grad = self.cf_model.pi_i_hat.grad
        grad_norm_sq = 0.0 if grad is None else float(grad.pow(2).sum().item())
        num_classes = int(target_output.size(1))
        denominator = float(num_epochs -1) if num_epochs > 1 else 1.0
        delta = (0.1 + math.log(max(num_classes, 1))) / denominator
        lr = delta / (grad_norm_sq + epsilon)

        self.cf_model.zero_grad(set_to_none=True)
        self.cf_model.reset_perturbation()
        self.set_beta(original_beta)

        return lr

    def resolve_node_learning_rate(
        self,
        lr_setting: float | str,
        num_epochs: int,
        target_node: int,
        epsilon: float = DEFAULT_DYNAMIC_LR_EPSILON,
    ) -> float:
        if isinstance(lr_setting, float):
            return lr_setting

        node_lr = self.compute_dynamic_lr(
            num_epochs=num_epochs,
            epsilon=epsilon,
        )
        print(f"Dynamic learning rate for target node {target_node}: {node_lr:.6g}")
        return node_lr

    def run_incremental_beta_search(
        self,
        cf_optimizer: str,
        node_idx: int,
        new_idx: int,
        lr: float,
        n_momentum: float,
        num_epochs: int,
        beta_min: float = DEFAULT_INCREMENTAL_BETA_MIN,
        beta_factor: float = DEFAULT_INCREMENTAL_BETA_FACTOR,
        beta_budget: int = DEFAULT_INCREMENTAL_BETA_BUDGET,
        beta_refinement_ratio: float = DEFAULT_INCREMENTAL_BETA_REFINEMENT_RATIO,
    ) -> tuple[List[List], bool, float]:
        trials_used = 0

        print(f"Starting incremental beta search for target node {node_idx}.")
        self.set_beta(0.0)
        best_cf_examples = self.explain(
            cf_optimizer=cf_optimizer,
            node_idx=node_idx,
            new_idx=new_idx,
            lr=lr,
            n_momentum=n_momentum,
            num_epochs=num_epochs,
        )
        possible = not self.cf_model.no_more_edits
        trials_used += 1

        if not best_cf_examples:
            print(
                "Incremental beta search stopped at beta=0.0 because no valid "
                "counterfactual was found."
            )
            return [], possible, 0.0

        beta_best = 0.0
        beta_lo = 0.0
        best_examples = best_cf_examples
        beta_hi: float | None = None
        beta = beta_min

        while trials_used < beta_budget:
            print(f"Testing beta={beta:.6g} for target node {node_idx}.")
            self.set_beta(beta)
            candidate_examples = self.explain(
                cf_optimizer=cf_optimizer,
                node_idx=node_idx,
                new_idx=new_idx,
                lr=lr,
                n_momentum=n_momentum,
                num_epochs=num_epochs,
            )
            trials_used += 1

            if candidate_examples:
                beta_best = beta
                beta_lo = beta
                best_examples = candidate_examples
                beta *= beta_factor
                continue

            beta_hi = beta
            break

        if beta_hi is None:
            print(
                f"Incremental beta search exhausted its trial budget with "
                f"best beta={beta_best:.6g}."
            )
            return best_examples, possible, beta_best

        if beta_lo == 0.0:
            print("No successful positive beta was found; returning beta=0.0.")
            return best_examples, possible, beta_best

        while (
            trials_used < beta_budget
            and beta_hi / beta_lo > beta_refinement_ratio
        ):
            beta_mid = math.sqrt(beta_lo * beta_hi)
            print(
                f"Refining beta in [{beta_lo:.6g}, {beta_hi:.6g}] with "
                f"midpoint {beta_mid:.6g}."
            )
            self.set_beta(beta_mid)
            candidate_examples = self.explain(
                cf_optimizer=cf_optimizer,
                node_idx=node_idx,
                new_idx=new_idx,
                lr=lr,
                n_momentum=n_momentum,
                num_epochs=num_epochs,
            )
            trials_used += 1

            if candidate_examples:
                beta_best = beta_mid
                beta_lo = beta_mid
                best_examples = candidate_examples
            else:
                beta_hi = beta_mid

        print(
            f"Selected beta={beta_best:.6g} for target node {node_idx} "
            f"after {trials_used} trial(s)."
        )
        return best_examples, possible, beta_best

    def explain(
        self,
        cf_optimizer: str,
        node_idx: int,
        new_idx: int,
        lr: float | str,
        n_momentum: float,
        num_epochs: int,
        patience: int = 5,
        dynamic_lr_epsilon: float = DEFAULT_DYNAMIC_LR_EPSILON,
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
            patience: Number of consecutive epochs with zero gradient and static prediction before stopping
        """
        self.node_idx = int(node_idx)
        self.new_idx = int(new_idx)
        self.cf_model.reset_perturbation()

        if isinstance(lr, float):
            lr = lr
        else:
            lr = self.resolve_node_learning_rate(
                lr_setting=lr,
                num_epochs=num_epochs,
                target_node=node_idx,
                epsilon=dynamic_lr_epsilon,
            )

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

        best_cf_example: List[List] = []
        best_loss = np.inf
        num_cf_examples = 0

        # Early stopping variables
        stop_counter = 0
        last_pred = -1

        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            new_example, loss_total, grad_is_zero, current_pred = self.train(
                epoch,
                num_epochs=num_epochs,
            )

            # If the CF model determined there are no further editable
            # node-hyperedge interactions for the target, stop searching.
            if getattr(self.cf_model, "no_available_edits", False):
                print("Stopping search: there are no available edits for target node. Node is isolated in the hypergraph.")
                break
            if getattr(self.cf_model, "no_more_edits", False):
                print("Stopping search: no more editable interactions for target node.")
                break

            if new_example and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1

            if grad_is_zero and current_pred == last_pred:
                stop_counter += 1
            else:
                stop_counter = 0  # Reset if gradient returns or prediction changes

            if stop_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(
                    f"Reason: Gradient zero and prediction stable for {patience} epochs."
                )
                break

            last_pred = current_pred

        print(f"{num_cf_examples} CF examples for node_idx = {self.node_idx}")
        print(" ")
        return best_cf_example

    def train(
        self,
        epoch: int,
        num_epochs: int,
    ) -> Tuple[List, float, bool, int]:
        """
        Single training epoch for the counterfactual model
        Returns:
            cf_stats: List of stats if a valid CF was found
            loss_total: Total loss value
            grad_is_zero: Boolean indicating if gradients were 0
            y_pred_new_actual: The current prediction (integer)
        """

        assert self.cf_optimizer is not None, "Call `explain` before `train`"

        # self.cf_model.train()
        self.cf_model.eval()
        self.cf_optimizer.zero_grad()

        # soft mask forward pass to compute losses and gradients
        output = self.cf_model.forward(self.sub_feat, self.sub_H)

        # hard mask forward pass to compute the actual prediction for the new subgraph structure
        output_actual, self.Pi = self.cf_model.forward_pred(self.sub_feat)

        log_prob_new = output[self.new_idx]
        log_prob_new_actual = output_actual[self.new_idx]

        y_pred_new = torch.argmax(log_prob_new)
        y_pred_new_actual = torch.argmax(log_prob_new_actual)

        loss_total, loss_pred, loss_graph_dist, cf_H = self.cf_model.loss(
            log_prob_new, self.y_pred_orig, y_pred_new_actual
        )
        loss_total.backward()

        pi_hat_grad = self.cf_model.pi_i_hat.grad
        grad_is_zero = False

        if pi_hat_grad is None:
            print(
                f"⚠️ WARNING (Epoch {epoch + 1}): pi_hat has no gradient (grad is None)"
            )
            grad_is_zero = True
        elif torch.all(pi_hat_grad == 0):
            print(f"⚠️WARNING (Epoch {epoch + 1}): pi_hat gradient is all zeros")
            grad_is_zero = True
        else:
            grad_norm = pi_hat_grad.norm().item()
            grad_max = pi_hat_grad.abs().max().item()
            print(f"pi_hat gradient norm: {grad_norm:.6f}, max: {grad_max:.6f}")

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
            if cf_H.is_sparse:
                cf_H_stored = cf_H.coalesce()
            else:
                cf_H_stored = cf_H.to_sparse().coalesce()

            if self.sub_H.is_sparse:
                sub_H_stored = self.sub_H.coalesce()
            else:
                sub_H_stored = self.sub_H.to_sparse().coalesce()

            cf_stats = [
                int(self.node_idx),
                int(self.new_idx),
                cf_H_stored.detach().cpu(),
                sub_H_stored.detach().cpu(),
                int(self.y_pred_orig.item()),
                int(y_pred_new.item()),
                int(y_pred_new_actual.item()),
                self.sub_labels[self.new_idx].detach().cpu().numpy(),
                int(self.sub_H.shape[0]),
                float(loss_total.item()),
                float(loss_pred.item()),
                float(loss_graph_dist.item()),
                self.log_prob_orig.detach().cpu(),
                log_prob_new.detach().cpu(),
                log_prob_new_actual.detach().cpu(),
            ]

        return (
            cf_stats,
            float(loss_total.item()),
            grad_is_zero,
            int(y_pred_new_actual.item()),
        )
