import math
from typing import List, Tuple


from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from utils import normalize_propagation


from .v1_strategy_sparse_hgcn_perturb import HGCN_Perturb as HGCN_Perturb_v1
from .v3_strategy_sparse_hgcn_perturb import HGCN_Perturb as HGCN_Perturb_v3


DEFAULT_INCREMENTAL_BETA_MIN = 1e-6
DEFAULT_INCREMENTAL_BETA_FACTOR = 2.0
DEFAULT_INCREMENTAL_BETA_BUDGET = 30
DEFAULT_INCREMENTAL_BETA_REFINEMENT_RATIO = 1.10
DYNAMIC_LR_MODE = "dynamic"
DYNAMIC_EPOCHWISE_LR_MODE = "dynamic-epochwise"
DYNAMIC_POWERS_OF_TWO_LR_MODE = "dynamic-powers-of-two"
DEFAULT_DYNAMIC_LR_MAX_LOGIT_STEP = 1.0
DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT = 6.0

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _inverse_sigmoid(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    return math.log(probability / (1.0 - probability))



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
        target_edge_debug: bool = False,
        quiet: bool = False,
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
            target_edge_debug: Whether to print per-epoch perturbation diagnostics
            quiet: Whether to suppress per-epoch/trial diagnostics
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
        self.strategy = strategy
        self.target_edge_debug = target_edge_debug
        self.quiet = quiet

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
        self.cf_model.verbose = not self.quiet

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

        # for name, param in self.model.named_parameters():
        #    print("orig model requires_grad: ", name, param.requires_grad)
        # for name, param in self.cf_model.named_parameters():
        #    print("cf model requires_grad: ", name, param.requires_grad)

        self.node_idx: int = -1
        self.new_idx: int = -1
        self.cf_optimizer: optim.Optimizer | None = None
        self._current_lr_debug_by_epoch: list[dict[str, object]] = []
        self._current_epochwise_lrs: list[float] = []
        self._current_lr_checkpoint_epochs: list[int] = []
        self._current_lr_checkpoint_values: list[float] = []
        self._dynamic_epochwise_lr_cache: dict[tuple[str, float, int], list[float]] = {}
        self._dynamic_epochwise_lr_debug_cache: dict[
            tuple[str, float, int], list[dict[str, object]]
        ] = {}
        self._dynamic_powers_lr_cache: dict[tuple[str, float, int], list[float]] = {}
        self._dynamic_powers_lr_checkpoint_cache: dict[
            tuple[str, float, int], tuple[list[int], list[float]]
        ] = {}
        self._dynamic_powers_lr_debug_cache: dict[
            tuple[str, float, int], list[dict[str, object]]
        ] = {}
        self._use_epochwise_dynamic_lr = False

    def _log(self, *items: object) -> None:
        if not self.quiet:
            print(*items)

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)
        self.cf_model.beta = self.beta

    def build_removed_incidence(self, cf_H: Tensor) -> Tensor:
        sub_H = self.sub_H.coalesce()
        cf_H = cf_H.coalesce()

        if sub_H.shape != cf_H.shape:
            raise ValueError(
                "The counterfactual incidence matrix must have the same shape as the original subgraph."
            )

        sub_indices = sub_H.indices()
        cf_indices = cf_H.indices()
        # This helper assumes the perturbation strategies preserve the original
        # sparse COO index pattern and only zero-out removed incidences in the
        # stored values. If a future refactor drops removed incidences from the
        # COO indices entirely, this direct value-wise subtraction is no longer
        # valid and the two sparse tensors must be aligned by index first.
        if not torch.equal(sub_indices, cf_indices):
            raise ValueError(
                "The counterfactual incidence matrix must preserve the original sparse index pattern. "
                "This helper assumes removed incidences remain as explicit zero-valued COO entries; "
                "if they are dropped from the sparse indices, the sparse tensors must be aligned by "
                "index before subtraction."
            )

        removed_values = sub_H.values() - cf_H.values()
        removed_mask = removed_values != 0

        if not torch.any(removed_mask):
            empty_indices = torch.empty(
                (2, 0),
                dtype=sub_indices.dtype,
                device=sub_H.device,
            )
            empty_values = torch.empty(
                (0,),
                dtype=sub_H.dtype,
                device=sub_H.device,
            )
            return torch.sparse_coo_tensor(
                empty_indices,
                empty_values,
                sub_H.size(),
                device=sub_H.device,
                dtype=sub_H.dtype,
            ).coalesce()

        removed_indices = sub_indices[:, removed_mask]
        removed_values = removed_values[removed_mask]
        return torch.sparse_coo_tensor(
            removed_indices,
            removed_values,
            sub_H.size(),
            device=sub_H.device,
            dtype=sub_H.dtype,
        ).coalesce()

    def _dynamic_lr_delta(self, num_epochs: int) -> float:
        denominator = float(num_epochs - 1) if num_epochs > 1 else 1.0
        num_classes = int(self.log_prob_orig.numel())
        return (0.1 + math.log(max(num_classes, 1))) / denominator

    def _dynamic_lr_from_grad(
        self,
        grad: Tensor | None,
        num_epochs: int,
        max_logit_step: float = DEFAULT_DYNAMIC_LR_MAX_LOGIT_STEP,
        active_mask: Tensor | None = None,
    ) -> float:
        if grad is None:
            return 0.0

        if active_mask is not None:
            active_mask = active_mask.to(device=grad.device, dtype=torch.bool)
            grad = grad[active_mask]
            if grad.numel() == 0:
                return 0.0

        grad_norm_sq = float(grad.pow(2).sum().item())
        if not math.isfinite(grad_norm_sq) or grad_norm_sq <= 0.0:
            return 0.0

        raw_lr = self._dynamic_lr_delta(num_epochs) / grad_norm_sq
        grad_abs_max = float(grad.abs().max().item())
        if not math.isfinite(raw_lr):
            raw_lr = 0.0
        if not math.isfinite(grad_abs_max) or grad_abs_max <= 0.0:
            return raw_lr

        max_safe_lr = max_logit_step / grad_abs_max
        return min(raw_lr, max_safe_lr)

    def _build_lr_debug_entry(
        self,
        lr: float,
        grad: Tensor | None,
        active_mask: Tensor | None = None,
        pi_hat: Tensor | None = None,
    ) -> dict[str, object]:
        entry: dict[str, object] = {"lr": float(lr)}
        if not self.target_edge_debug:
            return entry

        grad_cpu = None if grad is None else grad.detach().cpu().clone()
        active_mask_cpu = None
        if active_mask is not None:
            active_mask_cpu = active_mask.detach().to(dtype=torch.bool).cpu().clone()
        if pi_hat is None:
            pi_hat = self.cf_model.pi_i_hat.detach()
        pi_hat_cpu = pi_hat.detach().cpu().clone()
        entry.update(
            {
                "pi_hat": pi_hat_cpu,
                "grad": grad_cpu,
                "active_mask": active_mask_cpu,
            }
        )
        return entry

    def _get_lr_debug_for_epoch(self, epoch: int) -> dict[str, object] | None:
        if not self._current_lr_debug_by_epoch:
            return None
        index = min(max(epoch, 0), len(self._current_lr_debug_by_epoch) - 1)
        return self._current_lr_debug_by_epoch[index]

    def _print_perturbation_debug(
        self,
        lr_debug: dict[str, object] | None,
    ) -> None:
        if not self.target_edge_debug:
            return
        debug_formatter = getattr(self.cf_model, "format_target_perturbation_debug", None)
        if not callable(debug_formatter):
            return
        for debug_line in debug_formatter(lr_debug=lr_debug):
            print(debug_line)

    @staticmethod
    def _format_epochwise_learning_rates(
        epochwise_lrs: list[float],
        epoch_numbers: list[int] | None = None,
    ) -> str:
        if not epochwise_lrs:
            return "  none"

        if epoch_numbers is None:
            epoch_numbers = list(range(1, len(epochwise_lrs) + 1))

        formatted_pairs = [
            f"{epoch_number:04d}:{lr_value:.6e}"
            for epoch_number, lr_value in zip(epoch_numbers, epochwise_lrs)
        ]
        row_size = 5
        rows = [
            "  " + ", ".join(formatted_pairs[start : start + row_size])
            for start in range(0, len(formatted_pairs), row_size)
        ]
        return "\n".join(rows)

    def _build_cf_optimizer(
        self,
        cf_optimizer: str,
        lr: float,
        n_momentum: float,
    ) -> optim.Optimizer:
        if cf_optimizer == "SGD" and n_momentum == 0.0:
            return optim.SGD(self.cf_model.parameters(), lr=lr)
        if cf_optimizer == "SGD" and n_momentum != 0.0:
            return optim.SGD(
                self.cf_model.parameters(),
                lr=lr,
                nesterov=True,
                momentum=n_momentum,
            )
        if cf_optimizer == "Adadelta":
            return optim.Adadelta(self.cf_model.parameters(), lr=lr)
        raise ValueError(f"Unsupported cf_optimizer '{cf_optimizer}'")

    @staticmethod
    def _set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _soft_mask_exhausted(self) -> bool:
        soft_mask = torch.sigmoid(self.cf_model.pi_i_hat.detach())
        if self.strategy == "v1":
            H_coalesced = self.cf_model.H.coalesce()
            indices = H_coalesced.indices()
            target_mask = indices[0] == self.cf_model.target_node
            target_cols = indices[1][target_mask]
            if target_cols.numel() == 0:
                return True
            return bool(torch.all(soft_mask[target_cols] < 1e-3).item())
        return bool(torch.all(soft_mask < 1e-3).item())

    def _learning_rate_scope_mask(self) -> Tensor:
        pi_hat = self.cf_model.pi_i_hat
        if self.strategy == "v1":
            mask = torch.zeros_like(pi_hat, dtype=torch.bool)
            H_coalesced = self.cf_model.H.coalesce()
            indices = H_coalesced.indices()
            target_mask = indices[0] == self.cf_model.target_node
            target_cols = indices[1][target_mask].to(pi_hat.device)
            if target_cols.numel() > 0:
                mask[target_cols] = True
            return mask
        if self.strategy == "v3":
            return torch.ones_like(pi_hat, dtype=torch.bool)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _dynamic_lr_active_mask(self) -> Tensor:
        pi_hat = self.cf_model.pi_i_hat.detach()
        return self._learning_rate_scope_mask() & (
            pi_hat.abs() < DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT
        )

    @staticmethod
    def _power_of_two_epochs(num_epochs: int) -> list[int]:
        epochs = []
        epoch = 1
        while epoch <= num_epochs:
            epochs.append(epoch)
            epoch *= 2
        return epochs

    @staticmethod
    def _expand_sparse_lr_schedule(
        checkpoint_epochs: list[int],
        checkpoint_lrs: list[float],
        num_epochs: int,
    ) -> list[float]:
        if not checkpoint_epochs or not checkpoint_lrs:
            return []

        expanded_lrs: list[float] = []
        checkpoint_index = 0
        current_lr = float(checkpoint_lrs[0])
        for epoch in range(1, num_epochs + 1):
            while (
                checkpoint_index + 1 < len(checkpoint_epochs)
                and checkpoint_epochs[checkpoint_index + 1] <= epoch
            ):
                checkpoint_index += 1
                current_lr = float(checkpoint_lrs[checkpoint_index])
            expanded_lrs.append(current_lr)
        return expanded_lrs

    @staticmethod
    def _expand_sparse_debug_schedule(
        checkpoint_epochs: list[int],
        checkpoint_debug: list[dict[str, object]],
        num_epochs: int,
    ) -> list[dict[str, object]]:
        if not checkpoint_epochs or not checkpoint_debug:
            return []

        expanded_debug: list[dict[str, object]] = []
        checkpoint_index = 0
        current_debug = checkpoint_debug[0]
        for epoch in range(1, num_epochs + 1):
            while (
                checkpoint_index + 1 < len(checkpoint_epochs)
                and checkpoint_epochs[checkpoint_index + 1] <= epoch
            ):
                checkpoint_index += 1
                current_debug = checkpoint_debug[checkpoint_index]
            expanded_debug.append(current_debug)
        return expanded_debug

    def compute_dynamic_lr(
        self,
        num_epochs: int,
    ) -> float:
        if num_epochs <= 0:
            raise ValueError(
                "num_epochs must be positive when computing a dynamic learning rate."
            )

        original_beta = self.beta
        self.cf_model.reset_perturbation()
        self.set_beta(0.0)
        self.cf_model.eval()
        self.cf_model.zero_grad(set_to_none=True)
        pi_hat_snapshot = (
            self.cf_model.pi_i_hat.detach().clone()
            if self.target_edge_debug
            else None
        )

        output = self.cf_model.forward(self.sub_feat, self.sub_H)
        target_output = output[self.target_node_sub_idx].unsqueeze(0)
        target_label = self.y_pred_orig.view(1)
        dynamic_lr_loss = -F.nll_loss(target_output, target_label)
        dynamic_lr_loss.backward()

        grad = self.cf_model.pi_i_hat.grad
        lr = self._dynamic_lr_from_grad(grad, num_epochs)
        self._current_lr_debug_by_epoch = []
        if self.target_edge_debug:
            self._current_lr_debug_by_epoch = [
                self._build_lr_debug_entry(
                    lr=lr,
                    grad=grad,
                    pi_hat=pi_hat_snapshot,
                )
            ]

        self.cf_model.zero_grad(set_to_none=True)
        self.cf_model.reset_perturbation()
        self.set_beta(original_beta)

        return lr

    def compute_dynamic_epochwise_lrs(
        self,
        cf_optimizer: str,
        n_momentum: float,
        num_epochs: int,
    ) -> list[float]:
        if num_epochs <= 0:
            raise ValueError(
                "num_epochs must be positive when computing dynamic-epochwise learning rates."
            )

        cache_key = (cf_optimizer, float(n_momentum), int(num_epochs))
        cached_lrs = self._dynamic_epochwise_lr_cache.get(cache_key)
        if cached_lrs is not None:
            self._current_epochwise_lrs = list(cached_lrs)
            self._current_lr_checkpoint_epochs = list(range(1, num_epochs + 1))
            self._current_lr_checkpoint_values = list(cached_lrs)
            if self.target_edge_debug:
                self._current_lr_debug_by_epoch = list(
                    self._dynamic_epochwise_lr_debug_cache.get(cache_key, [])
                )
            else:
                self._current_lr_debug_by_epoch = []
            return list(cached_lrs)

        original_beta = self.beta
        epochwise_lrs: list[float] = []
        lr_debug_by_epoch: list[dict[str, object]] = []
        target_label = self.y_pred_orig.view(1)

        try:
            self.cf_model.reset_perturbation()
            self.set_beta(0.0)
            self.cf_model.eval()
            self.cf_model.zero_grad(set_to_none=True)
            calibration_optimizer = self._build_cf_optimizer(
                cf_optimizer=cf_optimizer,
                lr=0.0,
                n_momentum=n_momentum,
            )

            for _epoch in range(num_epochs):
                calibration_optimizer.zero_grad(set_to_none=True)
                pi_hat_snapshot = (
                    self.cf_model.pi_i_hat.detach().clone()
                    if self.target_edge_debug
                    else None
                )
                active_mask = self._dynamic_lr_active_mask()

                output = self.cf_model.forward(self.sub_feat, self.sub_H)
                target_output = output[self.target_node_sub_idx].unsqueeze(0)
                dynamic_lr_loss = -F.nll_loss(target_output, target_label)
                dynamic_lr_loss.backward()

                grad = self.cf_model.pi_i_hat.grad
                lr = self._dynamic_lr_from_grad(
                    grad,
                    num_epochs,
                    active_mask=active_mask,
                )
                epochwise_lrs.append(float(lr))
                if self.target_edge_debug:
                    lr_debug_by_epoch.append(
                        self._build_lr_debug_entry(
                            lr=lr,
                            grad=grad,
                            active_mask=active_mask,
                            pi_hat=pi_hat_snapshot,
                        )
                    )

                self._set_optimizer_lr(calibration_optimizer, lr)
                clip_grad_norm_(self.cf_model.parameters(), 2.0)
                calibration_optimizer.step()
        finally:
            self.cf_model.zero_grad(set_to_none=True)
            self.cf_model.reset_perturbation()
            self.set_beta(original_beta)

        self._dynamic_epochwise_lr_cache[cache_key] = list(epochwise_lrs)
        if self.target_edge_debug:
            self._dynamic_epochwise_lr_debug_cache[cache_key] = list(lr_debug_by_epoch)
        self._current_epochwise_lrs = list(epochwise_lrs)
        self._current_lr_checkpoint_epochs = list(range(1, num_epochs + 1))
        self._current_lr_checkpoint_values = list(epochwise_lrs)
        self._current_lr_debug_by_epoch = lr_debug_by_epoch
        return list(epochwise_lrs)

    def compute_dynamic_powers_of_two_lrs(
        self,
        cf_optimizer: str,
        n_momentum: float,
        num_epochs: int,
    ) -> list[float]:
        if num_epochs <= 0:
            raise ValueError(
                "num_epochs must be positive when computing dynamic powers-of-two learning rates."
            )

        cache_key = (cf_optimizer, float(n_momentum), int(num_epochs))
        cached_lrs = self._dynamic_powers_lr_cache.get(cache_key)
        if cached_lrs is not None:
            checkpoint_epochs, checkpoint_lrs = (
                self._dynamic_powers_lr_checkpoint_cache[cache_key]
            )
            self._current_epochwise_lrs = list(cached_lrs)
            self._current_lr_checkpoint_epochs = list(checkpoint_epochs)
            self._current_lr_checkpoint_values = list(checkpoint_lrs)
            if self.target_edge_debug:
                self._current_lr_debug_by_epoch = list(
                    self._dynamic_powers_lr_debug_cache.get(cache_key, [])
                )
            else:
                self._current_lr_debug_by_epoch = []
            return list(cached_lrs)

        original_beta = self.beta
        checkpoint_epochs = self._power_of_two_epochs(num_epochs)
        checkpoint_lrs: list[float] = []
        checkpoint_debug: list[dict[str, object]] = []
        target_label = self.y_pred_orig.view(1)

        try:
            self.cf_model.reset_perturbation()
            self.set_beta(0.0)
            self.cf_model.eval()
            self.cf_model.zero_grad(set_to_none=True)
            calibration_optimizer = self._build_cf_optimizer(
                cf_optimizer=cf_optimizer,
                lr=0.0,
                n_momentum=n_momentum,
            )

            for _checkpoint_epoch in checkpoint_epochs:
                calibration_optimizer.zero_grad(set_to_none=True)
                pi_hat_snapshot = (
                    self.cf_model.pi_i_hat.detach().clone()
                    if self.target_edge_debug
                    else None
                )
                active_mask = self._dynamic_lr_active_mask()

                output = self.cf_model.forward(self.sub_feat, self.sub_H)
                target_output = output[self.target_node_sub_idx].unsqueeze(0)
                dynamic_lr_loss = -F.nll_loss(target_output, target_label)
                dynamic_lr_loss.backward()

                grad = self.cf_model.pi_i_hat.grad
                lr = self._dynamic_lr_from_grad(
                    grad,
                    num_epochs,
                    active_mask=active_mask,
                )
                checkpoint_lrs.append(float(lr))
                if self.target_edge_debug:
                    checkpoint_debug.append(
                        self._build_lr_debug_entry(
                            lr=lr,
                            grad=grad,
                            active_mask=active_mask,
                            pi_hat=pi_hat_snapshot,
                        )
                    )

                self._set_optimizer_lr(calibration_optimizer, lr)
                clip_grad_norm_(self.cf_model.parameters(), 2.0)
                calibration_optimizer.step()
        finally:
            self.cf_model.zero_grad(set_to_none=True)
            self.cf_model.reset_perturbation()
            self.set_beta(original_beta)

        expanded_lrs = self._expand_sparse_lr_schedule(
            checkpoint_epochs,
            checkpoint_lrs,
            num_epochs,
        )
        debug_by_epoch = []
        if self.target_edge_debug:
            debug_by_epoch = self._expand_sparse_debug_schedule(
                checkpoint_epochs,
                checkpoint_debug,
                num_epochs,
            )

        self._dynamic_powers_lr_cache[cache_key] = list(expanded_lrs)
        self._dynamic_powers_lr_checkpoint_cache[cache_key] = (
            list(checkpoint_epochs),
            list(checkpoint_lrs),
        )
        if self.target_edge_debug:
            self._dynamic_powers_lr_debug_cache[cache_key] = list(debug_by_epoch)
        self._current_epochwise_lrs = list(expanded_lrs)
        self._current_lr_checkpoint_epochs = list(checkpoint_epochs)
        self._current_lr_checkpoint_values = list(checkpoint_lrs)
        self._current_lr_debug_by_epoch = debug_by_epoch
        return list(expanded_lrs)

    def resolve_node_learning_rate(
        self,
        lr_setting: float | str,
        num_epochs: int,
        target_node: int,
    ) -> float:
        if isinstance(lr_setting, float):
            return lr_setting

        node_lr = self.compute_dynamic_lr(
            num_epochs=num_epochs,
        )
        self._log(f"Dynamic learning rate for target node {target_node}: {node_lr:.6g}")
        return node_lr

    def run_incremental_beta_search(
        self,
        cf_optimizer: str,
        node_idx: int,
        new_idx: int,
        lr: float | str,
        n_momentum: float,
        num_epochs: int,
        beta_min: float = DEFAULT_INCREMENTAL_BETA_MIN,
        beta_factor: float = DEFAULT_INCREMENTAL_BETA_FACTOR,
        beta_budget: int = DEFAULT_INCREMENTAL_BETA_BUDGET,
        beta_refinement_ratio: float = DEFAULT_INCREMENTAL_BETA_REFINEMENT_RATIO,
    ) -> tuple[List[List], bool, float]:
        trials_used = 0

        self._log(f"Starting incremental beta search for target node {node_idx}.")
        self.set_beta(0.0)
        best_cf_examples = self.explain(
            cf_optimizer=cf_optimizer,
            node_idx=node_idx,
            new_idx=new_idx,
            lr=lr,
            n_momentum=n_momentum,
            num_epochs=num_epochs,
            debug_learning_rates=True,
        )
        possible = bool(best_cf_examples) or not self.cf_model.no_more_edits
        trials_used += 1

        if not best_cf_examples:
            self._log(
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
            self._log(f"Testing beta={beta:.6g} for target node {node_idx}.")
            self.set_beta(beta)
            candidate_examples = self.explain(
                cf_optimizer=cf_optimizer,
                node_idx=node_idx,
                new_idx=new_idx,
                lr=lr,
                n_momentum=n_momentum,
                num_epochs=num_epochs,
                debug_learning_rates=False,
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
            self._log(
                f"Incremental beta search exhausted its trial budget with "
                f"best beta={beta_best:.6g}."
            )
            return best_examples, possible, beta_best

        if beta_lo == 0.0:
            self._log("No successful positive beta was found; returning beta=0.0.")
            return best_examples, possible, beta_best

        while trials_used < beta_budget and beta_hi / beta_lo > beta_refinement_ratio:
            beta_mid = math.sqrt(beta_lo * beta_hi)
            self._log(
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
                debug_learning_rates=False,
            )
            trials_used += 1

            if candidate_examples:
                beta_best = beta_mid
                beta_lo = beta_mid
                best_examples = candidate_examples
            else:
                beta_hi = beta_mid

        self._log(
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
        debug_learning_rates: bool = False,
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
        self._current_lr_debug_by_epoch = []
        self._current_epochwise_lrs = []
        self._current_lr_checkpoint_epochs = []
        self._current_lr_checkpoint_values = []
        self._use_epochwise_dynamic_lr = False

        if isinstance(lr, float):
            lr = lr
        elif lr == DYNAMIC_LR_MODE:
            lr = self.resolve_node_learning_rate(
                lr_setting=lr,
                num_epochs=num_epochs,
                target_node=node_idx,
            )
        elif lr in {DYNAMIC_EPOCHWISE_LR_MODE, DYNAMIC_POWERS_OF_TWO_LR_MODE}:
            lr_mode = lr
            if lr_mode == DYNAMIC_EPOCHWISE_LR_MODE:
                scheduled_lrs = self.compute_dynamic_epochwise_lrs(
                    cf_optimizer=cf_optimizer,
                    n_momentum=n_momentum,
                    num_epochs=num_epochs,
                )
                lr_mode_label = "dynamic epochwise"
            else:
                scheduled_lrs = self.compute_dynamic_powers_of_two_lrs(
                    cf_optimizer=cf_optimizer,
                    n_momentum=n_momentum,
                    num_epochs=num_epochs,
                )
                lr_mode_label = "dynamic powers-of-two"
            self._use_epochwise_dynamic_lr = True
            lr = scheduled_lrs[0] if scheduled_lrs else 0.0
            if debug_learning_rates and not self.quiet:
                formatted_lrs = self._format_epochwise_learning_rates(
                    self._current_lr_checkpoint_values,
                    self._current_lr_checkpoint_epochs,
                )
                self._log(
                    f"Computed {len(self._current_lr_checkpoint_values)} "
                    f"{lr_mode_label} learning rates "
                    f"for target node {node_idx}:\n"
                    f"{formatted_lrs}"
                )
        else:
            raise ValueError(f"Unsupported learning-rate mode '{lr}'")

        self.cf_optimizer = self._build_cf_optimizer(
            cf_optimizer=cf_optimizer,
            lr=lr,
            n_momentum=n_momentum,
        )

        best_cf_example: List[List] = []
        best_loss = np.inf
        num_cf_examples = 0

        # Early stopping variables
        stop_counter = 0
        last_pred = -1

        for epoch in tqdm(
            range(num_epochs),
            desc="Training epochs",
            disable=self.quiet,
        ):
            new_example, loss_total, grad_is_zero, current_pred = self.train(
                epoch,
                num_epochs=num_epochs,
            )

            if new_example and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1

            # If the CF model determined there are no further editable
            # node-hyperedge interactions for the target, stop searching.
            if getattr(self.cf_model, "no_available_edits", False):
                self._log(
                    "Stopping search: there are no available edits for target node. "
                    "Node is isolated in the hypergraph."
                )
                break
            if getattr(self.cf_model, "no_more_edits", False):
                self._log(
                    "Stopping search: no more editable interactions for target node."
                )
                break

            if grad_is_zero and current_pred == last_pred:
                stop_counter += 1
            else:
                stop_counter = 0  # Reset if gradient returns or prediction changes

            if stop_counter >= patience:
                self._log(f"\nEarly stopping triggered at epoch {epoch + 1}")
                self._log(
                    f"Reason: Gradient zero and prediction stable for {patience} epochs."
                )
                break

            last_pred = current_pred

        self._log(f"{num_cf_examples} CF examples for node_idx = {self.node_idx}")
        self._log(" ")
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
        lr_debug = self._get_lr_debug_for_epoch(epoch)
        if self._use_epochwise_dynamic_lr and self._current_epochwise_lrs:
            lr_index = min(max(epoch, 0), len(self._current_epochwise_lrs) - 1)
            self._set_optimizer_lr(
                self.cf_optimizer,
                self._current_epochwise_lrs[lr_index],
            )
        self.cf_optimizer.zero_grad()
        current_lr = float(self.cf_optimizer.param_groups[0]["lr"])

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
        grad_is_zero = False

        stop_requested = bool(
            getattr(self.cf_model, "no_available_edits", False)
            or getattr(self.cf_model, "no_more_edits", False)
        )

        if stop_requested:
            grad_is_zero = True
            self._log(
                f"Stopping optimization before backward at epoch {epoch + 1}: "
                "no more editable interactions remain for the target node."
            )
            self._print_perturbation_debug(lr_debug)
        else:
            loss_total.backward()

            pi_hat_grad = self.cf_model.pi_i_hat.grad
            self._print_perturbation_debug(lr_debug)

            if pi_hat_grad is None:
                self._log(
                    f"⚠️ WARNING (Epoch {epoch + 1}): pi_hat has no gradient (grad is None)"
                )
                grad_is_zero = True
            elif torch.all(pi_hat_grad == 0):
                self._log(f"⚠️WARNING (Epoch {epoch + 1}): pi_hat gradient is all zeros")
                grad_is_zero = True
            else:
                grad_norm = pi_hat_grad.norm().item()
                grad_max = pi_hat_grad.abs().max().item()
                self._log(f"pi_hat gradient norm: {grad_norm:.6f}, max: {grad_max:.6f}")

            clip_grad_norm_(self.cf_model.parameters(), 2.0)
            self.cf_optimizer.step()

        epoch_summary = [
            "Node idx: {}".format(self.node_idx),
            "New idx: {}".format(self.new_idx),
            "Epoch: {:04d}".format(epoch + 1),
        ]
        if not self._use_epochwise_dynamic_lr:
            epoch_summary.append("lr: {:.6g}".format(current_lr))
        epoch_summary.extend(
            [
                "loss: {:.4f}".format(loss_total.item()),
                "pred loss: {:.4f}".format(loss_pred.item()),
                "graph loss: {:.4f}".format(loss_graph_dist.item()),
            ]
        )
        self._log(*epoch_summary)
        self._log(
            "Output: {}\n".format(output[self.new_idx].data),
            "Output nondiff: {}\n".format(output_actual[self.new_idx].data),
            "orig pred: {}, new pred: {}, new pred nondiff: {}".format(
                self.y_pred_orig, y_pred_new, y_pred_new_actual
            ),
        )
        self._log(" ")

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

            removed_H = self.build_removed_incidence(cf_H)
            with torch.no_grad():
                S_removed = normalize_propagation(removed_H)
                removed_output = self.model(self.sub_feat, S_removed)
            log_prob_removed_only = removed_output[self.new_idx]

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
                log_prob_removed_only.detach().cpu(),
            ]

        return (
            cf_stats,
            float(loss_total.item()),
            grad_is_zero,
            int(y_pred_new_actual.item()),
        )
