from typing import Any

import torch

from utils import normalize_propagation

from baselines.hyperex.common import (
    build_explanation_hypergraph_from_alpha,
    build_local_to_global,
    compute_hyperedge_embeddings_global,
    extract_induced_edge_global_ids,
    local_class_probabilities,
    sparse_incidence_to_dense,
)


class HyperExExplainer:
    def __init__(
        self,
        model: torch.nn.Module,
        full_H: torch.Tensor,
        full_feat: torch.Tensor,
        comp_H: torch.Tensor,
        sub_feat: torch.Tensor,
        node_dict: dict[int, int],
        target_node_global: int,
        attention_module: torch.nn.Module,
        thresh_num: int = 10,
        device: torch.device = torch.device("cpu"),
        global_logits: torch.Tensor | None = None,
        global_log_p: torch.Tensor | None = None,
    ):
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.attention_module = attention_module
        self.attention_module.eval()

        self.full_H = full_H.coalesce().to(device)
        self.full_feat = full_feat.to(device)
        self.comp_H = comp_H.coalesce().to(device)
        self.sub_feat = sub_feat.to(device)
        self.device = device

        self.target_node_global = int(target_node_global)
        self.target_node_local = int(node_dict[self.target_node_global])
        self._node_dict = node_dict
        self.local_to_global = build_local_to_global(node_dict, device)
        self.thresh_num = int(thresh_num)

        self.num_links = int(self.comp_H._nnz())

        with torch.no_grad():
            if global_logits is None:
                S_full = normalize_propagation(self.full_H)
                global_logits = self.model(
                    self.full_feat, S_full, return_embeddings=True
                )
            if global_log_p is None:
                global_log_p = torch.log_softmax(global_logits, dim=1)

            self.global_logits = global_logits.to(device)
            self.global_log_p = global_log_p.to(device)
            self.global_probs = self.global_log_p.exp()
            self.log_p_orig = self.global_log_p[self.target_node_global].clone()
            self.y_pred_orig = int(self.log_p_orig.argmax().item())

    def explain(self) -> dict[str, Any]:
        if self.num_links == 0:
            return self._empty_result()

        with torch.no_grad():
            z_local = local_class_probabilities(
                self.global_logits, self.local_to_global
            )
            H_dense = sparse_incidence_to_dense(self.comp_H)
            h_global = compute_hyperedge_embeddings_global(
                self.full_H, self.global_logits
            )
            comp_edge_global_ids = extract_induced_edge_global_ids(
                self.full_H, self._node_dict
            )
            h_edges = h_global[comp_edge_global_ids]
            alpha, omega_m = self.attention_module.forward_dense(
                z_local,
                h_edges,
                H_dense,
            )
            expl_H = build_explanation_hypergraph_from_alpha(
                comp_H=self.comp_H,
                alpha=alpha,
                thresh_num=self.thresh_num,
            )
            rows, cols = self.comp_H.indices()
            edge_scores = omega_m[rows, cols].detach().cpu()
            local_norm = alpha[rows, cols].detach().cpu()

            S_expl = normalize_propagation(expl_H)
            log_p_expl = self.model(self.sub_feat, S_expl)[self.target_node_local]

        return {
            "expl_H": expl_H.detach().cpu(),
            "comp_H": self.comp_H.detach().cpu(),
            "log_p_expl": log_p_expl.detach().cpu(),
            "log_p_orig": self.log_p_orig.detach().cpu(),
            "y_pred_orig": self.y_pred_orig,
            "y_pred_expl": int(log_p_expl.argmax().item()),
            "num_links_expl": int(expl_H._nnz()),
            "num_links_comp": self.num_links,
            "edge_scores": edge_scores,
            "local_norm": local_norm,
            "alpha": alpha.detach().cpu(),
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "expl_H": self.comp_H.detach().cpu(),
            "comp_H": self.comp_H.detach().cpu(),
            "log_p_expl": self.log_p_orig.detach().cpu(),
            "log_p_orig": self.log_p_orig.detach().cpu(),
            "y_pred_orig": self.y_pred_orig,
            "y_pred_expl": self.y_pred_orig,
            "num_links_expl": 0,
            "num_links_comp": 0,
            "edge_scores": torch.zeros(0),
            "local_norm": torch.zeros(0),
            "alpha": torch.zeros(0),
        }
