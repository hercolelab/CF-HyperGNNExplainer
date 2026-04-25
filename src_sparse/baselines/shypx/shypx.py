from typing import Any, Dict

import torch
import torch.nn.functional as F

from utils import normalize_propagation


class SHypXExplainer:
    def __init__(
        self,
        model: torch.nn.Module,
        comp_H: torch.Tensor,
        sub_feat: torch.Tensor,
        target_node_local: int,
        lambda_pred: float = 1.0,
        lambda_size: float = 0.005,
        tau: float = 1.0,
        lr: float = 0.01,
        num_epochs: int = 400,
        init_prob: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.comp_H = comp_H.coalesce().to(device)
        self.sub_feat = sub_feat.to(device)
        self.target_node_local = target_node_local
        self.lambda_pred = lambda_pred
        self.lambda_size = lambda_size
        self.tau = tau
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

        self.num_links = int(self.comp_H._nnz())
        self.N, self.E = self.comp_H.size()

        S_comp = normalize_propagation(self.comp_H)
        with torch.no_grad():
            self.log_p_orig = model(self.sub_feat, S_comp)[
                self.target_node_local
            ].clone()
            self.y_pred_orig = self.log_p_orig.argmax().item()

        if self.num_links == 0:
            self._empty = True
            return
        self._empty = False

        indices = self.comp_H.indices()
        self.link_rows = indices[0]
        self.link_cols = indices[1]
        self.flat_idx = (self.link_rows * self.E + self.link_cols).long()

        init_logit = torch.log(
            torch.tensor(init_prob / (1.0 - init_prob), device=device)
        )
        self.logits = torch.full(
            (self.num_links,),
            init_logit.item(),
            device=device,
            requires_grad=True,
        )

    def explain(self) -> Dict[str, Any]:
        if self._empty:
            return self._empty_result()

        optimizer = torch.optim.Adam([self.logits], lr=self.lr)
        best_loss = float("inf")
        best_mask: torch.Tensor | None = None

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            y = self._gumbel_softmax_sample()
            H_dense = self._build_dense_H(y)
            S_dense = self._normalize_propagation_dense(H_dense)
            log_p_sub = self.model(self.sub_feat, S_dense)[self.target_node_local]
            log_p_sub = log_p_sub.clamp(min=-100)

            loss_pred = F.kl_div(
                self.log_p_orig.detach().unsqueeze(0),
                log_p_sub.unsqueeze(0),
                reduction="batchmean",
                log_target=True,
            )
            loss_size = y.sum()
            loss = self.lambda_pred * loss_pred + self.lambda_size * loss_size

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_mask = y.detach().clone()
                    print(
                        f"  Best checkpoint epoch={epoch}: loss={best_loss:.4f}, kl={loss_pred.item():.4f}, size={loss_size.item():.1f}"
                    )

        with torch.no_grad():
            if best_mask is None:
                best_mask = (torch.sigmoid(self.logits) >= 0.5).float()

            expl_H = self._mask_to_sparse(best_mask)
            expl_H = self._extract_connected_component(expl_H)
            comp_minus_expl_H = self._build_complement_hypergraph(expl_H)

            S_expl = normalize_propagation(expl_H)
            log_p_expl = self.model(self.sub_feat, S_expl)[self.target_node_local]
            S_comp_minus_expl = normalize_propagation(comp_minus_expl_H)
            log_p_comp_minus_expl = self.model(self.sub_feat, S_comp_minus_expl)[
                self.target_node_local
            ]
            num_links_expl = int((expl_H.coalesce().values() > 0.5).sum().item())
            num_links_comp_minus_expl = int(comp_minus_expl_H._nnz())

        return {
            "expl_H": expl_H.detach().cpu(),
            "comp_H": self.comp_H.detach().cpu(),
            "comp_minus_expl_H": comp_minus_expl_H.detach().cpu(),
            "log_p_expl": log_p_expl.detach().cpu(),
            "log_p_comp_minus_expl": log_p_comp_minus_expl.detach().cpu(),
            "log_p_orig": self.log_p_orig.detach().cpu(),
            "y_pred_orig": self.y_pred_orig,
            "y_pred_expl": log_p_expl.argmax().item(),
            "y_pred_comp_minus_expl": log_p_comp_minus_expl.argmax().item(),
            "num_links_expl": num_links_expl,
            "num_links_comp_minus_expl": num_links_comp_minus_expl,
            "num_links_comp": self.num_links,
            "best_loss": best_loss,
        }

    def _build_complement_hypergraph(self, expl_H: torch.Tensor) -> torch.Tensor:
        comp = self.comp_H.coalesce()
        expl = expl_H.coalesce()

        comp_rows, comp_cols = comp.indices()
        if comp_rows.numel() == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=comp.dtype, device=self.device),
                comp.size(),
                device=self.device,
                dtype=comp.dtype,
            ).coalesce()

        E = comp.size(1)
        comp_flat = comp_rows * E + comp_cols

        expl_rows, expl_cols = expl.indices()
        if expl_rows.numel() == 0:
            keep_mask = torch.ones(
                comp_rows.numel(), dtype=torch.bool, device=self.device
            )
        else:
            expl_flat = expl_rows * E + expl_cols
            keep_mask = ~torch.isin(comp_flat, expl_flat)

        rows = comp_rows[keep_mask]
        cols = comp_cols[keep_mask]
        vals = torch.ones(rows.numel(), dtype=comp.dtype, device=self.device)

        return torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            comp.size(),
            device=self.device,
            dtype=comp.dtype,
        ).coalesce()

    def _gumbel_softmax_sample(self) -> torch.Tensor:
        logits_2d = torch.stack([torch.zeros_like(self.logits), self.logits], dim=-1)
        samples = F.gumbel_softmax(logits_2d, tau=self.tau, hard=True, dim=-1)
        return samples[:, 1]

    def _build_dense_H(self, mask: torch.Tensor) -> torch.Tensor:
        flat_H = torch.zeros(self.N * self.E, device=self.device, dtype=mask.dtype)
        flat_H = flat_H.scatter(0, self.flat_idx, mask)
        return flat_H.view(self.N, self.E)

    @staticmethod
    def _normalize_propagation_dense(H: torch.Tensor) -> torch.Tensor:
        d_inv_sqrt = H.sum(dim=1).clamp(min=1).pow(-0.5)
        b_inv = H.sum(dim=0).clamp(min=1).pow(-1.0)

        # H_left = D^{-1/2} H B^{-1},  H_right = D^{-1/2} H
        H_left = d_inv_sqrt.unsqueeze(1) * H * b_inv.unsqueeze(0)
        H_right = d_inv_sqrt.unsqueeze(1) * H
        return H_left @ H_right.t()

    def _mask_to_sparse(self, mask: torch.Tensor) -> torch.Tensor:
        active = mask > 0.5
        rows = self.link_rows[active]
        cols = self.link_cols[active]
        num = int(active.sum().item())
        vals = torch.ones(num, dtype=self.comp_H.dtype, device=self.device)
        return torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            (self.N, self.E),
            device=self.device,
            dtype=self.comp_H.dtype,
        ).coalesce()

    def _extract_connected_component(self, expl_H: torch.Tensor) -> torch.Tensor:
        expl_H = expl_H.coalesce()
        indices = expl_H.indices()
        values = expl_H.values()

        active_mask = values > 0.5
        if not active_mask.any():
            return expl_H

        active_rows = indices[0][active_mask]
        active_cols = indices[1][active_mask]
        N, E = expl_H.size()

        node_visited = torch.zeros(N, dtype=torch.bool, device=self.device)
        node_visited[self.target_node_local] = True

        changed = True
        while changed:
            changed = False

            node_in_link = node_visited[active_rows]
            edge_reached = torch.zeros(E, dtype=torch.bool, device=self.device)
            if node_in_link.any():
                edge_reached[active_cols[node_in_link]] = True

            edge_in_link = edge_reached[active_cols]
            new_node_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
            if edge_in_link.any():
                new_node_mask[active_rows[edge_in_link]] = True

            newly_found = new_node_mask & ~node_visited
            if newly_found.any():
                node_visited |= newly_found
                changed = True

        node_in_link = node_visited[active_rows]
        edge_reached = torch.zeros(E, dtype=torch.bool, device=self.device)
        if node_in_link.any():
            edge_reached[active_cols[node_in_link]] = True
        cc_link_mask = edge_reached[active_cols]

        cc_rows = active_rows[cc_link_mask]
        cc_cols = active_cols[cc_link_mask]
        num_cc = int(cc_link_mask.sum().item())
        cc_vals = torch.ones(num_cc, dtype=expl_H.dtype, device=self.device)

        return torch.sparse_coo_tensor(
            torch.stack([cc_rows, cc_cols]),
            cc_vals,
            expl_H.size(),
            device=self.device,
            dtype=expl_H.dtype,
        ).coalesce()

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "expl_H": self.comp_H.detach().cpu(),
            "comp_H": self.comp_H.detach().cpu(),
            "comp_minus_expl_H": self.comp_H.detach().cpu(),
            "log_p_expl": self.log_p_orig.detach().cpu(),
            "log_p_comp_minus_expl": self.log_p_orig.detach().cpu(),
            "log_p_orig": self.log_p_orig.detach().cpu(),
            "y_pred_orig": self.y_pred_orig,
            "y_pred_expl": self.y_pred_orig,
            "y_pred_comp_minus_expl": self.y_pred_orig,
            "num_links_expl": 0,
            "num_links_comp_minus_expl": 0,
            "num_links_comp": 0,
            "best_loss": 0.0,
        }
