import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalize_propagation
DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT = 6

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

        # Start from a less saturated "keep hyperedge" initialization.
        self.pi_i_hat = nn.Parameter(
            torch.full((num_edges,), 4.0, device=H.device), requires_grad=True
        )

        self.pi_i = None
        self.H_tilde = None
        self.no_more_edits = False
        self.no_available_edits = False
        (
            self.edge_nodes,
            self.node_edges,
            self.edge_distances,
            self.distance_prev_edges,
            self.distance_next_edges,
        ) = self._build_edge_distance_metadata()
        self.target_edge_debug = True
        self.distance_prev_edge_counts = self._build_distance_prev_edge_counts()
        self.distance_prev_active_edge_counts = list(self.distance_prev_edge_counts)
        self.distance_prev_active_edges = [
            set(prev_edges) for prev_edges in self.distance_prev_edges
        ]

    def reset_perturbation(self) -> None:
        with torch.no_grad():
            self.pi_i_hat.fill_(4.0)
        self.pi_i = None
        self.H_tilde = None
        self.no_more_edits = False
        self.no_available_edits = False
        self.distance_prev_active_edge_counts = list(self.distance_prev_edge_counts)

    def set_target_edge_debug(self, enabled: bool) -> None:
        self.target_edge_debug = bool(enabled)
        if self.target_edge_debug:
            return

        # In non-debug runs the active-mask cascade only needs each edge's
        # next-distance dependents and a mutable predecessor count.
        self.edge_nodes = []
        self.node_edges = {}
        self.edge_distances = []
        self.distance_prev_edges = []
        self.distance_prev_active_edges = []

    def _build_edge_distance_metadata(
        self,
    ) -> tuple[
        list[set[int]],
        dict[int, set[int]],
        list[int | None],
        list[set[int]],
        list[set[int]],
    ]:
        H_indices = self.H.coalesce().indices().detach().cpu()
        num_edges = int(self.pi_i_hat.numel())
        edge_nodes = [set() for _ in range(num_edges)]
        node_edges: dict[int, set[int]] = {}
        for node_idx, edge_idx in zip(H_indices[0].tolist(), H_indices[1].tolist()):
            if 0 <= edge_idx < num_edges:
                edge_nodes[edge_idx].add(int(node_idx))
                node_edges.setdefault(int(node_idx), set()).add(int(edge_idx))

        edge_distances: list[int | None] = [None] * num_edges
        target_edges = sorted(node_edges.get(int(self.target_node), set()))
        queue = list(target_edges)
        for edge_idx in target_edges:
            edge_distances[edge_idx] = 0

        queue_index = 0
        while queue_index < len(queue):
            current_edge = queue[queue_index]
            queue_index += 1
            current_distance = edge_distances[current_edge]
            if current_distance is None:
                continue
            for node_idx in edge_nodes[current_edge]:
                for next_edge in node_edges.get(node_idx, set()):
                    if next_edge == current_edge:
                        continue
                    next_distance = current_distance + 1
                    if edge_distances[next_edge] is None:
                        edge_distances[next_edge] = next_distance
                        queue.append(next_edge)

        distance_prev_edges = [set() for _ in range(num_edges)]
        distance_next_edges = [set() for _ in range(num_edges)]
        for edge_idx, distance in enumerate(edge_distances):
            if distance is None:
                continue
            for node_idx in edge_nodes[edge_idx]:
                for adjacent_edge in node_edges.get(node_idx, set()):
                    if adjacent_edge == edge_idx:
                        continue
                    adjacent_distance = edge_distances[adjacent_edge]
                    if adjacent_distance == distance - 1:
                        distance_prev_edges[edge_idx].add(adjacent_edge)
                    elif adjacent_distance == distance + 1:
                        distance_next_edges[edge_idx].add(adjacent_edge)

        return (
            edge_nodes,
            node_edges,
            edge_distances,
            distance_prev_edges,
            distance_next_edges,
        )

    def _build_distance_prev_edge_counts(self) -> list[int]:
        return [
            1
            if distance == 0
            else len(prev_edges)
            if distance is not None
            else 0
            for distance, prev_edges in zip(self.edge_distances, self.distance_prev_edges)
        ]

    def compute_dynamic_lr_active_state(
        self,
        pi_hat: torch.Tensor | None = None,
        logit_limit: float = DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT,
    ) -> tuple[torch.Tensor, list[set[int]]]:
        if pi_hat is None:
            pi_hat = self.pi_i_hat.detach()
        else:
            pi_hat = pi_hat.detach()

        if not self.target_edge_debug:
            active_mask = self._compute_dynamic_lr_active_mask_with_counts(
                pi_hat,
                logit_limit,
            )
            return active_mask, []

        num_edges = len(self.edge_distances)
        if int(pi_hat.numel()) != num_edges:
            raise ValueError(
                "pi_hat size does not match the number of hyperedges in the neighborhood."
            )

        finite_pi_hat = torch.isfinite(pi_hat)
        base_active_tensor = finite_pi_hat & (pi_hat.abs() < float(logit_limit))
        base_active = [
            bool(is_active)
            for is_active in base_active_tensor.detach().cpu().reshape(-1).tolist()
        ]

        active = []
        for edge_idx, is_active in enumerate(base_active):
            distance = self.edge_distances[edge_idx]
            has_reachable_predecessor = (
                distance is not None
                and (distance == 0 or bool(self.distance_prev_edges[edge_idx]))
            )
            active.append(is_active and has_reachable_predecessor)

        distance_prev_active_edges = [
            set(prev_edges) for prev_edges in self.distance_prev_edges
        ]
        queue = [edge_idx for edge_idx, is_active in enumerate(active) if not is_active]
        queue_index = 0
        while queue_index < len(queue):
            inactive_edge = queue[queue_index]
            queue_index += 1
            for next_edge in self.distance_next_edges[inactive_edge]:
                distance_prev_active_edges[next_edge].discard(inactive_edge)
                if active[next_edge] and not distance_prev_active_edges[next_edge]:
                    active[next_edge] = False
                    queue.append(next_edge)

        self.distance_prev_active_edges = [
            set(prev_edges) for prev_edges in distance_prev_active_edges
        ]
        active_mask = torch.tensor(
            active,
            dtype=torch.bool,
            device=pi_hat.device,
        )
        return active_mask, distance_prev_active_edges

    def _compute_dynamic_lr_active_mask_with_counts(
        self,
        pi_hat: torch.Tensor,
        logit_limit: float,
    ) -> torch.Tensor:
        if int(pi_hat.numel()) != len(self.distance_prev_edge_counts):
            raise ValueError(
                "pi_hat size does not match the number of hyperedges in the neighborhood."
            )

        finite_pi_hat = torch.isfinite(pi_hat)
        base_active_tensor = finite_pi_hat & (pi_hat.abs() < float(logit_limit))
        active = [
            bool(is_active)
            for is_active in base_active_tensor.detach().cpu().reshape(-1).tolist()
        ]
        prev_active_counts = list(self.distance_prev_edge_counts)

        for edge_idx, prev_active_count in enumerate(prev_active_counts):
            if prev_active_count <= 0:
                active[edge_idx] = False

        queue = [edge_idx for edge_idx, is_active in enumerate(active) if not is_active]
        queue_index = 0
        while queue_index < len(queue):
            inactive_edge = queue[queue_index]
            queue_index += 1
            for next_edge in self.distance_next_edges[inactive_edge]:
                if prev_active_counts[next_edge] > 0:
                    prev_active_counts[next_edge] -= 1
                if active[next_edge] and prev_active_counts[next_edge] == 0:
                    active[next_edge] = False
                    queue.append(next_edge)

        self.distance_prev_active_edge_counts = list(prev_active_counts)
        return torch.tensor(
            active,
            dtype=torch.bool,
            device=pi_hat.device,
        )

    def dynamic_lr_active_mask(
        self,
        logit_limit: float = DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT,
    ) -> torch.Tensor:
        active_mask, _ = self.compute_dynamic_lr_active_state(
            logit_limit=logit_limit
        )
        return active_mask

    def _format_related_edges(self, edge_idx: int, related_edges: set[int]) -> str:
        edge_parts = []
        for related_edge in sorted(related_edges):
            shared_nodes = sorted(
                self.edge_nodes[related_edge] & self.edge_nodes[edge_idx]
            )
            edge_parts.append(f"edge{related_edge}:nodes{shared_nodes}")
        return "[" + ", ".join(edge_parts) + "]"

    def format_target_perturbation_debug(
        self,
        lr_debug: dict[str, object] | None = None,
    ) -> list[str]:
        lr_pi_hat = None
        lr_active_mask = None
        lr_grad = None
        if lr_debug is not None:
            lr_pi_hat = lr_debug.get("pi_hat")
            lr_active_mask = lr_debug.get("active_mask")
            lr_grad = lr_debug.get("grad")

        param_indices = torch.arange(
            self.pi_i_hat.numel(),
            device=self.pi_i_hat.device,
            dtype=torch.long,
        )
        if param_indices.numel() == 0:
            return ["Hyperedge debug: no hyperedges in the extracted neighborhood."]

        with torch.no_grad():
            pi_hat_values = self.pi_i_hat.detach().cpu()
            soft_values = torch.sigmoid(self.pi_i_hat).detach().cpu()
            hard_values = (soft_values >= 0.5).to(torch.int64)
            grad_values = None
            lr_active_values = None
            lr_grad_values = None
            lr_pi_hat_values = None
            if self.pi_i_hat.grad is not None:
                grad_values = self.pi_i_hat.grad.detach().cpu()
            if isinstance(lr_pi_hat, torch.Tensor):
                lr_pi_hat_values = lr_pi_hat.detach().cpu()
            if isinstance(lr_active_mask, torch.Tensor):
                lr_active_values = lr_active_mask.to(dtype=torch.bool).detach().cpu()
            if isinstance(lr_grad, torch.Tensor):
                lr_grad_values = lr_grad.detach().cpu()
            active_state_pi_hat = (
                lr_pi_hat if isinstance(lr_pi_hat, torch.Tensor) else self.pi_i_hat
            )
            computed_active_mask, distance_prev_active_edges = (
                self.compute_dynamic_lr_active_state(active_state_pi_hat)
            )
            if lr_active_values is None:
                lr_active_values = computed_active_mask.detach().cpu()

        debug_lines = []
        debug_lines.append(
            "Hyperedge debug "
            "(edge_idx, distance, distance_prev_edges, "
            "distance_prev_active_edges, distance_next_edges, "
            "calib_pi_hat, calib_active, calib_grad, "
            "current_pi_hat, current_sigmoid, current_hard, current_grad):"
        )
        for edge_idx in param_indices.detach().cpu().tolist():
            grad_str = "None"
            lr_pi_hat_str = "None"
            lr_active_str = "None"
            lr_grad_str = "None"
            distance = self.edge_distances[edge_idx]
            distance_str = "None" if distance is None else str(distance)
            prev_edges_str = self._format_related_edges(
                edge_idx, self.distance_prev_edges[edge_idx]
            )
            prev_active_edges_str = self._format_related_edges(
                edge_idx, distance_prev_active_edges[edge_idx]
            )
            next_edges_str = self._format_related_edges(
                edge_idx, self.distance_next_edges[edge_idx]
            )
            if grad_values is not None:
                grad_str = f"{float(grad_values[edge_idx]):+.6e}"
            if lr_pi_hat_values is not None:
                lr_pi_hat_str = f"{float(lr_pi_hat_values[edge_idx]):+.6f}"
            if lr_active_values is not None:
                lr_active_str = "1" if bool(lr_active_values[edge_idx]) else "0"
            if lr_grad_values is not None:
                lr_grad_str = f"{float(lr_grad_values[edge_idx]):+.6e}"
            debug_lines.append(
                f"  edge {edge_idx}: distance={distance_str}, "
                f"distance_prev_edges={prev_edges_str}, "
                f"distance_prev_active_edges={prev_active_edges_str}, "
                f"distance_next_edges={next_edges_str}, "
                f"calib_pi_hat={lr_pi_hat_str}, "
                f"calib_active={lr_active_str}, calib_grad={lr_grad_str}, "
                f"current_pi_hat={float(pi_hat_values[edge_idx]):+.6f}, "
                f"current_s={float(soft_values[edge_idx]):.6f}, "
                f"current_hard={int(hard_values[edge_idx])}, current_grad={grad_str}"
            )
        return debug_lines

    def forward(self, x, sub_H):
        sub_H = sub_H.coalesce()

        indices = sub_H.indices()
        values = sub_H.values().clone()

        col_indices = indices[1]
        values = values * F.sigmoid(self.pi_i_hat[col_indices])

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

        col_indices = indices[1]
        values = values * self.pi_i[col_indices]

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

        H_indices = self.H.indices()
        H_values = self.H.values()
        col_indices = H_indices[1]

        weights = 1 - self.pi_i[col_indices]
        loss_graph_dist = torch.sum(H_values * weights)

        soft_mask = F.sigmoid(self.pi_i_hat)
        active_upper = torch.sigmoid(
            soft_mask.new_tensor(DEFAULT_DYNAMIC_LR_ACTIVE_LOGIT_LIMIT)
        )
        active_lower = 1 - active_upper
        saturated_mask = (soft_mask < active_lower) | (soft_mask > active_upper)
        active_mask, _ = self.compute_dynamic_lr_active_state()
        no_reachable_active_edges = not bool(active_mask.any().item())

        if getattr(self, "verbose", True):
            target_incident_mask = torch.zeros_like(soft_mask, dtype=torch.bool)
            _, target_cols = extract_sparse_row(self.H, self.target_node)
            if target_cols.numel() > 0:
                target_incident_mask[target_cols.to(soft_mask.device)] = True

            def print_soft_mask_group(label, group_mask):
                group_values = soft_mask[group_mask]
                if group_values.numel() == 0:
                    print(f"{label}: count=0")
                    return

                group_saturated = saturated_mask[group_mask]
                group_below = group_values < active_lower
                group_above = group_values > active_upper
                group_active = ~group_saturated
                group_reachable_active = active_mask[group_mask]
                group_hard_removed = group_values < 0.5
                print(
                    f"{label}: count={group_values.numel()}, "
                    f"mean={group_values.detach().cpu().numpy().mean():.6f}, "
                    f"max={group_values.detach().cpu().numpy().max():.6f}, "
                    f"min={group_values.detach().cpu().numpy().min():.6f}, "
                    f"below_active_lower={int(group_below.sum().item())}, "
                    f"above_active_upper={int(group_above.sum().item())}, "
                    f"active_between={int(group_active.sum().item())}, "
                    f"reachable_active={int(group_reachable_active.sum().item())}, "
                    f"hard_removed={int(group_hard_removed.sum().item())}"
                )

            all_mask = torch.ones_like(soft_mask, dtype=torch.bool)
            print(
                "v3 no_more_edits thresholds: "
                f"active_lower={active_lower.item():.6f}, "
                f"active_upper={active_upper.item():.6f}, "
                f"all_saturated={bool(torch.all(saturated_mask).item())}, "
                f"reachable_active={int(active_mask.sum().item())}, "
                f"no_reachable_active={no_reachable_active_edges}"
            )
            print_soft_mask_group("v3 soft_mask all hyperedges", all_mask)
            print_soft_mask_group(
                "v3 soft_mask target-incident hyperedges",
                target_incident_mask,
            )
            print_soft_mask_group(
                "v3 soft_mask non-target-incident hyperedges",
                ~target_incident_mask,
            )
            print(loss_pred.item(), loss_graph_dist.item())

        if torch.all(saturated_mask) or no_reachable_active_edges:
            self.no_more_edits = True

        cf_H = self.H_tilde

        #  do not gate the prediction loss by `pred_same`.
        loss = loss_pred + self.beta * loss_graph_dist
        return loss, loss_pred, loss_graph_dist, cf_H


# if __name__ == "__main__":
#     from utils import normalize_propagation

#     H = torch.tensor(
#         [[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
#         dtype=torch.float32,
#     )
#     H = H.to_sparse()

#     S = normalize_propagation(H)

#     model = HGCN_Perturb(3, 2, 2, 5, 0.5, H, 0, 0.1)

#     out = model(torch.randn(6, 3), H)
#     print(out)
