#!/usr/bin/env python3
"""Run the LR/epoch explanation sweep and plot fidelity vs explanation size.

This script intentionally leaves the existing project code untouched. It runs
the sparse explainer once per learning rate at the maximum configured epoch,
snapshots intermediate epoch results, summarizes the requested metrics, and
writes CSV, a dependency-free SVG plot, and an aggregate pickle dump.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import pickle
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


LEARNING_RATES = (1.0, 10.0, 100.0)
EPOCH_COUNTS = (200, 400, 600)
# EPOCH_COUNTS = (1, 2, 3)
BETA = 0.001
EXPLAINER_STRATEGIES = ("v1", "v3")
DENOMINATOR_SCOPES = ("non_isolated", "possible")

# Plot labels and label sizes are configured here.
PLOT_CONFIG = {
    "x_axis_label": "Epochs",
    "fidelity_y_axis_label": "Fidelity",
    "size_y_axis_label": "Explanation size",
    "axis_label_size": 16,
    "tick_label_size": 12,
    "legend_label_size": 13,
    "fidelity_y_limits": (0.0, 1.0),
    "size_y_limits": None,
}

PLOT_COLORS = {
    1.0: "#2563eb",
    10.0: "#16a34a",
    100.0: "#dc2626",
    #10.0: "#7c3aed",
}


@dataclass(frozen=True)
class RunMetrics:
    explanation_strategy: str
    denominator_scope: str
    learning_rate: float
    epochs: int
    fidelity: float | None
    explanation_size: float | None
    num_cf_found: int
    fidelity_denominator: int
    num_targets: int
    num_non_isolated: int
    num_cf_possible: int
    result_path: Path


@dataclass
class SweepContext:
    model: torch.nn.Module
    data: object
    H: torch.Tensor
    y_log_prob_all: torch.Tensor
    y_pred_all: torch.Tensor
    target_nodes: list[int]
    device: torch.device


@dataclass
class NodeCheckpointResult:
    best_examples_by_epoch: dict[int, list[list[object]]]
    possible_by_epoch: dict[int, bool]
    elapsed_by_epoch: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CF-HyperGNNExplainer over the fixed learning-rate/"
            "epoch grid, then plot fidelity and explanation size."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the pretrained HGCN checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name passed to the sparse explainer.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for run pickles, CSV, and plot "
            "(default: results/lrepoch_<dataset>)."
        ),
    )
    parser.add_argument(
        "--dump-path",
        default=None,
        help=(
            "Destination pickle file for the aggregate LR/epoch sweep output "
            "(default: <output-dir>/lrepoch_dump.pkl)."
        ),
    )
    parser.add_argument(
        "--strategy",
        dest="strategies",
        nargs="+",
        choices=EXPLAINER_STRATEGIES,
        default=list(EXPLAINER_STRATEGIES),
        help=(
            "Explanation strategy or strategies used by the existing explainer "
            "(default: run both v1 and v3)."
        ),
    )
    parser.add_argument(
        "--cf-optimizer",
        choices=("SGD", "Adadelta"),
        default="SGD",
        help="Optimizer used by the existing explainer.",
    )
    parser.add_argument(
        "--n-momentum",
        type=float,
        default=0.0,
        help="Momentum for SGD.",
    )
    parser.add_argument(
        "--n-hops",
        type=int,
        default=4,
        help="Neighborhood radius for the explainer.",
    )
    parser.add_argument(
        "--target-node",
        type=int,
        default=None,
        help="Optional single target node for a smaller sweep.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used by the existing explainer.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run explanations even when a run pickle already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print max-epoch explainer settings and checkpoint outputs without running them.",
    )
    parser.add_argument(
        "--verbose-explainer",
        action="store_true",
        help="Show per-epoch sparse explainer diagnostics.",
    )
    return parser.parse_args()


def sanitize_name(text: str) -> str:
    sanitized = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text]
    value = "".join(sanitized)
    return value if value else "item"


def float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_architecture_args(model_path: Path) -> dict[str, object]:
    checkpoint = torch_load(model_path)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    return {
        "dropout": checkpoint_args.get("dropout", 0.5),
        "nhid": checkpoint_args.get("hidden", checkpoint_args.get("nhid", 64)),
        "nout": checkpoint_args.get("out_hidden", checkpoint_args.get("nout", 32)),
    }


def run_output_path(
    output_dir: Path,
    dataset: str,
    lr: float,
    epochs: int,
    explain_strategy: str,
) -> Path:
    dataset_token = sanitize_name(dataset.casefold())
    strategy_token = sanitize_name(explain_strategy)
    return output_dir / (
        f"cf_examples_{dataset_token}_beta{float_token(BETA)}_"
        f"lr{float_token(lr)}_epochs{epochs}_{strategy_token}.pkl"
    )


def build_explainer_command(
    *,
    repo_root: Path,
    model_path: Path,
    dataset: str,
    lr: float,
    epochs: int,
    output_path: Path,
    architecture_args: dict[str, object],
    args: argparse.Namespace,
    explain_strategy: str,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "src_sparse" / "main_explain.py"),
        "--dataset",
        dataset,
        "--n-hops",
        str(args.n_hops),
        "--beta",
        f"{BETA:g}",
        "--cf-optimizer",
        args.cf_optimizer,
        "--strategy",
        explain_strategy,
        "--lr",
        f"{lr:g}",
        "--n-momentum",
        str(args.n_momentum),
        "--num-epochs",
        str(epochs),
        "--dropout",
        str(architecture_args["dropout"]),
        "--nhid",
        str(architecture_args["nhid"]),
        "--nout",
        str(architecture_args["nout"]),
        "--ckpt-path",
        str(model_path),
        "--device",
        args.device,
        "--output-path",
        str(output_path),
    ]
    if args.target_node is not None:
        command.extend(["--target-node", str(args.target_node)])
    if not args.verbose_explainer:
        command.append("--quiet")
    return command


def epoch_checkpoints() -> tuple[int, ...]:
    checkpoints = tuple(sorted(set(int(value) for value in EPOCH_COUNTS)))
    if not checkpoints:
        raise ValueError("EPOCH_COUNTS must contain at least one value.")
    if checkpoints[0] <= 0:
        raise ValueError("EPOCH_COUNTS values must be positive.")
    return checkpoints


def ensure_src_sparse_on_path(repo_root: Path) -> None:
    src_sparse = str(repo_root / "src_sparse")
    if src_sparse not in sys.path:
        sys.path.insert(0, src_sparse)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    requested_device = torch.device(device_arg)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return requested_device


def resolve_planetoid_root(repo_root: Path) -> str:
    candidates = [
        repo_root / "src_sparse" / "data" / "Planetoid",
        repo_root / "data" / "Planetoid",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[0])


def load_sweep_context(
    *,
    repo_root: Path,
    model_path: Path,
    architecture_args: dict[str, object],
    args: argparse.Namespace,
) -> SweepContext:
    ensure_src_sparse_on_path(repo_root)
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    from hgcn import HGCN
    from utils import graph_to_hypergraph, normalize_propagation

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.dataset in ("Cora", "Citeseer", "Pubmed"):
        dataset = Planetoid(
            root=resolve_planetoid_root(repo_root),
            name=args.dataset,
            transform=NormalizeFeatures(),
        )
        data = dataset[0].to(device)
        H = graph_to_hypergraph(data.edge_index, data.num_nodes, device=device)
        nfeat = dataset.num_features
        nclass = dataset.num_classes
    else:
        from utils.allset_loader import load_allset_dataset

        data, H = load_allset_dataset(args.dataset, device=device)
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.train_mask = data.train_mask.to(device)
        data.val_mask = data.val_mask.to(device)
        data.test_mask = data.test_mask.to(device)
        nfeat = int(data.x.size(1))
        nclass = int(int(data.y.max().item()) + 1)

    model = HGCN(
        nfeat=nfeat,
        nhid=int(architecture_args["nhid"]),
        nout=int(architecture_args["nout"]),
        nclass=nclass,
        dropout=float(architecture_args["dropout"]),
    ).to(device)

    checkpoint = torch_load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S = normalize_propagation(H)
    with torch.no_grad():
        out = model(data.x, S)
        y_log_prob_all = out
        y_pred_all = torch.argmax(out, dim=1)

    if args.target_node is None:
        target_nodes = [int(idx.item()) for idx in torch.where(data.test_mask)[0]]
        if not target_nodes:
            raise ValueError(f"Dataset {args.dataset} has no test nodes.")
        print(f"Explaining {len(target_nodes)} node(s) from the test set.")
    else:
        if not 0 <= args.target_node < data.num_nodes:
            raise ValueError(
                f"target node {args.target_node} is outside the range of nodes in {args.dataset}"
            )
        target_nodes = [args.target_node]

    return SweepContext(
        model=model,
        data=data,
        H=H,
        y_log_prob_all=y_log_prob_all,
        y_pred_all=y_pred_all,
        target_nodes=target_nodes,
        device=device,
    )


def explain_with_epoch_checkpoints(
    *,
    explainer,
    cf_optimizer: str,
    node_idx: int,
    new_idx: int,
    lr: float,
    n_momentum: float,
    max_epochs: int,
    checkpoints: tuple[int, ...],
    elapsed_start: float,
    patience: int = 5,
) -> NodeCheckpointResult:
    from tqdm import tqdm

    checkpoint_set = set(checkpoints)
    best_examples_by_epoch: dict[int, list[list[object]]] = {}
    possible_by_epoch: dict[int, bool] = {}
    elapsed_by_epoch: dict[int, float] = {}
    best_cf_examples: list[list[object]] = []
    best_loss = math.inf
    num_cf_examples = 0
    stop_counter = 0
    last_pred = -1

    explainer.node_idx = int(node_idx)
    explainer.new_idx = int(new_idx)
    explainer.cf_model.reset_perturbation()
    explainer._current_lr_debug_by_epoch = []
    explainer._current_epochwise_lrs = []
    explainer._current_lr_checkpoint_epochs = []
    explainer._current_lr_checkpoint_values = []
    explainer._use_epochwise_dynamic_lr = False
    explainer.cf_optimizer = explainer._build_cf_optimizer(
        cf_optimizer=cf_optimizer,
        lr=float(lr),
        n_momentum=n_momentum,
    )

    def store_checkpoint(epoch_number: int) -> None:
        best_examples_by_epoch[epoch_number] = list(best_cf_examples)
        possible_by_epoch[epoch_number] = bool(best_cf_examples) or not getattr(
            explainer.cf_model,
            "no_more_edits",
            False,
        )
        elapsed_by_epoch[epoch_number] = time.time() - elapsed_start

    for epoch in tqdm(
        range(max_epochs),
        desc="Training epochs",
        disable=explainer.quiet,
    ):
        new_example, loss_total, grad_is_zero, current_pred = explainer.train(
            epoch,
            num_epochs=max_epochs,
        )

        if new_example and loss_total < best_loss:
            best_cf_examples.append(new_example)
            best_loss = loss_total
            num_cf_examples += 1

        epoch_number = epoch + 1
        if epoch_number in checkpoint_set:
            store_checkpoint(epoch_number)

        if getattr(explainer.cf_model, "no_available_edits", False):
            explainer._log(
                "Stopping search: there are no available edits for target node. "
                "Node is isolated in the hypergraph."
            )
            break
        if getattr(explainer.cf_model, "no_more_edits", False):
            explainer._log(
                "Stopping search: no more editable interactions for target node."
            )
            break

        if grad_is_zero and current_pred == last_pred:
            stop_counter += 1
        else:
            stop_counter = 0

        if stop_counter >= patience:
            explainer._log(f"\nEarly stopping triggered at epoch {epoch + 1}")
            explainer._log(
                f"Reason: Gradient zero and prediction stable for {patience} epochs."
            )
            break

        last_pred = current_pred

    for checkpoint in checkpoints:
        if checkpoint not in best_examples_by_epoch:
            store_checkpoint(checkpoint)

    explainer._log(f"{num_cf_examples} CF examples for node_idx = {explainer.node_idx}")
    explainer._log(" ")
    return NodeCheckpointResult(
        best_examples_by_epoch=best_examples_by_epoch,
        possible_by_epoch=possible_by_epoch,
        elapsed_by_epoch=elapsed_by_epoch,
    )


def run_lr_checkpoint_sweep(
    *,
    context: SweepContext,
    repo_root: Path,
    args: argparse.Namespace,
    lr: float,
    output_paths: dict[int, Path],
    checkpoints: tuple[int, ...],
    explain_strategy: str,
) -> None:
    ensure_src_sparse_on_path(repo_root)
    from cf_explanation.cf_explainer import CFExplainer
    from utils import get_hyper_neighbourhood_fast

    max_epochs = max(checkpoints)
    cf_examples_by_epoch: dict[int, list[object]] = {epoch: [] for epoch in checkpoints}
    num_successful_by_epoch = {epoch: 0 for epoch in checkpoints}
    possible_trials_by_epoch = {epoch: 0 for epoch in checkpoints}
    non_isolated_times_by_epoch: dict[int, list[float]] = {
        epoch: [] for epoch in checkpoints
    }
    possible_times_by_epoch: dict[int, list[float]] = {epoch: [] for epoch in checkpoints}
    isolated_nodes = 0
    total_start = time.time()

    for target_node in context.target_nodes:
        print(f"\n=== Running CF explainer for target node {target_node} ===")

        y_pred_orig = context.y_pred_all[target_node]
        log_prob_orig = context.y_log_prob_all[target_node]
        sub_H, sub_feat, sub_labels, node_dict = get_hyper_neighbourhood_fast(
            node_idx=target_node,
            H=context.H,
            n_hops=args.n_hops,
            features=context.data.x,
            labels=context.data.y,
        )
        sub_feat = sub_feat.to(context.device)
        sub_labels = sub_labels.to(context.device)
        target_node_sub_idx = node_dict[target_node]

        explainer = CFExplainer(
            model=context.model,
            sub_H=sub_H,
            sub_feat=sub_feat,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig,
            log_prob_orig=log_prob_orig,
            beta=BETA,
            target_node_sub_idx=target_node_sub_idx,
            device=context.device,
            strategy=explain_strategy,
            quiet=not args.verbose_explainer,
        )

        sub_H_coalesced = sub_H.coalesce()
        H_indices = sub_H_coalesced.indices()
        row_mask = H_indices[0] == target_node_sub_idx
        available_edges = H_indices[1][row_mask]
        if available_edges.numel() == 0:
            print(
                f"Target node {target_node} has no incident edges in the extracted subgraph. No edits are available."
            )
            isolated_nodes += 1
            for epoch in checkpoints:
                cf_examples_by_epoch[epoch].append(None)
            continue

        print(f"Fixed learning rate for target node {target_node}: {lr:.6g}")
        print(
            f"Running counterfactual search for target node {target_node} "
            f"with beta={BETA:.6g}, lr={lr:.6g}, and checkpoints "
            f"{', '.join(str(epoch) for epoch in checkpoints)}."
        )

        node_start = time.time()
        checkpoint_result = explain_with_epoch_checkpoints(
            explainer=explainer,
            cf_optimizer=args.cf_optimizer,
            node_idx=target_node,
            new_idx=target_node_sub_idx,
            lr=lr,
            n_momentum=args.n_momentum,
            max_epochs=max_epochs,
            checkpoints=checkpoints,
            elapsed_start=node_start,
        )
        node_elapsed = time.time() - node_start
        print(f"Node {target_node} run time: {node_elapsed:.2f}s")

        for epoch in checkpoints:
            elapsed = checkpoint_result.elapsed_by_epoch[epoch]
            possible = checkpoint_result.possible_by_epoch[epoch]
            best_cf_examples = checkpoint_result.best_examples_by_epoch[epoch]
            non_isolated_times_by_epoch[epoch].append(elapsed)
            if possible:
                possible_trials_by_epoch[epoch] += 1
                possible_times_by_epoch[epoch].append(elapsed)

            if not best_cf_examples:
                cf_examples_by_epoch[epoch].append(
                    {
                        "no_cf_found": True,
                        "possible": possible,
                        "node_idx": target_node,
                        "log_prob_orig": log_prob_orig.detach().cpu(),
                        "y_pred_orig": int(y_pred_orig.item()),
                    }
                )
                continue

            best_stats = best_cf_examples[-1]
            cf_examples_by_epoch[epoch].append([best_stats])
            num_successful_by_epoch[epoch] += 1

    total_elapsed = time.time() - total_start
    num_targets = len(context.target_nodes)
    num_non_isolated = num_targets - isolated_nodes
    print(f"\nTotal lr={lr:g} checkpoint run time: {total_elapsed:.2f}s")
    print(f"Isolated nodes: {isolated_nodes}/{num_targets}")
    print(f"Non-isolated nodes: {num_non_isolated}/{num_targets}")

    for epoch in checkpoints:
        non_isolated_times = non_isolated_times_by_epoch[epoch]
        possible_times = possible_times_by_epoch[epoch]
        avg_time_non_isolated = (
            sum(non_isolated_times) / len(non_isolated_times)
            if non_isolated_times
            else None
        )
        avg_time_possible = (
            sum(possible_times) / len(possible_times) if possible_times else None
        )
        print(
            f"Epoch {epoch}: counterfactual examples found "
            f"{num_successful_by_epoch[epoch]}/{num_non_isolated} "
            "(successful/non-isolated)."
        )

        output_path = output_paths[epoch]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(
                {
                    "dataset": args.dataset,
                    "strategy": explain_strategy,
                    "cf_examples_per_node": cf_examples_by_epoch[epoch],
                    "num_targets": num_targets,
                    "num_isolated": isolated_nodes,
                    "num_non_isolated": num_non_isolated,
                    "num_cf_possible": possible_trials_by_epoch[epoch],
                    "num_cf_found": num_successful_by_epoch[epoch],
                    "avg_time_non_isolated": avg_time_non_isolated,
                    "avg_time_possible": avg_time_possible,
                    "checkpoint_epoch": epoch,
                    "num_epochs": epoch,
                    "max_epochs": max_epochs,
                },
                handle,
            )
        print(f"Saved epoch {epoch} CF examples to {output_path}")


def sparse_value_map(
    tensor: torch.Tensor,
    eps: float = 1e-5,
) -> dict[tuple[int, int], float]:
    if not tensor.is_sparse:
        tensor = tensor.to_sparse()
    tensor = tensor.coalesce().cpu()
    indices = tensor.indices()
    values = tensor.values()
    entries: dict[tuple[int, int], float] = {}
    for idx in range(values.numel()):
        value = float(values[idx].item())
        if abs(value) > eps:
            entries[(int(indices[0, idx].item()), int(indices[1, idx].item()))] = value
    return entries


def incidence_diff_count(
    sub_h: torch.Tensor,
    cf_h: torch.Tensor,
    eps: float = 1e-5,
) -> int:
    sub_entries = sparse_value_map(sub_h, eps=eps)
    cf_entries = sparse_value_map(cf_h, eps=eps)
    keys = set(sub_entries) | set(cf_entries)
    return sum(
        1
        for key in keys
        if abs(sub_entries.get(key, 0.0) - cf_entries.get(key, 0.0)) > eps
    )


def removed_hyperedge_count(
    sub_h: torch.Tensor,
    cf_h: torch.Tensor,
    eps: float = 1e-5,
) -> int:
    if not sub_h.is_sparse:
        sub_h = sub_h.to_sparse()
    if not cf_h.is_sparse:
        cf_h = cf_h.to_sparse()
    sub_h = sub_h.coalesce().cpu()
    cf_h = cf_h.coalesce().cpu()

    sub_indices = sub_h.indices()
    sub_values = sub_h.values()
    sub_cols = {
        int(sub_indices[1, idx].item())
        for idx in range(sub_values.numel())
        if abs(float(sub_values[idx].item())) > eps
    }
    cf_indices = cf_h.indices()
    cf_values = cf_h.values()
    cf_cols = {
        int(cf_indices[1, idx].item())
        for idx in range(cf_values.numel())
        if abs(float(cf_values[idx].item())) > eps
    }
    return len(sub_cols - cf_cols)


def successful_rows(cf_examples_per_node: Iterable[object]) -> list[list[object]]:
    rows: list[list[object]] = []
    for example in cf_examples_per_node:
        if isinstance(example, dict) or example is None:
            continue
        if isinstance(example, (list, tuple)) and example:
            row = example[0]
            if isinstance(row, (list, tuple)) and len(row) >= 4:
                rows.append(list(row))
    return rows


def summarize_result(
    result_path: Path,
    *,
    lr: float,
    epochs: int,
    explain_strategy: str,
) -> list[RunMetrics]:
    with result_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        cf_examples_per_node = payload["cf_examples_per_node"]
        num_targets = int(payload.get("num_targets", len(cf_examples_per_node)))
        num_non_isolated = int(
            payload.get(
                "num_non_isolated",
                len(cf_examples_per_node)
                - sum(1 for item in cf_examples_per_node if item is None),
            )
        )
        num_cf_possible = payload.get("num_cf_possible")
    else:
        cf_examples_per_node = payload
        num_targets = len(cf_examples_per_node)
        num_non_isolated = len(cf_examples_per_node) - sum(
            1 for item in cf_examples_per_node if item is None
        )
        num_cf_possible = None

    rows = successful_rows(cf_examples_per_node)
    num_cf_found = len(rows)
    failure_entries = [
        item for item in cf_examples_per_node if isinstance(item, dict)
    ]
    possible_failures = [
        item for item in failure_entries if bool(item.get("possible"))
    ]
    if num_cf_possible is None:
        num_cf_possible = num_cf_found + len(possible_failures)
    num_cf_possible = int(num_cf_possible)

    sizes: list[float] = []
    for row in rows:
        cf_h = row[2]
        sub_h = row[3]
        if explain_strategy == "v3":
            sizes.append(float(removed_hyperedge_count(sub_h, cf_h)))
        else:
            sizes.append(float(incidence_diff_count(sub_h, cf_h)))

    denominator_by_scope = {
        "non_isolated": num_non_isolated,
        "possible": num_cf_possible,
    }
    metrics: list[RunMetrics] = []
    for denominator_scope in DENOMINATOR_SCOPES:
        fidelity_denominator = int(denominator_by_scope[denominator_scope])
        fidelity = (
            num_cf_found / fidelity_denominator
            if fidelity_denominator > 0
            else None
        )
        # Failed targets in the selected denominator contribute zero edits,
        # matching the convention used by the existing evaluator.
        explanation_size = (
            sum(sizes) / fidelity_denominator
            if fidelity_denominator > 0
            else None
        )
        metrics.append(
            RunMetrics(
                explanation_strategy=explain_strategy,
                denominator_scope=denominator_scope,
                learning_rate=lr,
                epochs=epochs,
                fidelity=fidelity,
                explanation_size=explanation_size,
                num_cf_found=num_cf_found,
                fidelity_denominator=fidelity_denominator,
                num_targets=num_targets,
                num_non_isolated=num_non_isolated,
                num_cf_possible=num_cf_possible,
                result_path=result_path,
            )
        )
    return metrics

def metric_sort_key(row: RunMetrics) -> tuple[int, int, float, int]:
    strategy_order = {value: index for index, value in enumerate(EXPLAINER_STRATEGIES)}
    scope_order = {value: index for index, value in enumerate(DENOMINATOR_SCOPES)}
    return (
        strategy_order.get(row.explanation_strategy, len(strategy_order)),
        scope_order.get(row.denominator_scope, len(scope_order)),
        row.learning_rate,
        row.epochs,
    )


def write_metrics_csv(metrics: list[RunMetrics], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "denominator_scope",
                "learning_rate",
                "epochs",
                "fidelity",
                "explanation_size",
                "num_cf_found",
                "fidelity_denominator",
                "num_targets",
                "num_non_isolated",
                "num_cf_possible",
                "result_path",
            ]
        )
        for row in sorted(metrics, key=metric_sort_key):
            writer.writerow(
                [
                    row.explanation_strategy,
                    row.denominator_scope,
                    f"{row.learning_rate:g}",
                    row.epochs,
                    format_number(row.fidelity),
                    format_number(row.explanation_size),
                    row.num_cf_found,
                    row.fidelity_denominator,
                    row.num_targets,
                    row.num_non_isolated,
                    row.num_cf_possible,
                    str(row.result_path),
                ]
            )


def metrics_record(row: RunMetrics) -> dict[str, object]:
    return {
        "strategy": row.explanation_strategy,
        "denominator_scope": row.denominator_scope,
        "lr": row.learning_rate,
        "epoch": row.epochs,
        "metrics": {
            "fidelity": row.fidelity,
            "explanation_size": row.explanation_size,
            "num_cf_found": row.num_cf_found,
            "fidelity_denominator": row.fidelity_denominator,
            "num_targets": row.num_targets,
            "num_non_isolated": row.num_non_isolated,
            "num_cf_possible": row.num_cf_possible,
        },
        "result_path": str(row.result_path),
    }


def write_metrics_dump(
    *,
    metrics: list[RunMetrics],
    dump_path: Path,
    args: argparse.Namespace,
    model_path: Path,
    output_dir: Path,
    csv_path: Path,
    plot_paths: dict[str, Path],
) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "model": str(model_path),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "plot_paths": {scope: str(path) for scope, path in plot_paths.items()},
        "learning_rates": list(LEARNING_RATES),
        "epoch_counts": list(EPOCH_COUNTS),
        "beta": BETA,
        "strategies": list(args.strategies),
        "cf_optimizer": args.cf_optimizer,
        "n_momentum": args.n_momentum,
        "n_hops": args.n_hops,
        "target_node": args.target_node,
        "device": args.device,
        "runs": [
            metrics_record(row)
            for row in sorted(
                metrics, key=metric_sort_key
            )
        ],
    }
    with dump_path.open("wb") as handle:
        pickle.dump(payload, handle)


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attr_text = " ".join(
        f'{name.replace("_", "-")}="{html.escape(str(value), quote=True)}"'
        for name, value in attrs.items()
        if value is not None
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_text}>{html.escape(text)}</text>'


def nice_ticks(y_min: float, y_max: float, count: int = 5) -> list[float]:
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        return [0.0, 1.0]
    if y_min == y_max:
        return [y_min]
    return [y_min + (y_max - y_min) * idx / (count - 1) for idx in range(count)]


def metric_limits(
    values: list[float],
    configured: tuple[float, float] | None,
) -> tuple[float, float]:
    if configured is not None:
        return configured
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return (0.0, 1.0)
    y_min = min(finite)
    y_max = max(finite)
    if y_min == y_max:
        padding = max(1.0, abs(y_min) * 0.1)
    else:
        padding = (y_max - y_min) * 0.08
    return (max(0.0, y_min - padding), y_max + padding)


def render_panel(
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    metrics: list[RunMetrics],
    value_getter,
    y_label: str,
    y_limits: tuple[float, float],
) -> list[str]:
    axis_label_size = int(PLOT_CONFIG["axis_label_size"])
    tick_label_size = int(PLOT_CONFIG["tick_label_size"])
    x_min = min(EPOCH_COUNTS)
    x_max = max(EPOCH_COUNTS)
    y_min, y_max = y_limits

    def sx(epoch: int) -> float:
        if x_max == x_min:
            return x0 + width / 2
        return x0 + (epoch - x_min) / (x_max - x_min) * width

    def sy(value: float) -> float:
        if y_max == y_min:
            return y0 + height / 2
        return y0 + height - (value - y_min) / (y_max - y_min) * height

    parts = [
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + height:.2f}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{x0:.2f}" y1="{y0 + height:.2f}" x2="{x0 + width:.2f}" y2="{y0 + height:.2f}" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in EPOCH_COUNTS:
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{y0 + height:.2f}" x2="{x:.2f}" y2="{y0 + height + 6:.2f}" stroke="#111827" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                x,
                y0 + height + 24,
                str(tick),
                text_anchor="middle",
                font_size=tick_label_size,
                fill="#374151",
            )
        )

    for tick in nice_ticks(y_min, y_max):
        y = sy(tick)
        parts.append(
            f'<line x1="{x0 - 6:.2f}" y1="{y:.2f}" x2="{x0:.2f}" y2="{y:.2f}" stroke="#111827" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x0 + width:.2f}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                x0 - 12,
                y + 4,
                format_number(tick),
                text_anchor="end",
                font_size=tick_label_size,
                fill="#374151",
            )
        )

    parts.append(
        svg_text(
            x0 + width / 2,
            y0 + height + 54,
            str(PLOT_CONFIG["x_axis_label"]),
            text_anchor="middle",
            font_size=axis_label_size,
            font_weight="600",
            fill="#111827",
        )
    )
    parts.append(
        svg_text(
            x0 - 62,
            y0 + height / 2,
            y_label,
            text_anchor="middle",
            font_size=axis_label_size,
            font_weight="600",
            fill="#111827",
            transform=f"rotate(-90 {x0 - 62:.2f} {y0 + height / 2:.2f})",
        )
    )

    metrics_by_series_epoch = {
        (row.explanation_strategy, row.learning_rate, row.epochs): row
        for row in metrics
    }
    for explain_strategy in EXPLAINER_STRATEGIES:
        dash_attr = ' stroke-dasharray="7 5"' if explain_strategy == "v3" else ""
        for lr in LEARNING_RATES:
            color = PLOT_COLORS[lr]
            points: list[tuple[float, float]] = []
            for epoch in EPOCH_COUNTS:
                row = metrics_by_series_epoch.get((explain_strategy, lr, epoch))
                value = None if row is None else value_getter(row)
                if value is None or not math.isfinite(float(value)):
                    continue
                points.append((sx(epoch), sy(float(value))))

            if len(points) >= 2:
                point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                parts.append(
                    f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
                )
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" stroke="white" stroke-width="1.5"/>'
                )

    return parts


def render_svg_plot(
    metrics: list[RunMetrics],
    plot_path: Path,
    *,
    denominator_scope: str,
) -> None:
    plot_metrics = [row for row in metrics if row.denominator_scope == denominator_scope]
    width = 1200
    height = 580
    left = 95
    right = 45
    top = 104
    bottom = 78
    gap = 95
    panel_width = (width - left - right - gap) / 2
    panel_height = height - top - bottom
    x_left = left
    x_right = left + panel_width + gap

    fidelity_values = [
        row.fidelity for row in plot_metrics if row.fidelity is not None
    ]
    size_values = [
        row.explanation_size for row in plot_metrics if row.explanation_size is not None
    ]
    fidelity_limits = metric_limits(
        [float(value) for value in fidelity_values],
        PLOT_CONFIG["fidelity_y_limits"],  # type: ignore[arg-type]
    )
    size_limits = metric_limits(
        [float(value) for value in size_values],
        PLOT_CONFIG["size_y_limits"],  # type: ignore[arg-type]
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    parts.append(
        svg_text(
            width / 2,
            34,
            f"Denominator: {denominator_scope}",
            text_anchor="middle",
            font_size=16,
            font_weight="600",
            fill="#111827",
        )
    )

    legend_y = 68
    legend_x = left
    legend_label_size = int(PLOT_CONFIG["legend_label_size"])
    legend_index = 0
    for explain_strategy in EXPLAINER_STRATEGIES:
        dash_attr = ' stroke-dasharray="7 5"' if explain_strategy == "v3" else ""
        for lr in LEARNING_RATES:
            x = legend_x + legend_index * 170
            color = PLOT_COLORS[lr]
            parts.append(
                f'<line x1="{x:.2f}" y1="{legend_y:.2f}" x2="{x + 26:.2f}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="2.8" stroke-linecap="round"{dash_attr}/>'
            )
            parts.append(
                f'<circle cx="{x + 13:.2f}" cy="{legend_y:.2f}" r="4" fill="{color}" stroke="white" stroke-width="1.2"/>'
            )
            parts.append(
                svg_text(
                    x + 36,
                    legend_y + 4,
                    f"{explain_strategy} lr={lr:g}",
                    font_size=legend_label_size,
                    fill="#111827",
                )
            )
            legend_index += 1

    parts.extend(
        render_panel(
            x0=x_left,
            y0=top,
            width=panel_width,
            height=panel_height,
            metrics=plot_metrics,
            value_getter=lambda row: row.fidelity,
            y_label=str(PLOT_CONFIG["fidelity_y_axis_label"]),
            y_limits=fidelity_limits,
        )
    )
    parts.extend(
        render_panel(
            x0=x_right,
            y0=top,
            width=panel_width,
            height=panel_height,
            metrics=plot_metrics,
            value_getter=lambda row: row.explanation_size,
            y_label=str(PLOT_CONFIG["size_y_axis_label"]),
            y_limits=size_limits,
        )
    )

    parts.append("</svg>")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    model_path = resolve_path(args.model, repo_root)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_dir = (
        resolve_path(args.output_dir, repo_root)
        if args.output_dir
        else repo_root / "results" / f"lrepoch_{sanitize_name(args.dataset.casefold())}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture_args = checkpoint_architecture_args(model_path)
    checkpoints = epoch_checkpoints()
    max_epochs = checkpoints[-1]
    metrics: list[RunMetrics] = []
    sweep_context: SweepContext | None = None

    for explain_strategy in args.strategies:
        for lr in LEARNING_RATES:
            output_paths = {
                epochs: run_output_path(
                    output_dir,
                    args.dataset,
                    lr,
                    epochs,
                    explain_strategy,
                )
                for epochs in checkpoints
            }
            command = build_explainer_command(
                repo_root=repo_root,
                model_path=model_path,
                dataset=args.dataset,
                lr=lr,
                epochs=max_epochs,
                output_path=output_paths[max_epochs],
                architecture_args=architecture_args,
                args=args,
                explain_strategy=explain_strategy,
            )
            checkpoint_label = ", ".join(str(epoch) for epoch in checkpoints)
            print(
                f"\n== strategy={explain_strategy}, lr={lr:g}, "
                f"max_epochs={max_epochs}, checkpoints=({checkpoint_label}), "
                f"beta={BETA:g} =="
            )
            print(
                "Equivalent max-epoch command:",
                " ".join(shlex.quote(part) for part in command),
            )
            print("Checkpoint outputs:")
            for epochs in checkpoints:
                print(f"  epoch {epochs}: {output_paths[epochs]}")

            if args.dry_run:
                continue

            missing_outputs = [
                output_path
                for output_path in output_paths.values()
                if not output_path.is_file()
            ]
            needs_run = args.rerun or bool(missing_outputs)
            if needs_run:
                if args.rerun:
                    print(
                        "Rerunning this strategy/learning rate and overwriting "
                        "checkpoint outputs."
                    )
                elif len(missing_outputs) != len(output_paths):
                    print(
                        "Regenerating this strategy/learning-rate checkpoint set "
                        "so all epochs come from the same optimization run."
                    )
                else:
                    print(
                        "No cached checkpoint outputs found for this "
                        "strategy/learning rate."
                    )

                if sweep_context is None:
                    sweep_context = load_sweep_context(
                        repo_root=repo_root,
                        model_path=model_path,
                        architecture_args=architecture_args,
                        args=args,
                    )
                run_lr_checkpoint_sweep(
                    context=sweep_context,
                    repo_root=repo_root,
                    args=args,
                    lr=float(lr),
                    output_paths=output_paths,
                    checkpoints=checkpoints,
                    explain_strategy=explain_strategy,
                )
            else:
                print("Using existing checkpoint outputs for this strategy/learning rate.")

            for epochs in checkpoints:
                output_path = output_paths[epochs]
                if not output_path.is_file():
                    raise FileNotFoundError(
                        f"Expected checkpoint output was not created: {output_path}"
                    )
                metrics.extend(
                    summarize_result(
                        output_path,
                        lr=lr,
                        epochs=epochs,
                        explain_strategy=explain_strategy,
                    )
                )

    if args.dry_run:
        print("\nDry run complete; no explanations, CSV, or plot were written.")
        return

    csv_path = output_dir / "lrepoch_metrics.csv"
    plot_paths = {
        "non_isolated": output_dir / "lrepoch_plot.svg",
        "possible": output_dir / "lrepoch_plot_possible.svg",
    }
    dump_path = (
        resolve_path(args.dump_path, repo_root)
        if args.dump_path
        else output_dir / "lrepoch_dump.pkl"
    )
    write_metrics_csv(metrics, csv_path)
    for denominator_scope, plot_path in plot_paths.items():
        render_svg_plot(metrics, plot_path, denominator_scope=denominator_scope)
    write_metrics_dump(
        metrics=metrics,
        dump_path=dump_path,
        args=args,
        model_path=model_path,
        output_dir=output_dir,
        csv_path=csv_path,
        plot_paths=plot_paths,
    )

    print(f"\nWrote metrics CSV: {csv_path}")
    for denominator_scope, plot_path in plot_paths.items():
        print(f"Wrote {denominator_scope} plot: {plot_path}")
    print(f"Wrote aggregate dump: {dump_path}")

if __name__ == "__main__":
    main()
