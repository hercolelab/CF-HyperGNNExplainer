import argparse
import os
import pickle
import time


import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


from cf_explanation.cf_explainer import CFExplainer
from hgcn import HGCN
from utils import (
    graph_to_hypergraph,
    normalize_propagation,
    get_hyper_neighbourhood_fast,
)


INCREMENTAL_BETA_MODE = "incremental"
DYNAMIC_LR_MODE = "dynamic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CF-HyperGNNExplainer on a pretrained HGCN model"
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        help="Name of the Planetoid dataset to load (e.g. Cora, Citeseer, Pubmed)",
    )
    parser.add_argument(
        "--target-node",
        type=int,
        default=None,
        help="Optional node to explain (default: run on all Planetoid test nodes)",
    )
    parser.add_argument(
        "--n-hops", type=int, default=4, help="Neighborhood radius for the explainer"
    )
    parser.add_argument(
        "--beta",
        type=str,
        default="0.5",
        help=(
            "Graph-distance loss weight as a float string, or 'incremental' "
            "to search the largest valid beta per node"
        ),
    )
    parser.add_argument(
        "--cf-optimizer",
        choices=("SGD", "Adadelta"),
        default="SGD",
        help="Optimizer for the counterfactual explainer",
    )
    parser.add_argument(
        "--strategy",
        choices=("v1", "v3"),
        default="v1",
        help="Explanation strategy to use (v1 or v3)",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="0.1",
        help=(
            "Explainer learning rate as a float string, or 'dynamic' to "
            "derive it from the initial gradient for each node"
        ),
    )
    parser.add_argument(
        "--n-momentum",
        type=float,
        default=0.0,
        help="Momentum for SGD (0 disables momentum)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=500, help="Number of explainer epochs"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability",
    )
    parser.add_argument(
        "--nhid",
        type=int,
        default=64,
        help="Hidden feature size for the base HGCN",
    )
    parser.add_argument(
        "--nout",
        type=int,
        default=32,
        help="Output hidden size used for the projection layer of the base HGCN",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Path to the pretrained HGCN checkpoint (default: ckpt.pt)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for inference and explanation. 'auto' selects CUDA if available",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Destination pickle file for CF examples (default: results/cf_examples_<dataset>_<timestamp>.pkl)",
    )
    return parser.parse_args()


def parse_scalar_or_mode(
    raw_value: str,
    *,
    arg_name: str,
    special_mode: str,
) -> float | str:
    value = raw_value.strip()
    if value.lower() == special_mode:
        return special_mode
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{arg_name} must be a float string or '{special_mode}', got {raw_value!r}."
        ) from exc


def parse_beta_setting(beta_arg: str) -> float | str:
    beta_setting = parse_scalar_or_mode(
        beta_arg,
        arg_name="--beta",
        special_mode=INCREMENTAL_BETA_MODE,
    )
    if isinstance(beta_setting, float) and beta_setting < 0.0:
        raise ValueError("--beta must be non-negative.")
    return beta_setting


def parse_lr_setting(lr_arg: str) -> float | str:
    lr_setting = parse_scalar_or_mode(
        lr_arg,
        arg_name="--lr",
        special_mode=DYNAMIC_LR_MODE,
    )
    if isinstance(lr_setting, float) and lr_setting < 0.0:
        raise ValueError("--lr must be non-negative.")
    return lr_setting


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    requested_device = torch.device(device_arg)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return requested_device


def resolve_planetoid_root() -> str:
    script_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(script_dir, "data", "Planetoid"),
        os.path.join(script_dir, "..", "data", "Planetoid"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    try:
        beta_setting = parse_beta_setting(args.beta)
        lr_setting = parse_lr_setting(args.lr)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if isinstance(beta_setting, float):
        print(f"Using fixed beta: {beta_setting:.6g}")
    else:
        print("Using incremental beta search.")
    if isinstance(lr_setting, float):
        print(f"Using fixed explainer learning rate: {lr_setting:.6g}")
    else:
        print("Using dynamic per-epoch explainer learning rates.")

    # Use Planetoid for Cora/Citeseer/Pubmed, otherwise load from AllSet
    if args.dataset in ("Cora", "Citeseer", "Pubmed"):
        dataset = Planetoid(
            root=resolve_planetoid_root(),
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
        nhid=args.nhid,
        nout=args.nout,
        nclass=nclass,
        dropout=args.dropout,
    ).to(device)

    ckpt_path = args.ckpt_path or "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S = normalize_propagation(H)

    with torch.no_grad():
        out = model(data.x, S)
        y_log_prob_all = out
        y_pred_all = torch.argmax(out, dim=1)

    if args.target_node is None:
        target_nodes = [int(idx) for idx in torch.where(data.test_mask)[0]]
        if not target_nodes:
            raise ValueError(f"Dataset {args.dataset} has no test nodes.")
        print(f"Explaining {len(target_nodes)} node(s) from the test set.")
    else:
        if not 0 <= args.target_node < data.num_nodes:
            raise ValueError(
                f"target node {args.target_node} is outside the range of nodes in {args.dataset}"
            )
        target_nodes = [args.target_node]

    cf_examples_per_node: list[list] = []
    num_successful = 0
    possible_trials = 0
    isolated_nodes = 0
    total_start = time.time()

    for target_node in target_nodes:
        print(f"\n=== Running CF explainer for target node {target_node} ===")

        y_pred_orig = y_pred_all[target_node]
        log_prob_orig = y_log_prob_all[target_node]

        sub_H, sub_feat, sub_labels, node_dict = get_hyper_neighbourhood_fast(
            node_idx=target_node,
            H=H,
            n_hops=args.n_hops,
            features=data.x,
            labels=data.y,
        )

        sub_feat = sub_feat.to(device)
        sub_labels = sub_labels.to(device)

        target_node_sub_idx = node_dict[target_node]
        initial_beta = 0.0 if beta_setting == INCREMENTAL_BETA_MODE else float(beta_setting)
        explainer = CFExplainer(
            model=model,
            sub_H=sub_H,
            sub_feat=sub_feat,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig,
            log_prob_orig=log_prob_orig,
            beta=initial_beta,
            target_node_sub_idx=target_node_sub_idx,
            device=device,
            strategy=args.strategy,
        )

        # Check whether the target node has any incident hyperedges in the
        # extracted subgraph (i.e., whether any node-hyperedge edits are
        # available). If there are none, the explainer cannot produce a CF.
        sub_H_coalesced = sub_H.coalesce()
        H_indices = sub_H_coalesced.indices()
        row_mask = H_indices[0] == target_node_sub_idx
        available_edges = H_indices[1][row_mask]
        initial_available = available_edges.numel() > 0
        if not initial_available:
            print(
                f"Target node {target_node} has no incident edges in the extracted subgraph. No edits are available."
            )
            isolated_nodes += 1
            continue

        node_lr = lr_setting
        if isinstance(node_lr, float):
            print(f"Fixed learning rate for target node {target_node}: {node_lr:.6g}")
        else:
            print(
                f"Using dynamic learning rate updates for target node {target_node}."
            )

        node_start = time.time()
        if beta_setting == INCREMENTAL_BETA_MODE:
            best_cf_examples, possible, selected_beta = explainer.run_incremental_beta_search(
                cf_optimizer=args.cf_optimizer,
                node_idx=target_node,
                new_idx=target_node_sub_idx,
                lr=node_lr,
                n_momentum=args.n_momentum,
                num_epochs=args.num_epochs,
            )
        else:
            selected_beta = float(beta_setting)
            print(
                f"Running counterfactual search for target node {target_node} "
                f"with beta={selected_beta:.6g} and "
                f"lr={'dynamic' if isinstance(node_lr, str) else f'{node_lr:.6g}'}"
                "."
            )
            best_cf_examples = explainer.explain(
                cf_optimizer=args.cf_optimizer,
                node_idx=target_node,
                new_idx=target_node_sub_idx,
                lr=node_lr,
                n_momentum=args.n_momentum,
                num_epochs=args.num_epochs,
            )
            possible = bool(best_cf_examples) or not explainer.cf_model.no_more_edits

        if possible:
            possible_trials += 1
        node_elapsed = time.time() - node_start
        print(f"Node {target_node} run time: {node_elapsed:.2f}s")

        if not best_cf_examples:
            print(
                "No counterfactual example changing the prediction was found for this node."
            )
            cf_examples_per_node.append([])
            continue

        print(f"Selected beta for target node {target_node}: {selected_beta:.6g}")
        if isinstance(node_lr, float):
            print(f"Learning rate for target node {target_node}: {node_lr:.6g}")
        else:
            print(
                f"Learning rate mode for target node {target_node}: dynamic (recomputed each epoch)"
            )

        best_stats = best_cf_examples[-1]
        cf_H_sparse = best_stats[2]
        cf_examples_per_node.append([best_stats])
        num_successful += 1

        cf_H = cf_H_sparse.to(device=device, dtype=sub_H.dtype).coalesce()

        with torch.no_grad():
            S_cf = normalize_propagation(cf_H)
            cf_out = model(sub_feat, S_cf)
            cf_pred = torch.argmax(cf_out, dim=1)
            print(
                f"Original model prediction on best CF hypergraph "
                f"(target node {target_node}, subgraph idx {target_node_sub_idx}): "
                f"{cf_pred[target_node_sub_idx].item()}"
            )

    total_elapsed = time.time() - total_start
    print(f"\nTotal explainer run time: {total_elapsed:.2f}s")
    num_targets = len(target_nodes)
    print(f"Isolated Nodes: {isolated_nodes}/{num_targets}")
    print(f"Nodes where counterfactuals were possible: {possible_trials}/{num_targets}")
    print(f"Counterfactual examples found: {num_successful}/{possible_trials} (successful/possible)")

    if cf_examples_per_node:
        if args.output_path:
            output_path = os.path.abspath(args.output_path)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        else:
            results_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "results")
            )
            os.makedirs(results_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"cf_examples_{args.dataset.lower()}_{timestamp}.pkl"
            output_path = os.path.join(results_dir, filename)

        with open(output_path, "wb") as f:
            pickle.dump(cf_examples_per_node, f)
        print(f"Saved CF examples (including empty entries) to {output_path}")


if __name__ == "__main__":
    main()
