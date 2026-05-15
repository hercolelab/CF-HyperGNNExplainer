import argparse
import os
import pickle
import time
from types import SimpleNamespace

import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from baselines.cf_gnnexplainer import CFExplainer, GCNSynthetic
from baselines.cf_gnnexplainer.utils import (
    get_neighbourhood,
    normalize_adj,
    star_expand_hypergraph,
)
from utils import graph_to_hypergraph


PLANETOID_DATASETS = ("Cora", "Citeseer", "Pubmed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original dense CF-GNNExplainer baseline on star-expanded hypergraphs."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "explain"),
        default="explain",
        help="Whether to train the star-expanded GCN or run CF-GNNExplainer.",
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        help="Name of the Planetoid or AllSet dataset",
    )
    parser.add_argument(
        "--target-node",
        type=int,
        default=None,
        help="Optional original hypergraph node to explain (default: all test nodes).",
    )
    parser.add_argument(
        "--n-hops",
        type=int,
        default=4,
        help="Neighborhood radius in the star-expanded graph.",
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--cf-optimizer",
        choices=("SGD", "Adadelta"),
        default="SGD",
        help="Optimizer for the counterfactual explainer.",
    )
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--n-momentum", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stop after this many stable zero-gradient CF epochs.",
    )
    parser.add_argument("--hidden", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Checkpoint to save in train mode or load in explain mode (default: ckpt.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of star-expanded GCN training epochs in train mode.",
    )
    parser.add_argument("--model-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--clip-grad-norm", type=float, default=2.0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


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


def load_dataset(dataset_name: str, device: torch.device):
    if dataset_name in PLANETOID_DATASETS:
        dataset = Planetoid(
            root=resolve_planetoid_root(),
            name=dataset_name,
            transform=NormalizeFeatures(),
        )
        data = dataset[0].to(device)
        H = graph_to_hypergraph(data.edge_index, data.num_nodes, device=device)
        return dataset, data, H

    from utils.allset_loader import load_allset_dataset

    data, H = load_allset_dataset(dataset_name, device=device)
    data.x = data.x.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)
    dataset = SimpleNamespace(
        num_features=int(data.x.size(1)),
        num_classes=int(int(data.y.max().item()) + 1),
    )
    return dataset, data, H


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def train_model(
    model: GCNSynthetic,
    graph,
    epochs: int,
    lr: float,
    weight_decay: float,
    clip_grad_norm: float,
) -> None:
    norm_adj = normalize_adj(graph.adj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(graph.x, norm_adj)
        loss = model.loss(output[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        if clip_grad_norm > 0:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        train_pred = output[graph.train_mask].argmax(dim=1)
        train_acc = (
            train_pred.eq(graph.y[graph.train_mask]).sum().item()
            / int(graph.train_mask.sum().item())
        )
        print(
            f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}"
        )


def default_output_path(dataset: str) -> str:
    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(results_dir, f"cf_gnn_examples_{dataset.lower()}_{timestamp}.pkl")


def default_checkpoint_path(dataset: str) -> str:
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(models_dir, f"cf_gnn_{dataset.lower()}_{timestamp}.pt")


def select_target_nodes(args: argparse.Namespace, data) -> list[int]:
    if args.target_node is not None:
        if not 0 <= args.target_node < data.num_nodes:
            raise ValueError(
                f"target node {args.target_node} is outside the range of nodes in {args.dataset}"
            )
        return [args.target_node]

    target_nodes = [int(idx) for idx in torch.where(data.test_mask)[0]]
    if not target_nodes:
        raise ValueError(f"Dataset {args.dataset} has no test nodes.")

    is_allset = args.dataset not in PLANETOID_DATASETS
    if is_allset and len(target_nodes) > 500:
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(target_nodes), generator=generator)[:500]
        target_nodes = sorted(target_nodes[int(i)] for i in perm)
        print(
            f"Subsampled {len(target_nodes)} test node(s) from AllSet dataset "
            f"{args.dataset} with seed {args.seed}."
        )
    return target_nodes


def save_trained_checkpoint(
    model: GCNSynthetic,
    args: argparse.Namespace,
    dataset,
) -> None:
    output_path = os.path.abspath(args.ckpt_path or default_checkpoint_path(args.dataset))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
        },
        output_path,
    )
    print(f"Saved CF-GNNExplainer GCN checkpoint to {output_path}")


def load_data_graph_and_model(args: argparse.Namespace, device: torch.device):
    dataset, data, H = load_dataset(args.dataset, device)
    graph = star_expand_hypergraph(
        H=H,
        features=data.x,
        labels=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )
    print(
        f"Star-expanded graph: {graph.num_original_nodes} original nodes, "
        f"{graph.num_hyperedge_nodes} hyperedge nodes, {graph.edge_index.size(1) // 2} links."
    )

    model = GCNSynthetic(
        nfeat=dataset.num_features,
        nhid=args.hidden,
        nout=args.hidden,
        nclass=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)
    return dataset, data, graph, model


def run_training(
    args: argparse.Namespace,
    dataset,
    graph,
    model: GCNSynthetic,
) -> None:
    train_model(
        model=model,
        graph=graph,
        epochs=args.epochs,
        lr=args.model_lr,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
    )
    save_trained_checkpoint(model, args, dataset)


def run_explain(
    args: argparse.Namespace,
    dataset,
    data,
    graph,
    model: GCNSynthetic,
    device: torch.device,
) -> None:
    ckpt_path = args.ckpt_path or "ckpt.pt"
    load_checkpoint(model, ckpt_path, device)
    print(f"Loaded CF-GNNExplainer GCN checkpoint from {ckpt_path}")
    model.eval()
    norm_adj = normalize_adj(graph.adj)
    with torch.no_grad():
        output = model(graph.x, norm_adj)
        y_log_prob_all = output
        y_pred_all = torch.argmax(output, dim=1)

    target_nodes = select_target_nodes(args, data)
    print(f"Explaining {len(target_nodes)} node(s) from the test set.")

    cf_examples_per_node: list = []
    num_successful = 0
    isolated_nodes = 0
    non_isolated_times: list[float] = []
    total_start = time.time()

    for target_node in target_nodes:
        print(f"\n=== Running CF-GNNExplainer for target node {target_node} ===")
        y_pred_orig = y_pred_all[target_node]
        log_prob_orig = y_log_prob_all[target_node]

        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
            node_idx=target_node,
            edge_index=graph.edge_index,
            n_hops=args.n_hops,
            features=graph.x,
            labels=graph.y,
        )
        sub_adj = sub_adj.to(device)
        sub_feat = sub_feat.to(device)
        sub_labels = sub_labels.to(device)
        target_node_sub_idx = int(node_dict[target_node])

        if sub_adj[target_node_sub_idx].sum().item() == 0:
            print(
                f"Target node {target_node} has no incident edges in the extracted star graph."
            )
            isolated_nodes += 1
            cf_examples_per_node.append(None)
            continue

        with torch.no_grad():
            sub_output = model(sub_feat, normalize_adj(sub_adj))
            if not args.quiet:
                print(f"Output original model, full adj: {output[target_node]}")
                print(
                    "Output original model, sub adj: "
                    f"{sub_output[target_node_sub_idx]}"
                )

        explainer = CFExplainer(
            model=model,
            sub_adj=sub_adj,
            sub_feat=sub_feat,
            n_hid=args.hidden,
            dropout=args.dropout,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig,
            log_prob_orig=log_prob_orig,
            num_classes=dataset.num_classes,
            beta=args.beta,
            device=device,
            quiet=args.quiet,
        )

        node_start = time.time()
        best_cf_examples = explainer.explain(
            node_idx=torch.tensor(target_node, device=device),
            cf_optimizer=args.cf_optimizer,
            new_idx=target_node_sub_idx,
            lr=args.lr,
            n_momentum=args.n_momentum,
            num_epochs=args.num_epochs,
            patience=args.patience,
        )
        node_elapsed = time.time() - node_start
        non_isolated_times.append(node_elapsed)
        print(f"Node {target_node} run time: {node_elapsed:.2f}s")

        if not best_cf_examples:
            print(
                "No counterfactual example changing the prediction was found for this node."
            )
            cf_examples_per_node.append(
                {
                    "no_cf_found": True,
                    "possible": True,
                    "node_idx": target_node,
                    "log_prob_orig": log_prob_orig.detach().cpu(),
                    "y_pred_orig": int(y_pred_orig.item()),
                }
            )
            continue

        best_stats = best_cf_examples[-1]
        cf_examples_per_node.append([best_stats])
        num_successful += 1

        cf_adj = best_stats[2].to(device=device, dtype=sub_adj.dtype)
        with torch.no_grad():
            cf_out = model(sub_feat, normalize_adj(cf_adj))
            cf_pred = torch.argmax(cf_out, dim=1)
            print(
                f"Original model prediction on best CF graph "
                f"(target node {target_node}, subgraph idx {target_node_sub_idx}): "
                f"{cf_pred[target_node_sub_idx].item()}"
            )

    total_elapsed = time.time() - total_start
    num_targets = len(target_nodes)
    num_non_isolated = num_targets - isolated_nodes
    avg_time_non_isolated = (
        sum(non_isolated_times) / len(non_isolated_times)
        if non_isolated_times
        else None
    )
    print(f"\nTotal explainer run time: {total_elapsed:.2f}s")
    print(f"Isolated nodes: {isolated_nodes}/{num_targets}")
    print(f"Non-isolated nodes: {num_non_isolated}/{num_targets}")
    print(
        f"Counterfactual examples found: {num_successful}/{num_non_isolated} (successful/non-isolated)"
    )

    output_path = (
        os.path.abspath(args.output_path)
        if args.output_path
        else default_output_path(args.dataset)
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "dataset": args.dataset,
                "baseline": "cf-gnnexplainer",
                "graph_conversion": "star_expansion",
                "cf_examples_per_node": cf_examples_per_node,
                "num_targets": num_targets,
                "num_isolated": isolated_nodes,
                "num_non_isolated": num_non_isolated,
                "num_cf_possible": num_non_isolated,
                "num_cf_found": num_successful,
                "avg_time_non_isolated": avg_time_non_isolated,
                "avg_time_possible": avg_time_non_isolated,
            },
            f,
        )
    print(f"Saved CF-GNNExplainer examples to {output_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset, data, graph, model = load_data_graph_and_model(args, device)

    if args.mode == "train":
        run_training(args, dataset, graph, model)
        return

    run_explain(args, dataset, data, graph, model, device)


if __name__ == "__main__":
    main()
