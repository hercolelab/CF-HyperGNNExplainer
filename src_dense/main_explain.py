import argparse
import os
import pickle
import time

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from cf_explanation.cf_explainer import CFExplainer
from hgcn import HGCN
from train import build_incidence_matrix
from utils.utils import get_hyper_neighbourhood, normalize_propagation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CF-HyperGNNExplainer on a pretrained HGCN model."
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
        help="Optional node to explain (default: run on all Planetoid test nodes).",
    )
    parser.add_argument(
        "--n-hops", type=int, default=4, help="Neighborhood radius for the explainer"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Weight for the graph-distance loss"
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
    parser.add_argument("--lr", type=float, default=0.1, help="Explainer learning rate")
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
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device used for inference and explanation. 'auto' selects CUDA or MPS if available",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Destination pickle file for CF examples (default: results/cf_examples_<dataset>_<timestamp>.pkl)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    requested_device = torch.device(device_arg)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if requested_device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return requested_device


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset = Planetoid(
        root=os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid"),
        name=args.dataset,
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    H = build_incidence_matrix(data.edge_index, data.num_nodes).to(device)
    model = HGCN(
        nfeat=dataset.num_features,
        nhid=args.nhid,
        nout=args.nout,
        nclass=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)

    ckpt_path = args.ckpt_path or "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S = normalize_propagation(H).to(device)
    with torch.no_grad():
        out = model(data.x, S)
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
    total_start = time.time()
    for target_node in target_nodes:
        print(f"\n=== Running CF explainer for target node {target_node} ===")

        y_pred_orig = y_pred_all[target_node]
        sub_H, sub_feat, sub_labels, node_dict = get_hyper_neighbourhood(
            node_idx=target_node,
            H=H,
            n_hops=args.n_hops,
            features=data.x,
            labels=data.y,
        )

        sub_H = sub_H.to(device)
        sub_feat = sub_feat.to(device)
        sub_labels = sub_labels.to(device)

        target_node_sub_idx = node_dict[target_node]
        explainer = CFExplainer(
            model=model,
            sub_H=sub_H,
            sub_feat=sub_feat,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig,
            beta=args.beta,
            target_node_sub_idx=target_node_sub_idx,
            device=device,
            strategy=args.strategy,
        )

        node_start = time.time()
        best_cf_examples = explainer.explain(
            cf_optimizer=args.cf_optimizer,
            node_idx=target_node,
            new_idx=target_node_sub_idx,
            lr=args.lr,
            n_momentum=args.n_momentum,
            num_epochs=args.num_epochs,
        )
        node_elapsed = time.time() - node_start
        print(f"Node {target_node} run time: {node_elapsed:.2f}s")

        if not best_cf_examples:
            print(
                "No counterfactual example changing the prediction was found for this node."
            )
            cf_examples_per_node.append([])
            continue

        best_stats = best_cf_examples[-1]
        cf_H_np = best_stats[2]
        cf_examples_per_node.append([best_stats])
        num_successful += 1

        cf_H = torch.tensor(cf_H_np, device=device, dtype=sub_H.dtype)
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
    print(f"Counterfactual examples found for {num_successful}/{num_targets} node(s).")

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
