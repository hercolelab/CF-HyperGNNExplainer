"""Run counterfactual explanation based on the dense HGCN perturbation model."""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument("--target-node", type=int, default=45, help="Node to explain")
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

    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid")
    dataset = Planetoid(
        root=data_root,
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

    if not 0 <= args.target_node < data.num_nodes:
        raise ValueError(
            f"target node {args.target_node} is outside the range of nodes in {args.dataset}"
        )

    y_pred_orig = y_pred_all[args.target_node]
    sub_H, sub_feat, sub_labels, node_dict = get_hyper_neighbourhood(
        node_idx=args.target_node,
        H=H,
        n_hops=args.n_hops,
        features=data.x,
        labels=data.y,
    )

    if args.target_node not in node_dict:
        raise ValueError(
            f"Target node {args.target_node} fell outside the {args.n_hops}-hop neighborhood."
        )

    sub_H = sub_H.to(device)
    sub_feat = sub_feat.to(device)
    sub_labels = sub_labels.to(device)

    target_node_sub_idx = node_dict[args.target_node]
    explainer = CFExplainer(
        model=model,
        sub_H=sub_H,
        sub_feat=sub_feat,
        sub_labels=sub_labels,
        y_pred_orig=y_pred_orig,
        beta=args.beta,
        target_node_sub_idx=target_node_sub_idx,
        device=device,
    )

    start = time.time()
    best_cf_examples = explainer.explain(
        cf_optimizer=args.cf_optimizer,
        node_idx=args.target_node,
        new_idx=target_node_sub_idx,
        lr=args.lr,
        n_momentum=args.n_momentum,
        num_epochs=args.num_epochs,
    )
    elapsed = time.time() - start
    print(f"Explainer run time: {elapsed:.2f}s")

    if not best_cf_examples:
        print("No counterfactual example changing the prediction was found.")
        return

    best_stats = best_cf_examples[-1]
    cf_H_np = best_stats[2]

    cf_H = torch.tensor(cf_H_np, device=device, dtype=sub_H.dtype)
    with torch.no_grad():
        S_cf = normalize_propagation(cf_H)
        cf_out = model(sub_feat, S_cf)
        cf_pred = torch.argmax(cf_out, dim=1)
        new_idx = target_node_sub_idx
        print(
            f"Original model prediction on best CF hypergraph "
            f"(target node {args.target_node} in subgraph idx {new_idx}): {cf_pred[new_idx].item()}"
        )


if __name__ == "__main__":
    main()
