import argparse
import os
import pickle
import time

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from baselines.get_computation_subhypergraph import get_computation_subhypergraph
from hgcn import HGCN
from baselines.shypx import SHypXExplainer
from utils import graph_to_hypergraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHypX explainer on a pretrained HGCN model."
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        help="Name of the Planetoid dataset (Cora, Citeseer, Pubmed)",
    )
    parser.add_argument(
        "--target-node",
        type=int,
        default=None,
        help="Single node to explain (default: all test nodes)",
    )
    parser.add_argument(
        "--n-hops",
        type=int,
        default=3,
        help="Computation subhypergraph depth (should match model depth)",
    )
    parser.add_argument(
        "--lambda-pred",
        type=float,
        default=1.0,
        help="Weight for the KL-divergence loss (λ_pred)",
    )
    parser.add_argument(
        "--lambda-size",
        type=float,
        default=0.005,
        help="Weight for the size penalty (λ_size)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Gumbel-Softmax temperature",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate for the logit parameters",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=400,
        help="Number of optimisation epochs per node",
    )
    parser.add_argument(
        "--init-prob",
        type=float,
        default=0.95,
        help="Initial link-inclusion probability (π_init)",
    )
    # ---- HGCN architecture (must match the checkpoint) ----
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability"
    )
    parser.add_argument(
        "--nhid", type=int, default=64, help="Hidden feature size for HGCN"
    )
    parser.add_argument(
        "--nout",
        type=int,
        default=32,
        help="Output projection size for HGCN",
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
        help="Device for inference / explanation",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Destination pickle file (default: results/shypx_<dataset>_<ts>.pkl)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        return torch.device("cpu")
    return dev


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
    H = graph_to_hypergraph(data.edge_index, data.num_nodes, device=device)

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

    if args.target_node is not None:
        if not 0 <= args.target_node < data.num_nodes:
            raise ValueError(
                f"target-node {args.target_node} outside [0, {data.num_nodes})"
            )
        target_nodes = [args.target_node]
    else:
        target_nodes = [int(idx) for idx in torch.where(data.test_mask)[0]]
        if not target_nodes:
            raise ValueError(f"Dataset {args.dataset} has no test nodes.")
        print(f"Explaining {len(target_nodes)} test node(s).")

    results: list = []
    total_start = time.time()

    for i, node_idx in enumerate(target_nodes):
        print(f"\n=== [{i + 1}/{len(target_nodes)}] Node {node_idx} ===")

        comp_H, sub_feat, _sub_labels, node_dict = get_computation_subhypergraph(
            node_idx=node_idx,
            H=H,
            n_hops=args.n_hops,
            features=data.x,
            labels=data.y,
        )

        if node_idx not in node_dict:
            print(
                f"  Node {node_idx} not found in computation subhypergraph — skipping."
            )
            results.append(None)
            continue

        target_local = node_dict[node_idx]

        explainer = SHypXExplainer(
            model=model,
            comp_H=comp_H,
            sub_feat=sub_feat.to(device),
            target_node_local=target_local,
            lambda_pred=args.lambda_pred,
            lambda_size=args.lambda_size,
            tau=args.tau,
            lr=args.lr,
            num_epochs=args.num_epochs,
            init_prob=args.init_prob,
            device=device,
        )

        node_start = time.time()
        result = explainer.explain()
        elapsed = time.time() - node_start

        result["node_idx"] = node_idx
        result["local_idx"] = target_local
        result["elapsed"] = elapsed
        results.append(result)

        print(
            f"  y_orig={result['y_pred_orig']}, y_expl={result['y_pred_expl']}, "
            f"links={result['num_links_expl']}/{result['num_links_comp']}, "
            f"loss={result['best_loss']:.4f}, time={elapsed:.2f}s"
        )

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.2f}s")

    valid = [r for r in results if r is not None]
    matches = sum(1 for r in valid if r["y_pred_orig"] == r["y_pred_expl"])
    print(f"Prediction match (Acc faithfulness): {matches}/{len(valid)}")

    if args.output_path:
        output_path = os.path.abspath(args.output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "results")
        )
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(
            results_dir, f"shypx_{args.dataset.lower()}_{timestamp}.pkl"
        )

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
