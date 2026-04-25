import argparse
import os
import pickle
import time

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from baselines.hyperex import (
    HyperExExplainer,
    build_attention_module,
    save_attention_checkpoint,
    train_hyperex_attention,
)
from hgcn import HGCN
from utils import (
    get_hyper_neighbourhood_fast,
    graph_to_hypergraph,
    normalize_propagation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HyperEx on a pretrained HGCN model."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "inference"),
        default="inference",
        help="Whether to train the HyperEx attention module or run inference.",
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
        help="Single node to explain (default: all test nodes in inference mode).",
    )
    parser.add_argument(
        "--n-hops",
        type=int,
        default=3,
        help="Computation subhypergraph depth (should match model depth).",
    )
    parser.add_argument(
        "--thresh-num",
        type=int,
        default=10,
        help=(
            "Top-k star-expansion edges (node-hyperedge pairs b_ij in B; same as "
            "nonzeros picked in the incidence matrix after scoring)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of HyperEx training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate for the HyperEx attention module.",
    )
    parser.add_argument(
        "--train-prop",
        type=float,
        default=0.5,
        help="Fraction of nodes used for HyperEx training.",
    )
    parser.add_argument(
        "--valid-prop",
        type=float,
        default=0.25,
        help="Fraction of nodes reserved for validation in the random split.",
    )
    parser.add_argument(
        "--node-samples",
        type=int,
        default=None,
        help="Optional cap on the number of training nodes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for the InfoNCE attention training objective.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="InfoNCE temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for HyperEx attention initialization.",
    )
    parser.add_argument(
        "--attention-ckpt",
        default=None,
        help="Attention checkpoint to load for inference or save after training.",
    )
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
        help="Device for training and explanation",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Destination pickle file (default: results/hyperex_<dataset>_<ts>.pkl)",
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


def default_results_dir() -> str:
    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def default_attention_ckpt(dataset: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        default_results_dir(),
        f"hyperex_attention_{dataset.lower()}_{timestamp}.pt",
    )


def default_output_path(dataset: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        default_results_dir(),
        f"hyperex_{dataset.lower()}_{timestamp}.pkl",
    )


def load_data_and_model(args: argparse.Namespace, device: torch.device):
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
    return dataset, data, H, model


def run_training(
    args: argparse.Namespace,
    dataset,
    data: torch.Tensor,
    H: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    attention_module = build_attention_module(
        num_classes=dataset.num_classes,
        device=device,
        max_hops=args.n_hops,
        checkpoint_path=args.attention_ckpt,
        strict=False,
    )
    attention_module = train_hyperex_attention(
        model=model,
        H=H,
        features=data.x,
        labels=data.y,
        n_hops=args.n_hops,
        thresh_num=args.thresh_num,
        epochs=args.epochs,
        lr=args.lr,
        train_prop=args.train_prop,
        valid_prop=args.valid_prop,
        node_samples=args.node_samples,
        batch_size=args.batch_size,
        tau=args.tau,
        seed=args.seed,
        device=device,
        attention_module=attention_module,
    )

    checkpoint_path = args.attention_ckpt or default_attention_ckpt(args.dataset)
    save_attention_checkpoint(attention_module, checkpoint_path)
    print(f"Saved HyperEx attention checkpoint to {checkpoint_path}")


def run_inference(
    args: argparse.Namespace,
    data: torch.Tensor,
    H: torch.Tensor,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
) -> None:
    attention_module = build_attention_module(
        num_classes=dataset.num_classes,
        device=device,
        max_hops=args.n_hops,
        checkpoint_path=args.attention_ckpt,
    )
    if args.attention_ckpt is None:
        print("Running HyperEx inference with an uninitialised attention module.")

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

    with torch.no_grad():
        S_full = normalize_propagation(H)
        global_logits = model(data.x, S_full, return_embeddings=True)
        global_log_p = torch.log_softmax(global_logits, dim=1)

    results: list = []
    total_start = time.time()

    for i, node_idx in enumerate(target_nodes):
        print(f"\n=== [{i + 1}/{len(target_nodes)}] Node {node_idx} ===")

        comp_H, sub_feat, _sub_labels, node_dict = get_hyper_neighbourhood_fast(
            node_idx=node_idx,
            H=H,
            n_hops=args.n_hops,
            features=data.x,
            labels=data.y,
        )

        if node_idx not in node_dict:
            print(
                f"  Node {node_idx} not found in computation subhypergraph; skipping."
            )
            results.append(None)
            continue

        explainer = HyperExExplainer(
            model=model,
            full_H=H,
            full_feat=data.x,
            comp_H=comp_H,
            sub_feat=sub_feat,
            node_dict=node_dict,
            target_node_global=node_idx,
            attention_module=attention_module,
            thresh_num=args.thresh_num,
            device=device,
            global_logits=global_logits,
            global_log_p=global_log_p,
        )

        node_start = time.time()
        result = explainer.explain()
        elapsed = time.time() - node_start

        result["node_idx"] = node_idx
        result["local_idx"] = node_dict[node_idx]
        result["elapsed"] = elapsed
        results.append(result)

        print(
            f"  y_orig={result['y_pred_orig']}, y_expl={result['y_pred_expl']}, "
            f"links={result['num_links_expl']}/{result['num_links_comp']}, "
            f"time={elapsed:.2f}s"
        )

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.2f}s")

    valid = [r for r in results if r is not None]
    matches = sum(1 for r in valid if r["y_pred_orig"] == r["y_pred_expl"])
    print(f"Prediction match (Acc faithfulness): {matches}/{len(valid)}")

    output_path = (
        os.path.abspath(args.output_path)
        if args.output_path
        else default_output_path(args.dataset)
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {output_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset, data, H, model = load_data_and_model(args, device)

    if args.mode == "train":
        run_training(args, dataset, data, H, model, device)
        return

    run_inference(args, data, H, model, dataset, device)


if __name__ == "__main__":
    main()
