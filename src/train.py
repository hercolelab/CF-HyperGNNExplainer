import argparse
import os
from typing import Any, Dict

import torch
from torch import Tensor
from tqdm.auto import tqdm
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from torch.nn.utils import clip_grad_norm_

from utils import normalize_propagation
from hgcn import HGCN


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_incidence_matrix(edge_index: Tensor, num_nodes: int) -> SparseTensor:
    """
    Convert CORA's edge_index to a hypergraph incidence matrix.
    Each node defines a hyperedge containing itself and its first-order neighbors.
    """
    edge_index = edge_index.cpu()
    row, col = edge_index

    adjacency = [set[Any]() for _ in range(num_nodes)]
    for src, dst in zip(row.tolist(), col.tolist()):
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    rows, cols = [], []
    for hyperedge_id in range(num_nodes):
        hyper_nodes = set(adjacency[hyperedge_id])
        hyper_nodes.add(hyperedge_id)
        for node_id in hyper_nodes:
            rows.append(node_id)
            cols.append(hyperedge_id)

    row_idx = torch.tensor(rows, dtype=torch.long)
    col_idx = torch.tensor(cols, dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)

    H = SparseTensor(
        row=row_idx, col=col_idx, value=values, sparse_sizes=(num_nodes, num_nodes)
    )
    return H


def accuracy(logits: Tensor, labels: Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return preds.eq(labels).sum().item() / labels.size(0)


def sanitize_checkpoint_name(text: str) -> str:
    sanitized = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            sanitized.append(ch)
        else:
            sanitized.append("_")
    result = "".join(sanitized)
    return result if result else "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HGCN.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--hidden",
        "--hidden-units",
        type=int,
        default=64,
        dest="hidden",
        help="Number of hidden units.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (L2 penalty).",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm. Set to 0 or negative to disable clipping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to use for training. 'auto' selects CUDA if available.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="hgcn",
        help="Base name for the saved model checkpoint.",
    )
    return parser.parse_args()


def evaluate(model: HGCN, data, S: Tensor) -> Dict[str, Dict[str, float]]:
    model.eval()
    metrics: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        out = model(data.x, S)
        for split in ("train", "val", "test"):
            mask = getattr(data, f"{split}_mask")
            if mask.sum().item() == 0:
                continue
            loss = model.loss(out[mask], data.y[mask]).item()
            acc = accuracy(out[mask], data.y[mask])
            metrics[split] = {"loss": loss, "acc": acc}
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(args.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = requested

    print(f"Using device: {device}")

    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid")
    dataset = Planetoid(root=data_root, name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    H = build_incidence_matrix(data.edge_index, data.num_nodes)
    S = normalize_propagation(H).to(device)

    model = HGCN(
        nfeat=dataset.num_features,
        nhid=args.hidden,
        nout=args.hidden,
        nclass=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    epochs = args.epochs

    for epoch in tqdm(range(1, epochs + 1), desc="Training epochs"):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, S)
        loss = model.loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        if args.clip_grad_norm > 0:
            clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])

        metrics = evaluate(model, data, S)
        val_loss = metrics.get("val", {}).get("loss", float("nan"))
        val_acc = metrics.get("val", {}).get("acc", float("nan"))

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    metrics = evaluate(model, data, S)
    test_metrics = metrics.get("test", {"loss": float("nan"), "acc": float("nan")})
    print(
        f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']:.4f}"
    )

    models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models")
    )
    os.makedirs(models_dir, exist_ok=True)

    clip_label = (
        f"clip{args.clip_grad_norm:g}" if args.clip_grad_norm > 0 else "clipNone"
    )
    checkpoint_name = (
        "_".join(
            [
                sanitize_checkpoint_name(args.model_name),
                f"seed{args.seed}",
                f"epochs{epochs}",
                f"lr{args.learning_rate:g}",
                f"hidden{args.hidden}",
                f"dropout{args.dropout:g}",
                f"wd{args.weight_decay:g}",
                clip_label,
            ]
        )
        + ".pt"
    )

    checkpoint_path = os.path.join(models_dir, checkpoint_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
