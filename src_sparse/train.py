import argparse
import os
from typing import Dict


import torch
from torch import Tensor
from tqdm.auto import tqdm
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


from torch.nn.utils import clip_grad_norm_


from utils import normalize_propagation, build_incidence_matrix
from hgcn import HGCN


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    Calculate the accuracy of the model's predictions
    Args:
        logits: The logits of the model's predictions
        labels: The labels of the data
    Returns:
        The accuracy of the model's predictions
    """
    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return preds.eq(labels).sum().item() / labels.size(0)


def sanitize_checkpoint_name(text: str) -> str:
    """
    Sanitize the checkpoint name to make it a valid filename
    """
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
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Name of the Planetoid dataset to load (e.g. Cora, Citeseer, Pubmed)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Number of hidden units for intermediate layers",
    )
    parser.add_argument(
        "--out-hidden",
        type=int,
        default=32,
        help="Number of hidden units for the output projection",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (L2 penalty)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm. Set to 0 or negative to disable clipping",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for training. 'auto' selects CUDA if available",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="hgcn",
        help="Base name for the saved model checkpoint",
    )
    return parser.parse_args()


def evaluate(model: HGCN, data, S: Tensor) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model on the data
    Args:
        model: The model to evaluate
        data: The data to evaluate the model on
        S: The propagation matrix, computed from the incidence matrix H
    Returns:
        A dictionary containing the metrics for each split (train, val, test)
    """
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

    # Use Planetoid for Cora/Citeseer/Pubmed, otherwise load from AllSet
    if args.dataset in ("Cora", "Citeseer", "Pubmed"):
        dataset = Planetoid(
            root=os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid"),
            name=args.dataset,
            transform=NormalizeFeatures(),
        )
        data = dataset[0].to(device)
        H = build_incidence_matrix(data.edge_index, data.num_nodes, device=device)
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
    S = normalize_propagation(H)

    model = HGCN(
        nfeat=nfeat,
        nhid=args.hidden,
        nout=args.out_hidden,
        nclass=nclass,
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
                f"dataset{args.dataset}",
                f"epochs{epochs}",
                f"lr{args.learning_rate:g}",
                f"nhid{args.hidden}",
                f"nout{args.out_hidden}",
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
