import os
from typing import Any, Dict

import torch
from torch import Tensor
from tqdm.auto import tqdm
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from hgcn import HGCN
from utils import normalize_propagation


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_incidence_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
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

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    size = (num_nodes, num_nodes)

    return torch.sparse_coo_tensor(indices, values, size)


def accuracy(logits: Tensor, labels: Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return preds.eq(labels).sum().item() / labels.size(0)


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
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid")
    dataset = Planetoid(root=data_root, name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    H = build_incidence_matrix(data.edge_index, data.num_nodes)
    S = normalize_propagation(H).to(device)

    model = HGCN(
        nfeat=dataset.num_features,
        nhid=64,
        nout=32,
        nclass=dataset.num_classes,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    epochs = 200

    for epoch in tqdm(range(1, epochs + 1), desc="Training epochs"):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, S)
        loss = model.loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
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


if __name__ == "__main__":
    main()
