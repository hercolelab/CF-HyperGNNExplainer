
import argparse
import csv
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import torch


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DEFAULT_DATASETS = (
    "cocitation-cora",
    "cocitation-citeseer",
    "cocitation-pubmed",
    "coauthorship-cora",
    "zoo",
    "mushrooms",
    "ntu2012",
    "modelnet40",
)
NUM_EPOCHS = 125
N_MOMENTUM = 0.9
LR_START = 0.1
LR_END = 10000.0
LR_REACH_EPOCH = 75
SATURATION_THRESHOLD = 1e-3
DEFAULT_CKPT_TEMPLATE = str(
    REPO_ROOT / "models" / "cf_gnn_{dataset_slug}.pt"
)
CSV_FIELDNAMES = [
    "dataset",
    "status",
    "checkpoint_path",
    "num_targets",
    "num_isolated",
    "num_considered",
    "num_saturated",
    "num_not_saturated",
    "total_elapsed_sec",
    "avg_epochs_run",
    "num_epochs",
    "n_momentum",
    "lr_start",
    "lr_reach_epoch",
    "lr_end",
    "saturation_threshold",
    "error",
]


def load_cf_gnn_main():
    module_path = SRC_DIR / "main_cf-gnn.py"
    spec = importlib.util.spec_from_file_location("main_cf_gnn", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CF-GNNExplainer saturation counting on the requested datasets. "
            "The explainer uses 125 epochs, SGD momentum 0.9, and a learning rate "
            "that grows linearly from 0.1 at epoch 1 to 10000 at epoch 75."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to process in order.",
    )
    parser.add_argument(
        "--ckpt-path-template",
        default=DEFAULT_CKPT_TEMPLATE,
        help=(
            "Checkpoint path or template for the trained star-expanded CF-GNN model. "
            "Templates may use {dataset}, {dataset_lower}, or {dataset_slug}."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Destination CSV path (default: results/cf_gnn_saturation_<timestamp>.csv).",
    )
    parser.add_argument(
        "--target-node",
        type=int,
        default=None,
        help="Optional node id to explain for every dataset instead of all selected test nodes.",
    )
    parser.add_argument("--n-hops", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--cf-optimizer",
        choices=("SGD", "Adadelta"),
        default="SGD",
        help="Optimizer for the counterfactual explainer. SGD uses momentum 0.9 here.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print one progress line every N considered nodes. Set 0 to disable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-epoch CF-GNNExplainer diagnostics.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error row and continue if a dataset fails.",
    )
    return parser.parse_args()


def default_output_csv() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "results" / f"cf_gnn_saturation_{timestamp}.csv"


def dataset_slug(dataset: str) -> str:
    return dataset.strip().lower().replace("/", "-").replace(" ", "-")


def resolve_checkpoint_path(template: str, dataset: str) -> Path:
    values = {
        "dataset": dataset,
        "dataset_lower": dataset.lower(),
        "dataset_slug": dataset_slug(dataset),
    }
    return Path(template.format(**values)).expanduser()


def lr_schedule(epoch_index: int) -> float:
    warmup_index = LR_REACH_EPOCH - 1
    if epoch_index >= warmup_index:
        return LR_END
    return LR_START + (LR_END - LR_START) * (epoch_index / warmup_index)


def build_dataset_args(args: argparse.Namespace, dataset_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset_name,
        target_node=args.target_node,
        n_hops=args.n_hops,
        beta=args.beta,
        cf_optimizer=args.cf_optimizer,
        hidden=args.hidden,
        dropout=args.dropout,
        seed=args.seed,
    )


def empty_row(dataset_name: str, ckpt_path: Path, status: str, error: str = "") -> dict:
    return {
        "dataset": dataset_name,
        "status": status,
        "checkpoint_path": str(ckpt_path),
        "num_targets": "",
        "num_isolated": "",
        "num_considered": "",
        "num_saturated": "",
        "num_not_saturated": "",
        "total_elapsed_sec": "",
        "avg_epochs_run": "",
        "num_epochs": NUM_EPOCHS,
        "n_momentum": N_MOMENTUM,
        "lr_start": LR_START,
        "lr_reach_epoch": LR_REACH_EPOCH,
        "lr_end": LR_END,
        "saturation_threshold": SATURATION_THRESHOLD,
        "error": error,
    }


def write_rows(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(
    *,
    cf_main,
    args: argparse.Namespace,
    dataset_name: str,
    device: torch.device,
) -> dict:
    torch.manual_seed(args.seed)
    dataset_args = build_dataset_args(args, dataset_name)
    ckpt_path = resolve_checkpoint_path(args.ckpt_path_template, dataset_name)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint for {dataset_name}: {ckpt_path}. "
            "Train one with src_sparse/main_cf-gnn.py --mode train "
            f"--dataset {dataset_name} --ckpt-path {ckpt_path}"
        )

    start = time.time()
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Loading checkpoint: {ckpt_path}")
    dataset, data, graph, model = cf_main.load_data_graph_and_model(dataset_args, device)
    cf_main.load_checkpoint(model, str(ckpt_path), device)
    model.eval()

    norm_adj = cf_main.normalize_adj(graph.adj)
    with torch.no_grad():
        output = model(graph.x, norm_adj)
        y_log_prob_all = output
        y_pred_all = torch.argmax(output, dim=1)

    target_nodes = cf_main.select_target_nodes(dataset_args, data)
    print(f"Considering {len(target_nodes)} target node(s).")

    num_isolated = 0
    num_considered = 0
    num_saturated = 0
    epochs_run_total = 0

    for target_node in target_nodes:
        y_pred_orig = y_pred_all[target_node]
        log_prob_orig = y_log_prob_all[target_node]

        sub_adj, sub_feat, sub_labels, node_dict = cf_main.get_neighbourhood(
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
            num_isolated += 1
            continue

        explainer = cf_main.CFExplainer(
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
            quiet=not args.verbose,
        )

        _, metadata = explainer.explain(
            node_idx=torch.tensor(target_node, device=device),
            cf_optimizer=args.cf_optimizer,
            new_idx=target_node_sub_idx,
            lr=LR_START,
            n_momentum=N_MOMENTUM,
            num_epochs=NUM_EPOCHS,
            patience=args.patience,
            lr_schedule=lr_schedule,
            saturation_threshold=SATURATION_THRESHOLD,
            return_metadata=True,
        )

        num_considered += 1
        epochs_run_total += int(metadata["epochs_run"])
        if bool(metadata["saturated"]):
            num_saturated += 1

        if args.progress_interval > 0 and num_considered % args.progress_interval == 0:
            print(
                f"Processed {num_considered} non-isolated node(s): "
                f"saturated={num_saturated}, "
                f"not_saturated={num_considered - num_saturated}."
            )

    elapsed = time.time() - start
    num_not_saturated = num_considered - num_saturated
    avg_epochs_run = epochs_run_total / num_considered if num_considered else 0.0
    print(
        f"{dataset_name}: saturated={num_saturated}, "
        f"not_saturated={num_not_saturated}, considered={num_considered}, "
        f"targets={len(target_nodes)}."
    )

    return {
        "dataset": dataset_name,
        "status": "success",
        "checkpoint_path": str(ckpt_path),
        "num_targets": len(target_nodes),
        "num_isolated": num_isolated,
        "num_considered": num_considered,
        "num_saturated": num_saturated,
        "num_not_saturated": num_not_saturated,
        "total_elapsed_sec": f"{elapsed:.6f}",
        "avg_epochs_run": f"{avg_epochs_run:.6f}",
        "num_epochs": NUM_EPOCHS,
        "n_momentum": N_MOMENTUM,
        "lr_start": LR_START,
        "lr_reach_epoch": LR_REACH_EPOCH,
        "lr_end": LR_END,
        "saturation_threshold": SATURATION_THRESHOLD,
        "error": "",
    }


def main() -> None:
    args = parse_args()
    cf_main = load_cf_gnn_main()
    device = cf_main.resolve_device(args.device)
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else default_output_csv()
    print(f"Using device: {device}")
    print(
        "Counterfactual settings: "
        f"epochs={NUM_EPOCHS}, momentum={N_MOMENTUM}, "
        f"lr={LR_START} -> {LR_END} by epoch {LR_REACH_EPOCH}, "
        f"saturation_threshold={SATURATION_THRESHOLD}."
    )

    rows = []
    for dataset_name in args.datasets:
        ckpt_path = resolve_checkpoint_path(args.ckpt_path_template, dataset_name)
        try:
            row = run_dataset(
                cf_main=cf_main,
                args=args,
                dataset_name=dataset_name,
                device=device,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"{dataset_name}: ERROR: {exc}")
            row = empty_row(dataset_name, ckpt_path, "error", str(exc))
        rows.append(row)
        write_rows(rows, output_csv)

    print("\nSaturation counts:")
    for row in rows:
        if row["status"] != "success":
            print(f"{row['dataset']}: error")
            continue
        print(
            f"{row['dataset']}: saturated={row['num_saturated']}, "
            f"not_saturated={row['num_not_saturated']}, "
            f"considered={row['num_considered']}"
        )
    print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    main()
