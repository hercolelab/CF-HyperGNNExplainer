import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from utils import graph_to_hypergraph
from utils.allset_loader import load_allset_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_SPARSE_DIR = REPO_ROOT / "src_sparse"
RESULTS_DIR = REPO_ROOT / "results" / "test_main_explain"
MANIFEST_DIR = REPO_ROOT / ".test_sparse_workflow"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"


def resolve_planetoid_root() -> Path:
    candidates = [
        SRC_SPARSE_DIR / "data" / "Planetoid",
        REPO_ROOT / "data" / "Planetoid",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def sanitize_name(text: str) -> str:
    sanitized = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            sanitized.append(ch)
        else:
            sanitized.append("_")
    result = "".join(sanitized)
    return result if result else "item"


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Training manifest not found at {MANIFEST_PATH}. Run test_train.py first."
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    manifest.setdefault("train", {})
    manifest.setdefault("explain", {})
    manifest.setdefault("artifact_paths", [])
    return manifest


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def register_artifact_path(manifest: dict[str, Any], path: Path) -> None:
    artifact_path = str(path.resolve())
    artifact_paths = set(manifest.get("artifact_paths", []))
    artifact_paths.add(artifact_path)
    manifest["artifact_paths"] = sorted(artifact_paths)


def load_dataset_and_incidence(dataset_name: str):
    device = torch.device("cpu")
    if dataset_name in {"Cora", "Citeseer", "Pubmed"}:
        dataset = Planetoid(
            root=str(resolve_planetoid_root()),
            name=dataset_name,
            transform=NormalizeFeatures(),
        )
        data = dataset[0]
        H = graph_to_hypergraph(data.edge_index, data.num_nodes, device=device)
        return data, H

    data, H = load_allset_dataset(dataset_name, device=device)
    return data, H


def choose_target_node(dataset_name: str) -> int:
    data, H = load_dataset_and_incidence(dataset_name)
    test_mask = data.test_mask.detach().cpu().bool().reshape(-1)
    test_nodes = torch.where(test_mask)[0]
    if test_nodes.numel() == 0:
        raise ValueError(f"Dataset {dataset_name} has no test nodes.")

    node_degrees = torch.sparse.sum(H, dim=1).to_dense().reshape(-1).cpu()
    viable_test_nodes = test_nodes[node_degrees[test_nodes] > 0]
    if viable_test_nodes.numel() > 0:
        return int(viable_test_nodes[0].item())
    return int(test_nodes[0].item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test explanation for the checkpoints created by test_train.py"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset subset to explain",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--n-hops", type=int, default=4)
    parser.add_argument("--beta", default="0.5")
    parser.add_argument("--cf-optimizer", choices=("SGD", "Adadelta"), default="SGD")
    parser.add_argument("--strategy", choices=("v1", "v3"), default="v1")
    parser.add_argument("--lr", default="0.1")
    parser.add_argument("--n-momentum", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    train_entries = manifest.get("train", {})

    if args.datasets is None:
        datasets = sorted(train_entries.keys())
    else:
        datasets = args.datasets

    failures: list[str] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        train_entry = train_entries.get(dataset)
        if train_entry is None:
            failures.append(dataset)
            print(f"No training entry found for {dataset}. Run test_train.py first.")
            continue
        if train_entry.get("status") != "success":
            failures.append(dataset)
            print(f"Skipping {dataset}: latest test_train.py run did not succeed.")
            continue

        checkpoint_path = Path(train_entry["checkpoint_path"])
        if not checkpoint_path.is_file():
            failures.append(dataset)
            print(f"Checkpoint missing for {dataset}: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {})
        target_node = choose_target_node(dataset)
        output_path = RESULTS_DIR / f"test_main_explain_{sanitize_name(dataset.casefold())}.pkl"

        register_artifact_path(manifest, output_path)
        explain_entry = {
            "dataset": dataset,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "output_path": str(output_path.resolve()),
            "target_node": target_node,
            "num_epochs": args.num_epochs,
            "beta": args.beta,
            "cf_optimizer": args.cf_optimizer,
            "strategy": args.strategy,
            "lr": args.lr,
            "n_momentum": args.n_momentum,
            "device": args.device,
            "status": "pending",
        }
        manifest["explain"][dataset] = explain_entry
        save_manifest(manifest)

        command = [
            sys.executable,
            str(SRC_SPARSE_DIR / "main_explain.py"),
            "--dataset",
            str(checkpoint_args.get("dataset", dataset)),
            "--target-node",
            str(target_node),
            "--n-hops",
            str(args.n_hops),
            "--beta",
            args.beta,
            "--cf-optimizer",
            args.cf_optimizer,
            "--strategy",
            args.strategy,
            "--lr",
            args.lr,
            "--n-momentum",
            str(args.n_momentum),
            "--num-epochs",
            str(args.num_epochs),
            "--dropout",
            str(checkpoint_args.get("dropout", 0.5)),
            "--nhid",
            str(checkpoint_args.get("hidden", 64)),
            "--nout",
            str(checkpoint_args.get("out_hidden", 32)),
            "--ckpt-path",
            str(checkpoint_path),
            "--device",
            args.device,
            "--output-path",
            str(output_path),
        ]

        print(f"\n== Explaining {dataset} (target node {target_node}) ==")
        print("Command:", " ".join(shlex.quote(part) for part in command))
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)

        explain_entry["returncode"] = completed.returncode
        explain_entry["status"] = (
            "success"
            if completed.returncode == 0 and output_path.is_file()
            else "failed"
        )
        manifest["explain"][dataset] = explain_entry
        save_manifest(manifest)

        if explain_entry["status"] == "success":
            print(f"Explanation output ready at {output_path}")
        else:
            failures.append(dataset)
            print(f"Explanation failed for {dataset}")

    print(f"\nManifest updated at {MANIFEST_PATH}")
    if failures:
        print("Explanation failed for:", ", ".join(failures))
        raise SystemExit(1)

    print(f"Explained {len(datasets)} dataset(s) successfully.")


if __name__ == "__main__":
    main()
