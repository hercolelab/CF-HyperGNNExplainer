import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def load_allset_dataset(
    dataset_name: str,
    base_data_dir: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[object, torch.sparse_coo_tensor]:
    """
    Load a dataset from the AllSet collection and produce a PyG-like `Data` object
    together with a hypergraph incidence matrix `H` (sparse COO tensor).

    The function looks for a folder named `dataset_name` (case-insensitive)
    under `<project>/data/AllSet_all_raw_data/AllSet_all_raw_data/` by default.

    Returns:
        data: lightweight object with `x`, `y`, `train_mask`, `val_mask`,
              `test_mask`, and `num_nodes` attributes (compatible with the
              rest of the project).
        H: sparse incidence matrix (shape: num_nodes x num_hyperedges)
    """

    # Resolve base directory
    if base_data_dir is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        base_data_dir = os.path.join(
            project_root, "data", "AllSet_all_raw_data", "AllSet_all_raw_data"
        )

    # If the expected folder doesn't exist inside the project, try climbing
    # parent directories to find a top-level `data/AllSet_all_raw_data/...` path
    if not os.path.isdir(base_data_dir):
        cur = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        found = None
        for _ in range(6):
            candidate = os.path.join(cur, "data", "AllSet_all_raw_data", "AllSet_all_raw_data")
            if os.path.isdir(candidate):
                found = candidate
                break
            cur = os.path.dirname(cur)
        if found:
            base_data_dir = found

    candidates = [dataset_name, dataset_name.lower(), dataset_name.capitalize()]
    dataset_folder = None
    for c in candidates:
        cand = os.path.join(base_data_dir, c)
        if os.path.isdir(cand):
            dataset_folder = cand
            break

    if dataset_folder is None:
        raise FileNotFoundError(
            f"Could not find dataset '{dataset_name}' under {base_data_dir}"
        )

    # Find content and edges files
    content_path = None
    edges_path = None
    for fname in os.listdir(dataset_folder):
        if fname.endswith(".content"):
            content_path = os.path.join(dataset_folder, fname)
        if fname.endswith(".edges"):
            edges_path = os.path.join(dataset_folder, fname)

    if content_path is None:
        raise FileNotFoundError(f"No .content file found in {dataset_folder}")

    # Parse content: assume first column is an id or name, last column is label
    features = []
    labels = []
    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            if len(toks) < 2:
                continue
            # Heuristic: skip first token (id or name), take last token as label
            if len(toks) >= 2:
                feat_toks = toks[1:-1]
                if len(feat_toks) == 0:
                    # fallback: maybe file doesn't include id column
                    feat_toks = toks[0:-1]
            else:
                feat_toks = []

            feats = [float(x) for x in feat_toks]
            lbl = int(toks[-1])
            features.append(feats)
            labels.append(lbl)

    if len(features) == 0:
        raise ValueError(f"No feature rows parsed from {content_path}")

    x = torch.tensor(np.array(features, dtype=np.float32), device=device)
    y_raw = np.array(labels, dtype=np.int64)

    # Remap labels to contiguous 0..C-1
    uniques, inverse = np.unique(y_raw, return_inverse=True)
    y = torch.tensor(inverse, dtype=torch.long, device=device)

    num_nodes = x.size(0)

    # Build incidence matrix H (if edges file present)
    if edges_path and os.path.isfile(edges_path):
        rows = []
        cols_raw = []
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                node_idx = int(parts[0])
                hed_raw = int(parts[1])
                rows.append(node_idx)
                cols_raw.append(hed_raw)

        if len(rows) == 0:
            # fallback to identity hyperedges
            idx = torch.arange(num_nodes, dtype=torch.long, device=device)
            indices = torch.stack([idx, idx], dim=0)
            vals = torch.ones(num_nodes, dtype=torch.float32, device=device)
            H = torch.sparse_coo_tensor(indices, vals, (num_nodes, num_nodes), device=device).coalesce()
        else:
            unique_cols = sorted(list(set(cols_raw)))
            raw_to_local = {r: i for i, r in enumerate(unique_cols)}
            cols = [raw_to_local[r] for r in cols_raw]
            indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
            vals = torch.ones(len(rows), dtype=torch.float32, device=device)
            H = torch.sparse_coo_tensor(indices, vals, (num_nodes, len(unique_cols)), device=device).coalesce()
    else:
        # No edges file: self hyperedges
        idx = torch.arange(num_nodes, dtype=torch.long, device=device)
        indices = torch.stack([idx, idx], dim=0)
        vals = torch.ones(num_nodes, dtype=torch.float32, device=device)
        H = torch.sparse_coo_tensor(indices, vals, (num_nodes, num_nodes), device=device).coalesce()

    # Create a lightweight PyG-like object
    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.y = y
    data.num_nodes = num_nodes

    # Create deterministic stratified splits: per-class 20% test, 10% val (min 1)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    num_classes = int(y.max().item()) + 1
    for c in range(num_classes):
        idxs = (y == c).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
        if len(idxs) == 0:
            continue
        rng = np.random.default_rng(seed=42 + c)
        perm = rng.permutation(idxs)
        total = len(perm)
        n_test = max(1, int(total * 0.2))
        n_val = max(1, int(total * 0.1))
        n_train = total - n_val - n_test
        if n_train <= 0:
            n_train = 1
            if total - n_train - n_test >= 0:
                n_val = max(0, total - n_train - n_test)
            else:
                n_test = max(0, total - n_train - n_val)

        train_idxs = perm[:n_train].tolist()
        val_idxs = perm[n_train : n_train + n_val].tolist()
        test_idxs = perm[n_train + n_val :].tolist()

        train_mask[train_idxs] = True
        val_mask[val_idxs] = True
        test_mask[test_idxs] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data, H


def _find_allset_base_dir() -> str:
    """Locate the AllSet_all_raw_data/AllSet_all_raw_data folder by climbing parents."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(
        project_root, "data", "AllSet_all_raw_data", "AllSet_all_raw_data"
    )
    if os.path.isdir(candidate):
        return candidate

    cur = project_root
    for _ in range(6):
        cand = os.path.join(cur, "data", "AllSet_all_raw_data", "AllSet_all_raw_data")
        if os.path.isdir(cand):
            return cand
        cur = os.path.dirname(cur)

    return candidate


def _safe_nnzsparse(H):
    try:
        Hc = H.coalesce()
        return int(Hc.values().numel())
    except Exception:
        try:
            return int(H._nnz())
        except Exception:
            return -1


def main() -> None:
    """CLI to test either datasets or a single node for a dataset.

    Usage:
      --test dataset      Run the dataset discovery/loading checks (previous behavior)
      --test node --dataset <name> --node-id <id>
                          Print information for the given node in the named dataset
    """

    import argparse
    import traceback
    import importlib

    parser = argparse.ArgumentParser(
        description="Test AllSet and Planetoid dataset loading and inspect nodes"
    )
    parser.add_argument(
        "--test",
        choices=("dataset", "node"),
        required=True,
        help="Type of test to run: 'dataset' to enumerate/load datasets, 'node' to inspect a specific node",
    )
    parser.add_argument("--dataset", help="Dataset name (for node inspection; Planetoid or AllSet folder name)")
    parser.add_argument("--node-id", type=int, help="Node id to inspect (required for --test node)")
    parser.add_argument(
        "--base-data-dir",
        default=None,
        help="Optional path to AllSet base directory (overrides discovery)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use (auto selects CUDA if available)",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    base_dir = args.base_data_dir or _find_allset_base_dir()
    print(f"Using device: {device}")
    print(f"AllSet base dir: {base_dir}")

    if args.test == "dataset":
        # Enumerate AllSet datasets
        if os.path.isdir(base_dir):
            entries = sorted(
                [e for e in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, e))]
            )
            if not entries:
                print("No subfolders found in AllSet base dir.")
            for name in entries:
                print(f"\n== AllSet: '{name}' ==")
                try:
                    data, H = load_allset_dataset(name, base_data_dir=base_dir, device=device)
                    print(f"  num_nodes: {data.num_nodes}")
                    print(f"  x.shape: {tuple(data.x.shape)}")
                    print(f"  y.shape: {tuple(data.y.shape)}")
                    if hasattr(data, "train_mask"):
                        t = int(data.train_mask.sum().item())
                        v = int(data.val_mask.sum().item())
                        te = int(data.test_mask.sum().item())
                        print(f"  train/val/test counts: {t}/{v}/{te}")
                    print(f"  H shape: {H.shape}, nnz: {_safe_nnzsparse(H)}")
                except Exception as e:
                    print(f"  Failed to load '{name}': {e}")
                    traceback.print_exc()
        else:
            print("AllSet base directory not found; skipping AllSet tests.")

        # Planetoid enumeration
        try:
            from torch_geometric.datasets import Planetoid
            from torch_geometric.transforms import NormalizeFeatures

            have_planetoid = True
        except Exception:
            print("torch_geometric not available; skipping Planetoid tests.")
            have_planetoid = False

        if have_planetoid:
            plan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid"))
            print(f"\nPlanetoid root: {plan_root}")
            for name in ("Cora", "Citeseer", "Pubmed"):
                print(f"\n== Planetoid: '{name}' ==")
                try:
                    dataset = Planetoid(root=plan_root, name=name, transform=NormalizeFeatures())
                    data = dataset[0].to(device)
                    print(f"  num_nodes: {data.num_nodes}")
                    print(f"  x.shape: {tuple(data.x.shape)}")
                    print(f"  dataset.num_features: {dataset.num_features}")
                    print(f"  dataset.num_classes: {dataset.num_classes}")

                    # Build incidence matrix if possible
                    try:
                        try:
                            from utils import build_incidence_matrix
                        except Exception:
                            from .utils import build_incidence_matrix  # type: ignore
                    except Exception:
                        try:
                            mod = importlib.import_module("src_sparse.utils.utils")
                            build_incidence_matrix = getattr(mod, "build_incidence_matrix")
                        except Exception as exc:
                            print(f"  Could not locate build_incidence_matrix: {exc}")
                            build_incidence_matrix = None

                    if build_incidence_matrix is not None:
                        try:
                            H = build_incidence_matrix(data.edge_index, data.num_nodes, device=device)
                            print(f"  H shape: {H.shape}, nnz: {_safe_nnzsparse(H)}")
                        except Exception as exc:
                            print(f"  Failed to build incidence matrix: {exc}")
                            traceback.print_exc()
                except Exception as e:
                    print(f"  Failed to load Planetoid '{name}': {e}")
                    traceback.print_exc()

        return

    # --- Node inspection mode ---
    if args.test == "node":
        if args.dataset is None or args.node_id is None:
            parser.error("--dataset and --node-id are required when --test node")

        name = args.dataset
        node_id = int(args.node_id)

        # Planetoid case
        if name in ("Cora", "Citeseer", "Pubmed"):
            try:
                from torch_geometric.datasets import Planetoid
                from torch_geometric.transforms import NormalizeFeatures
            except Exception:
                print("torch_geometric not available; cannot inspect Planetoid dataset")
                return

            plan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid"))
            try:
                dataset = Planetoid(root=plan_root, name=name, transform=NormalizeFeatures())
                data = dataset[0].to(device)
            except Exception as exc:
                print(f"Failed to load Planetoid '{name}': {exc}")
                traceback.print_exc()
                return

            if not (0 <= node_id < int(data.num_nodes)):
                print(f"node-id {node_id} is out of range (0..{int(data.num_nodes)-1})")
                return

            print(f"\nPlanetoid dataset '{name}' — node {node_id}")
            print(f"  num_nodes: {data.num_nodes}")
            print(f"  x.shape: {tuple(data.x.shape)}")
            xnode = data.x[node_id].cpu().numpy().tolist()
            print(f"  x[{node_id}] (len={len(xnode)}), first 10: {xnode[:10]}")
            if hasattr(data, "y"):
                print(f"  y[{node_id}]: {int(data.y[node_id].item())}")
            if hasattr(data, "train_mask"):
                print(f"  train/val/test membership: {bool(data.train_mask[node_id].item())}/{bool(data.val_mask[node_id].item())}/{bool(data.test_mask[node_id].item())}")

            # Graph neighbors
            try:
                ei = data.edge_index.cpu()
                row, col = ei
                neigh1 = col[(row == node_id)].cpu().numpy().tolist()
                neigh2 = row[(col == node_id)].cpu().numpy().tolist()
                neighbors = sorted(set(neigh1 + neigh2))
                print(f"  graph neighbors (count {len(neighbors)}), first 10: {neighbors[:10]}")
            except Exception:
                print("  Could not compute graph neighbors.")

            # Hyperedges via incidence matrix
            try:
                try:
                    from utils import build_incidence_matrix
                except Exception:
                    from .utils import build_incidence_matrix  # type: ignore
                H = build_incidence_matrix(data.edge_index, data.num_nodes, device=device)
                Hc = H.coalesce()
                idxs = Hc.indices()
                rows_np = idxs[0].cpu().numpy()
                cols_np = idxs[1].cpu().numpy()
                hedges = sorted(set(cols_np[rows_np == node_id].tolist()))
                print(f"  hyperedges containing node: {len(hedges)}, first 10: {hedges[:10]}")
                nodes_in_heds = sorted(set(rows_np[np.isin(cols_np, hedges)].tolist()))
                nodes_in_heds = [int(n) for n in nodes_in_heds if int(n) != node_id]
                print(f"  nodes in same hyperedges (count {len(nodes_in_heds)}), first 10: {nodes_in_heds[:10]}")
            except Exception:
                print("  Could not compute hypergraph info.")

            return

        # AllSet dataset case
        try:
            data, H = load_allset_dataset(name, base_data_dir=base_dir, device=device)
        except Exception as exc:
            print(f"Failed to load AllSet dataset '{name}': {exc}")
            traceback.print_exc()
            return

        if not (0 <= node_id < int(data.num_nodes)):
            print(f"node-id {node_id} is out of range (0..{int(data.num_nodes)-1})")
            return

        print(f"\nAllSet dataset '{name}' — node {node_id}")
        print(f"  num_nodes: {data.num_nodes}")
        print(f"  x.shape: {tuple(data.x.shape)}")
        xnode = data.x[node_id].cpu().numpy().tolist()
        print(f"  x[{node_id}] (len={len(xnode)}), first 10: {xnode[:10]}")
        print(f"  y[{node_id}]: {int(data.y[node_id].item())}")
        if hasattr(data, "train_mask"):
            print(f"  train/val/test membership: {bool(data.train_mask[node_id].item())}/{bool(data.val_mask[node_id].item())}/{bool(data.test_mask[node_id].item())}")

        try:
            Hc = H.coalesce()
            idxs = Hc.indices()
            rows_np = idxs[0].cpu().numpy()
            cols_np = idxs[1].cpu().numpy()
            hedges = sorted(set(cols_np[rows_np == node_id].tolist()))
            print(f"  hyperedges containing node: {len(hedges)}, first 10: {hedges[:10]}")
            nodes_in_heds = sorted(set(rows_np[np.isin(cols_np, hedges)].tolist()))
            nodes_in_heds = [int(n) for n in nodes_in_heds if int(n) != node_id]
            print(f"  nodes in same hyperedges (count {len(nodes_in_heds)}), first 10: {nodes_in_heds[:10]}")
        except Exception:
            print("  Could not compute hypergraph info for AllSet dataset.")



if __name__ == "__main__":
    main()
