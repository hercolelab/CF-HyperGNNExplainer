import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


_ALLSET_LE_DATASETS = frozenset(
    {
        "20newsw100",
        "modelnet40",
        "mushroom",
        "ntu2012",
        "zoo",
    }
)

_EXPECTED_ALLSET_STRUCTURE_STATS = {
    "20newsw100": {
        "num_nodes": 16242,
        "num_hyperedges": 100,
        "min_node_degree": 1,
        "max_node_degree": 44,
        "min_edge_degree": 29,
        "max_edge_degree": 2241,
    },
    "modelnet40": {
        "num_nodes": 12311,
        "num_hyperedges": 12311,
        "min_node_degree": 1,
        "max_node_degree": 30,
        "min_edge_degree": 5,
        "max_edge_degree": 5,
    },
    "mushroom": {
        "num_nodes": 8124,
        "num_hyperedges": 298,
        "min_node_degree": 5,
        "max_node_degree": 5,
        "min_edge_degree": 1,
        "max_edge_degree": 1808,
    },
    "ntu2012": {
        "num_nodes": 2012,
        "num_hyperedges": 2012,
        "min_node_degree": 1,
        "max_node_degree": 19,
        "min_edge_degree": 5,
        "max_edge_degree": 5,
    },
    "yelp": {
        "num_nodes": 50758,
        "num_hyperedges": 679302,
        "min_node_degree": 1,
        "max_node_degree": 7855,
        "min_edge_degree": 2,
        "max_edge_degree": 2838,
    },
    "house-committees": {
        "num_nodes": 1290,
        "num_hyperedges": 341,
        "min_node_degree": 0,
        "max_node_degree": 44,
        "min_edge_degree": 1,
        "max_edge_degree": 81,
    },
    "walmart-trips": {
        "num_nodes": 88860,
        "num_hyperedges": 69906,
        "min_node_degree": 0,
        "max_node_degree": 5733,
        "min_edge_degree": 2,
        "max_edge_degree": 25,
    },
    "zoo": {
        "num_nodes": 101,
        "num_hyperedges": 43,
        "min_node_degree": 17,
        "max_node_degree": 17,
        "min_edge_degree": 1,
        "max_edge_degree": 93,
    },
}


def _normalize_allset_dataset_name(name: str) -> str:
    normalized = name.strip().casefold().replace("_", "-").replace(" ", "-")
    aliases = {
        "20news": "20newsw100",
        "20news-w100": "20newsw100",
        "modelnet": "modelnet40",
        "mushrooms": "mushroom",
        "walmart": "walmart-trips",
        "walmart-trips-100": "walmart-trips",
        "house": "house-committees",
        "house-committees-100": "house-committees",
    }
    return aliases.get(normalized, normalized)


def _resolve_allset_base_data_dir(base_data_dir: Optional[str]) -> str:
    if base_data_dir is not None and os.path.isdir(base_data_dir):
        return os.path.abspath(base_data_dir)

    candidate = _find_allset_base_dir()
    if os.path.isdir(candidate):
        return candidate

    if base_data_dir is not None:
        return os.path.abspath(base_data_dir)

    return candidate


def _resolve_allset_dataset_folder(base_data_dir: str, dataset_name: str) -> Tuple[str, str]:
    base_data_dir = os.path.abspath(base_data_dir)
    normalized_target = _normalize_allset_dataset_name(dataset_name)

    if os.path.isdir(base_data_dir):
        base_name = os.path.basename(os.path.normpath(base_data_dir))
        if _normalize_allset_dataset_name(base_name) == normalized_target:
            return base_data_dir, base_name

    if not os.path.isdir(base_data_dir):
        raise FileNotFoundError(
            f"Could not find AllSet base directory at {base_data_dir}"
        )

    for entry in sorted(os.listdir(base_data_dir)):
        candidate = os.path.join(base_data_dir, entry)
        if not os.path.isdir(candidate):
            continue
        if _normalize_allset_dataset_name(entry) == normalized_target:
            return candidate, entry

    raise FileNotFoundError(
        f"Could not find dataset '{dataset_name}' under {base_data_dir}"
    )


def _build_deterministic_stratified_masks(
    labels: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    if labels.numel() == 0:
        return train_mask, val_mask, test_mask

    num_classes = int(labels.max().item()) + 1
    for class_idx in range(num_classes):
        idxs = (
            (labels == class_idx)
            .nonzero(as_tuple=False)
            .view(-1)
            .cpu()
            .numpy()
            .tolist()
        )
        if not idxs:
            continue

        rng = np.random.default_rng(seed=42 + class_idx)
        perm = rng.permutation(idxs)
        total = len(perm)
        n_test = max(1, int(total * 0.2))
        n_val = max(1, int(total * 0.1))
        n_train = total - n_val - n_test
        if n_train <= 0:
            n_train = 1
            remaining = total - n_train
            n_val = min(n_val, remaining)
            n_test = max(0, remaining - n_val)

        train_mask[perm[:n_train].tolist()] = True
        val_mask[perm[n_train : n_train + n_val].tolist()] = True
        test_mask[perm[n_train + n_val :].tolist()] = True

    return train_mask, val_mask, test_mask


def _move_simple_data_to_device(
    data: object,
    H: torch.Tensor,
    device: torch.device,
) -> Tuple[object, torch.Tensor]:
    for attr in ("x", "y", "train_mask", "val_mask", "test_mask"):
        if hasattr(data, attr):
            setattr(data, attr, getattr(data, attr).to(device))

    H = H.coalesce()
    if H.device != device:
        H = H.to(device)

    return data, H


def _looks_like_allset_le_dataset(content_path: str, edges_path: str) -> bool:
    try:
        idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
        idx_features_labels = np.atleast_2d(idx_features_labels)
        if idx_features_labels.shape[1] < 3:
            return False

        idx = np.asarray(idx_features_labels[:, 0], dtype=np.int64)
        if len(np.unique(idx)) != len(idx):
            return False

        edges_unordered = np.genfromtxt(edges_path, dtype=np.int64)
        if np.size(edges_unordered) == 0:
            return False
        edges_unordered = np.atleast_2d(edges_unordered)
        if edges_unordered.shape[1] != 2:
            return False

        idx_map = {raw_id: local_id for local_id, raw_id in enumerate(idx.tolist())}
        mapped_edges = []
        for raw_id in edges_unordered.reshape(-1).tolist():
            raw_id = int(raw_id)
            if raw_id not in idx_map:
                return False
            mapped_edges.append(idx_map[raw_id])

        edge_index = np.asarray(mapped_edges, dtype=np.int64).reshape(edges_unordered.shape).T
        if edge_index.shape[0] != 2:
            return False

        if int(edge_index[0].max()) != int(edge_index[1].min()) - 1:
            return False

        return len(np.unique(edge_index)) == int(edge_index.max()) + 1
    except Exception:
        return False


def _load_allset_le_dataset(
    dataset_folder: str,
    dataset_name: str,
    device: torch.device,
) -> Tuple[object, torch.Tensor]:
    content_path = os.path.join(dataset_folder, f"{dataset_name}.content")
    edges_path = os.path.join(dataset_folder, f"{dataset_name}.edges")

    idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
    idx_features_labels = np.atleast_2d(idx_features_labels)
    if idx_features_labels.shape[1] < 3:
        raise ValueError(
            f"Invalid LE dataset content format for {dataset_name}: expected id, features, label"
        )

    features = np.asarray(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels_raw = np.asarray(idx_features_labels[:, -1], dtype=np.float64).astype(np.int64)
    idx = np.asarray(idx_features_labels[:, 0], dtype=np.int64)
    idx_map = {raw_id: local_id for local_id, raw_id in enumerate(idx.tolist())}

    edges_unordered = np.genfromtxt(edges_path, dtype=np.int64)
    if np.size(edges_unordered) == 0:
        raise ValueError(f"No node-hyperedge pairs found in {edges_path}")
    edges_unordered = np.atleast_2d(edges_unordered)
    if edges_unordered.shape[1] != 2:
        raise ValueError(
            f"Invalid LE dataset edge format for {dataset_name}: expected two columns"
        )

    mapped_edges = np.asarray(
        [idx_map[int(raw_id)] for raw_id in edges_unordered.reshape(-1).tolist()],
        dtype=np.int64,
    ).reshape(edges_unordered.shape)

    edge_index = mapped_edges.T

    # Official AllSet preprocessing treats .content as a joint node/hyperedge id space.
    if int(edge_index[0].max()) != int(edge_index[1].min()) - 1:
        raise ValueError(
            f"Expected consecutive node and hyperedge id blocks for LE dataset {dataset_name}"
        )
    if len(np.unique(edge_index)) != int(edge_index.max()) + 1:
        raise ValueError(
            f"Expected consecutive ids after remapping LE dataset {dataset_name}"
        )

    num_nodes = int(edge_index[0].max()) + 1
    num_hyperedges = int(edge_index[1].max()) - num_nodes + 1

    x = torch.tensor(features[:num_nodes], dtype=torch.float32, device=device)
    label_values = labels_raw[:num_nodes]
    _, inverse = np.unique(label_values, return_inverse=True)
    y = torch.tensor(inverse, dtype=torch.long, device=device)

    rows = edge_index[0]
    cols = edge_index[1] - num_nodes
    indices = torch.from_numpy(
        np.vstack([rows, cols]).astype(np.int64, copy=False)
    ).to(device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    H = torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_hyperedges),
        device=device,
    ).coalesce()

    train_mask, val_mask, test_mask = _build_deterministic_stratified_masks(
        y,
        num_nodes,
        device,
    )

    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.y = y
    data.num_nodes = num_nodes
    data.n_x = num_nodes
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.train_percent = float(train_mask.sum().item()) / max(1, num_nodes)
    data.num_hyperedges = int(num_hyperedges)

    return data, H


def load_allset_dataset(
    dataset_name: str,
    base_data_dir: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[object, torch.sparse_coo_tensor]:
    """
    Load a dataset from the AllSet collection and produce a PyG-like `Data` object
    together with a hypergraph incidence matrix `H` (sparse COO tensor).

    The function resolves dataset names case-insensitively and dispatches to the
    correct parser for each raw format family used in the AllSet paper. In
    particular, datasets such as Zoo, Mushroom, ModelNet40, NTU2012, and
    20newsW100 are parsed with the original shared node/hyperedge id convention
    from the official AllSet codebase so their statistics match the paper.

    Returns:
        data: lightweight object with `x`, `y`, `train_mask`, `val_mask`,
              `test_mask`, and `num_nodes` attributes (compatible with the
              rest of the project).
        H: sparse incidence matrix (shape: num_nodes x num_hyperedges)
    """

    base_data_dir = _resolve_allset_base_data_dir(base_data_dir)
    dataset_folder, canonical_name = _resolve_allset_dataset_folder(
        base_data_dir,
        dataset_name,
    )
    normalized_name = _normalize_allset_dataset_name(canonical_name)

    if normalized_name == "yelp":
        data, H = load_yelp_dataset(path=dataset_folder, dataset=canonical_name)
        return _move_simple_data_to_device(data, H, device)

    if normalized_name == "house-committees":
        data, H = load_house_committees(path=dataset_folder, dataset=canonical_name)
        return _move_simple_data_to_device(data, H, device)

    if normalized_name == "walmart-trips":
        data, H = load_walmart_trips(path=dataset_folder, dataset=canonical_name)
        return _move_simple_data_to_device(data, H, device)

    if normalized_name in _ALLSET_LE_DATASETS:
        return _load_allset_le_dataset(dataset_folder, canonical_name, device)

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

    if edges_path and os.path.isfile(edges_path) and _looks_like_allset_le_dataset(
        content_path,
        edges_path,
    ):
        return _load_allset_le_dataset(dataset_folder, canonical_name, device)

    # Parse content: assume first column is an id or name, last column is label
    features = []
    labels = []
    row_ids = []
    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            if len(toks) < 2:
                continue
            try:
                row_ids.append(int(float(toks[0])))
            except (TypeError, ValueError):
                row_ids.append(None)
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
    _, inverse = np.unique(y_raw, return_inverse=True)
    y = torch.tensor(inverse, dtype=torch.long, device=device)

    num_nodes = x.size(0)
    row_id_to_local = None
    if all(row_id is not None for row_id in row_ids) and len(set(row_ids)) == len(row_ids):
        row_id_to_local = {int(row_id): idx for idx, row_id in enumerate(row_ids)}

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
                raw_node_idx = int(parts[0])
                node_idx = (
                    row_id_to_local.get(raw_node_idx, raw_node_idx)
                    if row_id_to_local is not None
                    else raw_node_idx
                )
                hed_raw = int(parts[1])
                if not (0 <= node_idx < num_nodes):
                    raise ValueError(
                        f"Encountered node id {raw_node_idx} outside 0..{num_nodes - 1} while loading {canonical_name}"
                    )
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

    train_mask, val_mask, test_mask = _build_deterministic_stratified_masks(
        y,
        num_nodes,
        device,
    )

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.n_x = num_nodes
    data.train_percent = float(train_mask.sum().item()) / max(1, num_nodes)
    data.num_hyperedges = int(H.shape[1])

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


def _summarize_degree_statistics(H: Tensor):
    node_degrees = torch.sparse.sum(H, dim=1).to_dense().reshape(-1)
    edge_degrees = torch.sparse.sum(H, dim=0).to_dense().reshape(-1)

    node_degrees_np = node_degrees.detach().cpu().numpy()
    edge_degrees_np = edge_degrees.detach().cpu().numpy()

    return {
        "min_node_degree": int(node_degrees.min().item()),
        "max_node_degree": int(node_degrees.max().item()),
        "median_node_degree": float(np.median(node_degrees_np)),
        "avg_node_degree": float(node_degrees.float().mean().item()),
        "min_edge_degree": int(edge_degrees.min().item()),
        "max_edge_degree": int(edge_degrees.max().item()),
        "median_edge_degree": float(np.median(edge_degrees_np)),
        "avg_edge_degree": float(edge_degrees.float().mean().item()),
    }


def _summarize_structure_statistics(data: object, H: Tensor) -> dict[str, int]:
    degree_stats = _summarize_degree_statistics(H)
    return {
        "num_nodes": int(getattr(data, "num_nodes")),
        "num_hyperedges": int(H.shape[1]),
        "min_node_degree": degree_stats["min_node_degree"],
        "max_node_degree": degree_stats["max_node_degree"],
        "min_edge_degree": degree_stats["min_edge_degree"],
        "max_edge_degree": degree_stats["max_edge_degree"],
    }


def _validate_allset_structure_statistics(
    dataset_name: str,
    data: object,
    H: Tensor,
) -> list[str]:
    expected = _EXPECTED_ALLSET_STRUCTURE_STATS.get(
        _normalize_allset_dataset_name(dataset_name)
    )
    if expected is None:
        return []

    actual = _summarize_structure_statistics(data, H)
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value}, got {actual_value}"
            )
    return mismatches


def _iter_cornell_hyperedges(
    hyperedges_path: str,
    num_nodes: int,
) -> list[int]:
    with open(hyperedges_path, 'r', encoding='utf-8') as f:
        for line in f:
            members = sorted(
                {
                    int(part)
                    for part in line.strip().split(',')
                    if part.strip()
                }
            )
            members = [member for member in members if 0 <= member < num_nodes]
            if not members:
                continue
            yield members


def load_citation_dataset(path='../data/AllSet_all_raw_data/AllSet_all_raw_data/cocitation', dataset = 'cora', train_percent = 0.5):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    import pickle

    def _resolve_root_dir(raw_path: str) -> str:
        if os.path.isdir(raw_path):
            return raw_path
        base_dir = _find_allset_base_dir()
        candidate = os.path.join(base_dir, os.path.basename(os.path.normpath(raw_path)))
        if os.path.isdir(candidate):
            return candidate
        return raw_path

    def _build_fallback_masks(labels_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        num_classes = int(labels_tensor.max().item()) + 1
        for class_idx in range(num_classes):
            idxs = (labels_tensor == class_idx).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
            if not idxs:
                continue
            rng = np.random.default_rng(seed=42 + class_idx)
            perm = rng.permutation(idxs)
            total = len(perm)
            n_test = max(1, int(total * 0.2))
            n_val = max(1, int(total * 0.1))
            n_train = total - n_val - n_test
            if n_train <= 0:
                n_train = 1
                remaining = total - n_train
                n_val = min(n_val, remaining)
                n_test = max(0, remaining - n_val)

            train_mask[perm[:n_train].tolist()] = True
            val_mask[perm[n_train : n_train + n_val].tolist()] = True
            test_mask[perm[n_train + n_val :].tolist()] = True

        return train_mask, val_mask, test_mask

    path = _resolve_root_dir(path)

    dataset_dir = None
    if os.path.isfile(os.path.join(path, 'features.pickle')):
        dataset_dir = path
    else:
        for candidate_name in (dataset, dataset.lower(), dataset.capitalize()):
            candidate_dir = os.path.join(path, candidate_name)
            if os.path.isdir(candidate_dir):
                dataset_dir = candidate_dir
                break

    if dataset_dir is None:
        raise FileNotFoundError(f"Could not find pickle-backed dataset '{dataset}' under {path}")

    print(f"Loading pickle-backed hypergraph dataset from {os.path.basename(dataset_dir)}")

    with open(os.path.join(dataset_dir, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
    if hasattr(features, 'toarray'):
        features = features.toarray()
    else:
        features = np.asarray(features)
    features = np.asarray(features, dtype=np.float32)

    with open(os.path.join(dataset_dir, 'labels.pickle'), 'rb') as f:
        labels = np.asarray(pickle.load(f), dtype=np.int64)

    num_nodes, feature_dim = features.shape
    if num_nodes != int(labels.shape[0]):
        raise ValueError(
            f"Feature/label size mismatch for {dataset_dir}: {num_nodes} nodes vs {labels.shape[0]} labels"
        )
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    _, remapped_labels = np.unique(labels, return_inverse=True)
    x = torch.from_numpy(features)
    y = torch.from_numpy(remapped_labels.astype(np.int64, copy=False))

    with open(os.path.join(dataset_dir, 'hypergraph.pickle'), 'rb') as f:
        hypergraph = pickle.load(f)

    def _hyperedge_sort_key(value):
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    rows = []
    cols = []
    hyperedge_count = 0
    for hyperedge_id in sorted(hypergraph.keys(), key=_hyperedge_sort_key):
        members = sorted(
            {
                int(node_idx)
                for node_idx in hypergraph[hyperedge_id]
                if 0 <= int(node_idx) < num_nodes
            }
        )
        if not members:
            continue
        rows.extend(members)
        cols.extend([hyperedge_count] * len(members))
        hyperedge_count += 1

    if rows:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(rows), dtype=torch.float32)
        H = torch.sparse_coo_tensor(indices, values, (num_nodes, hyperedge_count)).coalesce()
    else:
        idx = torch.arange(num_nodes, dtype=torch.long)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(num_nodes, dtype=torch.float32)
        H = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    split_dir = os.path.join(dataset_dir, 'splits')
    split_path = None
    if os.path.isdir(split_dir):
        split_files = sorted(
            [name for name in os.listdir(split_dir) if name.endswith('.pickle')],
            key=lambda name: int(os.path.splitext(name)[0]),
        )
        if split_files:
            split_path = os.path.join(split_dir, split_files[0])

    if split_path is not None:
        with open(split_path, 'rb') as f:
            split = pickle.load(f)

        train_indices = sorted(
            {
                int(node_idx)
                for node_idx in split.get('train', [])
                if 0 <= int(node_idx) < num_nodes
            }
        )
        test_indices = sorted(
            {
                int(node_idx)
                for node_idx in split.get('test', [])
                if 0 <= int(node_idx) < num_nodes
            }
        )

        val_indices = []
        if len(train_indices) > 1:
            rng = np.random.default_rng(seed=42)
            perm = rng.permutation(train_indices)
            val_size = min(max(1, int(round(len(train_indices) * 0.1))), len(train_indices) - 1)
            val_indices = perm[:val_size].tolist()
            train_indices = perm[val_size:].tolist()

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    else:
        train_mask, val_mask, test_mask = _build_fallback_masks(y)

    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.y = y
    data.num_nodes = num_nodes
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.n_x = num_nodes
    data.train_percent = float(train_mask.sum().item()) / max(1, num_nodes)
    data.num_hyperedges = int(H.shape[1])

    print(f"number of hyperedges: {data.num_hyperedges}")
    return data, H

def load_yelp_dataset(path='../data/AllSet_all_raw_data/AllSet_all_raw_data/yelp', dataset = 'yelp', 
        name_dictionary_size = 1000,
        train_percent = 0.5):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding. 
        - city, in one-hot coding. 
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    import re
    import unicodedata
    from collections import Counter

    import pandas as pd

    def _resolve_dataset_dir(raw_path: str) -> str:
        if os.path.isfile(os.path.join(raw_path, 'yelp_restaurant_latlong.csv')):
            return raw_path
        base_dir = _find_allset_base_dir()
        candidate = os.path.join(base_dir, os.path.basename(os.path.normpath(raw_path)))
        if os.path.isfile(os.path.join(candidate, 'yelp_restaurant_latlong.csv')):
            return candidate
        nested_candidate = os.path.join(raw_path, dataset)
        if os.path.isfile(os.path.join(nested_candidate, 'yelp_restaurant_latlong.csv')):
            return nested_candidate
        return raw_path

    def _build_masks(labels_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        num_classes = int(labels_tensor.max().item()) + 1
        for class_idx in range(num_classes):
            idxs = (labels_tensor == class_idx).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
            if not idxs:
                continue
            rng = np.random.default_rng(seed=42 + class_idx)
            perm = rng.permutation(idxs)
            total = len(perm)
            n_test = max(1, int(total * 0.2))
            n_val = max(1, int(total * 0.1))
            n_train = total - n_val - n_test
            if n_train <= 0:
                n_train = 1
                remaining = total - n_train
                n_val = min(n_val, remaining)
                n_test = max(0, remaining - n_val)

            train_mask[perm[:n_train].tolist()] = True
            val_mask[perm[n_train : n_train + n_val].tolist()] = True
            test_mask[perm[n_train + n_val :].tolist()] = True

        return train_mask, val_mask, test_mask

    path = _resolve_dataset_dir(path)
    print(f"Loading hypergraph dataset from {os.path.basename(path)}")

    latlong = pd.read_csv(
        os.path.join(path, 'yelp_restaurant_latlong.csv'),
        dtype=np.float32,
    ).to_numpy(dtype=np.float32, copy=True)

    loc = pd.read_csv(
        os.path.join(path, 'yelp_restaurant_locations.csv'),
        dtype=np.int64,
    )
    state_values = loc['state_int'].to_numpy(dtype=np.int64, copy=True) - 1
    city_values = loc['city_int'].to_numpy(dtype=np.int64, copy=True) - 1
    num_nodes = int(loc.shape[0])

    names = (
        pd.read_csv(os.path.join(path, 'yelp_restaurant_name.csv'), keep_default_na=False)['name']
        .fillna('')
        .astype(str)
        .tolist()
    )
    raw_labels = pd.read_csv(
        os.path.join(path, 'yelp_restaurant_business_stars.csv'),
        dtype=np.int64,
    )['business_stars'].to_numpy(dtype=np.int64, copy=True)

    if num_nodes != int(latlong.shape[0]) or num_nodes != int(len(names)) or num_nodes != int(raw_labels.shape[0]):
        raise ValueError(
            f"Yelp metadata size mismatch: latlong={latlong.shape[0]}, locations={num_nodes}, "
            f"names={len(names)}, labels={raw_labels.shape[0]}"
        )

    token_pattern = re.compile(r'[a-z0-9]+')
    stop_words = {
        'a', 'an', 'and', 'at', 'bar', 'cafe', 'co', 'de', 'for', 'grill', 'in', 'la',
        'llc', 'of', 'on', 'restaurant', 'shop', 'the', 'to', 'with'
    }
    tokenized_names = []
    token_counts = Counter()
    for name_text in names:
        normalized = unicodedata.normalize('NFKD', name_text).encode('ascii', 'ignore').decode('ascii').lower()
        tokens = [
            token
            for token in token_pattern.findall(normalized)
            if len(token) > 1 and token not in stop_words
        ]
        tokenized_names.append(tokens)
        token_counts.update(tokens)

    vocab_tokens = [token for token, _ in token_counts.most_common(max(0, int(name_dictionary_size)))]
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    num_states = int(state_values.max()) + 1 if num_nodes > 0 else 0
    num_cities = int(city_values.max()) + 1 if num_nodes > 0 else 0
    bow_dim = len(vocab)
    feature_dim = 2 + num_states + num_cities + bow_dim
    features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    features[:, :2] = latlong
    features[np.arange(num_nodes), 2 + state_values] = 1.0
    city_offset = 2 + num_states
    features[np.arange(num_nodes), city_offset + city_values] = 1.0
    bow_offset = city_offset + num_cities
    for row_idx, tokens in enumerate(tokenized_names):
        for token in tokens:
            col_idx = vocab.get(token)
            if col_idx is not None:
                features[row_idx, bow_offset + col_idx] += 1.0

    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    _, remapped_labels = np.unique(raw_labels, return_inverse=True)
    x = torch.from_numpy(features)
    y = torch.from_numpy(remapped_labels.astype(np.int64, copy=False))

    incidence = pd.read_csv(
        os.path.join(path, 'yelp_restaurant_incidence_H.csv'),
        usecols=['node', 'he', 'val'],
        dtype={'node': np.int64, 'he': np.int64, 'val': np.float32},
    )
    node_ids = incidence['node'].to_numpy(dtype=np.int64, copy=True) - 1
    hyperedge_ids = incidence['he'].to_numpy(dtype=np.int64, copy=True)
    values = incidence['val'].fillna(1.0).to_numpy(dtype=np.float32, copy=True)

    valid_mask = (node_ids >= 0) & (node_ids < num_nodes)
    node_ids = node_ids[valid_mask]
    hyperedge_ids = hyperedge_ids[valid_mask]
    values = values[valid_mask]

    if node_ids.size == 0:
        idx = torch.arange(num_nodes, dtype=torch.long)
        indices = torch.stack([idx, idx], dim=0)
        H = torch.sparse_coo_tensor(
            indices,
            torch.ones(num_nodes, dtype=torch.float32),
            (num_nodes, num_nodes),
        ).coalesce()
    else:
        unique_hyperedges, local_hyperedge_ids = np.unique(hyperedge_ids, return_inverse=True)
        indices = torch.from_numpy(
            np.vstack([node_ids, local_hyperedge_ids]).astype(np.int64, copy=False)
        )
        H = torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(values),
            (num_nodes, int(unique_hyperedges.size)),
        ).coalesce()

    train_mask, val_mask, test_mask = _build_masks(y)

    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.y = y
    data.num_nodes = num_nodes
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.n_x = num_nodes
    data.train_percent = float(train_mask.sum().item()) / max(1, num_nodes)
    data.num_hyperedges = int(H.shape[1])

    return data, H


def load_house_committees(path='../data/AllSet_all_raw_data/AllSet_all_raw_data/house-committees', dataset = 'house-committees', train_percent = 0.5):
    '''
    Load the house-committees hypergraph described by line-based node labels,
    node names, and comma-separated hyperedges.
    '''

    def _resolve_dataset_dir(raw_path: str) -> str:
        expected = f'node-labels-{dataset}.txt'
        if os.path.isfile(os.path.join(raw_path, expected)):
            return raw_path
        base_dir = _find_allset_base_dir()
        candidate = os.path.join(base_dir, os.path.basename(os.path.normpath(raw_path)))
        if os.path.isfile(os.path.join(candidate, expected)):
            return candidate
        nested_candidate = os.path.join(raw_path, dataset)
        if os.path.isfile(os.path.join(nested_candidate, expected)):
            return nested_candidate
        return raw_path

    def _build_masks(labels_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        num_classes = int(labels_tensor.max().item()) + 1
        for class_idx in range(num_classes):
            idxs = (labels_tensor == class_idx).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
            if not idxs:
                continue
            rng = np.random.default_rng(seed=42 + class_idx)
            perm = rng.permutation(idxs)
            total = len(perm)
            n_test = max(1, int(total * 0.2))
            n_val = max(1, int(total * 0.1))
            n_train = total - n_val - n_test
            if n_train <= 0:
                n_train = 1
                remaining = total - n_train
                n_val = min(n_val, remaining)
                n_test = max(0, remaining - n_val)

            train_mask[perm[:n_train].tolist()] = True
            val_mask[perm[n_train : n_train + n_val].tolist()] = True
            test_mask[perm[n_train + n_val :].tolist()] = True

        return train_mask, val_mask, test_mask

    path = _resolve_dataset_dir(path)
    print(f"Loading hypergraph dataset from {os.path.basename(path)}")

    labels_path = os.path.join(path, f'node-labels-{dataset}.txt')
    names_path = os.path.join(path, f'node-names-{dataset}.txt')
    hyperedges_path = os.path.join(path, f'hyperedges-{dataset}.txt')

    raw_label_keys = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [part.strip() for part in line.strip().split(',') if part.strip()]
            label_key = tuple(sorted(int(part) for part in parts))
            raw_label_keys.append(label_key if label_key else (-1,))

    node_names = []
    with open(names_path, 'r', encoding='utf-8') as f:
        for line in f:
            node_names.append(line.strip())

    num_nodes = len(raw_label_keys)
    if len(node_names) != num_nodes:
        raise ValueError(
            f"Node-name count mismatch for {dataset}: {len(node_names)} names vs {num_nodes} labels"
        )

    rows = []
    cols = []
    degree = np.zeros(num_nodes, dtype=np.float32)
    size_sum = np.zeros(num_nodes, dtype=np.float32)
    min_size = np.full(num_nodes, np.inf, dtype=np.float32)
    max_size = np.zeros(num_nodes, dtype=np.float32)
    num_hyperedges = 0

    for members in _iter_cornell_hyperedges(hyperedges_path, num_nodes):
        he_size = float(len(members))
        rows.extend(members)
        cols.extend([num_hyperedges] * len(members))
        degree[members] += 1.0
        size_sum[members] += he_size
        min_size[members] = np.minimum(min_size[members], he_size)
        max_size[members] = np.maximum(max_size[members], he_size)
        num_hyperedges += 1

    if rows:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(rows), dtype=torch.float32)
        H = torch.sparse_coo_tensor(indices, values, (num_nodes, num_hyperedges)).coalesce()
    else:
        idx = torch.arange(num_nodes, dtype=torch.long)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(num_nodes, dtype=torch.float32)
        H = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    avg_size = np.divide(size_sum, np.maximum(degree, 1.0), out=np.zeros_like(size_sum), where=degree > 0)
    min_size[~np.isfinite(min_size)] = 0.0
    name_word_count = np.array([len([part for part in name.split() if part]) for name in node_names], dtype=np.float32)
    name_char_count = np.array([len(name) for name in node_names], dtype=np.float32)

    features = np.stack(
        [degree, avg_size, min_size, max_size, name_word_count, name_char_count],
        axis=1,
    ).astype(np.float32, copy=False)
    scale = np.maximum(features.max(axis=0, keepdims=True), 1.0)
    features = features / scale

    unique_label_keys = sorted(set(raw_label_keys))
    label_to_idx = {label_key: idx for idx, label_key in enumerate(unique_label_keys)}
    y = torch.tensor([label_to_idx[label_key] for label_key in raw_label_keys], dtype=torch.long)
    x = torch.from_numpy(features)

    print(f"number of nodes:{num_nodes}, feature dimension: {features.shape[1]}")

    train_mask, val_mask, test_mask = _build_masks(y)

    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.y = y
    data.num_nodes = num_nodes
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.n_x = num_nodes
    data.train_percent = float(train_mask.sum().item()) / max(1, num_nodes)
    data.num_hyperedges = int(H.shape[1])

    return data, H


def load_walmart_trips(path='../data/AllSet_all_raw_data/AllSet_all_raw_data/walmart-trips', dataset = 'walmart-trips', train_percent = 0.5):
    '''
    Load the walmart-trips hypergraph described by line-based node labels and
    comma-separated hyperedges.
    '''

    def _resolve_dataset_dir(raw_path: str) -> str:
        expected = f'node-labels-{dataset}.txt'
        if os.path.isfile(os.path.join(raw_path, expected)):
            return raw_path
        base_dir = _find_allset_base_dir()
        candidate = os.path.join(base_dir, os.path.basename(os.path.normpath(raw_path)))
        if os.path.isfile(os.path.join(candidate, expected)):
            return candidate
        nested_candidate = os.path.join(raw_path, dataset)
        if os.path.isfile(os.path.join(nested_candidate, expected)):
            return nested_candidate
        return raw_path

    def _build_masks(labels_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        num_classes = int(labels_tensor.max().item()) + 1
        for class_idx in range(num_classes):
            idxs = (labels_tensor == class_idx).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
            if not idxs:
                continue
            rng = np.random.default_rng(seed=42 + class_idx)
            perm = rng.permutation(idxs)
            total = len(perm)
            n_test = max(1, int(total * 0.2))
            n_val = max(1, int(total * 0.1))
            n_train = total - n_val - n_test
            if n_train <= 0:
                n_train = 1
                remaining = total - n_train
                n_val = min(n_val, remaining)
                n_test = max(0, remaining - n_val)

            train_mask[perm[:n_train].tolist()] = True
            val_mask[perm[n_train : n_train + n_val].tolist()] = True
            test_mask[perm[n_train + n_val :].tolist()] = True

        return train_mask, val_mask, test_mask

    path = _resolve_dataset_dir(path)
    print(f"Loading hypergraph dataset from {os.path.basename(path)}")

    labels_path = os.path.join(path, f'node-labels-{dataset}.txt')
    hyperedges_path = os.path.join(path, f'hyperedges-{dataset}.txt')

    raw_label_keys = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [part.strip() for part in line.strip().split(',') if part.strip()]
            label_key = tuple(sorted(int(part) for part in parts))
            raw_label_keys.append(label_key if label_key else (-1,))

    num_nodes = len(raw_label_keys)
    rows = []
    cols = []
    degree = np.zeros(num_nodes, dtype=np.float32)
    size_sum = np.zeros(num_nodes, dtype=np.float32)
    size_sq_sum = np.zeros(num_nodes, dtype=np.float32)
    min_size = np.full(num_nodes, np.inf, dtype=np.float32)
    max_size = np.zeros(num_nodes, dtype=np.float32)
    num_hyperedges = 0

    for members in _iter_cornell_hyperedges(hyperedges_path, num_nodes):
        he_size = float(len(members))
        rows.extend(members)
        cols.extend([num_hyperedges] * len(members))
        degree[members] += 1.0
        size_sum[members] += he_size
        size_sq_sum[members] += he_size * he_size
        min_size[members] = np.minimum(min_size[members], he_size)
        max_size[members] = np.maximum(max_size[members], he_size)
        num_hyperedges += 1

    if rows:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(rows), dtype=torch.float32)
        H = torch.sparse_coo_tensor(indices, values, (num_nodes, num_hyperedges)).coalesce()
    else:
        idx = torch.arange(num_nodes, dtype=torch.long)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(num_nodes, dtype=torch.float32)
        H = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    avg_size = np.divide(size_sum, np.maximum(degree, 1.0), out=np.zeros_like(size_sum), where=degree > 0)
    rms_size = np.sqrt(
        np.divide(size_sq_sum, np.maximum(degree, 1.0), out=np.zeros_like(size_sq_sum), where=degree > 0)
    )
    min_size[~np.isfinite(min_size)] = 0.0
    bias = np.ones(num_nodes, dtype=np.float32)

    features = np.stack(
        [degree, avg_size, rms_size, min_size, max_size, bias],
        axis=1,
    ).astype(np.float32, copy=False)
    scale = np.maximum(features.max(axis=0, keepdims=True), 1.0)
    features = features / scale

    unique_label_keys = sorted(set(raw_label_keys))
    label_to_idx = {label_key: idx for idx, label_key in enumerate(unique_label_keys)}
    y = torch.tensor([label_to_idx[label_key] for label_key in raw_label_keys], dtype=torch.long)
    x = torch.from_numpy(features)

    print(f"number of nodes:{num_nodes}, feature dimension: {features.shape[1]}")

    train_mask, val_mask, test_mask = _build_masks(y)

    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.y = y
    data.num_nodes = num_nodes
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.n_x = num_nodes
    data.train_percent = float(train_mask.sum().item()) / max(1, num_nodes)
    data.num_hyperedges = int(H.shape[1])

    return data, H


def load_NTU2012(path='../data/AllSet_all_raw_data/AllSet_all_raw_data/NTU2012', dataset = 'NTU2012', train_percent = 0.5):
    '''
    Load the NTU2012 dataset, whose content file includes node rows followed by
    appended hyperedge rows, while the edges file maps nodes to hyperedge ids.
    '''

    def _resolve_dataset_dir(raw_path: str) -> str:
        expected = f'{dataset}.content'
        if os.path.isfile(os.path.join(raw_path, expected)):
            return raw_path
        base_dir = _find_allset_base_dir()
        candidate = os.path.join(base_dir, os.path.basename(os.path.normpath(raw_path)))
        if os.path.isfile(os.path.join(candidate, expected)):
            return candidate
        nested_candidate = os.path.join(raw_path, dataset)
        if os.path.isfile(os.path.join(nested_candidate, expected)):
            return nested_candidate
        return raw_path

    def _build_masks(labels_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        num_classes = int(labels_tensor.max().item()) + 1
        for class_idx in range(num_classes):
            idxs = (labels_tensor == class_idx).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
            if not idxs:
                continue
            rng = np.random.default_rng(seed=42 + class_idx)
            perm = rng.permutation(idxs)
            total = len(perm)
            n_test = max(1, int(total * 0.2))
            n_val = max(1, int(total * 0.1))
            n_train = total - n_val - n_test
            if n_train <= 0:
                n_train = 1
                remaining = total - n_train
                n_val = min(n_val, remaining)
                n_test = max(0, remaining - n_val)

            train_mask[perm[:n_train].tolist()] = True
            val_mask[perm[n_train : n_train + n_val].tolist()] = True
            test_mask[perm[n_train + n_val :].tolist()] = True

        return train_mask, val_mask, test_mask

    path = _resolve_dataset_dir(path)
    print(f"Loading hypergraph dataset from {os.path.basename(path)}")

    content_path = os.path.join(path, f'{dataset}.content')
    edges_path = os.path.join(path, f'{dataset}.edges')

    rows = []
    cols_raw = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            node_idx = int(parts[0])
            hyperedge_idx = int(parts[1])
            rows.append(node_idx)
            cols_raw.append(hyperedge_idx)

    if not rows:
        raise ValueError(f'No node-hyperedge pairs found in {edges_path}')

    num_nodes = min(cols_raw)
    unique_cols = sorted(set(cols_raw))
    raw_to_local = {raw_col: idx for idx, raw_col in enumerate(unique_cols)}
    cols = [raw_to_local[raw_col] for raw_col in cols_raw]
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    H = torch.sparse_coo_tensor(indices, values, (num_nodes, len(unique_cols))).coalesce()

    features_by_id = [None] * num_nodes
    labels_by_id = [None] * num_nodes
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 3:
                continue
            row_id = int(float(toks[0]))
            if not (0 <= row_id < num_nodes):
                continue
            features_by_id[row_id] = [float(value) for value in toks[1:-1]]
            labels_by_id[row_id] = int(float(toks[-1]))

    if any(feature_row is None for feature_row in features_by_id) or any(label is None for label in labels_by_id):
        raise ValueError(f'Could not parse all {num_nodes} node rows from {content_path}')

    features = np.asarray(features_by_id, dtype=np.float32)
    labels = np.asarray(labels_by_id, dtype=np.int64)
    _, remapped_labels = np.unique(labels, return_inverse=True)
    x = torch.from_numpy(features)
    y = torch.from_numpy(remapped_labels.astype(np.int64, copy=False))

    print(f"number of nodes:{num_nodes}, feature dimension: {features.shape[1]}")

    train_mask, val_mask, test_mask = _build_masks(y)

    class SimpleData:
        pass

    data = SimpleData()
    data.x = x
    data.y = y
    data.num_nodes = num_nodes
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.n_x = num_nodes
    data.train_percent = float(train_mask.sum().item()) / max(1, num_nodes)
    data.num_hyperedges = int(H.shape[1])

    return data, H


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
        validated_allset_datasets = 0
        validation_failures = []

        # Enumerate AllSet datasets
        if os.path.isdir(base_dir):
            entries = sorted(
                [e for e in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, e))]
            )
            if not entries:
                print("No subfolders found in AllSet base dir.")
            for name in entries:
                print(f"\n== AllSet: '{name}' ==")
                if name in ("cocitation", "coauthorship"):
                    family_dir = os.path.join(base_dir, name)
                    subdatasets = sorted(
                        [entry for entry in os.listdir(family_dir) if os.path.isdir(os.path.join(family_dir, entry))]
                    )
                    if not subdatasets:
                        print(f"  No subdatasets found in '{name}'.")
                    for subname in subdatasets:
                        print(f"  -- '{name}/{subname}' --")
                        try:
                            data, H = load_citation_dataset(path=family_dir, dataset=subname)
                        except Exception as e:
                            print(f"  Failed to load '{name}/{subname}': {e}")
                            traceback.print_exc()
                            continue

                        degree_stats = _summarize_degree_statistics(H)

                        print(f"    num_nodes: {data.num_nodes}")
                        print(f"    x.shape: {tuple(data.x.shape)}")
                        print(f"    y.shape: {tuple(data.y.shape)}")
                        print(f"    dataset.num_classes: {int(data.y.unique().numel())}")
                        print(f"    min_node_degree: {degree_stats['min_node_degree']}")
                        print(f"    max_node_degree: {degree_stats['max_node_degree']}")
                        print(f"    median_node_degree: {degree_stats['median_node_degree']:.6g}")
                        print(f"    avg_node_degree: {degree_stats['avg_node_degree']:.6g}")
                        print(f"    min_edge_degree: {degree_stats['min_edge_degree']}")
                        print(f"    max_edge_degree: {degree_stats['max_edge_degree']}")
                        print(f"    median_edge_degree: {degree_stats['median_edge_degree']:.6g}")
                        print(f"    avg_edge_degree: {degree_stats['avg_edge_degree']:.6g}")
                        if hasattr(data, "train_mask"):
                            t = int(data.train_mask.sum().item())
                            v = int(data.val_mask.sum().item())
                            te = int(data.test_mask.sum().item())
                            print(f"    train/val/test counts: {t}/{v}/{te}")
                        print(f"    H shape: {H.shape}, nnz: {_safe_nnzsparse(H)}")
                    continue

                try:
                    data, H = load_allset_dataset(name, base_data_dir=base_dir, device=device)
                except Exception as e:
                    print(f"  Failed to load '{name}': {e}")
                    traceback.print_exc()
                    continue
                degree_stats = _summarize_degree_statistics(H)

                print(f"  num_nodes: {data.num_nodes}")
                print(f"  x.shape: {tuple(data.x.shape)}")
                print(f"  y.shape: {tuple(data.y.shape)}")
                print(f"  dataset.num_classes: {int(data.y.unique().numel())}")
                print(f"  min_node_degree: {degree_stats['min_node_degree']}")
                print(f"  max_node_degree: {degree_stats['max_node_degree']}")
                print(f"  median_node_degree: {degree_stats['median_node_degree']:.6g}")
                print(f"  avg_node_degree: {degree_stats['avg_node_degree']:.6g}")
                print(f"  min_edge_degree: {degree_stats['min_edge_degree']}")
                print(f"  max_edge_degree: {degree_stats['max_edge_degree']}")
                print(f"  median_edge_degree: {degree_stats['median_edge_degree']:.6g}")
                print(f"  avg_edge_degree: {degree_stats['avg_edge_degree']:.6g}")
                if hasattr(data, "train_mask"):
                    t = int(data.train_mask.sum().item())
                    v = int(data.val_mask.sum().item())
                    te = int(data.test_mask.sum().item())
                    print(f"  train/val/test counts: {t}/{v}/{te}")
                print(f"  H shape: {H.shape}, nnz: {_safe_nnzsparse(H)}")

                mismatches = _validate_allset_structure_statistics(name, data, H)
                if _normalize_allset_dataset_name(name) in _EXPECTED_ALLSET_STRUCTURE_STATS:
                    validated_allset_datasets += 1
                    if mismatches:
                        print("  structure_check: FAIL")
                        for mismatch in mismatches:
                            print(f"    {mismatch}")
                        validation_failures.append((name, mismatches))
                    else:
                        print("  structure_check: PASS")
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
                                        
                    degree_stats = _summarize_degree_statistics(H)
                    print(f"  min_node_degree: {degree_stats['min_node_degree']}")
                    print(f"  max_node_degree: {degree_stats['max_node_degree']}")
                    print(f"  median_node_degree: {degree_stats['median_node_degree']:.6g}")
                    print(f"  avg_node_degree: {degree_stats['avg_node_degree']:.6g}")
                    print(f"  min_edge_degree: {degree_stats['min_edge_degree']}")
                    print(f"  max_edge_degree: {degree_stats['max_edge_degree']}")
                    print(f"  median_edge_degree: {degree_stats['median_edge_degree']:.6g}")
                    print(f"  avg_edge_degree: {degree_stats['avg_edge_degree']:.6g}")
                except Exception as e:
                    print(f"  Failed to load Planetoid '{name}': {e}")
                    traceback.print_exc()

        if validation_failures:
            print("\nAllSet structure validation failed for:")
            for dataset_name, mismatches in validation_failures:
                print(f"  - {dataset_name}")
                for mismatch in mismatches:
                    print(f"    {mismatch}")
            raise SystemExit(1)

        if validated_allset_datasets > 0:
            print(
                f"\nAll {validated_allset_datasets} checked AllSet datasets matched the reference structure statistics."
            )

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
