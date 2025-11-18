from __future__ import annotations

import argparse
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

HEADER = [
    "node_idx",
    "new_idx",
    "cf_adj",
    "sub_adj",
    "y_pred_orig",
    "y_pred_new",
    "y_pred_new_actual",
    "label",
    "num_nodes",
    "loss_total",
    "loss_pred",
    "loss_graph_dist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CF-HyperGNNExplainer pickle results."
    )
    parser.add_argument("--path", required=True, help="Pickle file with CF examples")
    parser.add_argument(
        "--dataset",
        default="Cora",
        help="Planetoid dataset name used during explanation (default: Cora)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device used for loading the model",
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


def load_dataset(args: argparse.Namespace, device: torch.device):
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "Planetoid")
    dataset = Planetoid(
        root=data_root,
        name=args.dataset,
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)
    return data


def load_cf_dataframe(cf_path: str) -> pd.DataFrame:
    cf_path = os.path.abspath(cf_path)
    with open(cf_path, "rb") as f:
        cf_examples = pickle.load(f)

    df_rows: List[List] = []
    for example in cf_examples:
        if isinstance(example, list) and example:
            df_rows.append(example[0])

    if not df_rows:
        return pd.DataFrame(columns=HEADER)

    df = pd.DataFrame(df_rows, columns=HEADER)
    return df


def add_num_edges(df: pd.DataFrame) -> pd.DataFrame:
    def count_hyperedges(sub_adj_obj) -> int:
        sub_adj = np.asarray(sub_adj_obj)
        if sub_adj.ndim == 1:
            sub_adj = sub_adj.reshape(1, -1)
        if sub_adj.size == 0:
            return 0
        return sub_adj.shape[1]

    df = df.copy()
    df["num_edges"] = df["sub_adj"].apply(count_hyperedges)
    return df


def prepare_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    int_cols = [
        "node_idx",
        "new_idx",
        "y_pred_orig",
        "y_pred_new",
        "y_pred_new_actual",
        "num_nodes",
    ]
    float_cols = ["loss_total", "loss_pred", "loss_graph_dist"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    args = parse_args()
    print(args)
    device = resolve_device(args.device)

    data = load_dataset(args, device)

    cf_df = load_cf_dataframe(args.path)
    cf_df = add_num_edges(cf_df)
    cf_df = prepare_numeric_columns(cf_df)

    idx_test = torch.where(data.test_mask)[0].cpu().numpy()

    if cf_df.empty:
        print(args.path)
        print(f"Num cf examples found: 0/{len(idx_test)}")
        print("Avg fidelity: 1.0")
        print("Average graph distance: nan, std: nan")
        print("Average sparsity: nan, std: nan")
        print("\n***************************************************************\n")
        return

    loss_graph_dist = cf_df["loss_graph_dist"].astype(float)
    num_edges_series = cf_df["num_edges"].replace({0: np.nan}).astype(float)
    sparsity = 1 - (loss_graph_dist / num_edges_series)

    print(args.path)
    num_cf = len(cf_df)
    num_targets = len(idx_test)
    success_rate = num_cf / num_targets if num_targets else float("nan")

    fidelity_mask = cf_df["y_pred_orig"] == cf_df["y_pred_new_actual"]
    fidelity = fidelity_mask.mean() if not cf_df.empty else float("nan")

    print(f"Num cf examples found: {num_cf}/{num_targets}")
    print(f"Success rate (cf found / targets): {success_rate}")
    print(
        "Fidelity (orig == cf prediction): {}".format(
            fidelity if not np.isnan(fidelity) else "nan"
        )
    )
    print(
        "Average graph distance: {}, std: {}".format(
            np.nanmean(loss_graph_dist), np.nanstd(loss_graph_dist)
        )
    )
    print(
        "Average sparsity: {}, std: {}".format(
            np.nanmean(sparsity), np.nanstd(sparsity)
        )
    )
    print(" ")
    print("***************************************************************")
    print(" ")


if __name__ == "__main__":
    main()
