#!/usr/bin/env python3
"""Run the sparse explainer beta ablation and print compact CSV metrics.

This script intentionally leaves the existing project files unchanged. It
runs ``src_sparse/main_explain.py`` for the three requested beta settings:

* 0.005
* 0.05
* incremental

Each run uses ``--lr dynamic`` and ``--num-epochs 50``. Normal execution writes
one result pickle per beta value plus an aggregate dump. ``--evaluate-only``
reads that aggregate dump and prints the requested CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


BETA_ARGUMENTS = ("0.1", "0.5")
LR_ARGUMENT = "dynamic"
EPOCHS = 50
SCOPES = ("possible",)
CSV_COLUMNS = [
    "beta_argument",
    "scope",
    "fidelity_accuracy_plus",
    "explanation_time",
    "graph_distance",
]
_TORCH: Any | None = None


def get_torch() -> Any:
    global _TORCH
    if _TORCH is None:
        import torch as torch_module

        _TORCH = torch_module
    return _TORCH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the requested beta settings available through "
            "src_sparse/main_explain.py."
        )
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        help="Dataset name passed to src_sparse/main_explain.py.",
    )
    parser.add_argument(
        "--ckpt-path",
        "--model",
        dest="ckpt_path",
        default="ckpt.pt",
        help="Path to the pretrained HGCN checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for beta result pickles, CSV, and aggregate dump "
            "(default: results/betaalbation_<dataset>)."
        ),
    )
    parser.add_argument(
        "--dump-path",
        default=None,
        help=(
            "Aggregate pickle dump path "
            "(default: <output-dir>/betaalbation_dump.pkl)."
        ),
    )
    parser.add_argument(
        "--results",
        "--dump",
        dest="results_path",
        default=None,
        help=(
            "Dump/result path read by --evaluate-only. If omitted, --dump-path "
            "or the default dump path is used."
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Read a previous betaalbation.py output and print CSV.",
    )
    parser.add_argument(
        "--strategy",
        choices=("v1", "v3"),
        default="v1",
        help="Explanation strategy passed to the sparse explainer.",
    )
    parser.add_argument(
        "--cf-optimizer",
        choices=("SGD", "Adadelta"),
        default="SGD",
        help="Optimizer passed to the sparse explainer.",
    )
    parser.add_argument("--n-momentum", type=float, default=0.0)
    parser.add_argument("--n-hops", type=int, default=4)
    parser.add_argument(
        "--target-node",
        type=int,
        default=None,
        help="Optional single target node for smaller experiments.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device passed to the sparse explainer.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override dropout from the checkpoint args.",
    )
    parser.add_argument(
        "--nhid",
        type=int,
        default=None,
        help="Override hidden size from the checkpoint args.",
    )
    parser.add_argument(
        "--nout",
        type=int,
        default=None,
        help="Override output hidden size from the checkpoint args.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Regenerate beta result pickles even when they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the three main_explain.py commands without running them.",
    )
    parser.add_argument(
        "--verbose-explainer",
        action="store_true",
        help="Show per-epoch sparse explainer diagnostics.",
    )
    parser.add_argument(
        "--target-edge-debug",
        action="store_true",
        help="Pass --target-edge-debug to the sparse explainer.",
    )
    return parser.parse_args()


def sanitize_name(text: str) -> str:
    sanitized = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text]
    value = "".join(sanitized)
    return value if value else "item"


def normalize_beta_argument(beta_arg: object) -> str:
    value = str(beta_arg).strip()
    if value.startswith("--"):
        value = value[2:]
    if value == "incremntal":
        value = "incremental"
    return value


def beta_token(beta_arg: str) -> str:
    return sanitize_name(
        normalize_beta_argument(beta_arg)
        .replace(".", "p")
        .replace("-", "m")
    )


def normalize_scope(scope: object) -> str:
    value = str(scope).strip().replace("_", "-")
    if value == "non-isolated":
        return value
    if value == "possible":
        return value
    return value


def resolve_path(path_text: str | None, repo_root: Path) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def default_output_dir(repo_root: Path, dataset: str) -> Path:
    dataset_token = sanitize_name(dataset.casefold())
    return repo_root / "results" / f"betaalbation_{dataset_token}"


def resolve_output_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.output_dir:
        output_dir = resolve_path(args.output_dir, repo_root)
        assert output_dir is not None
        return output_dir
    return default_output_dir(repo_root, args.dataset)


def resolve_dump_path(args: argparse.Namespace, repo_root: Path) -> Path:
    explicit = args.results_path or args.dump_path
    if explicit:
        resolved = resolve_path(explicit, repo_root)
        assert resolved is not None
        return resolved
    return resolve_output_dir(args, repo_root) / "betaalbation_dump.pkl"


def format_number(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def torch_load(path: Path) -> Any:
    torch = get_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_architecture_args(
    model_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    checkpoint = torch_load(model_path)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    return {
        "dropout": (
            args.dropout
            if args.dropout is not None
            else checkpoint_args.get("dropout", 0.5)
        ),
        "nhid": (
            args.nhid
            if args.nhid is not None
            else checkpoint_args.get("hidden", checkpoint_args.get("nhid", 64))
        ),
        "nout": (
            args.nout
            if args.nout is not None
            else checkpoint_args.get("out_hidden", checkpoint_args.get("nout", 32))
        ),
    }


def beta_output_path(
    output_dir: Path,
    dataset: str,
    beta_arg: str,
    explain_strategy: str,
    target_node: int | None,
) -> Path:
    dataset_token = sanitize_name(dataset.casefold())
    target_token = "" if target_node is None else f"_node{target_node}"
    return output_dir / (
        f"cf_examples_{dataset_token}_betaalbation_beta{beta_token(beta_arg)}_"
        f"lr{sanitize_name(LR_ARGUMENT)}_epochs{EPOCHS}_"
        f"{sanitize_name(explain_strategy)}{target_token}.pkl"
    )


def build_explainer_command(
    *,
    repo_root: Path,
    model_path: Path,
    output_path: Path,
    beta_arg: str,
    architecture_args: dict[str, object],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "src_sparse" / "main_explain.py"),
        "--dataset",
        args.dataset,
        "--n-hops",
        str(args.n_hops),
        "--beta",
        normalize_beta_argument(beta_arg),
        "--cf-optimizer",
        args.cf_optimizer,
        "--strategy",
        args.strategy,
        "--lr",
        LR_ARGUMENT,
        "--n-momentum",
        str(args.n_momentum),
        "--num-epochs",
        str(EPOCHS),
        "--dropout",
        str(architecture_args["dropout"]),
        "--nhid",
        str(architecture_args["nhid"]),
        "--nout",
        str(architecture_args["nout"]),
        "--ckpt-path",
        str(model_path),
        "--device",
        args.device,
        "--output-path",
        str(output_path),
    ]
    if args.target_node is not None:
        command.extend(["--target-node", str(args.target_node)])
    if args.target_edge_debug:
        command.append("--target-edge-debug")
    if not args.verbose_explainer:
        command.append("--quiet")
    return command


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def as_tensor(value: object) -> Any:
    torch = get_torch()
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def sparse_value_map(
    tensor_value: object,
    eps: float = 1e-5,
) -> dict[tuple[int, int], float]:
    tensor = as_tensor(tensor_value)
    if not tensor.is_sparse:
        tensor = tensor.to_sparse()
    tensor = tensor.coalesce().cpu()
    indices = tensor.indices()
    values = tensor.values()
    entries: dict[tuple[int, int], float] = {}
    for idx in range(values.numel()):
        value = float(values[idx].item())
        if abs(value) > eps:
            entries[(int(indices[0, idx].item()), int(indices[1, idx].item()))] = value
    return entries


def incidence_diff_count(
    sub_h: object,
    cf_h: object,
    eps: float = 1e-5,
) -> float:
    sub_entries = sparse_value_map(sub_h, eps=eps)
    cf_entries = sparse_value_map(cf_h, eps=eps)
    keys = set(sub_entries) | set(cf_entries)
    return float(
        sum(
            1
            for key in keys
            if abs(sub_entries.get(key, 0.0) - cf_entries.get(key, 0.0)) > eps
        )
    )


def removed_hyperedge_count(
    sub_h_value: object,
    cf_h_value: object,
    eps: float = 1e-5,
) -> float:
    sub_h = as_tensor(sub_h_value)
    cf_h = as_tensor(cf_h_value)
    if not sub_h.is_sparse:
        sub_h = sub_h.to_sparse()
    if not cf_h.is_sparse:
        cf_h = cf_h.to_sparse()
    sub_h = sub_h.coalesce().cpu()
    cf_h = cf_h.coalesce().cpu()

    sub_indices = sub_h.indices()
    sub_values = sub_h.values()
    sub_cols = {
        int(sub_indices[1, idx].item())
        for idx in range(sub_values.numel())
        if abs(float(sub_values[idx].item())) > eps
    }
    cf_indices = cf_h.indices()
    cf_values = cf_h.values()
    cf_cols = {
        int(cf_indices[1, idx].item())
        for idx in range(cf_values.numel())
        if abs(float(cf_values[idx].item())) > eps
    }
    return float(len(sub_cols - cf_cols))


def iter_success_rows(cf_examples_per_node: Iterable[object]) -> list[list[object]]:
    rows: list[list[object]] = []
    for example in cf_examples_per_node:
        if example is None or isinstance(example, dict):
            continue
        if isinstance(example, (list, tuple)) and example:
            row = example[0]
            if isinstance(row, (list, tuple)) and len(row) >= 7:
                rows.append(list(row))
    return rows


def iter_failure_entries(
    cf_examples_per_node: Iterable[object],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for example in cf_examples_per_node:
        if isinstance(example, dict):
            failures.append(example)
    return failures


def average(values: Iterable[float], denominator: int | None = None) -> float | None:
    value_list = list(values)
    if denominator is None:
        denominator = len(value_list)
    if denominator <= 0:
        return None
    return float(sum(value_list) / denominator)


def summarize_payload(
    payload: object,
    *,
    beta_arg: str,
    explain_strategy: str,
) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        cf_examples_per_node = payload["cf_examples_per_node"]
        num_targets = int(payload.get("num_targets", len(cf_examples_per_node)))
        num_non_isolated = payload.get("num_non_isolated")
        num_cf_possible = payload.get("num_cf_possible")
        avg_time_non_isolated = payload.get("avg_time_non_isolated")
        avg_time_possible = payload.get("avg_time_possible")
    else:
        cf_examples_per_node = payload
        num_targets = len(cf_examples_per_node)  # type: ignore[arg-type]
        num_non_isolated = None
        num_cf_possible = None
        avg_time_non_isolated = None
        avg_time_possible = None

    isolated_count = sum(1 for example in cf_examples_per_node if example is None)
    failure_entries = iter_failure_entries(cf_examples_per_node)
    possible_failures = [entry for entry in failure_entries if entry.get("possible")]
    success_rows = iter_success_rows(cf_examples_per_node)

    if num_non_isolated is None:
        num_non_isolated = num_targets - isolated_count
    if num_cf_possible is None:
        num_cf_possible = len(success_rows) + len(possible_failures)

    denominator_by_scope = {
        "non-isolated": int(num_non_isolated),
        "possible": int(num_cf_possible),
    }
    time_by_scope = {
        "non-isolated": avg_time_non_isolated,
        "possible": avg_time_possible,
    }

    success_fid_plus: list[float] = []
    success_graph_distances: list[float] = []
    for row in success_rows:
        y_pred_orig = int(row[4])
        y_pred_new_actual = int(row[6])
        success_fid_plus.append(float(y_pred_orig != y_pred_new_actual))

        cf_h = row[2]
        sub_h = row[3]
        if explain_strategy == "v3":
            graph_distance = removed_hyperedge_count(sub_h, cf_h)
        else:
            graph_distance = incidence_diff_count(sub_h, cf_h)
        success_graph_distances.append(graph_distance)

    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        denominator = denominator_by_scope[scope]
        if scope == "possible":
            failure_count = len(possible_failures)
        else:
            failure_count = len(failure_entries)

        fid_values = success_fid_plus + [0.0] * failure_count
        graph_values = success_graph_distances + [0.0] * failure_count
        rows.append(
            {
                "beta_argument": normalize_beta_argument(beta_arg),
                "scope": scope,
                "fidelity_accuracy_plus": average(fid_values, denominator),
                "explanation_time": time_by_scope[scope],
                "graph_distance": average(graph_values, denominator),
            }
        )
    return rows


def row_sort_key(row: dict[str, object]) -> tuple[int, int, str]:
    beta_arg = normalize_beta_argument(row.get("beta_argument", ""))
    scope = normalize_scope(row.get("scope", ""))
    beta_order = {
        normalize_beta_argument(value): index
        for index, value in enumerate(BETA_ARGUMENTS)
    }
    scope_order = {value: index for index, value in enumerate(SCOPES)}
    return (
        beta_order.get(beta_arg, len(beta_order)),
        scope_order.get(scope, len(scope_order)),
        beta_arg,
    )


def normalize_row(row: dict[str, object]) -> dict[str, object]:
    row = dict(row)
    if "beta_argument" not in row and "beta_arg" in row:
        row["beta_argument"] = row["beta_arg"]
    row["beta_argument"] = normalize_beta_argument(row.get("beta_argument", "unknown"))
    row["scope"] = normalize_scope(row.get("scope", "unknown"))
    return row


def write_csv_rows(rows: list[dict[str, object]], handle) -> None:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in sorted((normalize_row(row) for row in rows), key=row_sort_key):
        if row["scope"] not in SCOPES:
            continue
        writer.writerow([format_number(row.get(column)) for column in CSV_COLUMNS])


def write_csv_file(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        write_csv_rows(rows, handle)


def rows_from_dump(
    payload: object,
    repo_root: Path,
    default_beta_arg: str,
    default_explain_strategy: str,
) -> list[dict[str, object]]:
    if isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise ValueError("Aggregate dump 'rows' entry must be a list.")
        return [normalize_row(dict(row)) for row in rows]

    if isinstance(payload, dict) and "runs" in payload:
        all_rows: list[dict[str, object]] = []
        explain_strategy = str(payload.get("strategy", default_explain_strategy))
        for run in payload["runs"]:
            if not isinstance(run, dict):
                raise ValueError("Aggregate dump 'runs' entries must be dictionaries.")
            run_beta_arg = normalize_beta_argument(run.get("beta_argument", default_beta_arg))
            run_explain_strategy = str(run.get("strategy", explain_strategy))
            if "rows" in run:
                for row in run["rows"]:
                    raw_row = dict(row)
                    if "beta_argument" not in raw_row and "beta_arg" not in raw_row:
                        raw_row["beta_argument"] = run_beta_arg
                    row_dict = normalize_row(raw_row)
                    all_rows.append(row_dict)
                continue

            result_path_text = run.get("result_path")
            if not isinstance(result_path_text, str):
                raise ValueError("Run entry lacks a result_path.")
            result_path = resolve_path(result_path_text, repo_root)
            assert result_path is not None
            all_rows.extend(
                summarize_payload(
                    load_pickle(result_path),
                    beta_arg=run_beta_arg,
                    explain_strategy=run_explain_strategy,
                )
            )
        return all_rows

    if isinstance(payload, dict) and "cf_examples_per_node" in payload:
        beta_arg = normalize_beta_argument(payload.get("beta_argument", default_beta_arg))
        explain_strategy = str(payload.get("strategy", default_explain_strategy))
        return summarize_payload(
            payload,
            beta_arg=beta_arg,
            explain_strategy=explain_strategy,
        )

    if isinstance(payload, list):
        return summarize_payload(
            payload,
            beta_arg=default_beta_arg,
            explain_strategy=default_explain_strategy,
        )

    raise ValueError("Unsupported betaalbation/evaluation pickle format.")


def evaluate_only(args: argparse.Namespace, repo_root: Path) -> None:
    dump_path = resolve_dump_path(args, repo_root)
    rows = rows_from_dump(
        load_pickle(dump_path),
        repo_root,
        default_beta_arg="unknown",
        default_explain_strategy=args.strategy,
    )
    write_csv_rows(rows, sys.stdout)


def run_beta(
    *,
    repo_root: Path,
    model_path: Path,
    output_path: Path,
    beta_arg: str,
    architecture_args: dict[str, object],
    args: argparse.Namespace,
) -> None:
    command = build_explainer_command(
        repo_root=repo_root,
        model_path=model_path,
        output_path=output_path,
        beta_arg=beta_arg,
        architecture_args=architecture_args,
        args=args,
    )
    print(f"\n== beta argument: {normalize_beta_argument(beta_arg)} ==")
    print("Command:", " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return
    if output_path.is_file() and not args.rerun:
        print(f"Using existing result pickle: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"main_explain.py failed for beta {normalize_beta_argument(beta_arg)} "
            f"with return code {completed.returncode}."
        )
    if not output_path.is_file():
        raise FileNotFoundError(f"Expected result pickle was not created: {output_path}")


def run_comparison(args: argparse.Namespace, repo_root: Path) -> None:
    model_path = resolve_path(args.ckpt_path, repo_root)
    assert model_path is not None
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_dir = resolve_output_dir(args, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_args = checkpoint_architecture_args(model_path, args)

    rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []

    print(
        f"Comparing beta arguments {', '.join(BETA_ARGUMENTS)} with "
        f"lr={LR_ARGUMENT}, epochs={EPOCHS}, dataset={args.dataset}, "
        f"strategy={args.strategy}."
    )
    for beta_arg in BETA_ARGUMENTS:
        output_path = beta_output_path(
            output_dir=output_dir,
            dataset=args.dataset,
            beta_arg=beta_arg,
            explain_strategy=args.strategy,
            target_node=args.target_node,
        )
        run_beta(
            repo_root=repo_root,
            model_path=model_path,
            output_path=output_path,
            beta_arg=beta_arg,
            architecture_args=architecture_args,
            args=args,
        )

        run_record: dict[str, object] = {
            "strategy": args.strategy,
            "beta_argument": normalize_beta_argument(beta_arg),
            "result_path": str(output_path),
        }
        if not args.dry_run:
            run_rows = summarize_payload(
                load_pickle(output_path),
                beta_arg=beta_arg,
                explain_strategy=args.strategy,
            )
            rows.extend(run_rows)
            run_record["rows"] = run_rows
        runs.append(run_record)

    if args.dry_run:
        print("\nDry run complete; no explanations or dumps were written.")
        return

    csv_path = output_dir / "betaalbation_metrics.csv"
    dump_path = (
        resolve_path(args.dump_path, repo_root)
        if args.dump_path
        else output_dir / "betaalbation_dump.pkl"
    )
    assert dump_path is not None
    write_csv_file(rows, csv_path)

    dump_payload = {
        "dataset": args.dataset,
        "model": str(model_path),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "beta_arguments": list(BETA_ARGUMENTS),
        "lr": LR_ARGUMENT,
        "epochs": EPOCHS,
        "strategy": args.strategy,
        "cf_optimizer": args.cf_optimizer,
        "n_momentum": args.n_momentum,
        "n_hops": args.n_hops,
        "target_node": args.target_node,
        "device": args.device,
        "runs": runs,
        "rows": rows,
    }
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("wb") as handle:
        pickle.dump(dump_payload, handle)

    print(f"\nWrote metrics CSV: {csv_path}")
    print(f"Wrote aggregate dump: {dump_path}")
    print("Use --evaluate-only --results", dump_path, "to print the CSV.")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    if args.evaluate_only:
        evaluate_only(args, repo_root)
        return
    run_comparison(args, repo_root)


if __name__ == "__main__":
    main()
