#!/usr/bin/env python3
"""Compare the sparse explainer dynamic learning-rate strategies.

The experiment intentionally keeps the existing project files unchanged. It
runs ``src_sparse/main_explain.py`` once for each dynamic LR mode with the
fixed settings requested for this comparison:

* beta = 0.001
* num_epochs = 50

Normal execution writes one result pickle per LR strategy plus an aggregate
dump. ``--evaluate-only`` reads that aggregate dump and prints compact CSV
metrics.
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

import torch


BETA = 0.001
EPOCHS = 50
LR_STRATEGIES = ("dynamic", "dynamic-powers-of-two", "dynamic-epochwise")
EXPLAINER_STRATEGIES = ("v1", "v3")
SCOPES = ("non_isolated", "possible", "found")
CSV_COLUMNS = [
    "strategy",
    "lr_strategy",
    "scope",
    "fidelity_accuracy_plus",
    "denominator",
    "explanation_time",
    "graph_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and evaluate the three dynamic learning-rate strategies "
            "available through src_sparse/main_explain.py."
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
            "Directory for strategy result pickles, CSV, and aggregate dump "
            "(default: results/lrstrategy_<dataset>)."
        ),
    )
    parser.add_argument(
        "--dump-path",
        default=None,
        help=(
            "Aggregate pickle dump path "
            "(default: <output-dir>/lrstrategy_dump.pkl)."
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
        help="Read a previous lrstrategy.py output and print the requested CSV.",
    )
    parser.add_argument(
        "--strategy",
        dest="strategies",
        nargs="+",
        choices=EXPLAINER_STRATEGIES,
        default=list(EXPLAINER_STRATEGIES),
        help=(
            "Explanation strategy or strategies passed to the sparse explainer "
            "(default: run both v1 and v3)."
        ),
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
        help="Regenerate strategy result pickles even when they already exist.",
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


def float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def resolve_path(path_text: str | None, repo_root: Path) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def default_output_dir(repo_root: Path, dataset: str) -> Path:
    dataset_token = sanitize_name(dataset.casefold())
    return repo_root / "results" / f"lrstrategy_{dataset_token}"


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
    return resolve_output_dir(args, repo_root) / "lrstrategy_dump.pkl"


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


def strategy_output_path(
    output_dir: Path,
    dataset: str,
    lr_strategy: str,
    explain_strategy: str,
    target_node: int | None,
) -> Path:
    dataset_token = sanitize_name(dataset.casefold())
    lr_token = sanitize_name(lr_strategy)
    target_token = "" if target_node is None else f"_node{target_node}"
    return output_dir / (
        f"cf_examples_{dataset_token}_lrstrategy_beta{float_token(BETA)}_"
        f"epochs{EPOCHS}_{lr_token}_{explain_strategy}{target_token}.pkl"
    )


def build_explainer_command(
    *,
    repo_root: Path,
    model_path: Path,
    output_path: Path,
    lr_strategy: str,
    architecture_args: dict[str, object],
    args: argparse.Namespace,
    explain_strategy: str,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "src_sparse" / "main_explain.py"),
        "--dataset",
        args.dataset,
        "--n-hops",
        str(args.n_hops),
        "--beta",
        f"{BETA:g}",
        "--cf-optimizer",
        args.cf_optimizer,
        "--strategy",
        explain_strategy,
        "--lr",
        lr_strategy,
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


def get_incidence_diff(sub_h: torch.Tensor, cf_h: torch.Tensor) -> float:
    sub_dense = sub_h.to_dense().cpu()
    cf_dense = cf_h.to_dense().cpu()
    diff = torch.abs(sub_dense - cf_dense)
    return float(torch.sum((diff > 1e-5).float()).item())


def get_hyperedge_diff(sub_h: torch.Tensor, cf_h: torch.Tensor) -> float:
    sub_h = sub_h.coalesce()
    cf_h = cf_h.coalesce()

    sub_indices = sub_h.indices()
    if sub_indices.numel() == 0:
        return 0.0
    sub_cols = set(int(value) for value in torch.unique(sub_indices[1]).cpu().tolist())

    cf_indices = cf_h.indices()
    cf_values = cf_h.values()
    if cf_indices.numel() == 0:
        present_cf_cols: set[int] = set()
    else:
        nonzero_mask = cf_values.abs() > 1e-5
        present_cf_cols = (
            set(int(value) for value in cf_indices[1][nonzero_mask].cpu().tolist())
            if bool(nonzero_mask.any().item())
            else set()
        )

    return float(sum(1 for col in sub_cols if col not in present_cf_cols))


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


def iter_failure_entries(cf_examples_per_node: Iterable[object]) -> list[dict[str, Any]]:
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
    lr_strategy: str,
    explain_strategy: str,
) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        cf_examples_per_node = payload["cf_examples_per_node"]
        num_targets = int(payload.get("num_targets", len(cf_examples_per_node)))
        num_non_isolated = payload.get("num_non_isolated")
        num_cf_possible = payload.get("num_cf_possible")
        num_cf_found = payload.get("num_cf_found")
        explanation_time = payload.get("avg_time_possible")
        if explanation_time is None:
            explanation_time = payload.get("avg_time_non_isolated")
    else:
        cf_examples_per_node = payload
        num_targets = len(cf_examples_per_node)  # type: ignore[arg-type]
        num_non_isolated = None
        num_cf_possible = None
        num_cf_found = None
        explanation_time = None

    isolated_count = sum(1 for example in cf_examples_per_node if example is None)
    failure_entries = iter_failure_entries(cf_examples_per_node)
    possible_failures = [entry for entry in failure_entries if entry.get("possible")]
    success_rows = iter_success_rows(cf_examples_per_node)

    if num_non_isolated is None:
        num_non_isolated = num_targets - isolated_count
    if num_cf_found is None:
        num_cf_found = len(success_rows)
    if num_cf_possible is None:
        num_cf_possible = len(success_rows) + len(possible_failures)

    denominator_by_scope = {
        "non_isolated": int(num_non_isolated),
        "possible": int(num_cf_possible),
        "found": int(num_cf_found),
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
            graph_distance = get_hyperedge_diff(sub_h, cf_h)
        else:
            graph_distance = get_incidence_diff(sub_h, cf_h)
        success_graph_distances.append(graph_distance)

    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        denominator = denominator_by_scope[scope]
        if scope == "found":
            failure_count = 0
        elif scope == "possible":
            failure_count = len(possible_failures)
        else:
            failure_count = len(failure_entries)

        fid_values = success_fid_plus + [0.0] * failure_count
        graph_values = success_graph_distances + [0.0] * failure_count
        rows.append(
            {
                "strategy": explain_strategy,
                "lr_strategy": lr_strategy,
                "scope": scope,
                "fidelity_accuracy_plus": average(fid_values, denominator),
                "denominator": denominator,
                "explanation_time": explanation_time,
                "graph_distance": average(graph_values, denominator),
            }
        )
    return rows


def row_sort_key(row: dict[str, object]) -> tuple[int, int, int, str]:
    strategy = str(row.get("strategy", ""))
    lr_strategy = str(row.get("lr_strategy", ""))
    scope = str(row.get("scope", ""))
    strategy_order = {value: index for index, value in enumerate(EXPLAINER_STRATEGIES)}
    lr_strategy_order = {value: index for index, value in enumerate(LR_STRATEGIES)}
    scope_order = {value: index for index, value in enumerate(SCOPES)}
    return (
        strategy_order.get(strategy, len(strategy_order)),
        lr_strategy_order.get(lr_strategy, len(lr_strategy_order)),
        scope_order.get(scope, len(scope_order)),
        lr_strategy,
    )

def write_csv_rows(rows: list[dict[str, object]], handle) -> None:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in sorted(rows, key=row_sort_key):
        writer.writerow([format_number(row.get(column)) for column in CSV_COLUMNS])


def write_csv_file(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        write_csv_rows(rows, handle)


def rows_from_dump(
    payload: object,
    repo_root: Path,
    default_explain_strategy: str,
) -> list[dict[str, object]]:
    if isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise ValueError("Aggregate dump 'rows' entry must be a list.")
        default_strategy = str(payload.get("strategy", default_explain_strategy))
        return [
            {"strategy": default_strategy, **dict(row)}
            if "strategy" not in dict(row)
            else dict(row)
            for row in rows
        ]

    if isinstance(payload, dict) and "runs" in payload:
        all_rows: list[dict[str, object]] = []
        explain_strategy = str(payload.get("strategy", default_explain_strategy))
        for run in payload["runs"]:
            if not isinstance(run, dict):
                raise ValueError("Aggregate dump 'runs' entries must be dictionaries.")
            if "rows" in run:
                run_explain_strategy = str(run.get("strategy", explain_strategy))
                for row in run["rows"]:
                    row_dict = dict(row)
                    row_dict.setdefault("strategy", run_explain_strategy)
                    all_rows.append(row_dict)
                continue

            result_path_text = run.get("result_path")
            if not isinstance(result_path_text, str):
                raise ValueError("Run entry lacks a result_path.")
            result_path = resolve_path(result_path_text, repo_root)
            assert result_path is not None
            lr_strategy = str(run.get("lr_strategy", "unknown"))
            run_explain_strategy = str(run.get("strategy", explain_strategy))
            all_rows.extend(
                summarize_payload(
                    load_pickle(result_path),
                    lr_strategy=lr_strategy,
                    explain_strategy=run_explain_strategy,
                )
            )
        return all_rows

    if isinstance(payload, dict) and "cf_examples_per_node" in payload:
        lr_strategy = str(payload.get("lr_strategy", "unknown"))
        explain_strategy = str(payload.get("strategy", default_explain_strategy))
        return summarize_payload(
            payload,
            lr_strategy=lr_strategy,
            explain_strategy=explain_strategy,
        )

    if isinstance(payload, list):
        return summarize_payload(
            payload,
            lr_strategy="unknown",
            explain_strategy=default_explain_strategy,
        )

    raise ValueError("Unsupported lrstrategy/evaluation pickle format.")


def evaluate_only(args: argparse.Namespace, repo_root: Path) -> None:
    dump_path = resolve_dump_path(args, repo_root)
    default_strategy = args.strategies[0] if len(args.strategies) == 1 else "v1"
    rows = rows_from_dump(load_pickle(dump_path), repo_root, default_strategy)
    write_csv_rows(rows, sys.stdout)


def run_strategy(
    *,
    repo_root: Path,
    model_path: Path,
    output_path: Path,
    lr_strategy: str,
    architecture_args: dict[str, object],
    args: argparse.Namespace,
    explain_strategy: str,
) -> None:
    command = build_explainer_command(
        repo_root=repo_root,
        model_path=model_path,
        output_path=output_path,
        lr_strategy=lr_strategy,
        architecture_args=architecture_args,
        args=args,
        explain_strategy=explain_strategy,
    )
    print(f"\n== strategy: {explain_strategy}; lr strategy: {lr_strategy} ==")
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
            f"main_explain.py failed for lr strategy {lr_strategy} "
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

    strategy_label = ", ".join(args.strategies)
    print(
        f"Comparing dynamic LR strategies with beta={BETA:g}, "
        f"epochs={EPOCHS}, dataset={args.dataset}, "
        f"strategies=({strategy_label})."
    )
    for explain_strategy in args.strategies:
        for lr_strategy in LR_STRATEGIES:
            output_path = strategy_output_path(
                output_dir=output_dir,
                dataset=args.dataset,
                lr_strategy=lr_strategy,
                explain_strategy=explain_strategy,
                target_node=args.target_node,
            )
            run_strategy(
                repo_root=repo_root,
                model_path=model_path,
                output_path=output_path,
                lr_strategy=lr_strategy,
                architecture_args=architecture_args,
                args=args,
                explain_strategy=explain_strategy,
            )

            run_record: dict[str, object] = {
                "strategy": explain_strategy,
                "lr_strategy": lr_strategy,
                "result_path": str(output_path),
            }
            if not args.dry_run:
                run_rows = summarize_payload(
                    load_pickle(output_path),
                    lr_strategy=lr_strategy,
                    explain_strategy=explain_strategy,
                )
                rows.extend(run_rows)
                run_record["rows"] = run_rows
            runs.append(run_record)

    if args.dry_run:
        print("\nDry run complete; no explanations or dumps were written.")
        return

    csv_path = output_dir / "lrstrategy_metrics.csv"
    dump_path = (
        resolve_path(args.dump_path, repo_root)
        if args.dump_path
        else output_dir / "lrstrategy_dump.pkl"
    )
    assert dump_path is not None
    write_csv_file(rows, csv_path)

    dump_payload = {
        "dataset": args.dataset,
        "model": str(model_path),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "beta": BETA,
        "epochs": EPOCHS,
        "lr_strategies": list(LR_STRATEGIES),
        "strategies": list(args.strategies),
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
