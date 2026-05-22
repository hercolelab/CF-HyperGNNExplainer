#!/usr/bin/env python3
"""Build the combined beta-ablation CSV with sparsity from saved pickles.

This script does not train models, run explainers, or call the evaluator. It
reads the existing ``betaalbation_dump.pkl`` files produced by
``betaalbation.py``, loads each referenced result pickle, and recomputes only
the sparsity metric from the stored ``sub_H`` and ``cf_H`` tensors.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable

from betaalbation import (
    CSV_COLUMNS as BETA_CSV_COLUMNS,
    average,
    format_number,
    iter_failure_entries,
    iter_success_rows,
    load_pickle,
    normalize_row,
    normalize_scope,
    resolve_path,
    row_sort_key,
    sanitize_name,
    sparse_value_map,
    summarize_payload,
)
from run_betaalbation_all import DATASETS


CSV_COLUMNS = [
    "dataset",
    *BETA_CSV_COLUMNS,
    "sparsity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the same combined beta-ablation CSV produced by "
            "run_betaalbation_all.py, with an additional sparsity column, "
            "using only saved result pickles."
        )
    )
    parser.add_argument(
        "--output-csv",
        "--combined-csv",
        default="results/betaalbation_all_metrics_with_sparsity.csv",
        help="Destination for the merged CSV with sparsity.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing betaalbation_<dataset> result folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="Datasets to include. Defaults to the same list as run_betaalbation_all.py.",
    )
    parser.add_argument(
        "--dump-path",
        action="append",
        default=[],
        help=(
            "Explicit betaalbation_dump.pkl path to read. Can be passed more "
            "than once. When set, --datasets and --results-dir are ignored."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing dataset dump files instead of failing.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the CSV to stdout instead of writing --output-csv.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def default_dump_path(results_dir: Path, dataset: str) -> Path:
    dataset_token = sanitize_name(dataset.casefold())
    return results_dir / f"betaalbation_{dataset_token}" / "betaalbation_dump.pkl"


def collect_dump_paths(args: argparse.Namespace, root: Path) -> list[Path]:
    if args.dump_path:
        return [resolve_repo_path(path_text, root) for path_text in args.dump_path]

    results_dir = resolve_repo_path(args.results_dir, root)
    dump_paths: list[Path] = []
    missing_paths: list[Path] = []
    for dataset in args.datasets:
        dump_path = default_dump_path(results_dir, dataset)
        if dump_path.is_file():
            dump_paths.append(dump_path)
        else:
            missing_paths.append(dump_path)

    if missing_paths and not args.allow_missing:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing beta ablation dump file(s). Re-run with --allow-missing "
            f"to skip them:\n{missing}"
        )

    return dump_paths


def incidence_diff_and_entries(
    sub_h: object,
    cf_h: object,
    eps: float = 1e-5,
) -> tuple[float, float]:
    sub_entries = sparse_value_map(sub_h, eps=eps)
    cf_entries = sparse_value_map(cf_h, eps=eps)
    keys = set(sub_entries) | set(cf_entries)
    diff_count = float(
        sum(
            1
            for key in keys
            if abs(sub_entries.get(key, 0.0) - cf_entries.get(key, 0.0)) > eps
        )
    )
    num_entries = float(sum(sub_entries.values()))
    return diff_count, num_entries


def sparse_tensor(value: object) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not tensor.is_sparse:
        tensor = tensor.to_sparse()
    return tensor.coalesce().cpu()


def hyperedge_diff_and_entries(
    sub_h_value: object,
    cf_h_value: object,
    eps: float = 1e-5,
) -> tuple[float, float]:
    sub_h = sparse_tensor(sub_h_value)
    cf_h = sparse_tensor(cf_h_value)

    sub_indices = sub_h.indices()
    if sub_indices.numel() == 0:
        return 0.0, 0.0

    sub_cols = set(int(col) for col in sub_indices[1].tolist())

    cf_indices = cf_h.indices()
    cf_values = cf_h.values()
    if cf_indices.numel() == 0:
        cf_cols: set[int] = set()
    else:
        nonzero_mask = cf_values.abs() > eps
        if bool(nonzero_mask.any()):
            cf_cols = set(int(col) for col in cf_indices[1][nonzero_mask].tolist())
        else:
            cf_cols = set()

    removed_hyperedges = float(len(sub_cols - cf_cols))
    total_hyperedges = float(len(sub_cols))
    return removed_hyperedges, total_hyperedges


def success_sparsity_values(
    success_rows: Iterable[list[object]],
    strategy: str,
) -> list[float]:
    values: list[float] = []
    for row in success_rows:
        cf_h = row[2]
        sub_h = row[3]
        if strategy == "v3":
            graph_distance, num_entries = hyperedge_diff_and_entries(sub_h, cf_h)
        else:
            graph_distance, num_entries = incidence_diff_and_entries(sub_h, cf_h)

        if num_entries > 0:
            values.append(float(1 - graph_distance / num_entries))
        else:
            values.append(math.nan)
    return values


def finite_average(values: Iterable[float], denominator: int | None = None) -> float | None:
    value_list = [value for value in values if math.isfinite(value)]
    if denominator is None:
        denominator = len(value_list)
    if denominator <= 0:
        return None
    return float(sum(value_list) / denominator)


def sparsity_by_scope(payload: object, strategy: str) -> dict[str, float | None]:
    if isinstance(payload, dict):
        cf_examples_per_node = payload["cf_examples_per_node"]
        num_targets = int(payload.get("num_targets", len(cf_examples_per_node)))
        num_non_isolated = payload.get("num_non_isolated")
        num_cf_possible = payload.get("num_cf_possible")
    else:
        cf_examples_per_node = payload
        num_targets = len(cf_examples_per_node)  # type: ignore[arg-type]
        num_non_isolated = None
        num_cf_possible = None

    isolated_count = sum(1 for example in cf_examples_per_node if example is None)
    failure_entries = iter_failure_entries(cf_examples_per_node)
    possible_failures = [entry for entry in failure_entries if entry.get("possible")]
    success_rows = iter_success_rows(cf_examples_per_node)
    success_values = success_sparsity_values(success_rows, strategy)

    if num_non_isolated is None:
        num_non_isolated = num_targets - isolated_count
    if num_cf_possible is None:
        num_cf_possible = len(success_rows) + len(possible_failures)

    return {
        "possible": finite_average(
            [*success_values, *([1.0] * len(possible_failures))],
            int(num_cf_possible),
        ),
        "non-isolated": finite_average(
            [*success_values, *([1.0] * len(failure_entries))],
            int(num_non_isolated),
        ),
        "found": finite_average(success_values, len(success_values)),
    }


def dataset_from_dump(payload: object, dump_path: Path) -> str:
    if isinstance(payload, dict) and payload.get("dataset") is not None:
        return str(payload["dataset"])

    parent_name = dump_path.parent.name
    prefix = "betaalbation_"
    if parent_name.startswith(prefix):
        return parent_name[len(prefix) :]
    return parent_name


def rows_from_run(
    run: dict[str, object],
    dump_payload: dict[str, object],
    dataset: str,
    root: Path,
) -> list[dict[str, object]]:
    result_path_text = run.get("result_path")
    if not isinstance(result_path_text, str):
        raise ValueError("Run entry lacks a result_path; cannot compute sparsity.")

    result_path = resolve_path(result_path_text, root)
    assert result_path is not None
    result_payload = load_pickle(result_path)

    beta_arg = run.get("beta_argument", "unknown")
    strategy = str(run.get("strategy", dump_payload.get("strategy", "v3")))
    raw_rows = run.get("rows")
    if isinstance(raw_rows, list):
        base_rows = [normalize_row(dict(row)) for row in raw_rows]
    else:
        base_rows = summarize_payload(
            result_payload,
            beta_arg=str(beta_arg),
            explain_strategy=strategy,
        )

    sparsity_values = sparsity_by_scope(result_payload, strategy)
    rows: list[dict[str, object]] = []
    for row in base_rows:
        row = normalize_row(row)
        scope = normalize_scope(row["scope"])
        row["dataset"] = dataset
        row["sparsity"] = sparsity_values.get(scope)
        rows.append(row)
    return rows


def rows_from_dump(dump_path: Path, root: Path) -> list[dict[str, object]]:
    payload = load_pickle(dump_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected aggregate dump dictionary: {dump_path}")

    dataset = dataset_from_dump(payload, dump_path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(
            f"Dump does not contain run records with result_path entries: {dump_path}"
        )

    rows: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError(f"Dump run entry is not a dictionary: {dump_path}")
        rows.extend(rows_from_run(run, payload, dataset, root))
    return rows


def dataset_order(datasets: list[str]) -> dict[str, int]:
    return {dataset: index for index, dataset in enumerate(datasets)}


def combined_sort_key(
    row: dict[str, object],
    order_by_dataset: dict[str, int],
) -> tuple[int, str, tuple[int, int, str]]:
    dataset = str(row.get("dataset", ""))
    return (
        order_by_dataset.get(dataset, len(order_by_dataset)),
        dataset,
        row_sort_key(row),
    )


def write_csv(rows: list[dict[str, object]], handle, datasets: list[str]) -> None:
    order_by_dataset = dataset_order(datasets)
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: combined_sort_key(item, order_by_dataset)):
        writer.writerow(
            {
                column: format_number(row.get(column))
                for column in CSV_COLUMNS
            }
        )


def main() -> None:
    args = parse_args()
    root = repo_root()
    dump_paths = collect_dump_paths(args, root)

    rows: list[dict[str, object]] = []
    for dump_path in dump_paths:
        rows.extend(rows_from_dump(dump_path, root))

    if args.stdout:
        import sys

        write_csv(rows, sys.stdout, args.datasets)
        return

    output_csv = resolve_repo_path(args.output_csv, root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        write_csv(rows, handle, args.datasets)

    print(f"Wrote combined CSV with sparsity: {output_csv}")


if __name__ == "__main__":
    main()
