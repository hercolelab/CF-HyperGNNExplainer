#!/usr/bin/env python3
"""Train all requested datasets, run beta ablation, and merge CSV metrics."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from betaalbation import (
    format_number,
    iter_failure_entries,
    iter_success_rows,
    load_pickle,
    normalize_scope,
    sparse_value_map,
)


DATASETS = (
    "cocitation-cora",
    "cocitation-citeseer",
    "cocitation-pubmed",
    "coauthorship-cora",
    "coauthorship-dblp",
    "zoo",
    "mushrooms",
    "ntu2012",
    "modelnet40",
    "house",
)
CHECKPOINT_RE = re.compile(r"Saved checkpoint to\s+(?P<path>.+)")
METRICS_CSV_RE = re.compile(r"Wrote metrics CSV:\s+(?P<path>.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run src_sparse/train.py and betaalbation.py for each dataset, then "
            "merge all betaalbation CSVs with an extra dataset column."
        )
    )
    parser.add_argument(
        "--combined-csv",
        default="results/betaalbation_all_metrics.csv",
        help="Path for the merged CSV.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="Datasets to process. Defaults to the full requested list.",
    )
    parser.add_argument(
        "--uv",
        default="uv",
        help="uv executable to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training, ablation, or merging.",
    )
    return parser.parse_args()


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], repo_root: Path) -> str:
    print("\n$ " + quote_command(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    returncode = process.wait()
    output = "".join(output_lines)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {returncode}: {quote_command(command)}"
        )
    return output


def resolve_reported_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text.strip())
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def parse_checkpoint_path(output: str, repo_root: Path) -> Path:
    matches = CHECKPOINT_RE.findall(output)
    if not matches:
        raise RuntimeError("Could not find a 'Saved checkpoint to ...' line.")
    return resolve_reported_path(matches[-1], repo_root)


def parse_metrics_csv_path(output: str, dataset: str, repo_root: Path) -> Path:
    matches = METRICS_CSV_RE.findall(output)
    if matches:
        return resolve_reported_path(matches[-1], repo_root)
    return (
        repo_root
        / "results"
        / f"betaalbation_{dataset.casefold()}"
        / "betaalbation_metrics.csv"
    ).resolve()


def train_models(
    datasets: list[str],
    uv: str,
    repo_root: Path,
    dry_run: bool,
) -> dict[str, Path]:
    model_by_dataset: dict[str, Path] = {}
    for dataset in datasets:
        command = [uv, "run", "./src_sparse/train.py", "--dataset", dataset]
        print("Training ", dataset)
        if dry_run:
            print("$ " + quote_command(command))
            model_by_dataset[dataset] = Path(
                f"<checkpoint returned by train.py for {dataset}>"
            )
            continue

        output = run_command(command, repo_root)
        checkpoint_path = parse_checkpoint_path(output, repo_root)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint was not created: {checkpoint_path}")
        model_by_dataset[dataset] = checkpoint_path
    return model_by_dataset


def run_betaalbations(
    model_by_dataset: dict[str, Path],
    uv: str,
    repo_root: Path,
    dry_run: bool,
) -> dict[str, Path]:
    csv_by_dataset: dict[str, Path] = {}
    for dataset, model_path in model_by_dataset.items():
        command = [
            uv,
            "run",
            "betaalbation.py",
            "--model",
            str(model_path),
            "--dataset",
            dataset,
            "--rerun",
        ]
        if dry_run:
            print("$ " + quote_command(command))
            continue

        output = run_command(command, repo_root)
        csv_path = parse_metrics_csv_path(output, dataset, repo_root)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metrics CSV was not created: {csv_path}")
        csv_by_dataset[dataset] = csv_path
    return csv_by_dataset


def sparse_tensor(value: object) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not tensor.is_sparse:
        tensor = tensor.to_sparse()
    return tensor.coalesce().cpu()


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

    return float(len(sub_cols - cf_cols)), float(len(sub_cols))


def success_sparsity_values(success_rows: list[list[object]], strategy: str) -> list[float]:
    sparsity_values: list[float] = []
    for row in success_rows:
        cf_h = row[2]
        sub_h = row[3]
        if strategy == "v3":
            graph_distance, num_entries = hyperedge_diff_and_entries(sub_h, cf_h)
        else:
            graph_distance, num_entries = incidence_diff_and_entries(sub_h, cf_h)

        if num_entries > 0:
            sparsity_values.append(float(1 - graph_distance / num_entries))
        else:
            sparsity_values.append(math.nan)
    return sparsity_values


def average_finite(values: list[float], denominator: int) -> float | None:
    finite_values = [value for value in values if math.isfinite(value)]
    if denominator <= 0 or not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


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
        "possible": average_finite(
            [*success_values, *([1.0] * len(possible_failures))],
            int(num_cf_possible),
        ),
        "non-isolated": average_finite(
            [*success_values, *([1.0] * len(failure_entries))],
            int(num_non_isolated),
        ),
        "found": average_finite(success_values, len(success_values)),
    }


def sparsity_lookup_from_dump(
    dump_path: Path,
    repo_root: Path,
) -> dict[tuple[str, str], float | None]:
    payload = load_pickle(dump_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected aggregate dump dictionary: {dump_path}")

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(
            f"Dump does not contain run records with result_path entries: {dump_path}"
        )

    lookup: dict[tuple[str, str], float | None] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError(f"Dump run entry is not a dictionary: {dump_path}")

        result_path_text = run.get("result_path")
        if not isinstance(result_path_text, str):
            raise ValueError("Run entry lacks a result_path; cannot compute sparsity.")

        result_path = resolve_reported_path(result_path_text, repo_root)
        result_payload = load_pickle(result_path)
        beta_argument = str(run.get("beta_argument", "unknown"))
        strategy = str(run.get("strategy", payload.get("strategy", "v3")))
        for scope, sparsity in sparsity_by_scope(result_payload, strategy).items():
            lookup[(beta_argument, normalize_scope(scope))] = sparsity
    return lookup


def merge_csvs(
    csv_by_dataset: dict[str, Path],
    combined_csv: Path,
    repo_root: Path,
) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = ["dataset"]

    for dataset, csv_path in csv_by_dataset.items():
        dump_path = csv_path.parent / "betaalbation_dump.pkl"
        if not dump_path.is_file():
            raise FileNotFoundError(f"Aggregate dump was not created: {dump_path}")
        sparsity_lookup = sparsity_lookup_from_dump(dump_path, repo_root)

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue

            for fieldname in reader.fieldnames:
                if fieldname != "dataset" and fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            if "sparsity" not in fieldnames:
                fieldnames.append("sparsity")

            for row in reader:
                merged_row = {"dataset": dataset}
                for fieldname, value in row.items():
                    if fieldname != "dataset":
                        merged_row[fieldname] = value
                beta_argument = str(row.get("beta_argument", "unknown"))
                scope = normalize_scope(row.get("scope", "unknown"))
                merged_row["sparsity"] = format_number(
                    sparsity_lookup.get((beta_argument, scope))
                )
                rows.append(merged_row)

    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    with combined_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    combined_csv = resolve_reported_path(args.combined_csv, repo_root)

    print("== Training models ==")
    model_by_dataset = train_models(
        datasets=args.datasets,
        uv=args.uv,
        repo_root=repo_root,
        dry_run=args.dry_run,
    )

    print("\n== Running beta ablations ==")
    csv_by_dataset = run_betaalbations(
        model_by_dataset=model_by_dataset,
        uv=args.uv,
        repo_root=repo_root,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run complete; no commands were executed.")
        return

    print("\n== Merging CSVs ==")
    merge_csvs(csv_by_dataset, combined_csv, repo_root)
    print(f"Wrote combined CSV: {combined_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
