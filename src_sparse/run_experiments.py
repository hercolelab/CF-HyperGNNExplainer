import argparse
import contextlib
import csv
import itertools
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_SPARSE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "results" / "experiments"

RUNNER_ONLY_KEYS = {
    "evaluate-strategy",
    "output-path",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    options: dict[str, Any]


def normalize_key(key: Any) -> str:
    return str(key).replace("_", "-")


def normalize_options(options: dict[str, Any]) -> dict[str, Any]:
    return {normalize_key(key): value for key, value in options.items()}


def format_value(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def sanitize_name(text: Any) -> str:
    chars = [
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-"
        for ch in format_value(text)
    ]
    return "".join(chars).strip("-") or "value"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def expand_experiments(config: dict[str, Any]) -> list[Experiment]:
    defaults = normalize_options(config.get("defaults", {}))
    experiments = [
        build_named_experiment(defaults, raw_experiment)
        for raw_experiment in config.get("experiments", [])
    ]

    for raw_sweep in config.get("sweeps", []):
        experiments.extend(expand_sweep(defaults, raw_sweep))

    seen_names = set()
    duplicates = []
    for experiment in experiments:
        if experiment.name in seen_names:
            duplicates.append(experiment.name)
        seen_names.add(experiment.name)

    if duplicates:
        raise ValueError(f"Duplicate experiment names: {', '.join(sorted(duplicates))}")

    return experiments


def build_named_experiment(
    defaults: dict[str, Any], raw_experiment: dict[str, Any]
) -> Experiment:
    experiment = normalize_options(raw_experiment)
    name = experiment.pop("name", None)
    options = defaults | experiment
    return Experiment(str(name or derive_name("experiment", options)), options)


def expand_sweep(
    defaults: dict[str, Any], raw_sweep: dict[str, Any]
) -> list[Experiment]:
    sweep = normalize_options(raw_sweep)
    sweep_name = str(sweep["name"])
    base = defaults | normalize_options(sweep.get("base", {}))
    grid = normalize_options(sweep["grid"])
    grid_keys = list(grid)

    experiments = []
    for values in itertools.product(*(grid[key] for key in grid_keys)):
        grid_options = dict(zip(grid_keys, values))
        name = derive_name(sweep_name, grid_options)
        experiments.append(Experiment(name, base | grid_options))
    return experiments


def derive_name(prefix: str, options: dict[str, Any]) -> str:
    suffix = "__".join(
        f"{sanitize_name(key)}-{sanitize_name(value)}"
        for key, value in options.items()
    )
    return f"{sanitize_name(prefix)}__{suffix}" if suffix else sanitize_name(prefix)


def build_main_command(
    *,
    experiment: Experiment,
    output_path: Path,
    python_executable: str,
    src_dir: Path,
) -> list[str]:
    command = [python_executable, str(src_dir / "main_explain.py")]
    for key, value in experiment.options.items():
        if key in RUNNER_ONLY_KEYS or value is None:
            continue
        if value is True:
            command.append(f"--{key}")
        elif value is not False:
            command.extend([f"--{key}", format_value(value)])

    command.extend(["--output-path", str(output_path)])
    return command


def build_evaluate_command(
    *,
    experiment: Experiment,
    results_path: Path,
    python_executable: str,
    src_dir: Path,
) -> list[str]:
    strategy = experiment.options.get(
        "evaluate-strategy", experiment.options.get("strategy", "v1")
    )
    return [
        python_executable,
        str(src_dir / "evaluate.py"),
        "--results",
        str(results_path),
        "--strategy",
        format_value(strategy),
    ]


def write_resolved_config(experiment: Experiment, path: Path) -> None:
    payload = {"name": experiment.name} | experiment.options
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_command(command: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(shlex.quote(part) for part in command))
        handle.write("\n")


def stream_lines(pipe, stream, log_file, chunks: list[str]) -> None:
    for line in pipe:
        chunks.append(line)
        if log_file is not None:
            log_file.write(line)
            log_file.flush()
        stream.write(line)
        stream.flush()


def run_logged_command(
    command: list[str],
    cwd: Path,
    log_prefix: Path,
    *,
    log_stdout: bool = False,
):
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    env = os.environ | {"PYTHONUNBUFFERED": "1"}

    stdout_log_ctx = (
        log_prefix.with_suffix(".stdout.log").open("w", encoding="utf-8")
        if log_stdout
        else contextlib.nullcontext(None)
    )

    with (
        stdout_log_ctx as stdout_log,
        log_prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr_log,
        subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        ) as process,
    ):
        stdout_thread = threading.Thread(
            target=stream_lines,
            args=(process.stdout, sys.stdout, stdout_log, stdout_chunks),
        )
        stderr_thread = threading.Thread(
            target=stream_lines,
            args=(process.stderr, sys.stderr, stderr_log, stderr_chunks),
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def extract_evaluation_csv(stdout: str) -> str:
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("dataset,"):
            data_lines = []
            for following in lines[index + 1 :]:
                if not following.strip():
                    break
                data_lines.append(following)
            return "\n".join([line, *data_lines]) + "\n"
    return ""


def parse_evaluation_rows(csv_text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(csv_text.splitlines()))


def run_experiment(
    *,
    experiment: Experiment,
    runs_dir: Path,
    python_executable: str,
    src_dir: Path,
    log_stdout: bool = False,
) -> list[dict[str, Any]]:
    command_cwd = Path.cwd()
    run_dir = runs_dir / experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "cf_examples.pkl"
    evaluation_path = run_dir / "evaluate.csv"
    write_resolved_config(experiment, run_dir / "config.yaml")
    print(f"\n== Running {experiment.name} ==")

    main_command = build_main_command(
        experiment=experiment,
        output_path=results_path,
        python_executable=python_executable,
        src_dir=src_dir,
    )
    write_command(main_command, run_dir / "main_explain.command")
    print("Explain:", " ".join(shlex.quote(part) for part in main_command))
    main_result = run_logged_command(
        main_command,
        command_cwd,
        run_dir / "main_explain",
        log_stdout=log_stdout,
    )

    base_record = {
        "name": experiment.name,
        "status": "explain_failed",
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "evaluation_path": str(evaluation_path),
    }

    if main_result.returncode != 0 or not results_path.is_file():
        return [base_record | prefixed_options(experiment.options)]

    evaluate_command = build_evaluate_command(
        experiment=experiment,
        results_path=results_path,
        python_executable=python_executable,
        src_dir=src_dir,
    )
    write_command(evaluate_command, run_dir / "evaluate.command")
    print("Evaluate:", " ".join(shlex.quote(part) for part in evaluate_command))
    evaluate_result = run_logged_command(
        evaluate_command,
        command_cwd,
        run_dir / "evaluate",
        log_stdout=log_stdout,
    )

    if evaluate_result.returncode != 0:
        return [
            base_record
            | {"status": "evaluate_failed"}
            | prefixed_options(experiment.options)
        ]

    evaluation_csv = extract_evaluation_csv(evaluate_result.stdout)
    evaluation_path.write_text(evaluation_csv, encoding="utf-8")
    rows = parse_evaluation_rows(evaluation_csv)
    if not rows:
        return [
            base_record
            | {"status": "evaluate_failed"}
            | prefixed_options(experiment.options)
        ]

    options = prefixed_options(experiment.options)
    return [
        base_record | {"status": "success"} | row | options for row in rows
    ]


def prefixed_options(options: dict[str, Any]) -> dict[str, str]:
    return {f"config.{key}": format_value(value) for key, value in options.items()}


def write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def default_runs_dir(config_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS_ROOT / f"{config_path.stem}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse CF-HyperGNNExplainer experiment batches"
    )
    parser.add_argument("config", type=Path, help="YAML experiment batch config")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory where experiment folders are written",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch main_explain.py and evaluate.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching experiments",
    )
    parser.add_argument(
        "--log-stdout",
        action="store_true",
        help="Write child stdout to *.stdout.log (stderr is always logged to *.stderr.log)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiments = expand_experiments(config)
    runs_dir = args.runs_dir or default_runs_dir(args.config)
    if not experiments:
        raise SystemExit("No experiments found in the batch config.")

    if args.dry_run:
        for experiment in experiments:
            results_path = runs_dir / experiment.name / "cf_examples.pkl"
            main_command = build_main_command(
                experiment=experiment,
                output_path=results_path,
                python_executable=args.python_executable,
                src_dir=SRC_SPARSE_DIR,
            )
            evaluate_command = build_evaluate_command(
                experiment=experiment,
                results_path=results_path,
                python_executable=args.python_executable,
                src_dir=SRC_SPARSE_DIR,
            )
            print(" ".join(shlex.quote(part) for part in main_command))
            print(" ".join(shlex.quote(part) for part in evaluate_command))
        return

    runs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for experiment in experiments:
        records.extend(
            run_experiment(
                experiment=experiment,
                runs_dir=runs_dir,
                python_executable=args.python_executable,
                src_dir=SRC_SPARSE_DIR,
                log_stdout=args.log_stdout,
            )
        )
    write_manifest(records, runs_dir / "manifest.csv")
    print(f"\nManifest written to {runs_dir / 'manifest.csv'}")

    failed = sorted(
        {record["name"] for record in records if record["status"] != "success"}
    )
    if failed:
        raise SystemExit(f"Failed experiments: {', '.join(failed)}")


if __name__ == "__main__":
    main()
