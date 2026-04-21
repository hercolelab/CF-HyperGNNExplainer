import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_SPARSE_DIR = REPO_ROOT / "src_sparse"
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_DIR = REPO_ROOT / ".test_sparse_workflow"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"

DEFAULT_DATASETS = [
    "Cora",
    "Citeseer",
    "Pubmed",
    "20newsW100",
    "ModelNet40",
    "Mushroom",
    "NTU2012",
    "house-committees",
    "walmart-trips",
    "yelp",
    "zoo",
]

TEST_MODEL_NAME_PREFIX = "test_train_smoke_hgcn"


def sanitize_checkpoint_name(text: str) -> str:
    sanitized = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            sanitized.append(ch)
        else:
            sanitized.append("_")
    result = "".join(sanitized)
    return result if result else "model"


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {
            "version": 1,
            "train": {},
            "explain": {},
            "artifact_paths": [],
        }

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    manifest.setdefault("version", 1)
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


def build_checkpoint_path(args: argparse.Namespace, dataset: str) -> Path:
    clip_label = (
        f"clip{args.clip_grad_norm:g}" if args.clip_grad_norm > 0 else "clipNone"
    )
    checkpoint_name = (
        "_".join(
            [
                sanitize_checkpoint_name(args.model_name_prefix),
                f"seed{args.seed}",
                f"dataset{dataset}",
                f"epochs{args.epochs}",
                f"lr{args.learning_rate:g}",
                f"nhid{args.hidden}",
                f"nout{args.out_hidden}",
                f"dropout{args.dropout:g}",
                f"wd{args.weight_decay:g}",
                clip_label,
            ]
        )
        + ".pt"
    )
    return MODELS_DIR / checkpoint_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test training for all sparse-pipeline datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset names to train sequentially",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--model-name-prefix",
        default=TEST_MODEL_NAME_PREFIX,
        help="Prefix used to namespace checkpoints created by this script",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    failures: list[str] = []

    for dataset in args.datasets:
        checkpoint_path = build_checkpoint_path(args, dataset)
        register_artifact_path(manifest, checkpoint_path)

        entry = {
            "dataset": dataset,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "epochs": args.epochs,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "hidden": args.hidden,
            "out_hidden": args.out_hidden,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "clip_grad_norm": args.clip_grad_norm,
            "device": args.device,
            "model_name_prefix": args.model_name_prefix,
            "status": "pending",
        }
        manifest["train"][dataset] = entry
        save_manifest(manifest)

        command = [
            sys.executable,
            str(SRC_SPARSE_DIR / "train.py"),
            "--dataset",
            dataset,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--learning-rate",
            str(args.learning_rate),
            "--hidden",
            str(args.hidden),
            "--out-hidden",
            str(args.out_hidden),
            "--dropout",
            str(args.dropout),
            "--weight-decay",
            str(args.weight_decay),
            "--clip-grad-norm",
            str(args.clip_grad_norm),
            "--device",
            args.device,
            "--model-name",
            args.model_name_prefix,
        ]

        print(f"\n== Training {dataset} ==")
        print("Command:", " ".join(shlex.quote(part) for part in command))
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)

        entry["returncode"] = completed.returncode
        entry["status"] = (
            "success"
            if completed.returncode == 0 and checkpoint_path.is_file()
            else "failed"
        )
        manifest["train"][dataset] = entry
        save_manifest(manifest)

        if entry["status"] == "success":
            print(f"Checkpoint ready at {checkpoint_path}")
        else:
            failures.append(dataset)
            print(f"Training failed for {dataset}")

    print(f"\nManifest updated at {MANIFEST_PATH}")
    if failures:
        print("Training failed for:", ", ".join(failures))
        raise SystemExit(1)

    print(f"Trained {len(args.datasets)} dataset(s) successfully.")


if __name__ == "__main__":
    main()