import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results" / "test_main_explain"
MANIFEST_DIR = REPO_ROOT / ".test_sparse_workflow"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"
TEST_MODEL_NAME_PREFIX = "test_train_smoke_hgcn"


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
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_paths_to_remove() -> tuple[list[Path], list[Path]]:
    manifest = load_manifest()
    file_paths: set[Path] = set()
    dir_paths: set[Path] = set()

    for raw_path in manifest.get("artifact_paths", []):
        file_paths.add(Path(raw_path))

    prefix = sanitize_name(TEST_MODEL_NAME_PREFIX) + "_"
    if MODELS_DIR.is_dir():
        for checkpoint_path in MODELS_DIR.glob("*.pt"):
            if checkpoint_path.name.startswith(prefix):
                file_paths.add(checkpoint_path)

    if RESULTS_DIR.exists():
        dir_paths.add(RESULTS_DIR)
    if MANIFEST_DIR.exists():
        dir_paths.add(MANIFEST_DIR)

    return sorted(file_paths), sorted(dir_paths, reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove only the artifacts created by test_train.py and test_main_explain.py"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_paths, dir_paths = collect_paths_to_remove()

    if not file_paths and not dir_paths:
        print("No test artifacts found.")
        return

    for path in file_paths:
        print(f"FILE  {path}")
    for path in dir_paths:
        print(f"DIR   {path}")

    if args.dry_run:
        print("Dry run only; nothing was deleted.")
        return

    removed_files = 0
    removed_dirs = 0

    for path in file_paths:
        if path.is_file():
            path.unlink()
            removed_files += 1

    for path in dir_paths:
        if path.is_dir():
            shutil.rmtree(path)
            removed_dirs += 1

    print(f"Removed {removed_files} file(s) and {removed_dirs} directorie(s).")


if __name__ == "__main__":
    main()