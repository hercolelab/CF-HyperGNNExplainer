#!/usr/bin/env python
# coding: utf-8

"""Create 50/25/25 split pickles for citation/coauthorship datasets.

The raw HyperGCN-style datasets bundled with AllSet contain split pickles with a
small fixed train set. AllSet's PyG training path samples 50/25/25 splits at
runtime, but this helper persists those splits next to ``features.pickle``,
``labels.pickle`` and ``hypergraph.pickle`` by default.
"""

import argparse
import os
import os.path as osp
import pickle
import zipfile

import numpy as np


DATASETS = (
    ("coauthorship", "cora"),
    ("coauthorship", "dblp"),
    ("cocitation", "citeseer"),
    ("cocitation", "cora"),
    ("cocitation", "pubmed"),
)


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
DEFAULT_RAW_PARENT = osp.join(REPO_ROOT, "data", "AllSet_all_raw_data")
DEFAULT_RAW_ROOT = osp.join(DEFAULT_RAW_PARENT, "AllSet_all_raw_data")
DEFAULT_RAW_ZIP = osp.join(DEFAULT_RAW_PARENT, "AllSet_all_raw_data.zip")
DEFAULT_OUT_ROOT = osp.join(REPO_ROOT, "data", "AllSet_50_25_25_splits")


def _dedupe(paths):
    seen = set()
    for path in paths:
        if path is None:
            continue
        normalized = osp.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def _dataset_dir(root, family, dataset):
    return osp.join(root, family, dataset)


def find_raw_root(raw_root):
    candidates = []
    if raw_root is not None:
        candidates.extend((raw_root, osp.join(raw_root, "AllSet_all_raw_data")))
    candidates.extend(
        (
            DEFAULT_RAW_ROOT,
            DEFAULT_RAW_PARENT,
            osp.join(REPO_ROOT, "data", "AllSet_all_raw_data", "AllSet_all_raw_data"),
        )
    )

    for candidate in _dedupe(candidates):
        if not osp.isdir(candidate):
            continue
        if any(
            osp.isfile(
                osp.join(_dataset_dir(candidate, family, dataset), "labels.pickle")
            )
            for family, dataset in DATASETS
        ):
            return candidate

    return None


def find_raw_zip(raw_zip, raw_root):
    candidates = [raw_zip]
    if raw_root is not None:
        candidates.extend(
            (
                osp.join(raw_root, "AllSet_all_raw_data.zip"),
                osp.join(osp.dirname(raw_root), "AllSet_all_raw_data.zip"),
            )
        )
    candidates.extend(
        (
            DEFAULT_RAW_ZIP,
            osp.join(REPO_ROOT, "data", "raw_data", "AllSet_all_raw_data.zip"),
        )
    )

    for candidate in _dedupe(candidates):
        if osp.isfile(candidate):
            return candidate

    return None


def find_dataset_dir(raw_root, family, dataset):
    if raw_root is None:
        return None

    for root in _dedupe((raw_root, osp.join(raw_root, "AllSet_all_raw_data"))):
        dataset_dir = _dataset_dir(root, family, dataset)
        if osp.isfile(osp.join(dataset_dir, "labels.pickle")):
            return dataset_dir

    return None


def _zip_label_members(family, dataset):
    rel_path = "/".join((family, dataset, "labels.pickle"))
    return (
        rel_path,
        "/".join(("AllSet_all_raw_data", rel_path)),
        "/".join(("AllSet_all_raw_data", "AllSet_all_raw_data", rel_path)),
    )


def _read_pickle_from_zip(zip_path, members):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for member in members:
            if member not in names:
                continue
            with zf.open(member, "r") as f:
                return pickle.load(f)

    raise FileNotFoundError(
        "Could not find any of {} in {}".format(", ".join(members), zip_path)
    )


def load_labels(raw_root, raw_zip, family, dataset):
    dataset_dir = find_dataset_dir(raw_root, family, dataset)
    disk_path = (
        osp.join(dataset_dir, "labels.pickle") if dataset_dir is not None else None
    )

    if disk_path is not None and osp.isfile(disk_path):
        with open(disk_path, "rb") as f:
            return pickle.load(f)

    if raw_zip is not None and osp.isfile(raw_zip):
        return _read_pickle_from_zip(raw_zip, _zip_label_members(family, dataset))

    raise FileNotFoundError(
        "Could not find labels.pickle for {}/{} in raw_root={!r} or raw_zip={!r}".format(
            family, dataset, raw_root, raw_zip
        )
    )


def make_split(labels, train_prop, valid_prop, rng, stratified=False):
    labels = np.asarray(labels).reshape(-1)
    labeled_nodes = np.where(labels >= 0)[0]
    if len(labeled_nodes) == 0:
        raise ValueError("Cannot create a split for a dataset without labeled nodes")

    if not stratified:
        perm = rng.permutation(labeled_nodes)
        train_num = int(len(perm) * train_prop)
        valid_num = int(len(perm) * valid_prop)
        return {
            "train": perm[:train_num].astype(np.int64),
            "valid": perm[train_num : train_num + valid_num].astype(np.int64),
            "test": perm[train_num + valid_num :].astype(np.int64),
        }

    train_idx, valid_idx, test_idx = [], [], []
    for label in np.unique(labels[labeled_nodes]):
        cls_nodes = labeled_nodes[labels[labeled_nodes] == label]
        cls_perm = rng.permutation(cls_nodes)
        train_num = int(len(cls_perm) * train_prop)
        valid_num = int(len(cls_perm) * valid_prop)
        train_idx.append(cls_perm[:train_num])
        valid_idx.append(cls_perm[train_num : train_num + valid_num])
        test_idx.append(cls_perm[train_num + valid_num :])

    return {
        "train": rng.permutation(np.concatenate(train_idx)).astype(np.int64),
        "valid": rng.permutation(np.concatenate(valid_idx)).astype(np.int64),
        "test": rng.permutation(np.concatenate(test_idx)).astype(np.int64),
    }


def save_splits(labels, out_dir, runs, seed, train_prop, valid_prop, stratified):
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    for run in range(1, runs + 1):
        rng = np.random.RandomState(seed + run - 1)
        split = make_split(labels, train_prop, valid_prop, rng, stratified)
        out_path = osp.join(out_dir, "{}.pickle".format(run))
        with open(out_path, "wb") as f:
            pickle.dump(
                {key: value.tolist() for key, value in split.items()},
                f,
                protocol=4,
            )
        summaries.append(
            (
                len(split["train"]),
                len(split["valid"]),
                len(split["test"]),
            )
        )

    return summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 50/25/25 split pickles for AllSet "
            "citation/coauthorship datasets."
        )
    )
    parser.add_argument(
        "--raw-root",
        default=DEFAULT_RAW_ROOT,
        help=(
            "Unzipped AllSet_all_raw_data directory containing coauthorship/ "
            "and cocitation/. The parent extraction directory is also accepted."
        ),
    )
    parser.add_argument(
        "--raw-zip",
        default=DEFAULT_RAW_ZIP,
        help="Fallback raw zip used when --raw-root is not present.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help=(
            "Optional output root. By default, split files are written into each "
            "found dataset's splits/ directory. When only a zip is available, the "
            "fallback output root is {}."
        ).format(DEFAULT_OUT_ROOT),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of split files to create.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--train-prop", type=float, default=0.5)
    parser.add_argument("--valid-prop", type=float, default=0.25)
    parser.add_argument(
        "--stratified",
        action="store_true",
        help=(
            "Split independently per class instead of matching AllSet's "
            "unbalanced runtime split."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if (
        args.train_prop <= 0
        or args.valid_prop < 0
        or args.train_prop + args.valid_prop >= 1
    ):
        raise ValueError(
            "--train-prop and --valid-prop must leave a non-empty test split"
        )

    raw_root = find_raw_root(args.raw_root)
    raw_zip = find_raw_zip(args.raw_zip, raw_root)
    if raw_root is None and raw_zip is None:
        raise FileNotFoundError(
            "Could not find AllSet labels under --raw-root={!r} or --raw-zip={!r}".format(
                args.raw_root, args.raw_zip
            )
        )

    jobs = []
    for family, dataset in DATASETS:
        dataset_dir = find_dataset_dir(raw_root, family, dataset)
        labels = load_labels(raw_root, raw_zip, family, dataset)
        if args.out_root is not None:
            out_dir = osp.join(args.out_root, family, dataset, "splits")
        elif dataset_dir is not None:
            out_dir = osp.join(dataset_dir, "splits")
        else:
            out_dir = osp.join(DEFAULT_OUT_ROOT, family, dataset, "splits")
        jobs.append((family, dataset, labels, out_dir))

    for family, dataset, labels, out_dir in jobs:
        summaries = save_splits(
            labels,
            out_dir,
            args.runs,
            args.seed,
            args.train_prop,
            args.valid_prop,
            args.stratified,
        )
        train_num, valid_num, test_num = summaries[0]
        total = train_num + valid_num + test_num
        print(
            "{}/{}: wrote {} splits to {} ({}/{}/{}, {:.2f}/{:.2f}/{:.2f}%)".format(
                family,
                dataset,
                args.runs,
                out_dir,
                train_num,
                valid_num,
                test_num,
                100.0 * train_num / total,
                100.0 * valid_num / total,
                100.0 * test_num / total,
            )
        )


if __name__ == "__main__":
    main()
