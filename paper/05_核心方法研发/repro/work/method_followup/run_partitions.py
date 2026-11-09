"""Independent index-based CIFAR-10 partitions for frozen-method validation.

This adapter preserves run_controls.py's full training CLI and control rules.
Only runner.prepare is replaced in this process; no active runner, codec, or
control source is modified. Example (not run automatically):

    python run_partitions.py --dataset Cifar10IndexA0p1 \
        --split-seed 20261121 --seeds 20261131,20261132,20261133 \
        --variants topk_ef,align90 --rounds 30 --fraction .05 \
        --mixing 0 --out independent_partition_20261121

--split-seed controls client assignment and train/holdout membership.
--seeds controls model initialization and local training only. The complete
original-image indices are embedded in every run's data_manifest/config.json.
Images are shared as uint8 in memory and normalized on access; no image shards
or duplicate floating-point NPZ files are written. Official TEST labels are
not used by prepare and remain reserved for the existing --test-final path.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import struct
import sys
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

import run_controls as controls


HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "tinyproto_diag" / "dataset" / "Cifar10" / "rawdata"
DATASET_ALPHAS = {
    "Cifar10IndexA0p1": .1,
    "Cifar10IndexA0p5": .5,
    "Cifar10IndexA1": 1.,
}
NUM_CLIENTS = 20
NUM_CLASSES = 10
TRAIN_FRACTION = .75
MIN_CLIENT_SAMPLES = 100
VALIDATION_SEED = 73021
MIXING_SEED_DOMAIN = 0x4D495831  # MIX1: separate stream from partition draws.
MAX_PARTITION_ATTEMPTS = 1000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _index_hash(indices: Sequence[int] | np.ndarray) -> str:
    """SHA256 of ordered original-row IDs encoded as uint32 little endian."""
    return hashlib.sha256(np.asarray(indices, dtype="<u4").tobytes()).hexdigest()


def _nested_index_hash(groups: Sequence[np.ndarray]) -> str:
    """Bind client boundaries as uint32 lengths, followed by each ID list."""
    digest = hashlib.sha256()
    for ids in groups:
        digest.update(struct.pack("<I", len(ids)))
        digest.update(np.asarray(ids, dtype="<u4").tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_official_train():
    """Load official TRAIN once per process; never load the test examples."""
    source = CIFAR10(root=str(RAW), train=True, download=False)
    images = np.asarray(source.data, dtype=np.uint8)
    labels = np.asarray(source.targets, dtype=np.int64)
    _require(images.shape == (50000, 32, 32, 3), "unexpected official train image shape")
    _require(labels.shape == (50000,), "expected 50,000 official train labels")
    _require(np.array_equal(np.bincount(labels, minlength=10), np.repeat(5000, 10)),
             "unexpected official CIFAR-10 train class counts")
    source_files = [
        {"name": f"data_batch_{i}",
         "sha256": _file_sha256(RAW / "cifar-10-batches-py" / f"data_batch_{i}")}
        for i in range(1, 6)
    ]
    provenance = {
        "official_split": "CIFAR-10 train (50,000 original rows)",
        "official_train_batch_files": source_files,
        "official_train_labels_sha256": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
        "image_storage": "one shared uint8 NHWC source array; FP32 normalization on access",
        "normalization": "(FP32(uint8_CHW)/255 - 0.5)/0.5",
    }
    return images, labels, provenance


def partition_indices(labels, *, alpha, split_seed, mixing=0., validation_size=2000):
    """Build and verify all memberships using original official-train rows.

    Returns final client train groups, unchanged client holdout groups,
    stratified validation original-row IDs and a JSON-serializable manifest.
    No image values or prediction outputs affect partitioning.
    """
    labels = np.asarray(labels, dtype=np.int64)
    _require(labels.shape == (50000,), "partitioning requires all 50,000 official-train rows")
    _require(bool(np.all((labels >= 0) & (labels < NUM_CLASSES))), "labels must be in [0,9]")
    _require(math.isfinite(alpha) and alpha > 0, "alpha must be positive and finite")
    _require(isinstance(split_seed, (int, np.integer)) and split_seed >= 0,
             "partition seed must be a nonnegative integer")
    _require(math.isfinite(mixing) and 0 <= mixing <= 1, "mixing must be finite and in [0,1]")
    _require(isinstance(validation_size, (int, np.integer)) and validation_size >= NUM_CLASSES,
             "validation size must be an integer of at least 10")

    rng = np.random.default_rng(int(split_seed))
    for attempt in range(1, MAX_PARTITION_ATTEMPTS + 1):
        groups = [[] for _ in range(NUM_CLIENTS)]
        for label in range(NUM_CLASSES):
            ids = np.flatnonzero(labels == label)
            rng.shuffle(ids)
            allocation = rng.multinomial(len(ids), rng.dirichlet(np.repeat(alpha, NUM_CLIENTS)))
            offset = 0
            for client, amount in enumerate(allocation):
                groups[client].extend(ids[offset:offset + amount].tolist())
                offset += int(amount)
        if min(map(len, groups)) >= MIN_CLIENT_SAMPLES:
            break
    else:
        raise RuntimeError(f"no valid 20-client alpha={alpha} partition in {MAX_PARTITION_ATTEMPTS} attempts")

    train, holdout = [], []
    original_group_counts = []
    for group in groups:
        ids = np.asarray(group, dtype=np.int64)
        rng.shuffle(ids)
        cut = max(1, min(len(ids) - 1, int(round(TRAIN_FRACTION * len(ids)))))
        train.append(ids[:cut].copy())
        holdout.append(ids[cut:].copy())
        original_group_counts.append(np.bincount(labels[ids], minlength=NUM_CLASSES).tolist())

    base_train_hash = _nested_index_hash(train)
    before_train = np.sort(np.concatenate(train))
    before_sizes = [len(ids) for ids in train]
    if mixing > 0:
        mix_rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), MIXING_SEED_DOMAIN]))
        kept, moved, required = [], [], []
        for ids in train:
            order = mix_rng.permutation(len(ids))
            amount = int(round(mixing * len(ids)))
            moved.append(ids[order[:amount]])
            kept.append(ids[order[amount:]])
            required.append(amount)
        pool = np.concatenate(moved)
        pool = pool[mix_rng.permutation(len(pool))]
        train, start = [], 0
        for retained, amount in zip(kept, required):
            train.append(np.concatenate((retained, pool[start:start + amount])))
            start += amount
        _require(start == len(pool), "mixing failed to consume the pool exactly once")

    all_train = np.concatenate(train)
    all_holdout = np.concatenate(holdout)
    combined = np.concatenate((all_train, all_holdout))
    _require(len(train) == NUM_CLIENTS and len(holdout) == NUM_CLIENTS, "wrong client count")
    _require([len(ids) for ids in train] == before_sizes, "mixing changed client sample counts")
    _require(np.array_equal(np.sort(all_train), before_train), "mixing changed the global train ID multiset")
    _require(len(combined) == 50000 and np.array_equal(np.sort(combined), np.arange(50000)),
             "train and holdout must cover each original train row exactly once")
    _require(np.intersect1d(all_train, all_holdout).size == 0, "train/holdout overlap")
    _require(validation_size <= len(all_holdout), "requested validation exceeds the actual holdout pool")

    val_rng = np.random.default_rng(VALIDATION_SEED)
    validation_parts = []
    per_class = validation_size // NUM_CLASSES
    for label in range(NUM_CLASSES):
        available = all_holdout[labels[all_holdout] == label]
        validation_parts.append(val_rng.permutation(available)[:per_class])
    validation = np.concatenate(validation_parts)
    if len(validation) < validation_size:
        rest = np.setdiff1d(all_holdout, validation)
        validation = np.concatenate((validation, val_rng.permutation(rest)[:validation_size-len(validation)]))
    _require(len(validation) == validation_size and len(np.unique(validation)) == validation_size,
             "validation must contain the requested number of unique rows")
    _require(np.isin(validation, all_holdout).all() and not np.isin(validation, all_train).any(),
             "validation rows must come only from client holdouts")

    manifest = {
        "partition_scheme": "classwise Dirichlet multinomial, retried until every client has >=100 total rows",
        "alpha": float(alpha), "num_clients": NUM_CLIENTS, "num_classes": NUM_CLASSES,
        "partition_seed": int(split_seed), "split_seed": int(split_seed),
        "partition_attempts": attempt, "minimum_client_total_samples": MIN_CLIENT_SAMPLES,
        "client_train_fraction": TRAIN_FRACTION,
        "training_seed_policy": "--seeds affects model/local training only; it does not alter this data assignment",
        "mixing": float(mixing),
        "mixing_seed_sequence": [int(split_seed), MIXING_SEED_DOMAIN],
        "mixing_policy": "pool a rounded fraction per client, shuffle without replacement, return the same counts",
        "source_train_samples": 50000, "train_samples": len(all_train),
        "holdout_samples": len(all_holdout), "client_sizes": before_sizes,
        "client_holdout_sizes": [len(ids) for ids in holdout],
        "original_client_label_counts": original_group_counts,
        "client_label_counts": [np.bincount(labels[ids], minlength=NUM_CLASSES).tolist() for ids in train],
        "validation_source": "fixed-seed class-stratified subset of official-TRAIN client holdouts",
        "validation_seed": VALIDATION_SEED, "validation_samples": len(validation),
        "validation_class_counts": np.bincount(labels[validation], minlength=NUM_CLASSES).tolist(),
        "index_namespace": "zero-based original row in torchvision CIFAR10(train=True), before any shuffle",
        "index_hash_encoding": "uint32 little endian; grouped hashes prefix each client list with uint32 length",
        "unmixed_train_indices_sha256": base_train_hash,
        "train_indices_sha256": _nested_index_hash(train),
        "holdout_indices_sha256": _nested_index_hash(holdout),
        "validation_ids_sha256": _index_hash(validation),
        "assignment_label_sha256": _nested_index_hash([labels[ids] for ids in train]),
        "train_indices": [ids.tolist() for ids in train],
        "holdout_indices": [ids.tolist() for ids in holdout],
        "validation_indices": validation.tolist(),
        "integrity_checks": {
            "original_train_total_50000": True,
            "train_holdout_disjoint": True,
            "each_original_row_assigned_once": True,
            "validation_unique_and_holdout_only": True,
            "mixing_preserves_each_client_train_count": True,
            "mixing_preserves_global_train_id_multiset": True,
        },
        "test_used_for_selection": False,
        "official_test_policy": "not loaded by prepare; available only through inherited --test-final evaluation",
        "partition_generator_name": Path(__file__).name,
        "partition_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return train, holdout, validation, manifest


class IndexedCifar10(Dataset):
    """A view onto shared uint8 source images and original sample IDs."""

    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_id = int(self.indices[position])
        image = torch.from_numpy(self.images[original_id]).permute(2, 0, 1).to(dtype=torch.float32)
        image.div_(255.).sub_(.5).div_(.5)
        target = torch.tensor(int(self.labels[original_id]), dtype=torch.long)
        return image, target


def prepare(dataset, mixing, split_seed, validation_size):
    """Drop-in replacement for runner.prepare with the original API."""
    if dataset not in DATASET_ALPHAS:
        raise ValueError(f"index adapter requires --dataset in {tuple(DATASET_ALPHAS)}")
    images, labels, provenance = load_official_train()
    train_ids, _, validation_ids, manifest = partition_indices(
        labels, alpha=DATASET_ALPHAS[dataset], split_seed=split_seed,
        mixing=mixing, validation_size=validation_size)
    manifest.update({"source": dataset, **provenance})
    trainsets = [IndexedCifar10(images, labels, ids) for ids in train_ids]
    validation = IndexedCifar10(images, labels, validation_ids)
    return trainsets, validation, manifest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" not in argv and "-h" not in argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dataset", default="Cifar10IndexA0p1", choices=tuple(DATASET_ALPHAS))
        known, _ = parser.parse_known_args(argv)
        if not any(arg == "--dataset" or arg.startswith("--dataset=") for arg in argv):
            argv.extend(("--dataset", known.dataset))
    previous_prepare = controls.runner.prepare
    controls.runner.prepare = prepare
    try:
        return controls.main(argv)
    finally:
        controls.runner.prepare = previous_prepare


if __name__ == "__main__":
    main()
