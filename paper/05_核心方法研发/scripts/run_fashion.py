"""FashionMNIST task adapter for the frozen sparse-FL experiment stack.

Examples (never launched automatically):
    python run_fashion.py --dataset FashionIndexA0p1 --split-seed 20261124 \
        --seeds 20261134 --variants topk_ef,align90,lagrange90_ef \
        --rounds 30 --fraction .05 --out fashion_partition_20261124

Only this process is patched. run_followup, run_controls, run_lagrangian and
all codecs remain unchanged. Official TRAIN supplies both local training and
development holdouts. Official TEST is loaded lazily only for --test-final.

All methods use the existing 136,874-parameter SmallCNN. Fashion input is padded
in original uint8 zero space from 28x28 to 32x32, replicated to three channels,
then normalized to [-1,1]. This is a fixed cross-task backbone protocol, not a
claim that the architecture or preprocessing is optimal for FashionMNIST.
"""

from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import run_lagrangian as lagrangian
import run_followup as runner


HERE = Path(__file__).resolve().parent
RAW = HERE / "datasets" / "FashionMNIST" / "raw"
DOWNLOAD_MANIFEST = RAW.parent / "download_manifest.json"
DATASET_ALPHAS = {"FashionIndexA0p1": .1, "FashionIndexA0p5": .5, "FashionIndexA1": 1.}
NUM_CLIENTS = 20
NUM_CLASSES = 10
TRAIN_FRACTION = .75
MIN_CLIENT_SAMPLES = 100
MAX_PARTITION_ATTEMPTS = 1000
VALIDATION_SEED = 73021
MIXING_SEED_DOMAIN = 0x4D495831
EXPECTED_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
_TEST_FINAL_ALLOWED = False
_CIFAR_TEST_FACTORY = None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def source_file_manifest():
    """Hash compressed sources without loading or evaluating TEST examples."""
    files = []
    for name, expected in EXPECTED_MD5.items():
        path = RAW / name
        md5 = hashlib.md5()
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                md5.update(block)
                sha.update(block)
        _require(md5.hexdigest() == expected, f"FashionMNIST source MD5 mismatch: {name}")
        files.append({"file": name, "compressed_bytes": path.stat().st_size,
                      "sha256": sha.hexdigest(), "md5": md5.hexdigest(),
                      "expected_official_md5": expected, "official_md5_verified": True})
    return {
        "official_repository": "https://github.com/zalandoresearch/fashion-mnist",
        "official_md5_reference": "torchvision.datasets.FashionMNIST.resources",
        "source_gzip_files": files,
        "download_manifest_sha256": _sha256(DOWNLOAD_MANIFEST) if DOWNLOAD_MANIFEST.exists() else None,
    }


def _read_idx(path, *, magic, expected_shape):
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    dimensions = len(expected_shape)
    header_size = 4 * (dimensions + 1)
    _require(len(raw) >= header_size, f"truncated IDX header: {path}")
    header = struct.unpack(">" + "I" * (dimensions + 1), raw[:header_size])
    _require(header[0] == magic and tuple(header[1:]) == tuple(expected_shape),
             f"unexpected IDX magic/shape in {path}: {header}")
    expected_bytes = header_size + math.prod(expected_shape)
    _require(len(raw) == expected_bytes, f"wrong IDX payload length in {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(expected_shape).copy()


@lru_cache(maxsize=1)
def load_official_train():
    source_file_manifest()
    images = _read_idx(RAW / "train-images-idx3-ubyte.gz", magic=2051,
                       expected_shape=(60000, 28, 28))
    labels = _read_idx(RAW / "train-labels-idx1-ubyte.gz", magic=2049,
                       expected_shape=(60000,)).astype(np.int64)
    _require(np.array_equal(np.bincount(labels, minlength=10), np.repeat(6000, 10)),
             "unexpected FashionMNIST training class counts")
    return images, labels


def load_official_test():
    """Lazy final-evaluation factory; blocked outside --test-final execution."""
    if not _TEST_FINAL_ALLOWED:
        raise RuntimeError("FashionMNIST official TEST requires explicit --test-final")
    source_file_manifest()
    images = _read_idx(RAW / "t10k-images-idx3-ubyte.gz", magic=2051,
                       expected_shape=(10000, 28, 28))
    labels = _read_idx(RAW / "t10k-labels-idx1-ubyte.gz", magic=2049,
                       expected_shape=(10000,)).astype(np.int64)
    _require(np.array_equal(np.bincount(labels, minlength=10), np.repeat(1000, 10)),
             "unexpected FashionMNIST official test class counts")
    return FashionIndexDataset(images, labels, np.arange(10000))


def _index_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def _nested_hash(groups):
    digest = hashlib.sha256()
    for group in groups:
        digest.update(struct.pack("<I", len(group)))
        digest.update(np.asarray(group, dtype="<u4").tobytes())
    return digest.hexdigest()


def partition_indices(labels, *, alpha, split_seed, mixing=0., validation_size=2000):
    """Index-only 60,000-row Dirichlet partition and leakage-free holdouts."""
    labels = np.asarray(labels, dtype=np.int64)
    _require(labels.shape == (60000,), "expected all 60,000 official TRAIN rows")
    _require(bool(np.all((labels >= 0) & (labels < 10))), "Fashion labels must be in [0,9]")
    _require(math.isfinite(alpha) and alpha > 0, "alpha must be positive and finite")
    _require(isinstance(split_seed, (int, np.integer)) and split_seed >= 0,
             "partition seed must be a nonnegative integer")
    _require(math.isfinite(mixing) and 0 <= mixing <= 1, "mixing must be in [0,1]")
    _require(isinstance(validation_size, (int, np.integer)) and validation_size >= 10,
             "validation size must be an integer of at least 10")
    rng = np.random.default_rng(int(split_seed))
    for attempt in range(1, MAX_PARTITION_ATTEMPTS + 1):
        groups = [[] for _ in range(NUM_CLIENTS)]
        for label in range(NUM_CLASSES):
            ids = np.flatnonzero(labels == label)
            rng.shuffle(ids)
            sizes = rng.multinomial(len(ids), rng.dirichlet(np.repeat(alpha, NUM_CLIENTS)))
            offset = 0
            for client, size in enumerate(sizes):
                groups[client].extend(ids[offset:offset+size].tolist())
                offset += int(size)
        if min(map(len, groups)) >= MIN_CLIENT_SAMPLES:
            break
    else:
        raise RuntimeError("could not form a valid FashionMNIST partition")

    train, holdout, original_counts = [], [], []
    for group in groups:
        ids = np.asarray(group, dtype=np.int64)
        rng.shuffle(ids)
        cut = max(1, min(len(ids)-1, int(round(TRAIN_FRACTION * len(ids)))))
        train.append(ids[:cut].copy())
        holdout.append(ids[cut:].copy())
        original_counts.append(np.bincount(labels[ids], minlength=10).tolist())
    before_sizes = [len(ids) for ids in train]
    original_train = np.sort(np.concatenate(train))
    original_hash = _nested_hash(train)
    if mixing:
        mix_rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), MIXING_SEED_DOMAIN]))
        kept, moved, required = [], [], []
        for ids in train:
            order = mix_rng.permutation(len(ids))
            count = int(round(mixing * len(ids)))
            moved.append(ids[order[:count]])
            kept.append(ids[order[count:]])
            required.append(count)
        pool = np.concatenate(moved)
        pool = pool[mix_rng.permutation(len(pool))]
        train, offset = [], 0
        for retained, count in zip(kept, required):
            train.append(np.concatenate((retained, pool[offset:offset+count])))
            offset += count
        _require(offset == len(pool), "mixing did not consume the pool exactly once")
    all_train, all_holdout = np.concatenate(train), np.concatenate(holdout)
    combined = np.concatenate((all_train, all_holdout))
    _require([len(ids) for ids in train] == before_sizes, "mixing changed client train counts")
    _require(np.array_equal(np.sort(all_train), original_train), "mixing changed global train membership")
    _require(len(combined) == 60000 and np.array_equal(np.sort(combined), np.arange(60000)),
             "each official TRAIN row must occur exactly once")
    _require(not np.intersect1d(all_train, all_holdout).size, "train and holdout overlap")
    _require(validation_size <= len(all_holdout), "validation exceeds actual holdout pool")
    vrng = np.random.default_rng(VALIDATION_SEED)
    selected = []
    for label in range(10):
        available = all_holdout[labels[all_holdout] == label]
        selected.append(vrng.permutation(available)[:validation_size//10])
    validation = np.concatenate(selected)
    if len(validation) < validation_size:
        rest = np.setdiff1d(all_holdout, validation)
        validation = np.concatenate((validation, vrng.permutation(rest)[:validation_size-len(validation)]))
    _require(len(validation) == validation_size and len(np.unique(validation)) == validation_size,
             "validation must contain requested unique rows")
    _require(np.isin(validation, all_holdout).all() and not np.isin(validation, all_train).any(),
             "validation must come solely from holdout rows")
    manifest = {
        "partition_scheme": "classwise Dirichlet multinomial; retry until each client has at least 100 total rows",
        "alpha": float(alpha), "partition_seed": int(split_seed), "split_seed": int(split_seed),
        "partition_attempts": attempt, "num_clients": 20, "num_classes": 10,
        "minimum_client_total_samples": MIN_CLIENT_SAMPLES, "client_train_fraction": .75,
        "training_seed_policy": "--seeds affects initialization/local training only; --split-seed changes data membership",
        "mixing": float(mixing), "mixing_seed_sequence": [int(split_seed), MIXING_SEED_DOMAIN],
        "mixing_policy": "reallocate a rounded per-client fraction without replacement, preserving each client count",
        "source_train_samples": 60000, "train_samples": len(all_train), "holdout_samples": len(all_holdout),
        "client_sizes": before_sizes, "client_holdout_sizes": [len(ids) for ids in holdout],
        "original_client_label_counts": original_counts,
        "client_label_counts": [np.bincount(labels[ids], minlength=10).tolist() for ids in train],
        "validation_source": "class-stratified subset of official-TRAIN local holdouts",
        "validation_seed": VALIDATION_SEED, "validation_samples": len(validation),
        "validation_class_counts": np.bincount(labels[validation], minlength=10).tolist(),
        "index_namespace": "zero-based original FashionMNIST official TRAIN IDX row before any shuffle",
        "index_hash_encoding": "uint32 little endian; grouped lists prefix each client with uint32 length",
        "unmixed_train_indices_sha256": original_hash,
        "train_indices_sha256": _nested_hash(train), "holdout_indices_sha256": _nested_hash(holdout),
        "validation_ids_sha256": _index_hash(validation),
        "assignment_label_sha256": _nested_hash([labels[ids] for ids in train]),
        "train_indices": [ids.tolist() for ids in train],
        "holdout_indices": [ids.tolist() for ids in holdout], "validation_indices": validation.tolist(),
        "integrity_checks": {
            "official_train_total_60000": True, "each_official_train_row_once": True,
            "train_holdout_disjoint": True, "validation_unique_holdout_only": True,
            "mixing_preserves_each_client_train_count": True,
            "mixing_preserves_global_train_id_multiset": True,
        },
        "test_used_for_selection": False,
        "official_test_policy": "10,000 official TEST rows loaded only by --test-final; no CIFAR test fallback",
        "partition_generator_name": Path(__file__).name,
        "partition_generator_sha256": _sha256(Path(__file__)),
    }
    return train, holdout, validation, manifest


class FashionIndexDataset(Dataset):
    """Shared uint8 grayscale source with fixed cross-task input conversion."""

    def __init__(self, images, labels, indices):
        self.images, self.labels = images, labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_id = int(self.indices[position])
        # Pad before normalization: original zero becomes -1 at the border.
        gray = torch.from_numpy(self.images[original_id]).unsqueeze(0)
        padded = F.pad(gray, (2, 2, 2, 2), mode="constant", value=0)
        image = padded.repeat(3, 1, 1).to(dtype=torch.float32)
        image.div_(255.).sub_(.5).div_(.5)
        return image, torch.tensor(int(self.labels[original_id]), dtype=torch.long)


def source_and_override_metadata():
    source_names = ("run_fashion.py", "run_followup.py", "run_controls.py",
                    "run_lagrangian.py", "guarded_codec.py", "lagrangian_codec.py")
    return {
        "fashion_adapter_name": "run_fashion.py", "fashion_adapter_version": 1,
        "fashion_source_sha256": {name: _sha256(HERE/name) for name in source_names},
        "fashion_model_source_sha256": _sha256(HERE.parent/"tinyproto_diag"/"tools"/"fedavg_update_diagnostic.py"),
        "fashion_test_dispatch_module_sha256": _sha256(HERE.parent/"tinyproto_diag"/"tools"/"sparse_update_benchmark.py"),
        "fashion_runtime_overrides": ["runner.prepare=run_fashion.prepare",
            "runner.one_run=Fashion metadata wrapper around run_lagrangian.one_run",
            "runner.snapshot_run=Fashion metadata wrapper around run_lagrangian.snapshot_run",
            "sparse_update_benchmark.load_global_testset=run_fashion.dispatch_global_testset"],
        "fashion_backbone": "unchanged SmallCNN, 136874 parameters, shared fixed cross-task backbone",
        "fashion_architecture_claim": "comparison protocol only; not an optimized FashionMNIST architecture",
        "fashion_preprocessing": "uint8 gray28 -> zero-pad2 to gray32 -> repeat3 -> FP32 -> (x/255-.5)/.5",
        "fashion_official_test_gate": "explicit --test-final only; never evaluated at import/prepare",
    }


def prepare(dataset, mixing, split_seed, validation_size):
    if dataset not in DATASET_ALPHAS:
        raise ValueError(f"Fashion adapter requires --dataset in {tuple(DATASET_ALPHAS)}")
    images, labels = load_official_train()
    train_ids, _, validation_ids, manifest = partition_indices(
        labels, alpha=DATASET_ALPHAS[dataset], split_seed=split_seed,
        mixing=mixing, validation_size=validation_size)
    manifest.update({"source": dataset, "task": "FashionMNIST",
                     **source_file_manifest(), **source_and_override_metadata()})
    trainsets = [FashionIndexDataset(images, labels, ids) for ids in train_ids]
    validation = FashionIndexDataset(images, labels, validation_ids)
    return trainsets, validation, manifest


def dispatch_global_testset(dataset_name):
    """Route Fashion final evaluation explicitly; never silently use CIFAR."""
    if not _TEST_FINAL_ALLOWED:
        raise RuntimeError("official test factory requires explicit --test-final")
    if dataset_name.startswith("Fashion"):
        if dataset_name not in DATASET_ALPHAS:
            raise ValueError(f"unknown Fashion dataset name {dataset_name!r}")
        return load_official_test()
    if dataset_name.startswith("Cifar10") and _CIFAR_TEST_FACTORY is not None:
        return _CIFAR_TEST_FACTORY(dataset_name)
    raise ValueError(f"unsupported final-evaluation dataset {dataset_name!r}")


def _with_metadata(args):
    params = copy.copy(args)
    for key, value in source_and_override_metadata().items():
        setattr(params, key, value)
    return params


def main(argv=None):
    global _TEST_FINAL_ALLOWED, _CIFAR_TEST_FACTORY
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        return lagrangian.main(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=tuple(DATASET_ALPHAS), default="FashionIndexA0p1")
    parser.add_argument("--test-final", action="store_true")
    parser.add_argument("--variants", required=True)
    known, _ = parser.parse_known_args(argv)
    if not [item.strip() for item in known.variants.split(",") if item.strip()]:
        parser.error("explicit --variants must contain at least one method")
    if not any(arg == "--dataset" or arg.startswith("--dataset=") for arg in argv):
        argv.extend(("--dataset", known.dataset))

    # Import registers a function only; CIFAR's test images are not loaded.
    import sparse_update_benchmark as benchmark
    old_prepare, old_factory = runner.prepare, benchmark.load_global_testset
    old_one_run, old_snapshot = runner.one_run, runner.snapshot_run
    old_allowed, old_cifar = _TEST_FINAL_ALLOWED, _CIFAR_TEST_FACTORY
    lagrangian.install()
    lag_one_run, lag_snapshot = runner.one_run, runner.snapshot_run

    def one_run(args, trainsets, validation, manifest, variant, seed, out):
        return lag_one_run(_with_metadata(args), trainsets, validation, manifest, variant, seed, out)

    def snapshot_run(args, trainsets, validation, manifest, seed, out):
        return lag_snapshot(_with_metadata(args), trainsets, validation, manifest, seed, out)

    runner.prepare = prepare
    runner.one_run, runner.snapshot_run = one_run, snapshot_run
    benchmark.load_global_testset = dispatch_global_testset
    _TEST_FINAL_ALLOWED, _CIFAR_TEST_FACTORY = known.test_final, old_factory
    old_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        return runner.main()
    finally:
        sys.argv = old_argv
        runner.prepare, benchmark.load_global_testset = old_prepare, old_factory
        runner.one_run, runner.snapshot_run = old_one_run, old_snapshot
        _TEST_FINAL_ALLOWED, _CIFAR_TEST_FACTORY = old_allowed, old_cifar


if __name__ == "__main__":
    main()
