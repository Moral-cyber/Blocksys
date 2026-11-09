"""Generate a leakage-free CIFAR-10 Dirichlet federated split.

Only the official CIFAR-10 *training* split is assigned to clients.  The
official 10,000-example test split is never written into client training or
local test shards and is loaded independently by sparse_update_benchmark.py.
"""

from pathlib import Path
import argparse
import json

import numpy as np
from torchvision.datasets import CIFAR10


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "Cifar10" / "rawdata"


def make_split(alpha=0.1, n_clients=20, seed=20261102, train_fraction=0.8,
               min_client_samples=100):
    name = f"Cifar10CleanA{alpha:g}".replace(".", "p")
    out = ROOT / "dataset" / name
    config_path = out / "config.json"
    if config_path.exists():
        print(f"exists: {out}")
        return out

    dataset = CIFAR10(root=str(RAW), train=True, download=False)
    images = np.asarray(dataset.data).transpose(0, 3, 1, 2).astype(np.float32)
    images = (images / 255.0 - 0.5) / 0.5
    labels = np.asarray(dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(seed)

    # Retry the Dirichlet draw until every client has enough examples for a
    # local train/test split. Class-wise allocation keeps the global label
    # counts intact while producing controllable statistical heterogeneity.
    for attempt in range(1000):
        client_ids = [[] for _ in range(n_clients)]
        for label in range(10):
            ids = np.flatnonzero(labels == label)
            rng.shuffle(ids)
            counts = rng.multinomial(len(ids), rng.dirichlet(np.repeat(alpha, n_clients)))
            offset = 0
            for client, count in enumerate(counts):
                client_ids[client].extend(ids[offset:offset + count].tolist())
                offset += int(count)
        sizes = np.asarray([len(ids) for ids in client_ids], dtype=np.int64)
        if int(sizes.min()) >= min_client_samples:
            break
    else:
        raise RuntimeError(f"could not produce a valid split for alpha={alpha}")

    train_dir = out / "train"
    test_dir = out / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    statistic = []
    train_sizes = []
    test_sizes = []
    for client, ids in enumerate(client_ids):
        ids = np.asarray(ids, dtype=np.int64)
        rng.shuffle(ids)
        cut = max(1, min(len(ids) - 1, int(round(len(ids) * train_fraction))))
        train_ids, test_ids = ids[:cut], ids[cut:]
        with (train_dir / f"{client}.npz").open("wb") as f:
            np.savez_compressed(f, data={"x": images[train_ids], "y": labels[train_ids]})
        with (test_dir / f"{client}.npz").open("wb") as f:
            np.savez_compressed(f, data={"x": images[test_ids], "y": labels[test_ids]})
        pairs, counts = np.unique(labels[ids], return_counts=True)
        statistic.append([[int(k), int(v)] for k, v in zip(pairs, counts)])
        train_sizes.append(int(len(train_ids)))
        test_sizes.append(int(len(test_ids)))

    config = {
        "num_clients": n_clients,
        "num_classes": 10,
        "non_iid": True,
        "balance": False,
        "partition": "dir",
        "alpha": alpha,
        "seed": seed,
        "source_train_samples": int(len(images)),
        "source_test_samples": 10000,
        "client_total_samples": int(sizes.sum()),
        "client_train_samples": int(sum(train_sizes)),
        "client_test_samples": int(sum(test_sizes)),
        "global_test_samples": 10000,
        "class_counts": statistic,
        "train_sizes": train_sizes,
        "test_sizes": test_sizes,
        "test_source": "client-local holdout from official CIFAR-10 train split",
        "global_test_source": "official CIFAR-10 test split",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"generated {out}: total={int(sizes.sum())}, train={sum(train_sizes)}, test={sum(test_sizes)}, sizes={sizes.tolist()}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    make_split(args.alpha, args.clients, args.seed, args.train_fraction)


if __name__ == "__main__":
    main()
