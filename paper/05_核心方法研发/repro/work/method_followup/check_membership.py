"""Read-only content audit of the supplied federated CIFAR-10 split."""
from pathlib import Path
import collections
import hashlib
import json
import pickle
import numpy as np

ROOT = Path("/Users/wind/Documents/Codex/2026-11-02/zhe/work/tinyproto_diag")
RAW = ROOT / "dataset/Cifar10/rawdata/cifar-10-batches-py"
SPLIT = ROOT / "dataset/Cifar10CleanA0p1"


def digest(image, label):
    return hashlib.sha256(image.tobytes() + bytes([int(label)])).hexdigest()


sources = {}
for source, names in (
    ("official_train", [f"data_batch_{i}" for i in range(1, 6)]),
    ("official_test", ["test_batch"]),
):
    counter = collections.Counter()
    for name in names:
        with (RAW / name).open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        for image, label in zip(batch[b"data"], batch[b"labels"]):
            counter[digest(image, label)] += 1
    sources[source] = counter

shard_counters = {}
normalization_exact = True
shape_dtype = set()
sizes = {}
for split in ("train", "test"):
    counter = collections.Counter()
    split_sizes = []
    for client in range(20):
        with np.load(SPLIT / split / f"{client}.npz", allow_pickle=True) as archive:
            data = archive["data"].tolist()
        x, y = data["x"], data["y"]
        shape_dtype.add((str(x.dtype), tuple(x.shape[1:]), str(y.dtype)))
        split_sizes.append(len(y))
        raw = np.rint((x * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        restored = (raw.astype(np.float32) / 255.0 - 0.5) / 0.5
        normalization_exact &= np.array_equal(restored, x)
        for image, label in zip(raw, y):
            counter[digest(image.reshape(-1), label)] += 1
    shard_counters[split] = counter
    sizes[split] = split_sizes

train, holdout = shard_counters["train"], shard_counters["test"]
official_train, official_test = sources["official_train"], sources["official_test"]
combined = train + holdout
shared_official = set(official_train) & set(official_test)
summary = {
    "method": "SHA-256 of uint8 CHW image bytes plus label; shard normalization inverted and verified exactly",
    "normalization_exact": bool(normalization_exact),
    "shape_dtype": [[a, list(b), c] for a, b, c in sorted(shape_dtype)],
    "sizes": sizes,
    "official_train_samples": sum(official_train.values()),
    "official_test_samples": sum(official_test.values()),
    "client_train_samples": sum(train.values()),
    "client_holdout_samples": sum(holdout.values()),
    "all_shards_multiset_equals_official_train": combined == official_train,
    "train_images_not_in_official_train": sum(v for k, v in train.items() if k not in official_train),
    "holdout_images_not_in_official_train": sum(v for k, v in holdout.items() if k not in official_train),
    "content_overlap_train_holdout_unique_hashes": len(set(train) & set(holdout)),
    "content_overlap_train_official_test_unique_hashes": len(set(train) & set(official_test)),
    "content_overlap_all_shards_official_test_unique_hashes": len(set(combined) & set(official_test)),
    "content_overlap_original_official_train_test_unique_hashes": len(shared_official),
    "duplicate_extra_samples_official_train": sum(official_train.values()) - len(official_train),
    "duplicate_extra_samples_client_train": sum(train.values()) - len(train),
    "duplicate_extra_samples_client_holdout": sum(holdout.values()) - len(holdout),
}
out = Path("/Users/wind/Documents/Codex/2026-11-02/zhe/outputs/method_followup/checks/dataset_membership_current.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
