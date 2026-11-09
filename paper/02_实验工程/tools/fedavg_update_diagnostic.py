"""Pilot diagnostic for sparse model updates under Non-IID client data.

This is deliberately separate from the TinyProto implementation. It answers
whether parameter Top-K compression creates a measurable client-level cost
under the same actual byte budget. It is a pilot, not a final benchmark.
"""

from pathlib import Path
import argparse
import copy
import csv
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


class SmallCNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_client(dataset_name, client_id, train):
    split = "train" if train else "test"
    path = ROOT / "dataset" / dataset_name / split / f"{client_id}.npz"
    with path.open("rb") as f:
        data = np.load(f, allow_pickle=True)["data"].tolist()
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)
    return TensorDataset(x, y)


def vectorize(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def assign_vector(model, vector):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vector[offset:offset + n].view_as(p))
            offset += n


def compress(vector, fraction, residual=None, mode="topk"):
    candidate = vector if residual is None else vector + residual
    k = max(1, int(candidate.numel() * fraction))
    idx = torch.topk(candidate.abs(), k=k, sorted=False).indices
    out = torch.zeros_like(candidate)
    out[idx] = candidate[idx]
    next_residual = None if residual is None else candidate - out
    return out, next_residual, int(k)


def train_local(global_model, trainset, seed):
    torch.manual_seed(seed)
    local = copy.deepcopy(global_model)
    local.train()
    loader = DataLoader(trainset, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(local.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        optimizer.zero_grad()
        loss_fn(local(x), y).backward()
        optimizer.step()
    return local


@torch.no_grad()
def evaluate(model, testsets):
    model.eval()
    client_acc = []
    class_hits = np.zeros(10, dtype=np.int64)
    class_total = np.zeros(10, dtype=np.int64)
    for testset in testsets:
        loader = DataLoader(testset, batch_size=128)
        correct = total = 0
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += len(y)
            for label in range(10):
                mask = y == label
                class_hits[label] += int(((pred == y) & mask).sum())
                class_total[label] += int(mask.sum())
        client_acc.append(correct / total if total else 0.0)
    class_acc = np.divide(class_hits, class_total, out=np.zeros_like(class_hits, dtype=float), where=class_total > 0)
    return {
        "overall_acc": float(np.mean(client_acc)),
        "client_acc_std": float(np.std(client_acc)),
        "client_acc_min": float(np.min(client_acc)),
        "client_acc": client_acc,
        "class_acc": class_acc.tolist(),
    }


def run(dataset_name, method, rounds, fraction, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = json.loads((ROOT / "dataset" / dataset_name / "config.json").read_text())
    n_clients = int(config["num_clients"])
    trainsets = [load_client(dataset_name, i, True) for i in range(n_clients)]
    testsets = [load_client(dataset_name, i, False) for i in range(n_clients)]
    sample_counts = np.asarray([len(x) for x in trainsets], dtype=float)
    weights = sample_counts / sample_counts.sum()

    global_model = SmallCNN()
    residuals = [torch.zeros_like(vectorize(global_model)) for _ in range(n_clients)]
    rows = []
    total_bytes = 0
    dense_bytes = vectorize(global_model).numel() * 4

    for rnd in range(rounds):
        base = vectorize(global_model)
        updates = []
        round_bytes = 0
        for client_id, trainset in enumerate(trainsets):
            local = train_local(global_model, trainset, seed + rnd * 100 + client_id)
            delta = vectorize(local) - base
            if method == "dense":
                sent, residuals[client_id], k = delta, None, delta.numel()
                candidate = delta
                payload_bytes = dense_bytes
            elif method == "topk":
                sent, residuals[client_id], k = compress(delta, fraction, None, method)
                candidate = delta
                payload_bytes = k * (4 + 4)  # value + flat index
            elif method == "topk_ef":
                candidate = delta + residuals[client_id]
                sent, residuals[client_id], k = compress(delta, fraction, residuals[client_id], method)
                payload_bytes = k * (4 + 4)
            else:
                raise ValueError(method)
            updates.append(sent)
            round_bytes += payload_bytes
            candidate_norm = float(torch.linalg.vector_norm(candidate))
            sent_norm = float(torch.linalg.vector_norm(sent))
            cosine = float(torch.dot(candidate, sent) / (candidate_norm * sent_norm + 1e-12))
            rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "alpha": config["alpha"],
                "method": method,
                "round": rnd,
                "client_id": client_id,
                "client_samples": len(trainset),
                "update_norm": candidate_norm,
                "sent_norm_ratio": sent_norm / (candidate_norm + 1e-12),
                "update_sent_cosine": cosine,
                "selected_fraction": k / candidate.numel(),
            })

        aggregate = sum(float(w) * update for w, update in zip(weights, updates))
        assign_vector(global_model, base + aggregate)
        total_bytes += round_bytes
        metrics = evaluate(global_model, testsets)
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "alpha": config["alpha"],
            "method": method,
            "round": rnd,
            "client_id": "__global__",
            "client_samples": int(sample_counts.sum()),
            "round_bytes": round_bytes,
            "dense_bytes": dense_bytes * n_clients,
            "cumulative_bytes": total_bytes,
            **{k: v for k, v in metrics.items() if k not in ("client_acc", "class_acc")},
            "class_acc": metrics["class_acc"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seeds", type=str, default="20261102")
    parser.add_argument("--datasets", type=str, default="PilotA0p1,PilotA0p5,PilotA1")
    parser.add_argument("--output-stem", type=str, default="fedavg_update_diagnostic")
    args = parser.parse_args()
    all_rows = []
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for seed in seeds:
        for dataset in datasets:
            for method in ("dense", "topk", "topk_ef"):
                print(f"running seed={seed} {dataset} {method}", flush=True)
                all_rows.extend(run(dataset, method, args.rounds, args.fraction, seed))

    out_json = ROOT / "results" / f"{args.output_stem}.json"
    out_csv = ROOT / "results" / f"{args.output_stem}.csv"
    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    fields = sorted({key for row in all_rows for key in row if not isinstance(row.get(key), list)})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: v for k, v in row.items() if k in fields})
    print(out_csv)


if __name__ == "__main__":
    main()
