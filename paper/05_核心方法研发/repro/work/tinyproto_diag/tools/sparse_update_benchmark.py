"""Compare sparse FedAvg update encoders under an explicit byte budget.

The script reuses the pilot's model, data loader, local trainer, and evaluator.
Block allocation methods are heuristics: energy estimates update magnitude and
novelty estimates directional difference from the previous global update.
"""

from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10

from fedavg_update_diagnostic import (
    ROOT,
    SmallCNN,
    assign_vector,
    evaluate,
    load_client,
    train_local,
    vectorize,
)


METHODS = (
    "dense",
    "global_topk",
    "global_topk_ef",
    "block_topk",
    "block_topk_ef",
    "energy_novelty",
    "energy_novelty_ef",
)


def make_blocks(length, block_size):
    return [(start, min(start + block_size, length)) for start in range(0, length, block_size)]


def load_global_testset(dataset_name):
    """Load the canonical CIFAR-10 test split, independent of client shards.

    The generated federated files are formed by splitting each client's local
    data into train/test subsets.  They are useful for client-level dispersion
    metrics, but the headline global accuracy must use the original 10,000
    CIFAR-10 test examples exactly once.
    """
    if not dataset_name.startswith("Cifar10"):
        raise ValueError("canonical global evaluation is currently implemented for Cifar10")
    raw_root = ROOT / "dataset" / "Cifar10" / "rawdata"
    dataset = CIFAR10(root=str(raw_root), train=False, download=False)
    images = np.asarray(dataset.data).transpose(0, 3, 1, 2).astype(np.float32)
    images = (images / 255.0 - 0.5) / 0.5
    labels = np.asarray(dataset.targets, dtype=np.int64)
    return TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))


def allocate_quotas(scores, total, capacities):
    """Allocate exactly total coordinates according to nonnegative block scores."""
    quotas = np.zeros(len(scores), dtype=np.int64)
    remaining = min(int(total), int(np.sum(capacities)))
    weights = np.maximum(np.asarray(scores, dtype=np.float64), 1e-12)
    capacities = np.asarray(capacities, dtype=np.int64)

    while remaining:
        active_ids = np.flatnonzero(quotas < capacities)
        if not len(active_ids):
            break
        active_weights = weights[active_ids]
        shares = remaining * active_weights / active_weights.sum()
        floors = np.floor(shares).astype(np.int64)
        additions = np.minimum(floors, capacities[active_ids] - quotas[active_ids])
        if additions.sum():
            quotas[active_ids] += additions
            remaining -= int(additions.sum())
            if not remaining:
                break
        # Once every proportional floor is zero (or all nonzero floors have
        # saturated their blocks), distribute the remainder by largest
        # fractional share.  `active_ids` and `shares` are kept aligned; the
        # previous implementation recomputed active IDs before zipping them
        # with stale shares, which could assign the wrong block quotas.
        active_ids = np.flatnonzero(quotas < capacities)
        if not len(active_ids):
            break
        active_weights = weights[active_ids]
        shares = remaining * active_weights / active_weights.sum()
        order = np.argsort(-(shares - np.floor(shares)), kind="stable")
        take = min(remaining, len(order))
        chosen = active_ids[order[:take]]
        quotas[chosen] += 1
        remaining -= int(take)
    return quotas


def block_scores(candidate, blocks, reference, novelty_weight):
    energy = []
    novelty = []
    for start, end in blocks:
        current = candidate[start:end]
        energy.append(float(torch.linalg.vector_norm(current)) / max(1, end - start) ** 0.5)
        if reference is None:
            novelty.append(0.0)
            continue
        previous = reference[start:end]
        denom = torch.linalg.vector_norm(current) * torch.linalg.vector_norm(previous)
        if float(denom) <= 1e-12:
            cosine = 0.0
        else:
            cosine = float(torch.dot(current, previous) / denom)
        novelty.append(float(np.clip(1.0 - cosine, 0.0, 2.0) / 2.0))

    energy = np.asarray(energy, dtype=np.float64)
    novelty = np.asarray(novelty, dtype=np.float64)
    # Multiplicative modulation keeps energy as the main signal and gives
    # directionally novel blocks at most (1 + novelty_weight) times the share.
    scores = (energy + 1e-12) * (1.0 + novelty_weight * novelty)
    return scores, energy, novelty


def encode_update(delta, fraction, method, residual, blocks, reference, novelty_weight,
                  packet_header_bytes, block_header_bytes):
    use_ef = method.endswith("_ef")
    candidate = delta + residual if use_ef else delta
    length = candidate.numel()
    target_k = max(1, min(length, int(length * fraction)))
    zero = torch.zeros_like(candidate)

    if method == "dense":
        sent = candidate
        next_residual = torch.zeros_like(candidate)
        selected = length
        active_blocks = len(blocks)
        payload_bytes = length * candidate.element_size()
        index_bytes = 0
        block_bytes = 0
    elif method in ("global_topk", "global_topk_ef"):
        idx = torch.topk(candidate.abs(), k=target_k, sorted=False).indices
        sent = zero
        sent[idx] = candidate[idx]
        next_residual = candidate - sent if use_ef else torch.zeros_like(candidate)
        selected = target_k
        active_blocks = 0
        payload_bytes = selected * candidate.element_size()
        index_bytes = selected * 4
        block_bytes = 0
    else:
        if method.startswith("energy_novelty"):
            scores, energy, novelty = block_scores(candidate, blocks, reference, novelty_weight)
        else:
            scores = np.ones(len(blocks), dtype=np.float64)
            energy = novelty = None
        capacities = np.asarray([end - start for start, end in blocks], dtype=np.int64)
        quotas = allocate_quotas(scores, target_k, capacities)
        sent = zero
        for quota, (start, end) in zip(quotas, blocks):
            if quota:
                local_idx = torch.topk(candidate[start:end].abs(), k=int(quota), sorted=False).indices
                sent[start + local_idx] = candidate[start + local_idx]
        next_residual = candidate - sent if use_ef else torch.zeros_like(candidate)
        selected = int(quotas.sum())
        active_blocks = int(np.count_nonzero(quotas))
        payload_bytes = selected * candidate.element_size()
        index_bytes = selected * 4
        block_bytes = active_blocks * block_header_bytes

    total_bytes = packet_header_bytes + payload_bytes + index_bytes + block_bytes
    diagnostics = {
        "selected_coordinates": int(selected),
        "active_blocks": int(active_blocks),
        "payload_bytes": int(payload_bytes),
        "index_bytes": int(index_bytes),
        "block_header_bytes": int(block_bytes),
        "packet_header_bytes": int(packet_header_bytes),
        "encoded_bytes": int(total_bytes),
        "residual_norm": float(torch.linalg.vector_norm(next_residual)),
        "candidate_norm": float(torch.linalg.vector_norm(candidate)),
        "sent_norm": float(torch.linalg.vector_norm(sent)),
    }
    return sent, next_residual, diagnostics


def run(dataset_name, method, rounds, fraction, seed, block_size, novelty_weight,
        packet_header_bytes, block_header_bytes, client_limit=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = json.loads((ROOT / "dataset" / dataset_name / "config.json").read_text())
    n_clients = int(config["num_clients"])
    if client_limit is not None:
        n_clients = min(n_clients, client_limit)
    trainsets = [load_client(dataset_name, i, True) for i in range(n_clients)]
    # Client shards remain available for dispersion/fairness diagnostics.  The
    # headline metric uses the canonical CIFAR-10 test split exactly once.
    testsets = [load_client(dataset_name, i, False) for i in range(n_clients)]
    global_testset = load_global_testset(dataset_name)
    sample_counts = np.asarray([len(x) for x in trainsets], dtype=np.float64)
    weights = sample_counts / sample_counts.sum()

    global_model = SmallCNN()
    initial = vectorize(global_model)
    residuals = [torch.zeros_like(initial) for _ in range(n_clients)]
    blocks = make_blocks(initial.numel(), block_size)
    reference = None
    rows = []
    total_bytes = 0
    dense_round_bytes = initial.numel() * initial.element_size() * n_clients + packet_header_bytes * n_clients

    for rnd in range(rounds):
        base = vectorize(global_model)
        updates = []
        round_bytes = 0
        round_residual_norms = []
        round_active_blocks = []
        for client_id, trainset in enumerate(trainsets):
            local = train_local(global_model, trainset, seed + rnd * 100 + client_id)
            delta = vectorize(local) - base
            candidate = delta + residuals[client_id] if method.endswith("_ef") else delta
            sent, residuals[client_id], diag = encode_update(
                delta, fraction, method, residuals[client_id], blocks, reference,
                novelty_weight, packet_header_bytes, block_header_bytes,
            )
            updates.append(sent)
            round_bytes += diag["encoded_bytes"]
            round_residual_norms.append(diag["residual_norm"])
            round_active_blocks.append(diag["active_blocks"])
            candidate_norm = diag["candidate_norm"]
            sent_norm = diag["sent_norm"]
            cosine = float(torch.dot(candidate, sent) /
                           (torch.linalg.vector_norm(candidate) * torch.linalg.vector_norm(sent) + 1e-12))
            rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "alpha": config.get("alpha"),
                "method": method,
                "round": rnd,
                "client_id": client_id,
                "client_samples": len(trainset),
                **diag,
                "sent_norm_ratio": sent_norm / (candidate_norm + 1e-12),
                "update_sent_cosine": cosine,
            })

        aggregate = sum(float(w) * update for w, update in zip(weights, updates))
        assign_vector(global_model, base + aggregate)
        reference = aggregate.detach().clone()
        total_bytes += round_bytes
        global_metrics = evaluate(global_model, [global_testset])
        client_metrics = evaluate(global_model, testsets)
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "alpha": config.get("alpha"),
            "method": method,
            "round": rnd,
            "client_id": "__global__",
            "client_samples": int(sample_counts.sum()),
            "train_samples": int(sample_counts.sum()),
            "client_test_samples": int(sum(len(testset) for testset in testsets)),
            "global_test_samples": int(len(global_testset)),
            "global_test_source": "official CIFAR-10 test split",
            "round_bytes": int(round_bytes),
            "dense_round_bytes": int(dense_round_bytes),
            "compression_ratio_vs_dense": float(dense_round_bytes / max(round_bytes, 1)),
            "cumulative_bytes": int(total_bytes),
            "mean_residual_norm": float(np.mean(round_residual_norms)),
            "mean_active_blocks": float(np.mean(round_active_blocks)),
            "overall_acc": global_metrics["overall_acc"],
            "global_class_acc": global_metrics["class_acc"],
            "client_acc_std": client_metrics["client_acc_std"],
            "client_acc_min": client_metrics["client_acc_min"],
            "client_acc": client_metrics["client_acc"],
            "class_acc": client_metrics["class_acc"],
        })
        print(f"seed={seed} {dataset_name} {method} round={rnd + 1}/{rounds} "
              f"acc={global_metrics['overall_acc']:.4f} bytes={round_bytes}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seeds", type=str, default="20261102,20261103,20261104")
    parser.add_argument("--datasets", type=str, default="Cifar10")
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--novelty-weight", type=float, default=1.0)
    parser.add_argument("--packet-header-bytes", type=int, default=64)
    parser.add_argument("--block-header-bytes", type=int, default=8)
    parser.add_argument("--client-limit", type=int, default=None)
    parser.add_argument("--output-stem", type=str, default="sparse_update_benchmark")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if args.block_size < 1 or args.packet_header_bytes < 0 or args.block_header_bytes < 0:
        parser.error("block size must be positive and header sizes nonnegative")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")

    all_rows = []
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for seed in seeds:
        for dataset in datasets:
            for method in methods:
                print(f"starting seed={seed} dataset={dataset} method={method}", flush=True)
                all_rows.extend(run(
                    dataset, method, args.rounds, args.fraction, seed,
                    args.block_size, args.novelty_weight, args.packet_header_bytes,
                    args.block_header_bytes, args.client_limit,
                ))

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.output_stem}.json"
    out_csv = out_dir / f"{args.output_stem}.csv"
    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    fields = sorted({key for row in all_rows for key in row if not isinstance(row.get(key), list)})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in all_rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
