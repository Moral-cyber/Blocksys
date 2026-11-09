"""One-step counterfactuals with real EF state at a DEVELOPMENT checkpoint.

Never evaluates official test. The source run must explicitly have test_final
false. Counterfactuals share model, next local deltas, reference and residuals;
they are not independent trained methods or independent repetitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_accelerated as accelerated
import run_controls as controls
import run_followup as runner
import run_lagrangian as lagrangian


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(source, output, *, backend, variants, threads=2):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"use a new probe output: {output}")
    config = json.loads((source / "config.json").read_text())
    if config.get("test_final") is not False:
        raise ValueError("probe accepts only an explicit development run without final-test evaluation")
    if config.get("dataset") != "Cifar10CleanA0p1":
        raise ValueError("this first probe version supports only the frozen clean NPZ development split")
    if config.get("training_backend", "cpu") != backend:
        raise ValueError("next local update backend must match the source trajectory")
    if config["variant"] not in ("topk_ef", "align90", "align99"):
        raise ValueError("only ordinary-local-objective source states supported; no hidden control history")
    for key, name in (("runner_source_sha256", "run_followup.py"),
                      ("codec_source_sha256", "guarded_codec.py")):
        if config.get(key) != sha(Path(__file__).parent / name):
            raise ValueError(f"source hash drift: {name}")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested MPS unavailable; no fallback")
    result = json.loads((source / "result.json").read_text())
    if result.get("status") != "DONE" or len(result.get("rows", [])) != config["rounds"]:
        raise ValueError("source development run must be complete")
    if any(v.startswith(("prox", "regtopk")) for v in variants):
        raise ValueError("shared-state probe cannot simulate different local objectives or unobserved histories")
    torch.set_num_threads(threads)
    controls.install()
    trainsets, validation, manifest = runner.prepare(config["dataset"], config["mixing"],
        config["split_seed"], config["validation_size"])
    if manifest != config["data_manifest"]:
        raise ValueError("data manifest differs from the source run")
    checkpoint = torch.load(source / "checkpoint.pt", weights_only=False, map_location="cpu")
    if len(checkpoint["rows"]) != config["rounds"]:
        raise ValueError("checkpoint is not the declared endpoint")
    model = runner.SmallCNN()
    model.load_state_dict(checkpoint["model"])
    base = runner.vectorize(model).clone()
    residuals = [r.clone() for r in checkpoint["residuals"]]
    reference = checkpoint["previous"].clone()
    if len(residuals) != len(trainsets):
        raise ValueError("residual/client count mismatch")
    weights = [len(ds) / sum(map(len, trainsets)) for ds in trainsets]
    began = time.perf_counter()
    deltas = [accelerated.train_local(model, ds,
        config["seed"] + config["rounds"] * 100 + i,
        config["lr"], config["local_epochs"], backend=backend)
        for i, ds in enumerate(trainsets)]
    if not torch.equal(base, runner.vectorize(model)):
        raise AssertionError("local training changed the shared global model")
    output.mkdir(parents=True)
    torch.save({"base": base, "reference": reference, "residuals": residuals,
                "deltas": deltas, "weights": weights}, output / "shared_state.pt")
    rows = []
    for residual_mode in ("actual_EF", "zero_residual_control"):
        rems = residuals if residual_mode == "actual_EF" else [torch.zeros_like(r) for r in residuals]
        candidates = [d + r for d, r in zip(deltas, rems)]
        dense = sum(w * u for w, u in zip(weights, candidates))
        dense_norm2 = float(dense.double().square().sum())
        for variant in ["dense_candidate_reference", *variants]:
            if variant == "dense_candidate_reference":
                sent, diags = dense, []
                err, independent = 0., 0.
                wire = 20 * (64 + 4 * base.numel())
            else:
                encoded = [lagrangian.encode_variant(d, r, reference,
                    config["fraction"], variant, config["seed"] + config["rounds"] * 100 + i)
                    for i, (d, r) in enumerate(zip(deltas, rems))]
                for u, (s, r, diag) in zip(candidates, encoded):
                    if not torch.equal(u, s + r):
                        raise AssertionError("candidate/residual conservation failed")
                    if diag["selected_coordinates"] != max(1, int(base.numel() * config["fraction"])):
                        raise AssertionError("K mismatch")
                sent = sum(w * e[0] for w, e in zip(weights, encoded))
                err = float((dense - sent).double().square().sum())
                independent = sum(w*w * float((u-e[0]).double().square().sum())
                    for w, u, e in zip(weights, candidates, encoded))
                diags = [e[2] for e in encoded]
                wire = sum(d["encoded_bytes"] for d in diags)
            runner.assign_vector(model, base + sent)
            row = {"variant": variant, "residual_mode": residual_mode,
                "validation": runner.metrics(model, validation),
                "aggregate_error_squared": err,
                "aggregate_relative_error": (err / (dense_norm2 + 1e-30))**.5,
                "independent_error_energy": independent,
                "error_cross_term": err-independent,
                "aggregate_norm": float(sent.double().norm()),
                "reference_cosine_dense_candidate": float(torch.dot(reference.double(), dense.double()) /
                    (reference.double().norm() * dense.double().norm() + 1e-30)),
                "round_analytical_bytes": wire,
                "mean_filler_fraction": float(np.mean([d["filler_fraction"] for d in diags])) if diags else 0.,
                "mean_retained_topk_energy_ratio": float(np.mean([d["retained_topk_energy_ratio"] for d in diags])) if diags else None}
            rows.append(row)
        baseline = next(r for r in rows if r["residual_mode"] == residual_mode and r["variant"] == "topk_ef")
        for row in rows:
            if row["residual_mode"] == residual_mode:
                row["delta_loss_vs_topk"] = row["validation"]["loss"] - baseline["validation"]["loss"]
                row["delta_accuracy_pp_vs_topk"] = 100 * (row["validation"]["accuracy"] - baseline["validation"]["accuracy"])
                row["delta_aggregate_relative_error_vs_topk"] = row["aggregate_relative_error"] - baseline["aggregate_relative_error"]
    runner.assign_vector(model, base)
    metadata = {"status": "DONE", "source_run": str(source), "source_variant": config["variant"],
        "source_round": config["rounds"], "probe_round": config["rounds"]+1,
        "source_seed": config["seed"], "backend": backend, "variants": variants,
        "source_config_sha256": sha(source / "config.json"),
        "source_checkpoint_sha256": sha(source / "checkpoint.pt"),
        "shared_state_sha256": sha(output / "shared_state.pt"),
        "probe_source_sha256": sha(__file__),
        "dependency_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "data_manifest": manifest, "official_test_evaluated": False,
        "scope": "one shared next-step counterfactual from a completed DEVELOPMENT state; not independent training or replicates",
        "elapsed_seconds": time.perf_counter() - began, "rows": rows}
    runner.atomic_json(output / "result.json", metadata)
    print(json.dumps({k:metadata[k] for k in ("status", "source_variant", "probe_round", "official_test_evaluated", "elapsed_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "mps"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--variants", default="topk_ef,align90,align99,random90,innov90,reverse90_ef,shuffle90_ef,fixed50_ef,sign50_ef")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if "topk_ef" not in variants or args.threads < 1:
        parser.error("include topk_ef; threads must be positive")
    probe(args.source, args.output, backend=args.backend, variants=variants, threads=args.threads)


if __name__ == "__main__":
    main()
