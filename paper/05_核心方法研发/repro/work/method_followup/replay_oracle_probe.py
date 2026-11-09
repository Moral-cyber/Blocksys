"""Finite, TRAIN-gradient oracle replay from saved DEVELOPMENT tensors.

No optimizer or local training is run. Validation is used only after supports
have been selected. The empirical TRAIN gradient and dense aggregate reference
are centralized diagnostic oracles, not deployable/free-communication methods.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_followup as runner
from guarded_codec import encode


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
SCORE_KINDS = ("signed_product", "absolute_product", "sign_reference")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = float(a.norm() * b.norm())
    return float(torch.dot(a, b)) / denominator if denominator else None


def shared_core(u, fraction, tau):
    if u.ndim != 1 or u.dtype != torch.float32 or not bool(torch.isfinite(u).all()):
        raise ValueError("candidate must be a finite flattened FP32 tensor")
    if not 0 < fraction <= 1 or not 0 < tau <= 1:
        raise ValueError("fraction and tau must be in (0,1]")
    k = max(1, min(u.numel(), int(u.numel() * fraction)))
    top = torch.topk(u.abs(), k=k, sorted=True).indices
    energy = u[top].double().square()
    top_energy = float(energy.sum())
    if tau == 1 or top_energy == 0:
        core_size = k
    else:
        target = torch.tensor(tau * top_energy, dtype=torch.float64)
        core_size = min(k, int(torch.searchsorted(energy.cumsum(0), target)) + 1)
    return {"top": top, "core": top[:core_size], "k": k,
            "q": k - core_size, "top_energy": top_energy, "tau": tau}


def choose(u, plan, reference, score_kind):
    """Change only filler scoring; all variants share the exact same core/q."""
    if score_kind not in SCORE_KINDS:
        raise ValueError(score_kind)
    if reference.shape != u.shape or not bool(torch.isfinite(reference).all()):
        raise ValueError("invalid reference")
    fallback = plan["q"] == 0 or float(reference.double().square().sum()) == 0
    if fallback:
        indices = plan["top"]
    else:
        ud, gd = u.double(), reference.double()
        if score_kind == "signed_product":
            scores = ud * gd
        elif score_kind == "absolute_product":
            scores = (ud * gd).abs()
        else:
            scores = ud * gd.sign()
        available_mask = torch.ones(u.numel(), dtype=torch.bool)
        available_mask[plan["core"]] = False
        available = torch.nonzero(available_mask, as_tuple=False).flatten()
        filler = available[torch.topk(scores[available], k=plan["q"], sorted=False).indices]
        indices = torch.cat((plan["core"], filler))
    indices = indices.sort().values
    if indices.numel() != plan["k"] or indices.unique().numel() != plan["k"]:
        raise AssertionError("support cardinality changed")
    if not bool(torch.isin(plan["core"], indices).all()):
        raise AssertionError("shared core changed")
    sent = torch.zeros_like(u)
    sent[indices] = u[indices]
    residual = u - sent
    if not torch.equal(sent + residual, u):
        raise AssertionError("residual conservation failed")
    retained = float(sent.double().square().sum())
    if retained + 1e-12 * max(1., plan["top_energy"]) < plan["tau"] * plan["top_energy"]:
        raise AssertionError("energy guard failed")
    return sent, indices, {"k": plan["k"], "core_size": plan["core"].numel(),
        "filler_size": plan["q"], "reference_fallback": fallback,
        "retained_topk_energy_ratio": retained / plan["top_energy"] if plan["top_energy"] else 1.}


def full_train_gradient(model, trainsets, batch_size=128):
    """Exact empirical CE target; batch gradients are summed in FP64.

    Every TRAIN example has weight 1/N, not 1/(M*n_i). No step is applied.
    Eval mode has no behavioral effect for the frozen SmallCNN (no BN/dropout).
    """
    total = sum(map(len, trainsets))
    if not total:
        raise ValueError("empty TRAIN objective")
    before = runner.vectorize(model).clone()
    gradient = torch.zeros(before.numel(), dtype=torch.float64)
    loss_total, seen = 0., 0
    model.eval()
    for dataset in trainsets:
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            model.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
            loss.backward()
            batch_gradient = torch.cat([p.grad.detach().flatten().double() for p in model.parameters()])
            gradient.add_(batch_gradient)
            loss_total += float(loss.detach())
            seen += len(y)
    model.zero_grad(set_to_none=True)
    if seen != total or not torch.equal(before, runner.vectorize(model)):
        raise AssertionError("TRAIN gradient count/model invariant failed")
    return gradient / total, {"samples": total, "loss": loss_total / total,
        "weight_per_example": 1 / total, "gradient_accumulator_dtype": "float64",
        "backprop_dtype": str(next(model.parameters()).dtype), "batch_size": batch_size,
        "optimizer_steps": 0, "model_mode": "eval"}


def fixture():
    """Small checks with analytically distinct scoring and unequal client sizes."""
    torch.set_num_threads(1)
    u = torch.tensor([10., 1., .99, .98])
    reference = torch.tensor([0., .1, 2., -10.])
    plan = shared_core(u, .5, .9)
    expected = {"signed_product": [0, 2], "absolute_product": [0, 3], "sign_reference": [0, 1]}
    for kind, target in expected.items():
        sent, indices, _ = choose(u, plan, reference, kind)
        assert indices.tolist() == target, (kind, indices)
        frozen = encode(u, torch.zeros_like(u), fraction=.5,
            method="guard_alignment_ef", reference=reference, tau=.9)
        if kind == "signed_product":
            assert torch.equal(sent, frozen.sent)
        _, tau1, _ = choose(u, shared_core(u, .5, 1.), reference, kind)
        assert torch.equal(tau1, plan["top"].sort().values)
        _, zero, _ = choose(u, plan, torch.zeros_like(reference), kind)
        assert torch.equal(zero, plan["top"].sort().values)

    torch.manual_seed(73031)
    model = nn.Linear(3, 2).double()
    x = torch.randn(7, 3, dtype=torch.float64)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    datasets = [TensorDataset(x[:2], y[:2]), TensorDataset(x[2:], y[2:])]
    gradient, info = full_train_gradient(model, datasets, batch_size=3)
    model.zero_grad(set_to_none=True)
    direct_loss = nn.functional.cross_entropy(model(x), y)
    direct_loss.backward()
    direct = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.allclose(gradient, direct, rtol=1e-12, atol=1e-12)
    assert math.isclose(info["loss"], float(direct_loss.detach()), rel_tol=1e-12, abs_tol=1e-12)
    model.zero_grad(set_to_none=True)
    wrong = sum(nn.functional.cross_entropy(model(ds.tensors[0]), ds.tensors[1]) for ds in datasets) / 2
    wrong.backward()
    equal_client = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert float((gradient - equal_client).norm()) > 1e-3
    return {"status": "PASS", "score_supports": expected,
        "shared_core": plan["core"].tolist(), "shared_q": plan["q"],
        "signed_matches_frozen_codec": True, "tau1_and_zero_reference_match_topk": True,
        "unequal_client_gradient_matches_all_sample_mean": True,
        "incorrect_equal_client_gradient_detected": True,
        "official_test_accessed": False, "validation_gradient_used": False}


def bind_locks(dataset, helper):
    lock_path = WORKSPACE / "outputs/method_followup/checks/source_and_dataset_lock.json"
    membership_path = WORKSPACE / "outputs/method_followup/checks/dataset_membership_current.json"
    lock = json.loads(lock_path.read_text())
    membership = json.loads(membership_path.read_text())
    if not membership["all_shards_multiset_equals_official_train"] or membership["content_overlap_train_holdout_unique_hashes"] != 0:
        raise ValueError("dataset membership check not clean")
    locked = {(WORKSPACE / row["path"]).resolve(): row["sha256"] for row in lock["files"]}
    root = runner.PROJECT / "dataset" / dataset
    # These 'test' shards are local holdouts of official TRAIN, never official test.
    consumed = [helper, root / "config.json"] + [root / split / f"{i}.npz"
        for split in ("train", "test") for i in range(20)]
    verified = []
    for path in consumed:
        path = Path(path).resolve()
        actual = sha(path)
        if locked.get(path) != actual:
            raise ValueError(f"consumed input is missing or differs from source/data lock: {path}")
        verified.append({"path": str(path), "sha256": actual})
    return {"source_and_dataset_lock": {"path": str(lock_path), "sha256": sha(lock_path)},
        "dataset_membership": {"path": str(membership_path), "sha256": sha(membership_path)},
        "verified_consumed_files": verified,
        "membership_sizes": membership["sizes"],
        "official_test_files_decoded_or_evaluated": False}


def replay(probe_dir, output, *, tau=.9, threads=1, batch_size=128):
    probe_dir, output = Path(probe_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("use a new output directory")
    if threads not in (1, 2):
        raise ValueError("finite CPU probe requires one or two threads")
    torch.set_num_threads(threads)
    began = time.perf_counter()
    original = json.loads((probe_dir / "result.json").read_text())
    source = Path(original["source_run"])
    if source.parent.resolve() != (HERE / "development_runs").resolve():
        raise ValueError("only explicit frozen DEVELOPMENT runs are accepted")
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("test_final") is not False or original.get("official_test_evaluated") is not False:
        raise ValueError("not an explicitly test-free development state")
    if config["dataset"] != "Cifar10CleanA0p1" or config.get("training_backend", "cpu") != "cpu":
        raise ValueError("this finite replay is for the frozen clean CPU development state")
    for key, path in (("source_config_sha256", config_path),
        ("source_checkpoint_sha256", source / "checkpoint.pt"),
        ("shared_state_sha256", probe_dir / "shared_state.pt")):
        if original[key] != sha(path):
            raise ValueError(f"probe provenance drift: {key}")
    for key, filename in (("runner_source_sha256", "run_followup.py"), ("codec_source_sha256", "guarded_codec.py")):
        if config[key] != sha(HERE / filename):
            raise ValueError(f"frozen source drift: {filename}")
    helper = Path(inspect.getsourcefile(runner.SmallCNN)).resolve()
    locks = bind_locks(config["dataset"], helper)
    trainsets, validation, manifest = runner.prepare(config["dataset"], config["mixing"],
        config["split_seed"], config["validation_size"])
    if manifest != config["data_manifest"] or manifest != original["data_manifest"]:
        raise ValueError("source TRAIN/validation manifest drift")
    if [len(ds) for ds in trainsets] != locks["membership_sizes"]["train"]:
        raise ValueError("TRAIN sizes differ from membership check")
    state = torch.load(probe_dir / "shared_state.pt", weights_only=False, map_location="cpu")
    base, g = state["base"], state["reference"]
    candidates = [d + r for d, r in zip(state["deltas"], state["residuals"])]
    total = sum(map(len, trainsets))
    weights = [len(ds) / total for ds in trainsets]
    if weights != state["weights"] or len(candidates) != len(trainsets):
        raise ValueError("client/sample weights changed")
    plans = [shared_core(u, config["fraction"], tau) for u in candidates]
    top_values = []
    for u, plan in zip(candidates, plans):
        top = torch.zeros_like(u)
        top[plan["top"]] = u[plan["top"]]
        top_values.append(top)
    aggregate = lambda vectors: sum(w * v for w, v in zip(weights, vectors))
    Q, U = aggregate(top_values), aggregate(candidates)
    current_delta = aggregate(state["deltas"])
    model = runner.SmallCNN()
    runner.assign_vector(model, base + Q)
    gradient, gradient_info = full_train_gradient(model, trainsets, batch_size=batch_size)
    h = -gradient
    print(json.dumps({"stage": "all_train_gradient_DONE", **gradient_info}, allow_nan=False), flush=True)
    references = {"previous_global": g, "aggregate_U_oracle": U, "train_negative_gradient_oracle": h}
    supports, rows = {}, []
    def evaluate(name, sent_values, indices, reference_name=None, score_kind=None, diagnostics=None):
        sent = aggregate(sent_values)
        delta = sent - Q
        parameter_delta = (base + sent).double() - (base + Q).double()
        error = float((U - sent).double().square().sum())
        diagonal = sum(w*w * float((u-s).double().square().sum())
            for w, u, s in zip(weights, candidates, sent_values))
        runner.assign_vector(model, base + sent)
        metrics = runner.metrics(model, validation)
        row = {"variant": name, "residual_mode": "actual_EF", "reference": reference_name,
            "score_kind": score_kind, "validation": metrics,
            "selection_change_norm": float(delta.double().norm()),
            "parameter_change_norm": float(parameter_delta.norm()),
            "gradient_dotdelta": float(torch.dot(gradient, delta.double())),
            "gradient_dot_parameter_delta": float(torch.dot(gradient, parameter_delta)),
            "negative_gradient_dotdelta": float(torch.dot(h, delta.double())),
            "previous_g_dotdelta": float(torch.dot(g.double(), delta.double())),
            "aggregate_U_dotdelta": float(torch.dot(U.double(), delta.double())),
            "fresh_local_delta_dot_change_weighted": sum(w*float(torch.dot(local.double(), (s-top).double()))
                for w, local, s, top in zip(weights, state["deltas"], sent_values, top_values)),
            "fresh_local_delta_dot_change_unweighted": sum(float(torch.dot(local.double(), (s-top).double()))
                for local, s, top in zip(state["deltas"], sent_values, top_values)),
            "aggregate_error_squared": error, "independent_error_energy": diagonal,
            "error_cross_term": error-diagonal,
            "aggregate_relative_error": math.sqrt(error/(float(U.double().square().sum())+1e-30)),
            "aggregate_norm": float(sent.double().norm()),
            "same_core_and_q": indices is not None,
            "diagnostic_only": name != "topk_ef" and not (reference_name == "previous_global" and score_kind == "signed_product"),
            "reference_requires_centralized_train_information": reference_name in ("aggregate_U_oracle", "train_negative_gradient_oracle")}
        if indices is not None:
            supports[name] = indices
            row.update({"support_sha256": [tensor_sha(x) for x in indices],
                "round_analytical_bytes": len(weights)*(64+8*plans[0]["k"]),
                "mean_support_replacements": float(np.mean([plan["k"] - int(torch.isin(idx, plan["top"]).sum()) for idx, plan in zip(indices, plans)])),
                "mean_retained_topk_energy_ratio": float(np.mean([float(s.double().square().sum())/plan["top_energy"] if plan["top_energy"] else 1. for s, plan in zip(sent_values, plans)]))})
        if diagnostics is not None:
            row["clients"] = diagnostics
        rows.append(row)
        return row
    baseline = evaluate("topk_ef", top_values, [p["top"].sort().values for p in plans])
    evaluate("dense_U_diagnostic", candidates, None)
    for reference_name, reference in references.items():
        for kind in SCORE_KINDS:
            encoded = [choose(u, plan, reference, kind) for u, plan in zip(candidates, plans)]
            if reference_name == "previous_global" and kind == "signed_product":
                for u, (sent, _, _) in zip(candidates, encoded):
                    old = encode(u, torch.zeros_like(u), fraction=config["fraction"],
                        method="guard_alignment_ef", reference=g, tau=tau)
                    if not torch.equal(sent, old.sent):
                        raise AssertionError("replay signed selector differs from frozen prefix")
            row = evaluate(f"{reference_name}__{kind}", [x[0] for x in encoded],
                [x[1] for x in encoded], reference_name, kind, [x[2] for x in encoded])
            print(json.dumps({"stage": "replay", "variant": row["variant"],
                "loss_difference": row["validation"]["loss"]-baseline["validation"]["loss"],
                "gradient_dotdelta": row["gradient_dotdelta"]}), flush=True)
    fresh = [choose(u, plan, local, "signed_product")
        for u, plan, local in zip(candidates, plans, state["deltas"])]
    row = evaluate("fresh_local_delta__signed_product_EXPLORATORY", [x[0] for x in fresh],
        [x[1] for x in fresh], "fresh_local_delta_exploratory", "signed_product", [x[2] for x in fresh])
    print(json.dumps({"stage": "replay", "variant": row["variant"],
        "loss_difference": row["validation"]["loss"]-baseline["validation"]["loss"],
        "gradient_dotdelta": row["gradient_dotdelta"]}), flush=True)
    for row in rows:
        row["delta_validation_loss_vs_topk"] = row["validation"]["loss"]-baseline["validation"]["loss"]
        row["delta_validation_accuracy_pp_vs_topk"] = 100*(row["validation"]["accuracy"]-baseline["validation"]["accuracy"])
    output.mkdir(parents=True)
    oracle_path = output / "oracle_and_supports.pt"
    torch.save({"gradient_at_topk": gradient, "negative_gradient": h, "Q": Q, "U": U,
        "current_delta_aggregate": current_delta, "previous_g": g,
        "shared_core_indices": [p["core"] for p in plans], "support_indices": supports}, oracle_path)
    dependencies = [Path(__file__).resolve(), HERE / "run_followup.py", HERE / "guarded_codec.py", helper]
    result = {"status": "DONE", "scope": "one saved DEVELOPMENT next-step replay; no closed-loop training or independent repeats",
        "source_probe": str(probe_dir), "source_run": str(source), "source_round": original["source_round"],
        "probe_round": original["probe_round"], "source_variant": original["source_variant"],
        "source_seed": original["source_seed"], "backend": "cpu", "threads": threads,
        "fraction": config["fraction"], "tau": tau, "parameter_count": base.numel(),
        "residual_mode": "actual_EF", "shared_K": plans[0]["k"],
        "shared_core_sizes": [p["core"].numel() for p in plans], "shared_q": [p["q"] for p in plans],
        "gradient_objective": "CE mean over all 37500 source client TRAIN examples, weight 1/N per example; at base+TopK aggregate",
        "gradient_info": gradient_info, "gradient_used_validation": False,
        "official_test_accessed": False, "confirmation_results_accessed": False,
        "oracles_are_not_deployable_communication_free_methods": True,
        "data_manifest": manifest, "data_lock_binding": locks,
        "reference_geometry": {"norms": {name: float(ref.double().norm()) for name, ref in references.items()},
            "previous_g_cos_U": cosine(g, U), "previous_g_cos_current_delta": cosine(g, current_delta),
            "previous_g_cos_negative_gradient": cosine(g, h), "U_cos_negative_gradient": cosine(U, h),
            "current_delta_cos_negative_gradient": cosine(current_delta, h),
            "fresh_client_delta_cos_candidate": [cosine(local, u) for local, u in zip(state["deltas"], candidates)]},
        "input_sha256": {str(path): sha(path) for path in (probe_dir/"result.json", probe_dir/"shared_state.pt", config_path, source/"checkpoint.pt")},
        "executed_dependency_sha256": {str(path): sha(path) for path in dependencies},
        "oracle_and_supports_sha256": sha(oracle_path),
        "runtime": {"executable": sys.executable, "python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "elapsed_seconds": time.perf_counter()-began, "rows": rows}
    runner.assign_vector(model, base)
    runner.atomic_json(output / "result.json", result)
    lines = ["# 有限 TRAIN oracle replay", "", "只有一个开发状态；不构成独立训练重复。实际 EF，固定同一核心/q/K。", "",
        "训练梯度来自全部源 TRAIN 样本在 Top-K 更新后模型的经验 CE；验证集只评价结果。集中式 oracle 不作为免费通信算法。", "",
        "| 参考 / 控制 | 验证 loss 差 | 准确率差 pp | grad·改动 | 改动范数 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['delta_validation_loss_vs_topk']:.6f} | {row['delta_validation_accuracy_pp_vs_topk']:.4f} | {row['gradient_dotdelta']:.6f} | {row['selection_change_norm']:.6f} |")
    lines += ["", "参考几何：", "", "```json", json.dumps(result["reference_geometry"], ensure_ascii=False, indent=2), "```", "",
        "负 grad·改动只说明训练目标的一阶预测有利；完整更新仍受曲率影响，验证结果也未必改善。", "oracle 也无益时，不应仅通过修历史参考继续拯救此选择方向。"]
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": "DONE", "output": str(output), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=.9)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not math.isfinite(args.tau) or not 0 < args.tau <= 1 or args.batch_size < 1:
        parser.error("invalid tau or batch size")
    replay(args.probe, args.output, tau=args.tau, threads=args.threads, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
