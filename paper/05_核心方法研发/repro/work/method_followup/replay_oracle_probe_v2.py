"""Bounded extension of a completed oracle replay; preserves all v1 files.

Reuses saved h(base+Q), computes exactly one extra full-TRAIN gradient at BASE,
and adds current dense delta / BASE-gradient controls with the same core/q/K.
No local training, validation gradient, official test, or closed-loop run.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import replay_oracle_probe as v1


def extend(parent_dir, output, threads=1, batch_size=128):
    parent_dir, output = Path(parent_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("use a new v2 output directory")
    if threads not in (1, 2):
        raise ValueError("CPU threads must be 1 or 2")
    torch.set_num_threads(threads)
    began = time.perf_counter()
    parent = json.loads((parent_dir / "result.json").read_text())
    if parent["status"] != "DONE" or parent["official_test_accessed"] or parent["gradient_used_validation"]:
        raise ValueError("parent must be an explicitly test/validation-gradient-free completed replay")
    for path, digest in {**parent["input_sha256"], **parent["executed_dependency_sha256"]}.items():
        if v1.sha(path) != digest:
            raise ValueError(f"parent input/dependency drift: {path}")
    if v1.sha(parent_dir / "oracle_and_supports.pt") != parent["oracle_and_supports_sha256"]:
        raise ValueError("parent oracle/support artifact changed")
    probe_dir = Path(parent["source_probe"])
    source = Path(parent["source_run"])
    if source.parent.resolve() != (v1.HERE / "development_runs").resolve():
        raise ValueError("explicit DEVELOPMENT sources only")
    config = json.loads((source / "config.json").read_text())
    if config["test_final"] is not False or config["dataset"] != "Cifar10CleanA0p1":
        raise ValueError("incompatible source")
    helper = Path(inspect.getsourcefile(v1.runner.SmallCNN)).resolve()
    locks = v1.bind_locks(config["dataset"], helper)
    trainsets, validation, manifest = v1.runner.prepare(config["dataset"], config["mixing"],
        config["split_seed"], config["validation_size"])
    if manifest != parent["data_manifest"]:
        raise ValueError("source data changed")
    state = torch.load(probe_dir / "shared_state.pt", map_location="cpu", weights_only=False)
    old_oracles = torch.load(parent_dir / "oracle_and_supports.pt", map_location="cpu", weights_only=False)
    base, g, weights = state["base"], state["reference"], state["weights"]
    if weights != [len(ds)/sum(map(len, trainsets)) for ds in trainsets]:
        raise ValueError("sample weights changed")
    candidates = [d+r for d, r in zip(state["deltas"], state["residuals"])]
    plans = [v1.shared_core(u, parent["fraction"], parent["tau"]) for u in candidates]
    if [p["q"] for p in plans] != parent["shared_q"]:
        raise ValueError("shared q changed")
    top_values = []
    for u, plan in zip(candidates, plans):
        sent = torch.zeros_like(u)
        sent[plan["top"]] = u[plan["top"]]
        top_values.append(sent)
    aggregate = lambda values: sum(w*v for w, v in zip(weights, values))
    Q, U, current_delta = aggregate(top_values), aggregate(candidates), aggregate(state["deltas"])
    for name, tensor in (("Q", Q), ("U", U), ("current_delta_aggregate", current_delta)):
        if not torch.equal(tensor, old_oracles[name]):
            raise ValueError(f"shared tensor differs from parent: {name}")
    gradient_topk = old_oracles["gradient_at_topk"]
    model = v1.runner.SmallCNN()
    v1.runner.assign_vector(model, base+Q)
    baseline_metrics = v1.runner.metrics(model, validation)
    baseline = next(row for row in parent["rows"] if row["variant"] == "topk_ef")
    if baseline_metrics != baseline["validation"]:
        raise ValueError("parent baseline validation did not replay exactly")
    # Only BASE and TRAIN enter this gradient calculation. Q is not its anchor.
    v1.runner.assign_vector(model, base)
    gradient_base, info_base = v1.full_train_gradient(model, trainsets, batch_size=batch_size)
    h_base = -gradient_base
    print(json.dumps({"stage": "all_train_gradient_at_BASE_DONE", **info_base}), flush=True)
    references = {"current_dense_delta_oracle": current_delta,
                  "BASE_train_negative_gradient_oracle": h_base}
    rows, supports = [], {}
    for reference_name, reference in references.items():
        for kind in v1.SCORE_KINDS:
            encoded = [v1.choose(u, plan, reference, kind) for u, plan in zip(candidates, plans)]
            sent_values = [item[0] for item in encoded]
            sent = aggregate(sent_values)
            delta = sent-Q
            parameter_delta = (base+sent).double()-(base+Q).double()
            error = float((U-sent).double().square().sum())
            diagonal = sum(w*w*float((u-s).double().square().sum())
                for w, u, s in zip(weights, candidates, sent_values))
            v1.runner.assign_vector(model, base+sent)
            metrics = v1.runner.metrics(model, validation)
            name = f"{reference_name}__{kind}"
            supports[name] = [item[1] for item in encoded]
            row = {"variant": name, "residual_mode": "actual_EF", "reference": reference_name,
                "score_kind": kind, "validation": metrics, "selection_change_norm": float(delta.double().norm()),
                "parameter_change_norm": float(parameter_delta.norm()),
                "gradient_dotdelta": float(torch.dot(gradient_topk, delta.double())),
                "gradient_dot_parameter_delta": float(torch.dot(gradient_topk, parameter_delta)),
                "BASE_gradient_dotdelta": float(torch.dot(gradient_base, delta.double())),
                "BASE_gradient_dot_parameter_delta": float(torch.dot(gradient_base, parameter_delta)),
                "previous_g_dotdelta": float(torch.dot(g.double(), delta.double())),
                "current_dense_delta_dotdelta": float(torch.dot(current_delta.double(), delta.double())),
                "fresh_local_delta_dot_change_weighted": sum(w*float(torch.dot(local.double(), (s-top).double()))
                    for w, local, s, top in zip(weights, state["deltas"], sent_values, top_values)),
                "fresh_local_delta_dot_change_unweighted": sum(float(torch.dot(local.double(), (s-top).double()))
                    for local, s, top in zip(state["deltas"], sent_values, top_values)),
                "aggregate_error_squared": error, "independent_error_energy": diagonal,
                "error_cross_term": error-diagonal,
                "aggregate_relative_error": math.sqrt(error/(float(U.double().square().sum())+1e-30)),
                "same_core_and_q": True, "diagnostic_only": True,
                "reference_requires_centralized_train_information": True,
                "round_analytical_bytes": len(weights)*(64+8*plans[0]["k"]),
                "mean_support_replacements": float(np.mean([plan["k"]-int(torch.isin(item[1], plan["top"]).sum()) for item, plan in zip(encoded, plans)])),
                "mean_retained_topk_energy_ratio": float(np.mean([item[2]["retained_topk_energy_ratio"] for item in encoded])),
                "support_sha256": [v1.tensor_sha(item[1]) for item in encoded], "clients": [item[2] for item in encoded],
                "delta_validation_loss_vs_topk": metrics["loss"]-baseline_metrics["loss"],
                "delta_validation_accuracy_pp_vs_topk": 100*(metrics["accuracy"]-baseline_metrics["accuracy"])}
            rows.append(row)
            print(json.dumps({"stage": "replay_v2", "variant": name,
                "loss_difference": row["delta_validation_loss_vs_topk"],
                "accuracy_difference_pp": row["delta_validation_accuracy_pp_vs_topk"]}), flush=True)
    output.mkdir(parents=True)
    new_oracles = output / "new_oracles_and_supports.pt"
    torch.save({"gradient_at_BASE": gradient_base, "negative_gradient_at_BASE": h_base,
        "current_dense_delta": current_delta, "support_indices": supports}, new_oracles)
    geometry = {"previous_g_cos_BASE_h": v1.cosine(g, h_base), "U_cos_BASE_h": v1.cosine(U, h_base),
        "current_dense_delta_cos_BASE_h": v1.cosine(current_delta, h_base),
        "BASE_h_cos_TopK_h": v1.cosine(h_base, -gradient_topk),
        "current_dense_delta_cos_TopK_h": v1.cosine(current_delta, -gradient_topk),
        "BASE_h_norm": float(h_base.norm()), "current_dense_delta_norm": float(current_delta.double().norm())}
    result = {"status": "DONE", "version": 2, "scope": "finite extension of one saved DEVELOPMENT state; inherited v1 rows are not new independent runs",
        "parent_replay_directory": str(parent_dir), "parent_result_sha256": v1.sha(parent_dir/"result.json"),
        "parent_oracles_sha256": v1.sha(parent_dir/"oracle_and_supports.pt"),
        "source_probe": str(probe_dir), "source_run": str(source), "source_variant": parent["source_variant"],
        "source_round": parent["source_round"], "probe_round": parent["probe_round"],
        "backend": "cpu", "threads": threads, "fraction": parent["fraction"], "tau": parent["tau"],
        "shared_K": parent["shared_K"], "shared_core_sizes": parent["shared_core_sizes"], "shared_q": parent["shared_q"],
        "residual_mode": "actual_EF", "gradient_BASE_info": info_base,
        "gradient_BASE_anchor_uses_only_base_model_not_Q": True,
        "gradient_used_validation": False, "official_test_accessed": False, "confirmation_results_accessed": False,
        "local_training_or_optimizer_steps": 0, "oracles_not_communication_free": True,
        "reference_geometry_v2": geometry, "reference_geometry_v1": parent["reference_geometry"],
        "data_manifest": manifest, "data_lock_binding": locks,
        "executed_dependency_sha256": {**parent["executed_dependency_sha256"], str(Path(__file__).resolve()): v1.sha(__file__)},
        "new_oracles_and_supports_sha256": v1.sha(new_oracles),
        "inherited_variants": [row["variant"] for row in parent["rows"]],
        "new_variants": [row["variant"] for row in rows], "rows": parent["rows"]+rows,
        "runtime": parent["runtime"], "elapsed_seconds_v2": time.perf_counter()-began}
    v1.runner.atomic_json(output/"result.json", result)
    lines = ["# 有限 TRAIN oracle replay v2", "", "同一源状态；继承 v1 的12行，仅新增6行。未重训客户端；新增一次全 TRAIN 的 BASE 梯度。", "",
        "BASE h 在原 w 处求梯度，Top-K h 在 w+Q 处求梯度；二者均是集中式诊断 oracle。current dense delta 也不视为免费通信参考。", "",
        "| 参考 / 控制 | 验证 loss 差 | 准确率差 pp | grad(w+Q)·改动 | grad(w)·改动 | 改动范数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['delta_validation_loss_vs_topk']:.6f} | {row['delta_validation_accuracy_pp_vs_topk']:.4f} | {row['gradient_dotdelta']:.6f} | {row['BASE_gradient_dotdelta']:.6f} | {row['selection_change_norm']:.6f} |")
    lines += ["", "参考几何：", "", "```json", json.dumps(geometry, indent=2), "```", "",
        "当前 dense delta 与局部 delta 都是多步本地优化后的更新，不等同于当前全局目标的负梯度。单状态有效也不构成稳定优势；未在验证集上计算梯度。"]
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": "DONE", "output": str(output), "elapsed_seconds_v2": result["elapsed_seconds_v2"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("invalid batch size")
    extend(args.parent, args.output, args.threads, args.batch_size)
