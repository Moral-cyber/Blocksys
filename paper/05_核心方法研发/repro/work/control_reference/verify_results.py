"""Independent read-only saved-result verification for existing control references.

Reads only saved config/result/checkpoint files, never training code or datasets.
The head-only dense run is a reused same-seed background, not a new repeat.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch

METHODS = ("plain_dense", "body_control", "full_control")
BACKGROUND = "head_control_dense_same_seed_background"
BODY, HEAD, TOTAL, ROUNDS = 136224, 650, 136874, 20
SIZES = {"plain_dense": 0, "body_control": BODY, "full_control": TOTAL, BACKGROUND: HEAD}
SOURCE_HASH = "5a6f1376c6e5b997cd05a044fbc52060382484e80118314f8599d2ff13490223"
PLAN_HASH = "d67374ec7621b9977b345be85c464b23c10192745f27c03d133576b1187b97db"
BACKGROUND_SOURCE_HASH = "8ee487ea523ee772a99e3555174cc9f41d2c9e1ade300f58b3df778562c63b82"
BACKGROUND_PLAN_HASH = "31b014425b47eb7af3fe36c51d755246e57127ebed68034f4e02572db277b48e"
STATIC_NOTES = [
    "run.py:43-51,89-98: checked last-650 head layout; none/body/full select 0/136224/136874 coordinate prefixes. Body correction stops before the head.",
    "run.py:75,85-103: fixed c-ci per round, one CE backward per actual batch, raw gradient accumulated before correction; ci_new=ci-c-delta_subset/(.03*actual_steps) is checked against raw gradient mean at 5e-6.",
    "run.py:143-146,196-220: with full participation and fixed p, sum(p*ci)=c implies sum(p*ci_new)=-sum(p*delta_subset/(eta*K)); prior and new invariants are checked. This does not establish partial-participation support.",
    "run.py:161-169,223-236: one analytical dense delta upload, analytical dense model download, plus 16+4d control download; per-client/global controls are each 4d and all 21 simulated control vectors cost 21*4d. The 64B header/K layout is assumed, not serialized.",
    "run.py:239-244: version/config/signature/weights/client_sizes/control tensors and complete metric rows are checkpointed; result uses those same metric rows.",
    "run.py:267-271; PLAN.md:3-13: fixed official-TRAIN development split, existing SCAFFOLD/SCALLION references; reused head_control_dense has the same seed/partition and is not an independent repeat or new algorithm.",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def group_hash(groups):
    buf = bytearray()
    for ids in groups:
        buf.extend(struct.pack("<I", len(ids)))
        buf.extend(np.asarray(ids, dtype="<u4").tobytes())
    return sha(buf)


def normal(value):
    return json.loads(json.dumps(value))


def maxerr(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.


def last5(rows):
    rows = rows[15:20]
    classes = np.asarray([r["validation"]["class_accuracy"] for r in rows], dtype=float)
    means = classes.mean(axis=0)
    return {"rounds": [r["round"] for r in rows],
        "ce": float(np.mean([r["validation"]["loss"] for r in rows])),
        "accuracy": float(np.mean([r["validation"]["accuracy"] for r in rows])),
        "class_accuracy_mean": means.tolist(),
        "macro_accuracy_mean": float(means.mean()),
        "worst_class_accuracy_mean_over_round_minima": float(classes.min(axis=1).mean()),
        "minimum_of_mean_class_accuracy": float(means.min()),
        "minimum_mean_class_indices": np.flatnonzero(means == means.min()).tolist(),
        "round_worst_class_indices": [np.flatnonzero(row == row.min()).tolist() for row in classes]}


def verify(root, background):
    checks, snapshots, arms, configs, histories = [], {}, {}, {}, {}

    def check(name, passed, detail=None):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    def read(path):
        data = path.read_bytes()
        snapshots[str(path.resolve())] = {"bytes": len(data), "sha256": sha(data)}
        return data

    for method in (*METHODS, BACKGROUND):
        isbg = method == BACKGROUND
        folder = background if isbg else root / method
        paths = [folder / n for n in ("config.json", "result.json", "checkpoint.pt")]
        if not all(p.exists() for p in paths):
            arms[method] = {"status": "NOT_STARTED_OR_NO_COMPLETE_SNAPSHOT", "rounds": 0}
            continue
        try:
            config, result = [json.loads(read(p)) for p in paths[:2]]
            checkpoint = torch.load(io.BytesIO(read(paths[2])), map_location="cpu", weights_only=False)
        except Exception as exc:
            arms[method] = {"status": "SNAPSHOT_READ_ERROR", "error": repr(exc)}
            continue
        configs[method] = config
        rows, ckrows = result["rows"], normal(checkpoint["rows"])
        histories[method] = rows
        man, size = config["manifest"], SIZES[method]
        sizes = man["client_sizes"]
        nclients = len(sizes)
        weights = np.asarray(sizes, dtype=float) / sum(sizes)
        steps = [math.ceil(n / 64) for n in sizes]
        prefix = method + ": "
        unsigned = {k: v for k, v in config.items() if k != "signature"}
        check(prefix + "config/result/checkpoint signature", sha(json.dumps(unsigned, sort_keys=True).encode()) == config["signature"] == result["signature"] == checkpoint["signature"])
        check(prefix + "frozen source/plan metadata", config["source_sha256"] == (BACKGROUND_SOURCE_HASH if isbg else SOURCE_HASH) and config["plan_sha256"] == (BACKGROUND_PLAN_HASH if isbg else PLAN_HASH))
        check(prefix + "fixed setup", config["method"] == ("head_control_dense" if isbg else method) and config["seed"] == 20261191 and config["rounds"] == ROUNDS and config["backend"] == "mps" and config["body"] == BODY and config["head"] == HEAD and config["batch"] == 64 and config["lr"] == .03 and config["local_epochs"] == 1 and nclients == 20)
        check(prefix + "sample weights", np.allclose(config["weights"], weights, rtol=0, atol=1e-15) and abs(sum(config["weights"]) - 1) < 1e-12)
        check(prefix + "official test excluded in metadata", config["official_test_loaded"] is False and result["official_test_loaded"] is False and man["test_used_for_selection"] is False and man["official_split"] == "CIFAR-10 train (50,000 original rows)")
        train, held, dev = man["train_indices"], man["holdout_indices"], man["validation_indices"]
        ti, hi = [i for group in train for i in group], [i for group in held for i in group]
        check(prefix + "development split integrity", len(ti) == len(set(ti)) == 37500 and len(hi) == len(set(hi)) == 12500 and set(ti).isdisjoint(hi) and set(ti + hi) == set(range(50000)) and len(dev) == len(set(dev)) == 2000 and set(dev) <= set(hi) and man["validation_class_counts"] == [200] * 10)
        check(prefix + "manifest ID hashes/sizes", group_hash(train) == man["train_indices_sha256"] and group_hash(held) == man["holdout_indices_sha256"] and sha(np.asarray(dev, dtype="<u4").tobytes()) == man["validation_ids_sha256"] and [len(ids) for ids in train] == sizes)
        check(prefix + "finite saved values", finite(rows) and finite(checkpoint))
        check(prefix + "round identifiers/status", len(rows) <= ROUNDS and [r["round"] for r in rows] == list(range(1, len(rows) + 1)) and result["status"] == ("DONE" if len(rows) == ROUNDS else "RUNNING"))
        common = min(len(rows), len(ckrows))
        check(prefix + "checkpoint/result common rows", rows[:common] == ckrows[:common], {"result_rounds": len(rows), "checkpoint_rounds": len(ckrows)})
        if result["status"] == "DONE":
            check(prefix + "final checkpoint/result complete rows", rows == ckrows)
        model = checkpoint["model"]
        check(prefix + "model parameter layout", sum(x.numel() for x in model.values()) == TOTAL and list(model)[-2:] == ["classifier.3.weight", "classifier.3.bias"] and sum(model[k].numel() for k in list(model)[-2:]) == HEAD)
        cis, c = checkpoint["cis"], checkpoint["c"]
        check(prefix + "control tensor layout", len(cis) == nclients and all(isinstance(v, torch.Tensor) and v.shape == (size,) and v.dtype == torch.float32 and v.device.type == "cpu" for v in [*cis, c]))
        weighted = sum((float(w) * ci for w, ci in zip(weights, cis)), torch.zeros(size))
        final_control_error = maxerr(weighted, c)
        check(prefix + "final independently weighted control", final_control_error <= 5e-6 and abs(float(c.norm()) - ckrows[-1]["control_norm"]) <= 5e-6)
        base_bytes, extra_down = 64 + 4 * TOTAL, 16 + 4 * size if size else 0
        if not isbg:
            check(prefix + "checkpoint/config identity and weights", checkpoint["version"] == config["checkpoint_version"] == 1 and normal(checkpoint["config"]) == config and checkpoint["weights"] == config["weights"] and checkpoint["client_sizes"] == sizes == config["client_sizes"])
            check(prefix + "final client control norms match checkpoint", all(abs(float(ci.norm()) - client["control_state_norm"]) <= 5e-6 for ci, client in zip(cis, ckrows[-1]["clients"])))
            check(prefix + "control dimensions/state accounting", config["parameters"] == TOTAL and config["control_size"] == size and config["control_subset"] == ("prefix" if size else "none") and config["control_state_per_client_bytes"] == config["global_control_state_bytes"] == 4 * size and config["simulation_control_state_bytes"] == (nclients + 1) * 4 * size)
            check(prefix + "analytical config communication", config["upload_bytes_analytical_per_client"] == config["download_model_bytes_analytical_per_client"] == base_bytes and config["upload_control_bytes"] == 0 and config["download_control_bytes_analytical_per_client"] == extra_down)
        else:
            check(prefix + "background exact dense transport", config["upload_cap_bytes"] is None and config["control_client_state_bytes"] == 2600 and all(torch.count_nonzero(v) == 0 for v in checkpoint["residuals"]))
        max_formula = max_prior = max_inv = 0.
        gradients = optimizers = upload = down_model = down_control = 0
        for row in rows:
            rp, clients = prefix + "round " + str(row["round"]) + ": ", row["clients"]
            check(rp + "actual K/gradients", len(clients) == nclients and [cl["client"] for cl in clients] == list(range(nclients)) and [cl["steps"] for cl in clients] == steps and [cl["gradient_evaluations"] for cl in clients] == steps)
            if not isbg:
                check(rp + "actual optimizer counts/seeds/samples/weights", [cl["optimizer_steps"] for cl in clients] == steps and [cl["seed"] for cl in clients] == [20261191 + 100 * (row["round"] - 1) + i for i in range(nclients)] and [cl["samples"] for cl in clients] == sizes and np.allclose([cl["weight"] for cl in clients], weights, rtol=0, atol=1e-15) and row["actual_gradient_evaluations"] == row["actual_optimizer_steps"] == sum(steps))
                check(rp + "fixed global anchor diagnostic", row["global_anchor_unchanged_during_clients"] is True and len(row["global_anchor_sha256"]) == 64)
                check(rp + "analytical communication", all(cl["upload_bytes_analytical"] == base_bytes for cl in clients) and row["upload_bytes_analytical"] == nclients * base_bytes and row["download_model_bytes_analytical"] == nclients * base_bytes and row["download_control_bytes_analytical"] == nclients * extra_down and row["download_bytes_analytical"] == nclients * (base_bytes + extra_down))
                max_prior = max(max_prior, row["prior_control_invariant_error"])
                optimizers += row["actual_optimizer_steps"]
            else:
                check(rp + "background analytical communication", all(cl["upload_bytes"] == base_bytes for cl in clients) and row["upload_bytes"] == nclients * base_bytes and row["download_bytes_analytical"] == nclients * (base_bytes + extra_down))
            max_formula = max(max_formula, *(cl["control_formula_max_error"] for cl in clients))
            max_inv = max(max_inv, row["control_invariant_error"])
            gradients += sum(cl["gradient_evaluations"] for cl in clients)
            upload += nclients * base_bytes
            down_model += nclients * base_bytes
            down_control += nclients * extra_down
            val = row["validation"]
            classes = np.asarray(val["class_accuracy"], dtype=float)
            check(rp + "class/macro/worst/micro saved metric arithmetic", val["samples"] == 2000 and classes.shape == (10,) and np.all((classes >= 0) & (classes <= 1)) and np.allclose(classes * 200, np.rint(classes * 200), rtol=0, atol=1e-8) and abs(classes.mean() - val["macro_accuracy"]) <= 1e-14 and abs(classes.mean() - val["accuracy"]) <= 3e-8 and classes.min() == val["worst_class_accuracy"] and val["loss"] >= 0)
        check(prefix + "recorded control diagnostics bounded", 0 <= max_formula <= 5e-6 and 0 <= max_prior <= 5e-6 and 0 <= max_inv <= 5e-6)
        if size == 0:
            check(prefix + "plain zero control", all(v.numel() == 0 for v in [*cis, c]) and max_formula == max_prior == max_inv == 0 and all(row["control_norm"] == 0 for row in rows))
        arms[method] = {"status": result["status"], "rounds": len(rows), "checkpoint_rounds": len(ckrows), "role": "reused same-seed/partition background; not independent repetition" if isbg else "existing SCAFFOLD control diagnostic baseline", "control_coordinates": size, "control_scope": "suffix head" if isbg else "none" if size == 0 else "prefix body" if size == BODY else "all parameters", "actual_steps_per_client_per_round": steps, "gradient_evaluations_total": gradients, "optimizer_steps_total": None if isbg else optimizers, "optimizer_steps_background_note": "not separately logged in reused background" if isbg else None, "upload_bytes_analytical_total": upload, "download_model_bytes_analytical_total": down_model, "download_control_bytes_analytical_total": down_control, "total_communication_bytes_analytical": upload + down_model + down_control, "extra_control_download_bytes_per_client_per_round": extra_down, "control_state_per_client_bytes": 4 * size, "global_control_state_bytes": 4 * size, "simulation_control_state_bytes": 4 * size * (nclients + 1), "control_formula_max_error_recorded": max_formula, "prior_control_invariant_max_error_recorded": max_prior, "control_invariant_max_error_recorded": max_inv, "checkpoint_weighted_control_max_error": final_control_error}
    if configs:
        baseline = configs.get("plain_dense", next(iter(configs.values())))
        for method, config in configs.items():
            check(method + ": common development protocol/partition", all(config[k] == baseline[k] for k in ("seed", "rounds", "backend", "manifest", "body", "head", "lr", "batch", "local_epochs", "weights", "evaluation", "official_test_loaded")))
            check(method + ": common helper identities", config["runtime"]["accelerated_source_sha256"] == baseline["runtime"]["accelerated_source_sha256"] and config["runtime"]["accelerated_model_source_sha256"] == baseline["runtime"]["accelerated_model_source_sha256"] and config["runtime"]["accelerated_dependency_versions"] == baseline["runtime"]["accelerated_dependency_versions"])
        anchors = [histories[m][0]["global_anchor_sha256"] for m in METHODS if histories.get(m)]
        check("new reference arms share saved initial global anchor", len(set(anchors)) <= 1)
    complete = all(arms[m].get("status") == "DONE" and arms[m].get("rounds") == arms[m].get("checkpoint_rounds") == ROUNDS for m in (*METHODS, BACKGROUND))
    comparison = None
    if complete:
        for method in arms:
            arms[method]["last5"] = last5(histories[method])
        plain = arms["plain_dense"]["last5"]
        comparison = {method: {"ce_reduction_vs_plain": plain["ce"] - arm["last5"]["ce"], "accuracy_gain_vs_plain": arm["last5"]["accuracy"] - plain["accuracy"], "mean_round_worst_class_gain_vs_plain": arm["last5"]["worst_class_accuracy_mean_over_round_minima"] - plain["worst_class_accuracy_mean_over_round_minima"], "per_class_accuracy_gain_vs_plain": (np.array(arm["last5"]["class_accuracy_mean"]) - plain["class_accuracy_mean"]).tolist()} for method, arm in arms.items() if method != "plain_dense"}
    failures = [c for c in checks if not c["pass"]]
    inputs_unchanged = all(sha(Path(name).read_bytes()) == item["sha256"] for name, item in snapshots.items())
    check("saved inputs unchanged after verification", inputs_unchanged)
    failures = [c for c in checks if not c["pass"]]
    return {"experiment_status": "ALL_DONE" if complete else "RUNNING", "artifact_check": "FAIL" if failures else "PASS", "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures, "review": "REVIEW_UNAVAILABLE", "scope": "independent saved-artifact arithmetic; existing control baselines; no new-algorithm GO claim", "arms": arms, "comparisons": comparison, "snapshots": snapshots, "saved_inputs_unchanged_after_verification": inputs_unchanged, "checks": checks, "separately_read_static_source_notes": STATIC_NOTES, "limits": ["Existing SCAFFOLD full/body/head control references, not a new algorithm or independent confirmation of a contribution.", "Outcomes apply to this fixed 20-round, lr=.03, local-epoch1 SmallCNN/CIFAR10 development protocol. They do not establish what longer schedules or other fixed protocols would do, or invalidate SCAFFOLD generally.", "Same seed20261191 and partition20261143; reused head_control_dense adds no independent seed. Rounds are not independent repeats.", "No training or dataset/official TEST load in this checker; saved metrics are checked/reaveraged, not recomputed from predictions.", "Per-step raw gradients and updates are not saved; local telescoping and historical server invariants are recorded training diagnostics. Final control weighting is independently recomputed.", "MPS local training is supported by saved metadata/static source; this CPU checker does not independently reproduce its trajectory.", "All communication values are analytical, including assumed 64B delta header with actual K. No serialization/network/authentication/energy/chain measurement.", "Fixed full participation and fixed sample weights are essential to server delta-only control reconstruction.", "No external cross-model reviewer is available: REVIEW_UNAVAILABLE."]}


def report(out):
    lines = ["# 既有控制基线：本地独立结果核验", "", f"实验状态：**{out['experiment_status']}**；保存产物检查 **{out['artifact_check']}（{out['checks_passed']}/{out['checks_total']}）**；外部审查 `REVIEW_UNAVAILABLE`。", "", "输入仅为四组已保存 config/result/checkpoint；不导入训练源文件，不加载数据、不训练。head背景复用同seed/分区，不构成独立重复。", "", "| 组 | 轮数/状态 | last5 CE | last5 acc% | mean round-worst% | min mean-class% | 梯度次数 | 解析总通信B |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, arm in out["arms"].items():
        m = arm.get("last5")
        nums = [f"{m['ce']:.10f}", f"{100*m['accuracy']:.6f}", f"{100*m['worst_class_accuracy_mean_over_round_minima']:.6f}", f"{100*m['minimum_of_mean_class_accuracy']:.6f}"] if m else ["—"] * 4
        lines.append(f"| {name} | {arm.get('rounds', 0)}/{arm['status']} | " + " | ".join(nums) + f" | {arm.get('gradient_evaluations_total', '—')} | {arm.get('total_communication_bytes_analytical', '—')} |")
    if out["experiment_status"] != "ALL_DONE":
        lines += ["", "RUNNING：三组新参考与既有head背景全部20轮DONE后才统一重算last5；无去留或方法优势结论。"]
    else:
        lines += ["", "逐类 last5 accuracy 均值（class 0–9，%）：", "", "| 组 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, arm in out["arms"].items():
            lines.append("| " + name + " | " + " | ".join(f"{100*x:.4f}" for x in arm["last5"]["class_accuracy_mean"]) + " |")
        lines += ["", "mean round-worst = 每轮10类最小值再取5轮均值；min mean-class = 先取每类5轮均值再取最小值，两者分别报告。"]
        for name, comp in out["comparisons"].items():
            lines.append(f"- {name} 相对plain：CE降低{comp['ce_reduction_vs_plain']:.10f}，accuracy差{100*comp['accuracy_gain_vs_plain']:.6f}pp，mean round-worst差{100*comp['mean_round_worst_class_gain_vs_plain']:.6f}pp。这是已有控制基线的同协议诊断，不支持新算法结论。")
    for name, arm in out["arms"].items():
        if "control_coordinates" not in arm:
            continue
        lines += ["", f"{name}：控制范围={arm['control_scope']}，维数={arm['control_coordinates']}；实际每client每轮steps={arm['actual_steps_per_client_per_round']}；gradient={arm['gradient_evaluations_total']}，optimizer={arm['optimizer_steps_total']}（背景未单独记录optimizer）。", f"解析上行={arm['upload_bytes_analytical_total']}B，模型下行={arm['download_model_bytes_analytical_total']}B，控制下行={arm['download_control_bytes_analytical_total']}B；额外控制下行={arm['extra_control_download_bytes_per_client_per_round']}B/client/round；每client控制state={arm['control_state_per_client_bytes']}B，全模拟控制state={arm['simulation_control_state_bytes']}B（含server）。", f"记录的公式/prior invariant/new invariant最大误差={arm['control_formula_max_error_recorded']:.3g}/{arm['prior_control_invariant_max_error_recorded']:.3g}/{arm['control_invariant_max_error_recorded']:.3g}；最终checkpoint独立加权control误差={arm['checkpoint_weighted_control_max_error']:.3g}。"]
    lines += ["", "单独只读静态核对（不属于脚本的运行输入）："] + ["- " + x for x in out["separately_read_static_source_notes"]]
    lines += ["", "证据边界："] + ["- " + x for x in out["limits"]]
    if out["failures"]:
        lines += ["", "失败项："] + [f"- {c['check']}: {c['detail']}" for c in out["failures"]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument("--input-dir", type=Path, default=here / "development_s20261191")
    p.add_argument("--background-dir", type=Path, default=here.parent / "head_control/development_s20261191/head_control_dense")
    p.add_argument("--output-dir", type=Path, default=here)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = verify(args.input_dir, args.background_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LOCAL_STATS_CHECK.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "LOCAL_RESULTS_CHECK.md").write_text(report(out))
    print(json.dumps({"experiment_status": out["experiment_status"], "artifact_check": out["artifact_check"], "checks_passed": out["checks_passed"], "checks_total": out["checks_total"], "failures": out["failures"][:5]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
