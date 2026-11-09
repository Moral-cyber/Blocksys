"""Independent NumPy recomputation of saved PFL development arrays; no torch/data.

Original source/PLAN, configs and checkpoint bytes are read/hash checked only.
No model import, checkpoint deserialization, image dataset or new training.
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

METHODS = ("ce_uniform3", "fedlc_uniform3", "balanced_uniform3", "balanced_dense")
VARIANTS = ("raw", "analytic_prior", "prior_only", "learned_head", "learned_head_plus_prior")
SOURCE_SHA = "69ddadabc8797646aa65874a2418baa77ccde1818d28bf8df00513bda5df4d15"
PLAN_SHA = "05d460669282ba30e3e7cdc03060b395069ba78740f2108b6f991c7bbb91d58a"
CE_TOL = 5e-6
ACC_TOL = 2e-14
STATIC_NOTES = [
    "probe.py:31,39-44: loads official TRAIN only; prior counts are bincount(labels[client TRAIN IDs]), checked against frozen training manifest, then shift=log((n_ic+1)/(n_i+10)). Holdout labels do not fit the prior.",
    "probe.py:45-54: head baseline freezes all parameters then unfreezes only final Linear, runs actual CPU forward/backward/SGD .03 for one TRAIN epoch, batch64. The frozen feature extractor is still forwarded on every TRAIN batch.",
    "probe.py:55-63: raw and adapted models both evaluate every holdout input, then five score arrays use the same holdout GT. Learned head is also offered the same analytic prior.",
    "probe.py:67-82: macro means give each of 20 clients equal weight; weighted means use actual holdout n_i/total; q10 is linear 10th percentile. Learned comparator takes max accuracy among learned_head/learned_head_plus_prior across three low-byte models; dense remains a diagnostic.",
    "probe.py:83-89: model/config input hashes are checked and source/PLAN identified; original diagnostic status is DONE. This pure NumPy checker independently recomputes saved-array scores, not the trained heads or frozen logits.",
    "PLAN.md:6-9 states 37500/12500; actual shared split is 37502/12498 due to per-client round(.75*n). Both original result personal_eval strings and PLAN retain 12500 and need this explicit warning.",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def nested_hash(groups):
    data = bytearray()
    for group in groups:
        data.extend(struct.pack("<I", len(group)))
        data.extend(np.asarray(group, dtype="<u4").tobytes())
    return sha(data)


def metrics(score, labels):
    # Preserve the FP32 score construction used by the author, then compute CE
    # in FP64 via stable log-sum-exp, without torch's reduction implementation.
    values = score.astype(np.float64)
    rowmax = values.max(axis=1)
    lse = rowmax + np.log(np.exp(values - rowmax[:, None]).sum(axis=1))
    losses = lse - values[np.arange(len(labels)), labels]
    correct = int(np.count_nonzero(np.argmax(score, axis=1) == labels))
    return {"accuracy": correct / len(labels), "loss": float(losses.mean()), "samples": int(len(labels)), "correct": correct, "loss_sum_fp64": float(losses.sum())}


def summarize(clients, variant):
    ms = [client["metrics"][variant] for client in clients]
    accuracies = np.array([m["accuracy"] for m in ms])
    losses = np.array([m["loss"] for m in ms])
    counts = np.array([m["samples"] for m in ms])
    ordered = np.sort(accuracies)
    pos = .1 * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    q10 = ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])
    return {"macro_accuracy": float(accuracies.mean()), "macro_loss": float(losses.mean()), "weighted_accuracy": sum(m["correct"] for m in ms) / int(counts.sum()), "weighted_loss": sum(m["loss_sum_fp64"] for m in ms) / int(counts.sum()), "client_accuracy_q10": float(q10), "correct_total": sum(m["correct"] for m in ms), "samples_total": int(counts.sum()), "loss_sum_fp64": sum(m["loss_sum_fp64"] for m in ms)}


def verify(here, diagnostic, sources):
    checks, snapshots, warnings, arms, manifests = [], {}, [], {}, {}
    max_client_ce = max_summary_ce = max_acc = max_prior = 0.
    common_npz = {}

    def check(name, passed, detail=None):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    def read(path):
        data = path.read_bytes()
        snapshots[str(path.resolve())] = {"bytes": len(data), "sha256": sha(data)}
        return data

    author = json.loads(read(diagnostic / "result.json"))
    check("author diagnostic DONE and scope flags", author["status"] == "DONE" and author["official_test_loaded"] is False and author["new_fl_training"] is False and author["extra_head_baseline_training"] is True)
    check("frozen source/PLAN identity", sha(read(here / "probe.py")) == author["source_sha256"] == SOURCE_SHA and sha(read(here / "PLAN.md")) == author["plan_sha256"] == PLAN_SHA)
    for method in METHODS:
        folder = sources / method
        conf = json.loads(read(folder / "config.json"))
        ckbytes = read(folder / "checkpoint.pt")
        flresult = json.loads(read(folder / "result.json"))
        man = conf["data_manifest"]
        manifests[method] = man
        prefix = method + ": "
        check(prefix + "exact model/config inputs match diagnostic hashes", author["input_sha256"][str((folder / "config.json").resolve())] == sha((folder / "config.json").read_bytes()) and author["input_sha256"][str((folder / "checkpoint.pt").resolve())] == sha(ckbytes))
        unsigned = {k: v for k, v in conf.items() if k != "signature"}
        check(prefix + "source model final20 metadata/signature", conf["method"] == method and conf["seed"] == 20261201 and conf["partition_seed"] == 20261193 and conf["rounds"] == 20 and flresult["status"] == "DONE" and len(flresult["rows"]) == 20 and flresult["rows"][-1]["round"] == 20 and conf["signature"] == flresult["signature"] == sha(json.dumps(unsigned, sort_keys=True).encode()) and conf["official_test_loaded"] is False and flresult["official_test_loaded"] is False)
        train, held, globaldev = man["train_indices"], man["holdout_indices"], man["validation_indices"]
        trainids, heldids = [i for group in train for i in group], [i for group in held for i in group]
        check(prefix + "all TRAIN/holdout IDs disjoint and exhaustive", len(trainids) == len(set(trainids)) == man["train_samples"] and len(heldids) == len(set(heldids)) == man["holdout_samples"] and set(trainids).isdisjoint(heldids) and set(trainids + heldids) == set(range(50000)) and len(globaldev) == len(set(globaldev)) == 2000 and set(globaldev) <= set(heldids))
        check(prefix + "saved ID hashes", nested_hash(train) == man["train_indices_sha256"] and nested_hash(held) == man["holdout_indices_sha256"] and sha(np.asarray(globaldev, dtype="<u4").tobytes()) == man["validation_ids_sha256"])
        check(prefix + "per-client rounded split", all(len(tr) == max(1, min(len(tr) + len(ho) - 1, round(.75 * (len(tr) + len(ho))))) for tr, ho in zip(train, held)))
        if man["train_samples"] != 37500 or man["holdout_samples"] != 12500:
            warnings.append({"method": method, "warning": "PLAN/result personal_eval says37500/12500; actual rounded split37502/12498", "actual_train": man["train_samples"], "actual_holdout": man["holdout_samples"], "prior_globaldev_reused": len(globaldev), "action": "disclose without altering old PLAN or gate"})
        source_clients = author["results"][method]["clients"]
        check(prefix + "complete20 clients/order and five variants", len(source_clients) == len(train) == len(held) == 20 and [c["client"] for c in source_clients] == list(range(20)) and all(set(c["metrics"]) == set(VARIANTS) for c in source_clients))
        clients = []
        gradients = train_forward_batches = eval_forward_batches = 0
        train_forward_samples = eval_forward_samples = 0
        seconds = 0.
        for i, (tr, ho, original) in enumerate(zip(train, held, source_clients)):
            cp = prefix + "client " + str(i) + ": "
            path = diagnostic / f"{method}_client{i}.npz"
            with np.load(io.BytesIO(read(path)), allow_pickle=False) as npz:
                arrays = {key: npz[key].copy() for key in npz.files}
            check(cp + "NPZ exact schema", set(arrays) == {"raw", "learned", "labels", "prior_shift", "holdout_ids", "train_counts"})
            raw, learned, labels = arrays["raw"], arrays["learned"], arrays["labels"]
            counts, ids, storedshift = arrays["train_counts"], arrays["holdout_ids"], arrays["prior_shift"]
            check(cp + "finite logits/GT shapes and types", raw.shape == learned.shape == (len(ho), 10) and raw.dtype == learned.dtype == np.float32 and np.isfinite(raw).all() and np.isfinite(learned).all() and labels.shape == (len(ho),) and np.issubdtype(labels.dtype, np.integer) and np.all((labels >= 0) & (labels < 10)) and storedshift.shape == (10,) and storedshift.dtype == np.float32)
            check(cp + "holdout IDs exact order and no train overlap", np.array_equal(ids, np.asarray(ho, dtype=np.uint32)) and len(np.unique(ids)) == len(ho) and set(map(int, ids)).isdisjoint(tr))
            expected_counts = np.asarray(man["client_label_counts"][i], dtype=np.int64)
            check(cp + "prior count origin matches TRAIN manifest", counts.shape == (10,) and np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0) and np.array_equal(counts, expected_counts) and int(counts.sum()) == len(tr) == original["client_train_samples"])
            held_counts = np.bincount(labels, minlength=10)
            original_counts = np.asarray(man["original_client_label_counts"][i], dtype=np.int64)
            check(cp + "holdout GT count reconciles with original group minus TRAIN", np.array_equal(counts + held_counts, original_counts))
            check(cp + "TRAIN counts also match every frozen FL round", all(row["clients"][i]["train_counts"] == counts.tolist() for row in flresult["rows"]))
            prior = (counts.astype(np.float64) + 1) / (len(tr) + 10)
            shift = np.log(prior).astype(np.float32)
            priorerr = float(np.max(np.abs(storedshift - shift)))
            max_prior = max(max_prior, priorerr)
            check(cp + "independent TRAIN-only Dirichlet1 logprior", np.array_equal(storedshift, shift) and np.isfinite(storedshift).all() and abs(prior.sum() - 1) < 1e-14, {"max_abs_error": priorerr})
            if i not in common_npz:
                common_npz[i] = {k: arrays[k] for k in ("labels", "holdout_ids", "train_counts", "prior_shift")}
            else:
                check(cp + "same GT/IDs/counts/prior across models", all(np.array_equal(arrays[key], common_npz[i][key]) for key in common_npz[i]))
            nsteps, evalbatches = math.ceil(len(tr) / 64), math.ceil(len(ho) / 256)
            check(cp + "actual adaptation count and zero analytic gradients", original["adaptation_gradient_evaluations"] == nsteps and original["analytic_extra_gradient_evaluations"] == 0 and original["holdout_samples"] == len(ho) and np.isfinite(original["adaptation_seconds"]) and original["adaptation_seconds"] >= 0)
            scores = {"raw": raw, "analytic_prior": np.add(raw, shift, dtype=np.float32), "prior_only": np.broadcast_to(shift, raw.shape), "learned_head": learned, "learned_head_plus_prior": np.add(learned, shift, dtype=np.float32)}
            recomputed = {}
            for variant, score in scores.items():
                m = metrics(score, labels)
                ref = original["metrics"][variant]
                ceerr, accerr = abs(m["loss"] - ref["loss"]), abs(m["accuracy"] - ref["accuracy"])
                max_client_ce, max_acc = max(max_client_ce, ceerr), max(max_acc, accerr)
                check(cp + variant + " independent CE/accuracy/count", m["samples"] == ref["samples"] == len(ho) and ceerr <= CE_TOL and accerr <= ACC_TOL, {"ce_abs_error": ceerr, "accuracy_abs_error": accerr})
                recomputed[variant] = m
            clients.append({"client": i, "train_samples": len(tr), "holdout_samples": len(ho), "train_counts": counts.tolist(), "holdout_label_counts": np.bincount(labels, minlength=10).tolist(), "prior_probability_fp64": prior.tolist(), "prior_shift_fp32": shift.astype(float).tolist(), "adaptation_gradient_evaluations_recorded": nsteps, "adaptation_feature_forward_batches_derived": nsteps, "adaptation_feature_forward_samples_derived": len(tr), "evaluation_two_model_forward_batches_derived": 2 * evalbatches, "evaluation_two_model_forward_samples_derived": 2 * len(ho), "analytic_extra_gradient_evaluations": 0, "metrics": recomputed})
            gradients += nsteps
            train_forward_batches += nsteps
            train_forward_samples += len(tr)
            eval_forward_batches += 2 * evalbatches
            eval_forward_samples += 2 * len(ho)
            seconds += original["adaptation_seconds"]
        summaries = {variant: summarize(clients, variant) for variant in VARIANTS}
        for variant, summary in summaries.items():
            reference = author["results"][method]["summaries"][variant]
            for key, target in reference.items():
                error = abs(summary[key] - target)
                tol = CE_TOL if key.endswith("loss") else ACC_TOL
                check(prefix + variant + ": " + key + " independent summary", error <= tol, {"abs_error": error})
                if key.endswith("loss"):
                    max_summary_ce = max(max_summary_ce, error)
                else:
                    max_acc = max(max_acc, error)
        arms[method] = {"train_samples": man["train_samples"], "holdout_samples": man["holdout_samples"], "prior_globaldev_overlap_samples": len(set(globaldev) & set(heldids)), "summaries": summaries, "clients": clients, "adaptation_gradient_evaluations_recorded_total": gradients, "adaptation_feature_forward_batches_derived": train_forward_batches, "adaptation_feature_forward_samples_derived": train_forward_samples, "evaluation_two_model_forward_batches_derived": eval_forward_batches, "evaluation_two_model_forward_samples_derived": eval_forward_samples, "analytic_extra_gradient_evaluations_total": 0, "adaptation_seconds_recorded_total": seconds}
    base_manifest = manifests[METHODS[0]]
    for method in METHODS:
        check(method + ": exact shared development partition", manifests[method] == base_manifest)
        check(method + ": prior-only invariant across models", arms[method]["summaries"]["prior_only"] == arms[METHODS[0]]["summaries"]["prior_only"])
    candidate = arms["balanced_uniform3"]["summaries"]["analytic_prior"]
    gates, comparisons = {}, {}
    for baseline in ("ce_uniform3", "fedlc_uniform3"):
        ref = arms[baseline]["summaries"]["analytic_prior"]
        gates[baseline] = {"ce": candidate["macro_loss"] < ref["macro_loss"], "accuracy": candidate["macro_accuracy"] >= ref["macro_accuracy"] + .005}
        comparisons[baseline] = {"macro_ce_reduction": ref["macro_loss"] - candidate["macro_loss"], "macro_accuracy_gain": candidate["macro_accuracy"] - ref["macro_accuracy"]}
    learned = [(arms[m]["summaries"][v]["macro_accuracy"], m, v) for m in METHODS[:3] for v in ("learned_head", "learned_head_plus_prior")]
    best = max(learned)
    all_learned = [(arms[m]["summaries"][v]["macro_accuracy"], m, v) for m in METHODS for v in ("learned_head", "learned_head_plus_prior")]
    best_including_dense = max(all_learned)
    check("including dense learned heads leaves best comparator unchanged", best_including_dense == best)
    gates["vs_learned"] = {"accuracy": candidate["macro_accuracy"] >= best[0] - .005}
    decision = "FOLLOW_UP_ONLY" if all(all(item.values()) for item in gates.values()) else "STOP_THIS_FIXED_PERSONALIZATION_RECIPE"
    check("independent frozen gates/decision equal recorded", gates == author["gates"] and decision == author["decision"])
    dense = arms["balanced_dense"]["summaries"]["analytic_prior"]
    dense_effect = {"quantized_minus_dense_macro_accuracy": candidate["macro_accuracy"] - dense["macro_accuracy"], "quantized_minus_dense_macro_ce": candidate["macro_loss"] - dense["macro_loss"]}
    unchanged = all(sha(Path(path).read_bytes()) == record["sha256"] for path, record in snapshots.items())
    check("all saved inputs/source/PLAN unchanged after verification", unchanged)
    failures = [item for item in checks if not item["pass"]]
    return {"status": "DONE", "artifact_check": "FAIL" if failures else "PASS", "protocol_fidelity": "WARN" if warnings else "PASS", "warnings": warnings, "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures, "review": "REVIEW_UNAVAILABLE", "evaluation": "final-round20 PFL client-TRAIN-holdout development; not global IID and not independent test", "decision": decision, "gates": gates, "comparisons": comparisons, "best_learned_lowbyte_comparator": {"method": best[1], "variant": best[2], "macro_accuracy": best[0], "candidate_accuracy_gap": candidate["macro_accuracy"] - best[0]}, "best_learned_including_dense_comparator": {"method": best_including_dense[1], "variant": best_including_dense[2], "macro_accuracy": best_including_dense[0]}, "dense_comparison": dense_effect, "max_client_ce_abs_difference_vs_torch_record": max_client_ce, "max_summary_ce_abs_difference_vs_torch_record": max_summary_ce, "max_accuracy_or_q10_abs_difference": max_acc, "max_fp32_prior_shift_abs_difference": max_prior, "ce_verification_tolerance": CE_TOL, "accuracy_verification_tolerance": ACC_TOL, "arms": arms, "aggregate_compute": {"models": 4, "clients_per_model": 20, "head_adaptation_gradient_evaluations_recorded": sum(arm["adaptation_gradient_evaluations_recorded_total"] for arm in arms.values()), "head_adaptation_feature_forward_batches_derived": sum(arm["adaptation_feature_forward_batches_derived"] for arm in arms.values()), "head_adaptation_feature_forward_samples_derived": sum(arm["adaptation_feature_forward_samples_derived"] for arm in arms.values()), "two_model_holdout_forward_batches_derived": sum(arm["evaluation_two_model_forward_batches_derived"] for arm in arms.values()), "two_model_holdout_forward_samples_derived": sum(arm["evaluation_two_model_forward_samples_derived"] for arm in arms.values()), "analytic_extra_gradient_evaluations": 0, "notes": "head baseline actually trains; derived forward counts follow code/data batch dimensions, not instrumentation or hardware FLOPs"}, "snapshots": snapshots, "all_inputs_unchanged": unchanged, "static_source_notes": STATIC_NOTES, "limits": ["This recomputes all400 client score metrics and100 summary metrics from80 saved NPZ using NumPy FP64 stable CE, preserving FP32 score additions. It does not reevaluate images/model features, rerun adaptation or reproduce FL.", "Checkpoint bytes are hash checked only; pure NumPy never deserializes torch checkpoints. Final20 provenance is supported by source result metadata, exact checkpoint hashes, and the earlier frozen global-run audit.", "Counts match saved TRAIN manifest, every frozen FL round, and reconcile with NPZ holdout GT counts to original group counts. Source computes them from TRAIN IDs. Original training labels are not in these NPZ, so this checker does not independently reconstruct TRAIN counts from original labels. Holdout GT/ID consistency is checked across all four models.", "12498 clientholdout samples include all2000 previously used globaldev samples; same seed/partition and frozen final-round models. This is non-independent PFL development, not new global IID or official test evidence.", "20 clients share each global training run and are not20 independent seeds; no seed-level significance or generalization claim is made.", "Analytic prior uses no extra gradient or feature forward beyond ordinary raw inference, but requires TRAIN counts and score addition. Learned-head comparator has actual extra head gradients and whole-body feature forwards; its plus-prior version is included.", "Gate is unchanged: balanced+prior must beat CE+prior and FedLC+prior by >=.5pp macro accuracy and lower macro CE, and stay within.5pp of best low-byte learned counterpart. Winning another comparator cannot rescue a failed gate.", "Source implementation uses learned comparator from three low-byte models; dense is diagnostic. Even including dense learned heads would not change best comparator or this STOP result.", "No official TEST, original dataset load, torch/model import, new FL/head training or source/PLAN change in this checker. External cross-model review is unavailable: REVIEW_UNAVAILABLE."]}


def report(out):
    lines = ["# 解析先验个性化：独立NumPy数组核验", "", f"状态 **{out['status']}**；数组/产物核验 **{out['artifact_check']}（{out['checks_passed']}/{out['checks_total']}）**；方案数字一致性 **{out['protocol_fidelity']}**；冻结决定 **{out['decision']}**；外部审查 `REVIEW_UNAVAILABLE`。", "", "这是最终轮20的PFL客户端分布开发诊断，不是global IID指标或独立test。实际TRAIN/holdout=37502/12498，原PLAN和result的37500/12500另列WARN；12498包含此前globaldev全部2000样本。未改旧PLAN。", "", "| 模型 | 分数 | macro acc% | macro CE | weighted acc% | weighted CE | client q10 acc% |", "|---|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for variant in VARIANTS:
            s = out["arms"][method]["summaries"][variant]
            lines.append(f"| {method} | {variant} | {100*s['macro_accuracy']:.6f} | {s['macro_loss']:.10f} | {100*s['weighted_accuracy']:.6f} | {s['weighted_loss']:.10f} | {100*s['client_accuracy_q10']:.6f} |")
    lines += ["", "固定门槛和复算："]
    for baseline, comp in out["comparisons"].items():
        lines.append(f"- 相对{baseline}+prior：macro CE降低{comp['macro_ce_reduction']:.10f}，macro accuracy差{100*comp['macro_accuracy_gain']:.6f}pp；gate={out['gates'][baseline]}。")
    best = out["best_learned_lowbyte_comparator"]
    lines.append(f"- best learned低字节对照={best['method']}/{best['variant']}，macro accuracy={100*best['macro_accuracy']:.6f}%；候选差{100*best['candidate_accuracy_gap']:.6f}pp；不低超过.5pp条件={out['gates']['vs_learned']['accuracy']}。")
    dense = out["dense_comparison"]
    lines.append(f"- 量化减dense：macro accuracy={100*dense['quantized_minus_dense_macro_accuracy']:.6f}pp、macro CE={dense['quantized_minus_dense_macro_ce']:.10f}，只诊断压缩代价。")
    lines += ["", f"400条client score指标、100项summary全部复算；最大CE误差client={out['max_client_ce_abs_difference_vs_torch_record']:.3g}、summary={out['max_summary_ce_abs_difference_vs_torch_record']:.3g}（核对容差{CE_TOL}）；accuracy/q10最大误差={out['max_accuracy_or_q10_abs_difference']:.3g}；FP32 prior shift最大误差={out['max_fp32_prior_shift_abs_difference']}。CE用独立FP64 logsumexp；score加法保留原FP32。", "", "宏平均每client等权；weighted用真实holdout样本数；q10由排序后的第1.9位置线性插值。JSON保留逐client正确分子、CE总和、真实分母、prior/count和完整summary。"]
    for method, arm in out["arms"].items():
        lines.append(f"- {method}：额外head适配记录{arm['adaptation_gradient_evaluations_recorded_total']}次梯度；据代码/实际batch数推导featureforward {arm['adaptation_feature_forward_batches_derived']}批/{arm['adaptation_feature_forward_samples_derived']}个TRAIN样本；raw+learned holdout推理{arm['evaluation_two_model_forward_batches_derived']}批/{arm['evaluation_two_model_forward_samples_derived']}个样本forward。")
    c = out["aggregate_compute"]
    lines += ["", f"4模型合计head适配{c['head_adaptation_gradient_evaluations_recorded']}次梯度与featureforward {c['head_adaptation_feature_forward_batches_derived']}批/{c['head_adaptation_feature_forward_samples_derived']}样本；两个模型holdout推理{c['two_model_holdout_forward_batches_derived']}批/{c['two_model_holdout_forward_samples_derived']}样本。解析prior额外梯度0；不能把learned head适配说成零计算。forward为代码与批大小推导，不是硬件FLOPs测量。", "", "只读源码证据："]
    lines.extend("- " + note for note in out["static_source_notes"])
    lines += ["", "证据边界："] + ["- " + note for note in out["limits"]]
    if out["failures"]:
        lines += ["", "失败项："] + [f"- {item['check']}: {item['detail']}" for item in out["failures"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--diagnostic-dir", type=Path, default=here / "diagnostic_v1")
    parser.add_argument("--source-model-dir", type=Path, default=here.parent / "balanced_quant/development_p20261193_s20261201")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    out = verify(here, args.diagnostic_dir, args.source_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LOCAL_CHECK.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "LOCAL_CHECK.md").write_text(report(out))
    print(json.dumps({k: out[k] for k in ("status", "artifact_check", "protocol_fidelity", "checks_passed", "checks_total", "decision", "gates", "comparisons", "aggregate_compute", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
