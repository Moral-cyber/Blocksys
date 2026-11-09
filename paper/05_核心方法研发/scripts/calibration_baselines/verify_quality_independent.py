"""Independent NumPy saved-array probability-quality verification.

No author metric/solver modules, torch, model, original images or TEST are used.
No calibrator is fitted. ECE bins are reconstructed by explicit interval masks.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def independent_score(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or logits.shape != (len(labels), 10) or not len(labels) or not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= 10)):
        raise ValueError("finite Nx10 scores and integer N-vector GT required")
    logp = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
    probabilities = np.exp(logp)
    correct = (np.argmax(logits, axis=1) == labels).astype(np.float64)
    confidence = probabilities.max(axis=1)
    target = np.eye(10)[labels]
    brier = np.sum((probabilities - target) ** 2, axis=1)
    bins = []
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & ((confidence < upper) if index < 14 else (confidence <= 1 + 1e-14))
        bins.append({"index": index, "lower": lower, "upper": upper, "right_closed": index == 14, "count": int(mask.sum()), "confidence_sum": float(np.sum(confidence[mask])), "correct_sum": float(np.sum(correct[mask]))})
    return {"samples": len(labels), "accuracy": float(correct.mean()), "nll": float(-logp[np.arange(len(labels)), labels].mean()), "brier_sum_classes": float(brier.mean()), "ece_15_equal_width": sum(abs(item["correct_sum"] - item["confidence_sum"]) for item in bins) / len(labels), "bins": bins}


def verify(root, out):
    checks, snapshots, summaries = [], {}, {}

    def check(name, passed, detail=None):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    def read(path):
        path = Path(path).resolve()
        if "test_batch" in str(path) or "t10k" in str(path) or path.name in {f"data_batch_{i}" for i in range(1, 6)}:
            raise ValueError("saved arrays only; image/TEST reads refused")
        data = path.read_bytes()
        snapshots[str(path)] = {"sha256": digest(data), "bytes": len(data)}
        return data

    def arrays(path):
        with np.load(io.BytesIO(read(path)), allow_pickle=False) as contents:
            return {key: contents[key].copy() for key in contents.files}

    report = json.loads(read(root / "result.json"))
    calroot = Path(report["calibration_root"])
    source = json.loads(read(calroot / "result.json"))
    original = Path(source["input_root"])
    original_result = json.loads(read(original / "result.json"))
    check("completed descriptive source and unchanged candidate decision", report["status"] == source["status"] == original_result["status"] == "DONE" and report["new_training"] is False and report["original_candidate_decision_unchanged"] == source["original_candidate_decision_unchanged"] == original_result["decision"])
    for filename, expected in report["source_identities"].items():
        check("protected identity " + filename, digest(read(filename)) == expected)
    # Independent analytic boundary cases, without calling the author function.
    uniform = independent_score(np.zeros((4, 10)), np.array([0, 1, 2, 3]))
    hot = np.full((2, 10), -1000.)
    hot[np.arange(2), [1, 2]] = 1000.
    onehot = independent_score(hot, np.array([1, 3]))
    check("uniform boundary and deterministic tie", abs(uniform["nll"] - np.log(10)) < 1e-14 and abs(uniform["brier_sum_classes"] - .9) < 1e-14 and abs(uniform["ece_15_equal_width"] - .15) < 1e-14 and uniform["accuracy"] == .25)
    check("confidence1 last-bin / multiclass Brier2 wrong-onehot", onehot["accuracy"] == .5 and onehot["brier_sum_classes"] == 1 and onehot["ece_15_equal_width"] == .5 and onehot["bins"][-1]["count"] == 2)
    maximum_metric_error = maximum_bin_sum_error = maximum_summary_error = 0.
    all_metrics, scored_pairs = {}, 0
    for method, arm in report["results"].items():
        client_metrics = []
        for i, client in enumerate(arm["clients"]):
            prefix = f"{method}/client{i}: "
            data, cal = arrays(original / f"{method}_client{i}.npz"), arrays(calroot / f"{method}_client{i}.npz")
            labels = data["labels"] if "labels" in data else data["holdout_labels"]
            y = data["train_labels"]
            counts = np.bincount(y, minlength=10)
            prior = np.log((counts + 1.) / (len(y) + 10)).astype(np.float32)
            check(prefix + "GT types/IDs and exact TRAIN prior", labels.dtype.kind in "iu" and labels.ndim == 1 and np.array_equal(cal["holdout_ids"], data["holdout_ids"]) and np.array_equal(cal["train_ids"], data["train_ids"]) and np.array_equal(prior, data["prior_shift"]) and np.array_equal(counts, data["train_counts"]))
            scores = {key.removesuffix("_holdout_logits"): value for key, value in cal.items() if key.endswith("_holdout_logits")}
            scores["prior_only"] = np.broadcast_to(prior, (len(labels), 10))
            check(prefix + "all expected saved score variants", set(scores) == set(client["metrics"]) == set(arm["summaries"]))
            metrics = {}
            for variant, logits in scores.items():
                metric = independent_score(logits, labels)
                recorded = client["metrics"][variant]
                err = max(abs(metric[key] - recorded[key]) for key in ("accuracy", "nll", "brier_sum_classes", "ece_15_equal_width"))
                maximum_metric_error = max(maximum_metric_error, err)
                check(prefix + variant + " independent NLL/Brier/ECE/accuracy", err < 1e-11 and metric["samples"] == recorded["samples"] == len(labels) and 0 <= metric["ece_15_equal_width"] <= 1 and 0 <= metric["brier_sum_classes"] <= 2 and metric["nll"] >= 0)
                bins_ok = sum(item["count"] for item in metric["bins"]) == len(labels)
                for item, saved in zip(metric["bins"], recorded["bins"]):
                    bins_ok &= all(item[key] == saved[key] for key in ("index", "lower", "upper", "right_closed", "count", "correct_sum"))
                    error = abs(item["confidence_sum"] - saved["confidence_sum"])
                    maximum_bin_sum_error = max(maximum_bin_sum_error, error)
                    bins_ok &= error < 1e-9
                check(prefix + variant + " all15 bins / empty bins / counts", bins_ok and len(recorded["bins"]) == 15)
                if variant != "prior_only":
                    calmetric = source["results"][method]["clients"][i]["metrics"][variant]
                    check(prefix + variant + " original accuracy/NLL retained", abs(metric["accuracy"] - calmetric["accuracy"]) < 1e-14 and abs(metric["nll"] - calmetric["loss"]) < 1e-11)
                metrics[variant] = metric
                scored_pairs += 1
            client_metrics.append(metrics)
        method_summaries = {}
        for variant, saved in arm["summaries"].items():
            metrics = [client[variant] for client in client_metrics]
            n = np.asarray([metric["samples"] for metric in metrics])
            pooled = np.sum([[item["correct_sum"] - item["confidence_sum"] for item in metric["bins"]] for metric in metrics], axis=0)
            values = {"pooled_ece_15_equal_width": float(np.sum(np.abs(pooled)) / n.sum())}
            for key in ("accuracy", "nll", "brier_sum_classes", "ece_15_equal_width"):
                x = np.asarray([metric[key] for metric in metrics])
                values["macro_" + key] = float(np.sum(x) / len(x))
                values["weighted_" + key] = float(np.sum(x * n) / n.sum())
            err = max(abs(values[key] - value) for key, value in saved.items())
            maximum_summary_error = max(maximum_summary_error, err)
            check(f"{method}/{variant}: all9 macro/weighted/pooled summaries", len(saved) == 9 and err < 1e-11 and values["pooled_ece_15_equal_width"] <= values["weighted_ece_15_equal_width"] + 1e-14)
            method_summaries[variant] = values
        summaries[method] = method_summaries
        all_metrics[method] = client_metrics
    unchanged = all(digest(Path(path).read_bytes()) == value["sha256"] for path, value in snapshots.items())
    check("all read inputs unchanged", unchanged)
    failures = [item for item in checks if not item["pass"]]
    result = {"status": "DONE", "artifact_check": "FAIL" if failures else "PASS", "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures, "review": "REVIEW_UNAVAILABLE", "training_seed": original_result["training_seed"], "original_candidate_decision_unchanged": original_result["decision"], "scored_client_variant_pairs": scored_pairs, "summary_values": sum(len(values) for arm in summaries.values() for values in arm.values()), "summaries": summaries, "max_metric_error": maximum_metric_error, "max_bin_confidence_sum_error": maximum_bin_sum_error, "max_summary_error": maximum_summary_error, "snapshots": snapshots, "inputs_unchanged": unchanged, "verifier_sha256": digest(Path(__file__).read_bytes()), "definitions": {"ECE": "15 fixed equal-width intervals [k/15,(k+1)/15), last includes1; empty bins contribute0; macro is mean client ECE, weighted is sample-weighted client ECE, pooled merges signed bin gaps before absolute value", "Brier": "mean sum_c(p_c-onehot_c)^2, unnormalized ten-class [0,2]", "prior_only": "one fixed TRAIN-count Dirichlet1 posterior-mean class probability per client, scored against the saved dataset GT"}, "limits": ["Descriptive saved-holdout metrics, no new gate or parameter selection. Lower NLL does not imply lower ECE or Brier.", "Saved logits are not independently established by CNN inference; no original images/TEST/model/torch/solver imports or fitting.", "score_quality.py casts labels to int64 before validation, so its standalone API does not reject fractional input labels. All actual saved labels here are integer vectors, so this is not an observed result defect.", "ECE is bin-dependent and clients are not independent training seeds. Pooled ECE may cancel client-specific bin gaps, and is not the sample-weighted mean of client ECE."]}
    if out.exists():
        raise ValueError("fresh independent output required")
    out.mkdir(parents=True)
    (out / "LOCAL_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    text = f"# 概率质量指标独立核验：seed{result['training_seed']}\n产物{result['artifact_check']} {result['checks_passed']}/{result['checks_total']}；{scored_pairs}组client×variant，全部15bin与{result['summary_values']}项macro/weighted/pooled汇总独立复算。最大指标差{maximum_metric_error:.2g}、汇总差{maximum_summary_error:.2g}、bin confidence_sum差{maximum_bin_sum_error:.2g}。\nECE为15固定等宽bin，最后含1，空bin贡献0；macro、weighted与pooled分开。Brier为10类平方误差总和、未除类别数，[0,2]；prior_only只由TRAIN计数预测、对保存dataset GT评分。原候选决定{original_result['decision']}不变；较低NLL不自动代表ECE更好。\n未导入作者score/solver、torch/model，未读原图/TEST，未拟合或训练；所有{len(snapshots)}输入hash前后相同。实际GT均整型；source通用接口先cast标签再检查，对fractional标签未显式拒绝，但本批无此输入。外部REVIEW_UNAVAILABLE。\n"
    (out / "LOCAL_CHECK.md").write_text(text)
    print(json.dumps({key: result[key] for key in ("status", "training_seed", "artifact_check", "checks_passed", "checks_total", "scored_client_variant_pairs", "summary_values", "max_metric_error", "max_summary_error", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    verify(arguments.quality_root.resolve(), arguments.output_dir.resolve())
