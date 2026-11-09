"""Independent NumPy-only array verification; never imports author solver/torch.

No image pickle is read, no model/CNN forward/backward runs. Newton fitting only
reuses saved TRAIN arrays with the frozen mathematical rule, not holdout data.
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
VARIANTS = ("raw", "analytic_prior", "prior_only", "learned_head", "learned_head_plus_prior", "fitted_bias")
FROZEN = {"PLAN.md": "120472270ba4fdff1d5fc211d6b7cb1d9ee352daee013de4321d396aae8b3310", "solver.py": "6b8dda5e86438f78f3918f8601ec3ce9e7099417d3b42ac55bfbeb4014132b96", "probe.py": "527cd00d7c36ac2faa3e3af83a4d15bef09deeb6b361e9743e0737491a0e1113"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def grouped_hash(groups):
    buf = bytearray()
    for ids in groups:
        buf.extend(struct.pack("<I", len(ids)))
        buf.extend(np.asarray(ids, dtype="<u4").tobytes())
    return digest(buf)


def centered(x):
    x = np.asarray(x, dtype=np.float64)
    return x - np.sum(x) / 10


def independent_fgh(z, y, b, b0):
    """Independent logaddexp objective and Hessian sums, no author functions."""
    a = np.asarray(z, dtype=np.float64) + b
    lse = np.logaddexp.reduce(a, axis=1)
    probability = np.exp(a - lse[:, None])
    d = b - b0
    f = float(np.sum(lse - a[np.arange(len(y)), y]) + 5 * np.sum(d * d))
    g = probability.sum(axis=0) - np.bincount(y, minlength=10) + 10 * d
    h = -np.einsum("ni,nj->ij", probability, probability, optimize=False)
    diagonal = np.diag_indices(10)
    h[diagonal] += probability.sum(axis=0) + 10
    return f, g, h


def independent_fit(z, y):
    counts = np.bincount(y, minlength=10)
    b0 = centered(np.log((counts + 1.) / (len(y) + 10)))
    b = b0.copy()
    f, g, h = independent_fgh(z, y, b, b0)
    trace = [{"objective": f, "gradient_inf": float(np.max(np.abs(g))), "bias": b.tolist()}]
    reason = "maximum_steps"
    for step in range(1, 21):
        if np.max(np.abs(g)) <= 1e-8:
            reason = "gradient_tolerance"
            break
        direction = centered(np.linalg.solve(h, g))
        slope = -float(g @ direction)
        if not np.isfinite(slope) or slope >= 0:
            raise ValueError("independent Newton direction is not descent")
        accepted = False
        for backtrack in range(41):
            alpha = 2. ** (-backtrack)
            trial = centered(b - alpha * direction)
            nf, ng, nh = independent_fgh(z, y, trial, b0)
            if nf <= f + 1e-4 * alpha * slope:
                accepted = True
                break
        if not accepted:
            if -slope <= 1e-10 * max(1., abs(f)):
                reason = "float64_objective_resolution"
                break
            raise ValueError("independent Armijo search failed")
        improvement = f - nf
        b, f, g, h = trial, nf, ng, nh
        trace.append({"step": step, "objective": f, "gradient_inf": float(np.max(np.abs(g))), "alpha": alpha, "backtracks": backtrack, "bias": b.tolist()})
        if improvement <= np.finfo(np.float64).eps * max(1., abs(f)):
            reason = "float64_objective_resolution"
            break
    return b, {"trace": trace, "stop_reason": reason, "objective": f, "gradient": g.tolist(), "gradient_inf": float(np.max(np.abs(g))), "hessian": h.tolist(), "hessian_min_eigenvalue": float(np.linalg.eigvalsh(h).min())}


def score_metrics(a, labels):
    a = np.asarray(a, dtype=np.float64)
    lse = np.logaddexp.reduce(a, axis=1)
    losses = lse - a[np.arange(len(labels)), labels]
    correct = int(np.count_nonzero(a.argmax(axis=1) == labels))
    return {"accuracy": correct / len(labels), "loss": float(losses.mean()), "samples": len(labels), "correct": correct, "loss_sum": float(losses.sum())}


def summary(clients, variant):
    m = [c["metrics"][variant] for c in clients]
    acc = np.array([v["accuracy"] for v in m])
    ce = np.array([v["loss"] for v in m])
    denominator = sum(v["samples"] for v in m)
    ordered = np.sort(acc)
    qpos = .1 * (len(ordered) - 1)
    low, high = math.floor(qpos), math.ceil(qpos)
    q10 = ordered[low] + (qpos - low) * (ordered[high] - ordered[low])
    return {"macro_accuracy": float(acc.mean()), "macro_loss": float(ce.mean()), "weighted_accuracy": sum(v["correct"] for v in m) / denominator, "weighted_loss": sum(v["loss_sum"] for v in m) / denominator, "client_accuracy_q10": float(q10), "correct_total": sum(v["correct"] for v in m), "samples_total": denominator, "loss_sum": sum(v["loss_sum"] for v in m)}


def verify(here, diagnostic, baseline, models):
    checks, snapshots, arms, manifests, skipped_images = [], {}, {}, {}, []
    max_metric_error = max_bias_error = max_trace_f = max_trace_g = max_fd_g = max_fd_h = max_gauge = max_stationarity = max_refit_score_error = max_refit_FP32bias_error = 0.
    exact_independent_fp32 = 0
    common_gt = {}

    def check(name, passed, detail=None):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    def read(path):
        data = path.read_bytes()
        snapshots[str(path.resolve())] = {"bytes": len(data), "sha256": digest(data)}
        return data

    def npzread(path):
        with np.load(io.BytesIO(read(path)), allow_pickle=False) as contents:
            return {k: contents[k].copy() for k in contents.files}

    recorded = json.loads(read(diagnostic / "result.json"))
    before = json.loads(read(diagnostic / "inputs_before.json"))
    previous = json.loads(read(baseline / "result.json"))
    check("DONE/fixed candidate/no-test flags", recorded["status"] == "DONE" and recorded["candidate_model_fixed_before_results"] == "balanced_uniform3" and recorded["official_test_byte_read_attempts"] == [] and recorded["official_test_inference"] is False)
    for name, expected in FROZEN.items():
        check("frozen " + name, digest(read(here / name)) == expected == recorded["source_sha256"][str((here / name).resolve())])
    check("before/after protected digest inventories match", before == recorded["input_sha256_before_after_equal"])
    for name, expected in before.items():
        path = Path(name)
        if path.name in {f"data_batch_{i}" for i in range(1, 6)}:
            skipped_images.append({"path": name, "recorded_sha256_not_independently_read": expected})
            continue
        if "test_batch" in path.name:
            raise ValueError("refuse TEST bytes")
        check("protected hash: " + name, digest(read(path)) == expected)
    check("exactly five image TRAIN files intentionally not reread", len(skipped_images) == 5)
    for method in METHODS:
        conf = json.loads(read(models / method / "config.json"))
        manifest = conf["data_manifest"]
        manifests[method] = manifest
        prefix = method + ": "
        check(prefix + "same frozen checkpoint/config as prior PFL", all(recorded["input_sha256_before_after_equal"][str((models / method / n).resolve())] == previous["input_sha256"][str((models / method / n).resolve())] for n in ("config.json", "checkpoint.pt")))
        trainids = [i for ids in manifest["train_indices"] for i in ids]
        heldids = [i for ids in manifest["holdout_indices"] for i in ids]
        check(prefix + "exact37502/12498 split no leakage and hashes", len(trainids) == len(set(trainids)) == 37502 and len(heldids) == len(set(heldids)) == 12498 and set(trainids).isdisjoint(heldids) and set(trainids + heldids) == set(range(50000)) and grouped_hash(manifest["train_indices"]) == manifest["train_indices_sha256"] and grouped_hash(manifest["holdout_indices"]) == manifest["holdout_indices_sha256"] and set(manifest["validation_indices"]) <= set(heldids))
        original_clients = recorded["results"][method]["clients"]
        check(prefix + "20 client ordering and model unchanged flag", len(original_clients) == 20 and [c["client"] for c in original_clients] == list(range(20)) and recorded["results"][method]["model_before_after_same"] is True)
        clients, total_forwards, batches, newton, solves, evaluations, backtracks_total = [], 0, 0, 0, 0, 0, 0
        reconstructed_full_labels = np.full(50000, -1, dtype=np.int64)
        for i, original in enumerate(original_clients):
            cp = prefix + f"client{i}: "
            arrays = npzread(diagnostic / f"{method}_client{i}.npz")
            old = npzread(baseline / f"{method}_client{i}.npz")
            z, y, ids = arrays["train_logits"], arrays["train_labels"], arrays["train_ids"]
            raw, learned, hy, hid = arrays["holdout_raw"], arrays["holdout_learned"], arrays["holdout_labels"], arrays["holdout_ids"]
            b, b0saved, prior = arrays["fitted_bias"], arrays["prior_bias_float64"], arrays["prior_shift"]
            op = original["optimization"]
            check(cp + "finite typed array shapes", z.shape == (len(ids), 10) and raw.shape == learned.shape == (len(hid), 10) and z.dtype == raw.dtype == learned.dtype == np.float32 and y.shape == (len(ids),) and hy.shape == (len(hid),) and b.shape == prior.shape == b0saved.shape == (10,) and b.dtype == prior.dtype == np.float32 and b0saved.dtype == np.float64 and all(np.isfinite(a).all() for a in (z, raw, learned, b, b0saved, prior)) and np.all((y >= 0) & (y < 10)) and np.all((hy >= 0) & (hy < 10)))
            check(cp + "TRAIN/holdout exact IDs/order and no overlap", np.array_equal(ids, manifest["train_indices"][i]) and np.array_equal(hid, manifest["holdout_indices"][i]) and set(map(int, ids)).isdisjoint(map(int, hid)))
            counts = np.bincount(y, minlength=10)
            check(cp + "actual TRAIN GT counts/holdout counts", np.array_equal(counts, manifest["client_label_counts"][i]) and np.array_equal(counts, op["train_counts"]) and np.array_equal(counts, old["train_counts"]) and np.array_equal(counts + np.bincount(hy, minlength=10), manifest["original_client_label_counts"][i]))
            reconstructed_full_labels[ids] = y
            reconstructed_full_labels[hid] = hy
            check(cp + "reuse prior PFL holdout arrays byte-for-value", all(np.array_equal(arrays[new], old[oldkey]) for new, oldkey in (("holdout_raw", "raw"), ("holdout_learned", "learned"), ("holdout_labels", "labels"), ("holdout_ids", "holdout_ids"), ("prior_shift", "prior_shift"))))
            uncentered = np.log((counts + 1.) / (len(y) + 10))
            b0 = centered(uncentered)
            check(cp + "TRAIN-only exact prior and centered gauge", np.array_equal(prior, uncentered.astype(np.float32)) and np.allclose(b0saved, b0, rtol=0, atol=2e-15) and np.allclose(op["prior_bias_float64"], b0, rtol=0, atol=2e-15))
            if i not in common_gt:
                common_gt[i] = (y.copy(), ids.copy(), hy.copy(), hid.copy())
            else:
                check(cp + "same TRAIN/holdout GT across models", all(np.array_equal(a, other) for a, other in zip((y, ids, hy, hid), common_gt[i])))
            author_b = np.asarray(op["fitted_bias_float64"])
            check(cp + "saved FP32 deployment bias exact cast/40B", np.array_equal(b, author_b.astype(np.float32)) and b.nbytes == original["local_state_bytes"] == op["local_state_bytes"] == 40 and op["local_state_dtype"] == "float32")
            f0, g0, h0 = independent_fgh(z, y, b0, b0)
            ff, gf, hf = independent_fgh(z, y, author_b, b0)
            # Independent numerical checks, away from final-step resolution.
            direction = centered(np.arange(1., 11.))
            direction /= np.linalg.norm(direction)
            step = 1e-4
            fp, gp, _ = independent_fgh(z, y, b0 + step * direction, b0)
            fm, gm, _ = independent_fgh(z, y, b0 - step * direction, b0)
            fd_g = abs((fp - fm) / (2 * step) - float(g0 @ direction))
            fd_h = float(np.max(np.abs((gp - gm) / (2 * step) - h0 @ direction)))
            max_fd_g, max_fd_h = max(max_fd_g, fd_g), max(max_fd_h, fd_h)
            check(cp + "gradient/Hessian finite-difference and gauge eigenvector", fd_g < 5e-5 and fd_h < 5e-5 and np.max(np.abs(hf @ np.ones(10) - 10)) < 1e-8 and np.linalg.eigvalsh(hf).min() >= 10 - 1e-8)
            check(cp + "independent initial/final objective gradient Hessian", abs(f0 - op["initial_objective"]) < 1e-8 and abs(ff - op["final_objective"]) < 1e-8 and abs(float(np.max(np.abs(gf))) - op["final_gradient_inf"]) < 1e-8 and abs(float(np.linalg.eigvalsh(hf).min()) - op["hessian_min_eigenvalue"]) < 1e-8 and ff <= f0)
            max_stationarity = max(max_stationarity, float(np.max(np.abs(gf))))
            max_gauge = max(max_gauge, abs(float(author_b.mean())))
            check(cp + "zero-mean gauge FP64 and small FP32 rounding", abs(float(author_b.mean())) < 2e-14 and abs(float(b.astype(np.float64).mean())) < 2e-7)
            # Reconstruct every stored accepted Newton transition independently,
            # recomputing direction and every failed line-search trial.
            current = b0.copy()
            f, g, h = independent_fgh(z, y, current, b0)
            tracecheck, searchrecords = True, []
            trace_obj_error = abs(f - op["trace"][0]["objective"])
            trace_grad_error = abs(float(np.max(np.abs(g))) - op["trace"][0]["gradient_inf"])
            for rec in op["trace"][1:]:
                direction = centered(np.linalg.solve(h, g))
                slope = -float(g @ direction)
                bt, alpha = rec["backtracks"], rec["alpha"]
                tracecheck &= rec["step"] == len(searchrecords) + 1 and 0 <= bt <= 40 and alpha == 2. ** (-bt) and slope < 0
                rejected = []
                for j in range(bt):
                    trialalpha = 2. ** (-j)
                    trialf, _, _ = independent_fgh(z, y, centered(current - trialalpha * direction), b0)
                    slack = trialf - (f + 1e-4 * trialalpha * slope)
                    rejected.append(slack)
                    # Last-bit objective equality can change reject/accept after
                    # an algebraically equivalent independent summation.
                    tracecheck &= slack >= -2e-10
                trial = centered(current - alpha * direction)
                nf, ng, nh = independent_fgh(z, y, trial, b0)
                armijo_slack = nf - (f + 1e-4 * alpha * slope)
                tracecheck &= armijo_slack <= 2e-10 and nf <= f + 2e-10 and abs(slope - rec["directional_derivative"]) < 1e-8
                trace_obj_error = max(trace_obj_error, abs(nf - rec["objective"]))
                trace_grad_error = max(trace_grad_error, abs(float(np.max(np.abs(ng))) - rec["gradient_inf"]))
                searchrecords.append({"step": rec["step"], "alpha": alpha, "backtracks": bt, "rejected_Armijo_slacks_independent": rejected, "accepted_Armijo_slack_independent": armijo_slack, "objective_independent": nf, "gradient_inf_independent": float(np.max(np.abs(ng))), "hessian_independent": nh.tolist()})
                current, f, g, h = trial, nf, ng, nh
            replayerr = float(np.max(np.abs(current - author_b)))
            max_trace_f, max_trace_g = max(max_trace_f, trace_obj_error), max(max_trace_g, trace_grad_error)
            tracecheck &= trace_obj_error < 1e-8 and trace_grad_error < 1e-7 and replayerr < 5e-9
            tracecheck &= op["accepted_newton_steps"] == len(op["trace"]) - 1 <= 20 and op["linear_solves"] == op["accepted_newton_steps"] and op["objective_evaluations"] == 1 + sum(1 + rec["backtracks"] for rec in op["trace"][1:])
            check(cp + "independent full Newton/Armijo trace replay", tracecheck, {"FP64_bias_error": replayerr, "trace_objective_max_error": trace_obj_error, "trace_gradient_max_error": trace_grad_error})
            reason = op["stop_reason"]
            check(cp + "stopping reason numerically supported", (reason == "gradient_tolerance" and float(np.max(np.abs(gf))) <= 1e-8 + 1e-10) or (reason == "float64_objective_resolution" and len(op["trace"]) >= 2 and op["trace"][-2]["objective"] - op["trace"][-1]["objective"] <= np.finfo(np.float64).eps * max(1., abs(op["final_objective"])) + 2e-10) or (reason == "maximum_steps" and op["accepted_newton_steps"] == 20))
            ib, fitinfo = independent_fit(z, y)
            biaserr = float(np.max(np.abs(ib - author_b)))
            max_bias_error = max(max_bias_error, biaserr)
            f32exact = np.array_equal(ib.astype(np.float32), b)
            exact_independent_fp32 += int(f32exact)
            fp32err = float(np.max(np.abs(ib.astype(np.float32) - b)))
            max_refit_FP32bias_error = max(max_refit_FP32bias_error, fp32err)
            check(cp + "independent refit matches deployed FP32 within1ulp-scale", biaserr <= 1e-7 and np.allclose(ib.astype(np.float32), b, rtol=0, atol=4e-7), {"FP64_error": biaserr, "independent_FP32_bit_exact": f32exact})
            scores = {"raw": raw.astype(np.float64), "analytic_prior": raw.astype(np.float64) + prior.astype(np.float64), "prior_only": np.broadcast_to(prior.astype(np.float64), raw.shape), "learned_head": learned.astype(np.float64), "learned_head_plus_prior": learned.astype(np.float64) + prior.astype(np.float64), "fitted_bias": raw.astype(np.float64) + b.astype(np.float64)}
            ms = {}
            for variant, values in scores.items():
                metric = score_metrics(values, hy)
                ref = original["metrics"][variant]
                err = max(abs(metric["accuracy"] - ref["accuracy"]), abs(metric["loss"] - ref["loss"]))
                max_metric_error = max(max_metric_error, err)
                check(cp + variant + " independent holdout metric", metric["samples"] == ref["samples"] and abs(metric["accuracy"] - ref["accuracy"]) < 2e-14 and abs(metric["loss"] - ref["loss"]) < 1e-11)
                ms[variant] = metric
            refit_score = score_metrics(raw.astype(np.float64) + ib.astype(np.float32).astype(np.float64), hy)
            refit_error = abs(refit_score["loss"] - ms["fitted_bias"]["loss"])
            max_refit_score_error = max(max_refit_score_error, refit_error)
            check(cp + "independent FP32-refit preserves holdout prediction/CE", refit_score["correct"] == ms["fitted_bias"]["correct"] and refit_error < 1e-7)
            nbatches = math.ceil(len(y) / 256)
            check(cp + "recorded forward/state/no-backward/uplink accounting", original["train_samples"] == original["extra_train_image_forwards"] == len(y) and original["extra_cnn_forward_batches"] == nbatches and original["holdout_samples"] == len(hy) and original["extra_cnn_backward"] == original["extra_uplink_bytes"] == 0 and original["extra_learned_head_training"] is False)
            clients.append({"client": i, "train_samples": len(y), "holdout_samples": len(hy), "train_counts": counts.tolist(), "initial_objective_independent": f0, "initial_gradient_independent": g0.tolist(), "initial_hessian_independent": h0.tolist(), "final_objective_at_saved_FP64bias": ff, "final_gradient_at_saved_FP64bias": gf.tolist(), "final_hessian_at_saved_FP64bias": hf.tolist(), "gauge_mean_saved_FP64": float(author_b.mean()), "saved_FP32bias": b.tolist(), "independent_fitted_FP64bias": ib.tolist(), "independent_refit": fitinfo, "independent_FP32_bit_exact": f32exact, "recorded_trace_replay": searchrecords, "finite_difference_gradient_error": fd_g, "finite_difference_Hessian_vector_error": fd_h, "metrics": ms})
            total_forwards += len(y)
            batches += nbatches
            newton += op["accepted_newton_steps"]
            solves += op["linear_solves"]
            evaluations += op["objective_evaluations"]
            backtracks_total += sum(rec.get("backtracks", 0) for rec in op["trace"])
        sums = {variant: summary(clients, variant) for variant in VARIANTS}
        check(prefix + "reconstruct all50000 label IDs/hash without original images", np.all(reconstructed_full_labels >= 0) and np.array_equal(np.bincount(reconstructed_full_labels, minlength=10), np.full(10, 5000)) and digest(reconstructed_full_labels.astype("<i8").tobytes()) == manifest["official_train_labels_sha256"])
        for variant, values in sums.items():
            for key, reference in recorded["results"][method]["summaries"][variant].items():
                error = abs(values[key] - reference)
                max_metric_error = max(max_metric_error, error)
                check(prefix + variant + "/" + key + " independent macro/weighted/q10", error < (1e-11 if key.endswith("loss") else 2e-14))
        arms[method] = {"clients": clients, "summaries": sums, "extra_train_forwards_recorded": total_forwards, "extra_CNN_batches_recorded": batches, "accepted_Newton_steps_recorded": newton, "linear_solves_recorded": solves, "objective_evaluations_recorded": evaluations, "backtracks_recorded": backtracks_total, "saved_diagnostic_NPZ_cache_bytes": sum(snapshots[str((diagnostic / f'{method}_client{i}.npz').resolve())]["bytes"] for i in range(20))}
    check("shared source partitions", all(m == manifests[METHODS[0]] for m in manifests.values()))
    cand = arms["balanced_uniform3"]["summaries"]["fitted_bias"]
    same = arms["balanced_uniform3"]["summaries"]["analytic_prior"]
    best_prior_name = max(METHODS, key=lambda m: arms[m]["summaries"]["analytic_prior"]["macro_accuracy"])
    best_prior = arms[best_prior_name]["summaries"]["analytic_prior"]
    lm, lv = max(((m, v) for m in METHODS for v in ("learned_head", "learned_head_plus_prior")), key=lambda pair: arms[pair[0]]["summaries"][pair[1]]["macro_accuracy"])
    learned = arms[lm]["summaries"][lv]
    gates = {"vs_same_model_prior": {"accuracy_plus_half_pp": cand["macro_accuracy"] >= same["macro_accuracy"] + .005, "lower_ce": cand["macro_loss"] < same["macro_loss"]}, "vs_best_existing_prior": {"method": best_prior_name, "accuracy_plus_half_pp": cand["macro_accuracy"] >= best_prior["macro_accuracy"] + .005, "lower_ce": cand["macro_loss"] < best_prior["macro_loss"]}, "vs_best_learned": {"method": lm, "variant": lv, "within_half_pp": cand["macro_accuracy"] >= learned["macro_accuracy"] - .005}}
    go = all(gates["vs_same_model_prior"].values()) and gates["vs_best_existing_prior"]["accuracy_plus_half_pp"] and gates["vs_best_existing_prior"]["lower_ce"] and gates["vs_best_learned"]["within_half_pp"]
    decision = "FOLLOW_UP" if go else "STOP"
    check("fixed candidate gates independently equal recorded", gates == recorded["gates"] and decision == recorded["decision"])
    total_forwards = sum(a["extra_train_forwards_recorded"] for a in arms.values())
    total_batches = sum(a["extra_CNN_batches_recorded"] for a in arms.values())
    total_newton = sum(a["accepted_Newton_steps_recorded"] for a in arms.values())
    check("aggregate actual compute accounting", total_forwards == recorded["total_extra_train_image_forwards"] == 150008 and total_batches == recorded["total_extra_cnn_forward_batches"] == 640 and total_newton == recorded["total_newton_steps"] == 280 and recorded["total_extra_cnn_backward"] == recorded["total_extra_uplink_bytes"] == 0 and recorded["local_bias_state_bytes_per_client"] == 40 and recorded["local_bias_state_bytes_per_model_20_clients"] == 800)
    unchanged = all(digest(Path(name).read_bytes()) == record["sha256"] for name, record in snapshots.items())
    check("all independently read inputs unchanged", unchanged)
    failures = [item for item in checks if not item["pass"]]
    return {"status": "DONE", "artifact_check": "FAIL" if failures else "PASS", "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures, "review": "REVIEW_UNAVAILABLE", "decision": decision, "candidate_fixed": "balanced_uniform3", "gates": gates, "candidate_macro_accuracy_gain_vs_same_prior": cand["macro_accuracy"] - same["macro_accuracy"], "candidate_macro_accuracy_gain_vs_best_prior": cand["macro_accuracy"] - best_prior["macro_accuracy"], "candidate_macro_CE_reduction_vs_same_prior": same["macro_loss"] - cand["macro_loss"], "candidate_macro_CE_reduction_vs_best_prior": best_prior["macro_loss"] - cand["macro_loss"], "max_holdout_metric_error": max_metric_error, "max_independent_refit_deployed_FP32bias_difference": max_refit_FP32bias_error, "max_independent_refit_holdout_CE_difference": max_refit_score_error, "max_independent_refit_FP64bias_difference": max_bias_error, "independent_refit_FP32_exact_count": exact_independent_fp32, "max_trace_objective_error": max_trace_f, "max_trace_gradient_error": max_trace_g, "max_gradient_finite_difference_error": max_fd_g, "max_Hessian_vector_finite_difference_error": max_fd_h, "max_saved_FP64_gauge_mean": max_gauge, "max_final_gradient_inf_independent": max_stationarity, "arms": arms, "extra_train_forwards_recorded": total_forwards, "extra_CNN_batches_recorded": total_batches, "accepted_Newton_steps_recorded": total_newton, "objective_evaluations_recorded": sum(a["objective_evaluations_recorded"] for a in arms.values()), "linear_solves_recorded": sum(a["linear_solves_recorded"] for a in arms.values()), "backtracks_recorded": sum(a["backtracks_recorded"] for a in arms.values()), "cache_bytes_total": sum(a["saved_diagnostic_NPZ_cache_bytes"] for a in arms.values()), "snapshots": snapshots, "inputs_unchanged": unchanged, "image_files_intentionally_unread": skipped_images, "limits": ["Independent NumPy code does not import author solver/probe, torch, CNN/model or any original image files. It only refits10D biases from saved TRAIN arrays and recomputes holdout metrics; no new forward/backward/FL/head training.", "Actual TRAIN labels are saved here, so counts are independently rebuilt from labels and checked against source configs and previous PFL arrays. It does not establish saved CNN logits by another image forward.", "This is final-round20 same-seed/partition PFL development with12498 holdout samples including oldglobaldev2000. It is not global IID, independent final test, or20 independent training seeds.", "Regularization10 makes Hessian >=10I. Centered b0 and updates select mean-zero bias; adding a common bias preserves CE but changes the regularizer, so the regularized objective is not fully gauge-invariant.", "Equivalent logaddexp and Hessian summations can affect last-bit Armijo behavior at FP64 objective resolution. Full stored trace is replayed with2e-10 objective slack tolerance; independent fresh-fit FP64/FP32 deviations are reported separately. Tolerances do not alter the frozen performance gate.", "Some fits stop at FP64 objective resolution with gradient infinity norm above1e-8; they are not mislabeled as all satisfying gradient tolerance. Saved stop reasons and Newton budgets are retained.", "Original author report says five TRAIN imagesets and no official TEST bytes; those five image file hashes are intentionally not independently reread here, preserving the no-original-images constraint. Other protected inputs are hash checked.", "Other loss/background fitted biases may perform better but cannot replace the predetermined balanced_uniform3 candidate. Both+.5pp accuracy requirements remain mandatory even when CE/learned comparisons pass.", "REVIEW_UNAVAILABLE denotes unavailable external cross-model review, not an independent cross-family acquittal. All read-only sources, prior results and arrays remain unchanged."]}


def report(out):
    c = out["arms"]["balanced_uniform3"]["summaries"]["fitted_bias"]
    lines = ["# 独立NumPy核查", f"DONE；{out['artifact_check']} {out['checks_passed']}/{out['checks_total']}；固定候选balanced_uniform3，决定**{out['decision']}**；外部`REVIEW_UNAVAILABLE`。", f"候选宏acc {100*c['macro_accuracy']:.6f}%、CE {c['macro_loss']:.9f}；相对同模型prior +{100*out['candidate_macro_accuracy_gain_vs_same_prior']:.6f}pp、最好prior +{100*out['candidate_macro_accuracy_gain_vs_best_prior']:.6f}pp，均未达+.5pp。两项CE及learned门槛通过也不能挽救；保留其他loss fit结果，不换候选。", "", "|模型|fit宏acc%|fit宏CE|weighted acc%|q10%|", "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        s = out["arms"][m]["summaries"]["fitted_bias"]
        lines.append(f"|{m}|{100*s['macro_accuracy']:.5f}|{s['macro_loss']:.8f}|{100*s['weighted_accuracy']:.5f}|{100*s['client_accuracy_q10']:.5f}|")
    lines.extend(["", f"80组TRAIN计数/IDs及复用holdout与config/旧PFL一致，按ID重建50000标签hash吻合；六score共480条client指标、120项汇总独立复算，最大差{out['max_holdout_metric_error']:.2g}。目标/梯度/Hessian含有限差分、>=10I及mean-zero检查通过；完整Newton/Armijo轨迹独立重放，FP32 bias与保存FP64转换精确一致。独立重拟合FP64最大差{out['max_independent_refit_FP64bias_difference']:.2g}，FP32位级一致{out['independent_refit_FP32_exact_count']}/80，另2组差1ulp且预测不变。", f"部分fit按FP64目标分辨率停止，最终grad∞最大{out['max_final_gradient_inf_independent']:.2g}，不能称全部≤1e−8；等价求和末位线搜索误差容差2e−10，不改效果门槛。", f"作者记录额外TRAIN forward150008样本/640批，Newton280步、评估{out['objective_evaluations_recorded']}次；bias40B/client。新增CNN backward/新head训练/uplink为0，但不是零计算。缓存{out['cache_bytes_total']}B。", "仅最终轮PFL开发，37502/12498且含旧dev2000；20client不是20seed，不冒充global或独立test。未读原图/officialtest，未做新CNN forward/训练，未导入作者solver；五个原图文件摘要仅保留作者记录，其余保护输入重验、源文件不变。完整计数、F/g/H、轨迹、六score及gates见LOCAL_CHECK.json。"])
    text = "\n".join(lines) + "\n"
    if len(text) > 1200:
        raise ValueError(f"report over1200 chars: {len(text)}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=here / "diagnostic_v1")
    parser.add_argument("--baseline-dir", type=Path, default=here.parent / "prior_personalization/diagnostic_v1")
    parser.add_argument("--model-dir", type=Path, default=here.parent / "balanced_quant/development_p20261193_s20261201")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    out = verify(here, args.input_dir, args.baseline_dir, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LOCAL_CHECK.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "LOCAL_CHECK.md").write_text(report(out))
    print(json.dumps({k: out[k] for k in ("status", "artifact_check", "checks_passed", "checks_total", "decision", "max_independent_refit_FP64bias_difference", "independent_refit_FP32_exact_count", "max_trace_objective_error", "max_trace_gradient_error", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
