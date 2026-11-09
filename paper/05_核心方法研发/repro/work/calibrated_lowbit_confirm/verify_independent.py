"""Independent saved-array/checkpoint verification; no author module imports.

Reads completed saved artifacts only. No original images, official TEST, CNN
construction, image forward/backward, FL training or head training are invoked.
Torch is used solely to deserialize CPU tensors with weights_only=True.
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

METHODS = ("ce_uniform3", "fedlc_uniform3", "fedlc_dense")
VARIANTS = ("raw", "analytic_prior", "prioranchor_bias", "matched_learned_head", "matched_learned_head_plus_prior")
N, CAP, CLIENTS, ROUNDS = 136874, 54808, 20, 20
HEADER = struct.Struct("<8sIIII40s")
FROZEN = {
    "PLAN.md": "ac6f2f4b9582e95897f9d9dedcd3af3481114c3824674cc1cee02832a3144597",
    "common.py": "3f2e4693488d7e5a35a182e5340edb9d73f5cafc45b7c0e61b1e2cc392a22d9a",
    "train.py": "7fe8777db0ab8b2309140b39026a9bdbec42cd09ba9d2b8ef0c89b831a5641bd",
    "evaluate.py": "afcb8a47dc9c303b3e7d558e4e91b50b560cd10dd0287956969d61ab27e13250",
}

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

def decode_independently(wire):
    """Parse little-endian fields and bit planes using NumPy, no author codec."""
    assert len(wire) >= HEADER.size
    magic, version, numel, blocksize, blocks, reserved = HEADER.unpack_from(wire)
    assert (magic, version, numel, blocksize, blocks) == (b"MPQCODEC", 1, N, 256, (N + 255) // 256)
    assert not any(reserved)
    offset, decoded, scales, packed_total = HEADER.size, [], [], 0
    zero_blocks, mincode, maxcode = 0, 8, -1
    for index in range(blocks):
        count = min(blocksize, numel - index * blocksize)
        assert offset + 5 <= len(wire)
        width, scale = wire[offset], struct.unpack_from("<f", wire, offset + 1)[0]
        assert width == 3 and math.isfinite(scale) and scale >= 0
        offset += 5
        nbytes = (count * width + 7) // 8
        packet = wire[offset:offset + nbytes]
        assert len(packet) == nbytes
        if count * width % 8:
            assert packet[-1] >> (count * width % 8) == 0
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8), bitorder="little")[:count * width]
        codes = (bits.reshape(count, width).astype(np.uint16) * (1 << np.arange(width))).sum(axis=1)
        assert np.all(codes < 7), "unused code 7 is noncanonical"
        if scale == 0:
            assert np.all(codes == 3), "zero-scale block must have zero codes"
            zero_blocks += 1
        values = (codes.astype(np.int16) - 3).astype(np.float32) * np.float32(scale)
        assert np.isfinite(values).all()
        mincode, maxcode = min(mincode, int(codes.min())), max(maxcode, int(codes.max()))
        decoded.append(values)
        scales.append(scale)
        offset += nbytes
        packed_total += nbytes
    assert offset == len(wire)
    vector = np.concatenate(decoded)
    assert vector.size == N
    return {"wire_bytes": len(wire), "decoded_coordinates": int(vector.size), "decoded_fp32_sha256": digest(vector.astype("<f4").tobytes()), "decoded_l2": float(np.linalg.norm(vector.astype(np.float64))), "all_finite": bool(np.isfinite(vector).all()), "header_bytes": HEADER.size, "blocks": blocks, "block_metadata_bytes": blocks * 5, "packed_value_bytes": packed_total, "tail_coordinates": N % 256, "uniform_bitwidth": 3, "zero_scale_blocks": zero_blocks, "min_scale": min(scales), "max_scale": max(scales), "code_range": [mincode, maxcode]}


def verify_bias(z, y, deployed, op, check, prefix):
    """Recompute the objective, derivatives and all stored Newton transitions."""
    counts = np.bincount(y, minlength=10)
    b0 = centered(np.log((counts + 1.) / (len(y) + 10)))
    b = np.asarray(op["fitted_bias_float64"], dtype=np.float64)
    check(prefix + "TRAIN-only centered prior", np.allclose(op["prior_bias_float64"], b0, rtol=0, atol=2e-15) and np.array_equal(op["train_counts"], counts))
    check(prefix + "saved FP32 bias cast and40B", np.array_equal(deployed, b.astype(np.float32)) and deployed.nbytes == op["local_state_bytes"] == 40 and op["local_state_dtype"] == "float32")
    f0, g0, h0 = independent_fgh(z, y, b0, b0)
    ff, gf, hf = independent_fgh(z, y, b, b0)
    direction = centered(np.arange(1., 11.))
    direction /= np.linalg.norm(direction)
    step = 1e-4
    fp, gp, _ = independent_fgh(z, y, b0 + step * direction, b0)
    fm, gm, _ = independent_fgh(z, y, b0 - step * direction, b0)
    fd_g = abs((fp - fm) / (2 * step) - float(g0 @ direction))
    fd_h = float(np.max(np.abs((gp - gm) / (2 * step) - h0 @ direction)))
    check(prefix + "finite-difference gradient/Hessian", fd_g < 5e-5 and fd_h < 5e-5)
    check(prefix + "strong convexity and centered gauge", np.linalg.eigvalsh(hf).min() >= 10 - 1e-8 and np.max(np.abs(hf @ np.ones(10) - 10)) < 1e-8 and abs(float(b.mean())) < 2e-14 and abs(float(deployed.astype(np.float64).mean())) < 2e-7)
    check(prefix + "independent initial/final F/g/H", abs(f0 - op["initial_objective"]) < 1e-8 and abs(ff - op["final_objective"]) < 1e-8 and abs(float(np.max(np.abs(gf))) - op["final_gradient_inf"]) < 1e-8 and abs(float(np.linalg.eigvalsh(hf).min()) - op["hessian_min_eigenvalue"]) < 1e-8 and ff <= f0)
    current = b0.copy()
    f, g, h = independent_fgh(z, y, current, b0)
    trace = op["trace"]
    trace_ok = trace[0]["step"] == 0
    ferror = abs(f - trace[0]["objective"])
    gerror = abs(float(np.max(np.abs(g))) - trace[0]["gradient_inf"])
    searchrecords = []
    for rec in trace[1:]:
        direction = centered(np.linalg.solve(h, g))
        slope = -float(g @ direction)
        bt, alpha = rec["backtracks"], rec["alpha"]
        trace_ok &= rec["step"] == len(searchrecords) + 1 and 0 <= bt <= 40 and alpha == 2. ** (-bt) and slope < 0
        rejected = []
        for j in range(bt):
            ta = 2. ** (-j)
            tf, _, _ = independent_fgh(z, y, centered(current - ta * direction), b0)
            slack = tf - (f + 1e-4 * ta * slope)
            rejected.append(slack)
            trace_ok &= slack >= -2e-10
        trial = centered(current - alpha * direction)
        nf, ng, nh = independent_fgh(z, y, trial, b0)
        slack = nf - (f + 1e-4 * alpha * slope)
        trace_ok &= slack <= 2e-10 and nf <= f + 2e-10 and abs(slope - rec["directional_derivative"]) < 1e-8
        ferror = max(ferror, abs(nf - rec["objective"]))
        gerror = max(gerror, abs(float(np.max(np.abs(ng))) - rec["gradient_inf"]))
        searchrecords.append({"step": rec["step"], "backtracks": bt, "alpha": alpha, "rejected_slacks": rejected, "accepted_slack": slack, "objective": nf, "gradient_inf": float(np.max(np.abs(ng)))})
        current, f, g, h = trial, nf, ng, nh
    replay_error = float(np.max(np.abs(current - b)))
    accepted = len(trace) - 1
    accepted_evals = 1 + sum(1 + rec["backtracks"] for rec in trace[1:])
    failed_search = op["linear_solves"] - accepted
    check(prefix + "complete Newton/Armijo trace replay", trace_ok and ferror < 1e-8 and gerror < 1e-7 and replay_error < 5e-9 and op["accepted_newton_steps"] == accepted <= 20 and failed_search in (0, 1) and op["objective_evaluations"] == accepted_evals + 41 * failed_search, {"max_F_error": ferror, "max_g_error": gerror, "bias_error": replay_error})
    reason = op["stop_reason"]
    resolution_ok = False
    if reason == "float64_objective_resolution":
        if failed_search:
            direction = centered(np.linalg.solve(hf, gf))
            slope = -float(gf @ direction)
            rejected_all = []
            for j in range(41):
                ta = 2. ** (-j)
                tf, _, _ = independent_fgh(z, y, centered(b - ta * direction), b0)
                rejected_all.append(tf - (ff + 1e-4 * ta * slope))
            resolution_ok = -slope <= 1e-10 * max(1., abs(ff)) and min(rejected_all) >= -2e-10
        elif len(trace) >= 2:
            resolution_ok = trace[-2]["objective"] - trace[-1]["objective"] <= np.finfo(np.float64).eps * max(1., abs(ff)) + 2e-10
    check(prefix + "recorded stopping reason supported", (reason == "gradient_tolerance" and float(np.max(np.abs(gf))) <= 1.01e-8 and not failed_search) or resolution_ok or (reason == "maximum_steps" and accepted == 20 and not failed_search))
    independent_b, info = independent_fit(z, y)
    refit_error = float(np.max(np.abs(independent_b - b)))
    refit_fp32_error = float(np.max(np.abs(independent_b.astype(np.float32) - deployed)))
    check(prefix + "independent fresh TRAIN refit agrees", refit_error <= 1e-7 and refit_fp32_error <= 4e-7, {"FP64_bias_error": refit_error, "FP32_bias_error": refit_fp32_error})
    return {"initial_objective": f0, "final_objective": ff, "final_gradient": gf.tolist(), "final_gradient_inf": float(np.max(np.abs(gf))), "final_Hessian": hf.tolist(), "finite_difference_gradient_error": fd_g, "finite_difference_Hessian_error": fd_h, "saved_gauge_mean": float(b.mean()), "trace_objective_error": ferror, "trace_gradient_error": gerror, "trace_replay": searchrecords, "accepted_Newton_steps": accepted, "linear_solves": op["linear_solves"], "objective_evaluations": op["objective_evaluations"], "backtracks": sum(rec.get("backtracks", 0) for rec in trace), "stop_reason": reason, "refit_FP64_error": refit_error, "refit_FP32_error": refit_fp32_error, "refit_FP32_exact": np.array_equal(independent_b.astype(np.float32), deployed), "independent_refit": info}, independent_b


def verify(here, models, evaluation):
    checks, snapshots, skipped_images, configurations, arms = [], {}, [], {}, {}

    def check(name, passed, detail=None):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    def read(path):
        path = Path(path).resolve()
        if "test_batch" in str(path) or path.name in {f"data_batch_{i}" for i in range(1, 6)}:
            raise ValueError("verifier refuses original image/TEST bytes")
        data = path.read_bytes()
        snapshots[str(path)] = {"bytes": len(data), "sha256": digest(data)}
        return data

    def npzread(path):
        with np.load(io.BytesIO(read(path)), allow_pickle=False) as arrays:
            return {key: arrays[key].copy() for key in arrays.files}

    recorded = json.loads(read(evaluation / "result.json"))
    check("DONE and fixed candidate/exploratory disclosure", recorded["status"] == "DONE" and recorded["candidate_fixed"] == "fedlc_uniform3 + prioranchor_bias" and "exploratory" in recorded["selection_disclosure"] and "balanced STOP unchanged" in recorded["selection_disclosure"])
    check("no TEST metadata", recorded["official_test_byte_reads"] is recorded["official_test_inference"] is False and recorded["identities_before_after_equal"] is True)
    for name, expected in FROZEN.items():
        path = here / name
        check("frozen source " + name, digest(read(path)) == expected == recorded["source_identities"][str(path.resolve())])
    identities = {**recorded["source_identities"], **recorded["data_identities"], **recorded["frozen_model_inputs"]}
    for filename, expected in identities.items():
        path = Path(filename)
        if path.name in {f"data_batch_{i}" for i in range(1, 6)}:
            skipped_images.append({"path": filename, "author_hash_only": expected})
        else:
            check("protected input " + filename, digest(read(path)) == expected)
    check("exactly five TRAIN image hashes intentionally not reread", len(skipped_images) == 5)
    common_gt = {}
    for method in METHODS:
        prefix = method + ": "
        folder = models / method
        conf = json.loads(read(folder / "config.json"))
        result = json.loads(read(folder / "result.json"))
        ck = torch.load(io.BytesIO(read(folder / "checkpoint.pt")), map_location="cpu", weights_only=True)
        configurations[method] = conf
        unsigned = {key: value for key, value in conf.items() if key != "signature"}
        manifest, rows = conf["data_manifest"], result["rows"]
        train, held, dev = manifest["train_indices"], manifest["holdout_indices"], manifest["validation_indices"]
        trainids, heldids = [i for ids in train for i in ids], [i for ids in held for i in ids]
        sizes, hsizes = list(map(len, train)), list(map(len, held))
        counts = np.asarray(manifest["client_label_counts"])
        steps = [math.ceil(size / 64) for size in sizes]
        dense = method.endswith("dense")
        packetbytes = 547560 if dense else 54067
        check(prefix + "self signature and checkpoint binding", digest(json.dumps(unsigned, sort_keys=True).encode()) == conf["signature"] == result["signature"] == ck["signature"] and ck["config"] == conf and ck["version"] == conf["version"] == 1)
        check(prefix + "fixed protocol", conf["method"] == method and conf["seed"] == 20261211 and conf["partition_seed"] == manifest["partition_seed"] == manifest["split_seed"] == 20261203 and conf["rounds"] == ROUNDS and conf["backend"] == "mps" and conf["parameters"] == N and conf["batch"] == 64 and conf["lr"] == .03 and conf["local_epochs"] == 1 and conf["loss"] == ("ce" if method.startswith("ce_") else "fedlc") and conf["upload_cap_bytes"] == (None if dense else CAP) and manifest["alpha"] == .1 and manifest["num_clients"] == CLIENTS)
        check(prefix + "frozen source/data agreement with evaluation", conf["source_identities"] == recorded["source_identities"] and conf["data_identities"] == recorded["data_identities"])
        check(prefix + "actual disjoint exhaustive partition", len(train) == len(held) == CLIENTS and len(trainids) == len(set(trainids)) == manifest["train_samples"] and len(heldids) == len(set(heldids)) == manifest["holdout_samples"] and set(trainids).isdisjoint(heldids) and set(trainids + heldids) == set(range(50000)) and len(dev) == len(set(dev)) == 2000 and set(dev) <= set(heldids))
        check(prefix + "partition hashes, sizes and rounding", grouped_hash(train) == manifest["train_indices_sha256"] == manifest["unmixed_train_indices_sha256"] and grouped_hash(held) == manifest["holdout_indices_sha256"] and digest(np.asarray(dev, dtype="<u4").tobytes()) == manifest["validation_ids_sha256"] and sizes == manifest["client_sizes"] and hsizes == manifest["client_holdout_sizes"] and all(t == round(.75 * (t + h)) for t, h in zip(sizes, hsizes)))
        check(prefix + "actual counts and fixed weights", counts.shape == (CLIENTS, 10) and np.all(counts >= 0) and np.array_equal(counts.sum(1), sizes) and np.allclose(conf["weights"], np.asarray(sizes) / sum(sizes), rtol=0, atol=1e-15) and sum(sizes) == recorded["training_samples_actual"] and sum(hsizes) == recorded["holdout_samples_actual"])
        check(prefix + "TRAIN-only data metadata", conf["official_test_byte_read"] is conf["official_test_inference"] is result["official_test_byte_read"] is manifest["test_used_for_selection"] is False and manifest["official_split"] == "CIFAR10 TRAIN only; strict five-file read" and len(manifest["official_train_batch_files"]) == 5)
        check(prefix + "final complete history", result["status"] == "DONE" and rows == ck["rows"] and [row["round"] for row in rows] == list(range(1, ROUNDS + 1)))
        tensors = list(ck["model"].values())
        residuals = ck["residuals"]
        check(prefix + "finite CPU FP32 model/residual layout", sum(value.numel() for value in tensors) == N and all(value.dtype == torch.float32 and value.device.type == "cpu" and torch.isfinite(value).all() for value in tensors) and len(residuals) == CLIENTS and all(value.shape == (N,) and value.dtype == torch.float32 and value.device.type == "cpu" and torch.isfinite(value).all() for value in residuals))
        check(prefix + "final EF norms/dense zero", all(abs(float(value.norm()) - row["residual_norm"]) <= 5e-6 for value, row in zip(residuals, rows[-1]["clients"])) and (not dense or all(torch.count_nonzero(value) == 0 for value in residuals)))
        expected_shift = (np.zeros_like(counts, dtype=float) if method.startswith("ce_") else -np.maximum(counts, 1).astype(float) ** (-.25)).astype(np.float32)
        for row in rows:
            rp, clientrows = prefix + f"round{row['round']}: ", row["clients"]
            check(rp + "client order, shared seeds and actual gradients", [client["client"] for client in clientrows] == list(range(CLIENTS)) and [client["seed"] for client in clientrows] == [20261211 + 100 * (row["round"] - 1) + i for i in range(CLIENTS)] and [client["steps"] for client in clientrows] == [client["gradient_evaluations"] for client in clientrows] == steps and row["actual_gradient_evaluations"] == sum(steps))
            check(rp + "actual TRAIN counts and exact FP32 loss shift", np.array_equal([client["train_counts"] for client in clientrows], counts) and np.array_equal(np.asarray([client["logit_shift"] for client in clientrows], dtype=np.float32), expected_shift))
            check(rp + "wire/downlink sums and EF recorded diagnostics", all(client["upload_bytes"] == packetbytes and 0 <= client["ef_max_error"] <= 5e-6 and math.isfinite(client["residual_norm"]) and client["residual_norm"] >= 0 and client["upload_accounting"] == ("analytical dense" if dense else "actual decoded wire") for client in clientrows) and row["upload_bytes"] == CLIENTS * packetbytes and row["download_bytes_analytical"] == CLIENTS * 547560 and row["global_anchor_unchanged_between_clients"] is True and (not dense or all(client["ef_max_error"] == client["residual_norm"] == 0 for client in clientrows)))
            val = row["validation"]
            klass = np.asarray(val["class_accuracy"])
            check(rp + "saved dev metric arithmetic", val["samples"] == 2000 and klass.shape == (10,) and np.all((klass >= 0) & (klass <= 1)) and np.allclose(klass * 200, np.rint(klass * 200), rtol=0, atol=1e-8) and abs(klass.mean() - val["accuracy"]) <= 3e-8 and abs(klass.mean() - val["macro_accuracy"]) < 1e-14 and klass.min() == val["worst_class_accuracy"] and math.isfinite(val["loss"]) and val["loss"] >= 0)
        packet = None
        if not dense:
            packet = decode_independently(read(folder / "first_payload.bin"))
            check(prefix + "independent first packet decode", packet["wire_bytes"] == packetbytes <= CAP and packet["header_bytes"] + packet["block_metadata_bytes"] + packet["packed_value_bytes"] == packetbytes)
        original_clients = recorded["results"][method]["clients"]
        check(prefix + "evaluation20 clients and unchanged global flag", len(original_clients) == CLIENTS and [client["client"] for client in original_clients] == list(range(CLIENTS)) and recorded["results"][method]["frozen_global_model_unchanged"] is True)
        clients, full_labels = [], np.full(50000, -1, dtype=np.int64)
        for i, original in enumerate(original_clients):
            cp = prefix + f"client{i}: "
            arrays = npzread(evaluation / f"{method}_client{i}.npz")
            z, y, ids = arrays["train_logits"], arrays["train_labels"], arrays["train_ids"]
            raw, learned, hy, hid = arrays["raw"], arrays["learned"], arrays["labels"], arrays["holdout_ids"]
            bias, prior = arrays["fitted_bias"], arrays["prior_shift"]
            check(cp + "finite typed array shapes", z.shape == (len(ids), 10) and raw.shape == learned.shape == (len(hid), 10) and z.dtype == raw.dtype == learned.dtype == bias.dtype == prior.dtype == np.float32 and y.shape == (len(ids),) and hy.shape == (len(hid),) and y.dtype.kind in "iu" and hy.dtype.kind in "iu" and bias.shape == prior.shape == (10,) and all(np.isfinite(value).all() for value in (z, raw, learned, bias, prior)) and np.all((y >= 0) & (y < 10)) and np.all((hy >= 0) & (hy < 10)))
            check(cp + "exact IDs/order and actual labels/counts", np.array_equal(ids, train[i]) and np.array_equal(hid, held[i]) and np.array_equal(np.bincount(y, minlength=10), counts[i]) and np.array_equal(arrays["train_counts"], counts[i]) and np.array_equal(np.bincount(y, minlength=10) + np.bincount(hy, minlength=10), manifest["original_client_label_counts"][i]))
            full_labels[ids], full_labels[hid] = y, hy
            if i not in common_gt:
                common_gt[i] = (y.copy(), ids.copy(), hy.copy(), hid.copy())
            else:
                check(cp + "same GT/order across methods", all(np.array_equal(a, b) for a, b in zip((y, ids, hy, hid), common_gt[i])))
            check(cp + "TRAIN-only analytic prior", np.array_equal(prior, np.log((counts[i] + 1.) / (len(y) + 10)).astype(np.float32)))
            bias_info, refit_bias = verify_bias(z, y, bias, original["bias_optimization"], check, cp)
            scores = {"raw": raw.astype(np.float64), "analytic_prior": raw.astype(np.float64) + prior.astype(np.float64), "prioranchor_bias": raw.astype(np.float64) + bias.astype(np.float64), "matched_learned_head": learned.astype(np.float64), "matched_learned_head_plus_prior": learned.astype(np.float64) + prior.astype(np.float64)}
            metrics = {}
            for variant, values in scores.items():
                metric = score_metrics(values, hy)
                reference = original["metrics"][variant]
                check(cp + variant + " independent metric", metric["samples"] == reference["samples"] and abs(metric["accuracy"] - reference["accuracy"]) < 2e-14 and abs(metric["loss"] - reference["loss"]) < 1e-11)
                metrics[variant] = metric
            refit_metrics = score_metrics(raw.astype(np.float64) + refit_bias.astype(np.float32).astype(np.float64), hy)
            bias_info["refit_holdout_CE_difference"] = abs(refit_metrics["loss"] - metrics["prioranchor_bias"]["loss"])
            check(cp + "independent refit preserves holdout decisions", refit_metrics["correct"] == metrics["prioranchor_bias"]["correct"] and bias_info["refit_holdout_CE_difference"] < 1e-7)
            cost, head = original["bias_extra_train_forward"], original["matched_head_training"]
            check(cp + "actual forward/gradient/state accounting", original["train_samples"] == cost["image_forwards"] == head["train_image_forwards"] == len(y) and cost["batch_calls"] == math.ceil(len(y) / 256) and original["holdout_samples"] == len(hy) and original["bias_new_cnn_backward"] == original["bias_extra_uplink_bytes"] == 0 and original["bias_local_state_bytes"] == 40 and head["gradient_evaluations"] == head["optimizer_steps"] == math.ceil(len(y) / 64) and head["state_bytes"] == 2600 and head["seed"] == 20261211 + 100000 + i and all(original[key]["image_forwards"] == len(hy) and original[key]["batch_calls"] == math.ceil(len(hy) / 256) for key in ("raw_holdout_forward", "learned_holdout_forward")))
            clients.append({"client": i, "train_samples": len(y), "holdout_samples": len(hy), "metrics": metrics, "bias_verification": bias_info})
        check(prefix + "reconstruct official50000 label hash", np.all(full_labels >= 0) and np.array_equal(np.bincount(full_labels, minlength=10), np.full(10, 5000)) and digest(full_labels.astype("<i8").tobytes()) == manifest["official_train_labels_sha256"] and grouped_hash([full_labels[ids] for ids in train]) == manifest["assignment_label_sha256"] and np.bincount(full_labels[dev], minlength=10).tolist() == manifest["validation_class_counts"] == [200] * 10)
        summaries = {variant: summary(clients, variant) for variant in VARIANTS}
        for variant, values in summaries.items():
            for key, reference in recorded["results"][method]["summaries"][variant].items():
                check(prefix + variant + "/" + key + " independent summary", abs(values[key] - reference) < (1e-11 if key.endswith("loss") else 2e-14))
        arms[method] = {"summaries": summaries, "clients": clients, "training_gradient_evaluations_recorded": sum(row["actual_gradient_evaluations"] for row in rows), "training_upload_bytes": sum(row["upload_bytes"] for row in rows), "training_download_bytes_analytical": sum(row["download_bytes_analytical"] for row in rows), "first_packet": packet}
    base = configurations[METHODS[0]]
    for method, conf in configurations.items():
        check(method + " common protocol/partition/source/data", all(conf[key] == base[key] for key in ("seed", "partition_seed", "rounds", "backend", "runtime", "data_manifest", "source_identities", "data_identities", "weights", "parameters", "batch", "lr", "local_epochs")))
    candidate = arms["fedlc_uniform3"]["summaries"]["prioranchor_bias"]
    prior = arms["fedlc_uniform3"]["summaries"]["analytic_prior"]
    ce = arms["ce_uniform3"]["summaries"]["prioranchor_bias"]
    dense = arms["fedlc_dense"]["summaries"]["prioranchor_bias"]
    strongest = max(("matched_learned_head", "matched_learned_head_plus_prior"), key=lambda name: arms["fedlc_uniform3"]["summaries"][name]["macro_accuracy"])
    head = arms["fedlc_uniform3"]["summaries"][strongest]
    gates = {"same_loss_prior_accuracy_plus_half_pp": candidate["macro_accuracy"] >= prior["macro_accuracy"] + .005, "same_loss_prior_lower_ce": candidate["macro_loss"] < prior["macro_loss"], "ce_bias_accuracy_noninferior_half_pp": candidate["macro_accuracy"] >= ce["macro_accuracy"] - .005, "ce_bias_ce_within_001": candidate["macro_loss"] <= ce["macro_loss"] + .01, "matched_head_accuracy_noninferior_half_pp": candidate["macro_accuracy"] >= head["macro_accuracy"] - .005, "quant_dense_accuracy_noninferior_half_pp": candidate["macro_accuracy"] >= dense["macro_accuracy"] - .005}
    decision = "FOLLOW_UP" if all(gates.values()) else "STOP"
    check("frozen gates and decision independently recomputed", gates == recorded["gates"] and decision == recorded["decision"] and strongest == recorded["matched_head_strongest_variant"])
    check("fixed final protocol metadata", recorded["training_seed"] == base["seed"] and recorded["partition_seed"] == base["partition_seed"] and recorded["training_rounds"] == ROUNDS)
    train_total = base["data_manifest"]["train_samples"]
    head_steps = sum(math.ceil(size / 64) for size in base["data_manifest"]["client_sizes"]) * len(METHODS)
    check("aggregate bias/head compute accounting", recorded["bias_total_train_image_forwards"] == train_total * len(METHODS) and recorded["bias_total_cnn_backward"] == recorded["bias_extra_uplink_bytes"] == 0 and recorded["bias_state_bytes_per_client"] == 40 and recorded["matched_head_total_gradient_evaluations"] == head_steps)
    unchanged = all(digest(Path(path).read_bytes()) == value["sha256"] for path, value in snapshots.items())
    check("all independently read inputs unchanged", unchanged)
    failures = [item for item in checks if not item["pass"]]
    details = [client["bias_verification"] for arm in arms.values() for client in arm["clients"]]
    return {"status": "DONE", "artifact_check": "FAIL" if failures else "PASS", "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures, "review": "REVIEW_UNAVAILABLE", "candidate_fixed": "fedlc_uniform3 + prioranchor_bias", "decision": decision, "gates": gates, "strongest_matched_head": strongest, "candidate_gain_vs_prior_pp": 100 * (candidate["macro_accuracy"] - prior["macro_accuracy"]), "candidate_CE_reduction_vs_prior": prior["macro_loss"] - candidate["macro_loss"], "candidate_minus_CE_bias_pp": 100 * (candidate["macro_accuracy"] - ce["macro_accuracy"]), "candidate_minus_dense_bias_pp": 100 * (candidate["macro_accuracy"] - dense["macro_accuracy"]), "actual_train_samples": train_total, "actual_holdout_samples": base["data_manifest"]["holdout_samples"], "arms": arms, "bias_extra_TRAIN_forwards_recorded": train_total * len(METHODS), "bias_forward_batches_recorded": len(METHODS) * sum(math.ceil(size / 256) for size in base["data_manifest"]["client_sizes"]), "matched_head_gradients_recorded": head_steps, "accepted_Newton_steps": sum(value["accepted_Newton_steps"] for value in details), "objective_evaluations": sum(value["objective_evaluations"] for value in details), "linear_solves": sum(value["linear_solves"] for value in details), "backtracks": sum(value["backtracks"] for value in details), "max_trace_F_error": max(value["trace_objective_error"] for value in details), "max_trace_g_error": max(value["trace_gradient_error"] for value in details), "max_refit_FP64_error": max(value["refit_FP64_error"] for value in details), "refit_FP32_exact_count": sum(value["refit_FP32_exact"] for value in details), "max_final_gradient_inf": max(value["final_gradient_inf"] for value in details), "snapshots": snapshots, "inputs_unchanged": unchanged, "original_image_hashes_not_reread": skipped_images, "limits": ["No author solver/codec/evaluator imports, original images/official TEST reads, CNN construction or image forward/backward/FL/head training. Torch solely deserializes saved CPU checkpoint tensors with weights_only=True.", "Saved TRAIN/holdout IDs and labels reconstruct the official50000 label hash. Saved logits are not established by an independent image forward; no official TEST observation is claimed.", "All20 rounds have logged count/seed/shift/gradient/byte arithmetic checked; final residual norms and both first packets are independently parsed. Only first packets are retained: full per-round EF, local deltas and aggregation cannot be replayed from final artifacts, and logged invariants remain author-recorded diagnostics.", "NumPy refit uses only saved TRAIN arrays. Equivalent logaddexp summations may alter last-bit Armijo behavior at FP64 resolution; replay tolerance2e-10 does not change the fixed performance thresholds.", "One exploratory partition20261203 and trainingseed20261211 selected after old-split results. Clients/rounds are not independent training seeds; no generalization or atomic novelty claim. Old balanced/prior/bias STOP decisions remain unchanged.", "FOLLOW_UP if every fixed gate passes still requires zero-anchor lambda10 bias, vector scaling/BCTS, strong FedCal vector, freshseed20261212/13, second task, attribution and total-cost comparisons. No rescue parameter selection.", "REVIEW_UNAVAILABLE records unavailable external cross-model review; local artifact PASS is not external scientific acceptance."]}


def make_report(result):
    candidate = result["arms"]["fedlc_uniform3"]["summaries"]["prioranchor_bias"]
    lines = ["# 新分区确认独立核查", f"DONE；产物{result['artifact_check']} {result['checks_passed']}/{result['checks_total']}；固定FedLC uniform3＋prior-anchor bias：**{result['decision']}**；外部REVIEW_UNAVAILABLE。", f"宏acc {100*candidate['macro_accuracy']:.6f}%、CE {candidate['macro_loss']:.9f}；比同loss prior {result['candidate_gain_vs_prior_pp']:+.6f}pp、CE减少{result['candidate_CE_reduction_vs_prior']:.9f}；比CE+bias {result['candidate_minus_CE_bias_pp']:+.6f}pp、dense+bias {result['candidate_minus_dense_bias_pp']:+.6f}pp。", "六个冻结门槛：" + "；".join(f"{key}={'通过' if value else '未达'}" for key, value in result["gates"].items()) + "。", f"实际TRAIN/holdout={result['actual_train_samples']}/{result['actual_holdout_samples']}；60组保存数组重建50000标签hash、300条client指标与75项汇总独立复算。TRAIN-only prior、F/g/H有限差分、H≥10I、零均值及全部Newton/Armijo轨迹核验；独立重新拟合FP64最大差{result['max_refit_FP64_error']:.2g}，FP32位级一致{result['refit_FP32_exact_count']}/60。", f"三组训练日志的共同seed/counts/shift/梯度/字节、checkpoint最终布局/EF norm与首包解析核查。作者bias额外TRAIN forward {result['bias_extra_TRAIN_forwards_recorded']}样本/{result['bias_forward_batches_recorded']}批，CNN反传0；matched head {result['matched_head_gradients_recorded']}梯度；Newton {result['accepted_Newton_steps']}步、目标评估{result['objective_evaluations']}次；bias40B/client。", "未读原图/official TEST，未做新CNN/训练；仅保存首包，不能从最终产物独立重演所有轮local delta/EF/聚合。一个探索性新split/seed，非最终test，clients不是seeds；旧STOP不变。FOLLOW_UP也须零anchor、BCTS/FedCal、两freshseed、第二任务及归属/总成本对照。源/输入重验不变；完整证据见LOCAL_CHECK.json。"]
    lines.insert(-1, f"部分fit以FP64目标分辨率停止，最终梯度∞范数最大{result['max_final_gradient_inf']:.2g}，非全部≤1e−8；轨迹等价求和容差2e−10不改效果门槛。")
    return "\n".join(lines) + "\n"


def self_test():
    rng = np.random.default_rng(818)
    z = rng.normal(size=(83, 10)).astype(np.float32)
    y = rng.integers(0, 10, size=83)
    b0 = centered(np.log((np.bincount(y, minlength=10) + 1.) / 93))
    f, g, h = independent_fgh(z, y, b0, b0)
    step, direction = 1e-4, centered(np.arange(10.))
    direction /= np.linalg.norm(direction)
    fp, gp, _ = independent_fgh(z, y, b0 + step * direction, b0)
    fm, gm, _ = independent_fgh(z, y, b0 - step * direction, b0)
    assert abs((fp - fm) / (2 * step) - g @ direction) < 1e-6
    assert np.max(np.abs((gp - gm) / (2 * step) - h @ direction)) < 1e-6
    b, info = independent_fit(z, y)
    assert info["objective"] < f and abs(b.mean()) < 1e-14 and info["hessian_min_eigenvalue"] >= 10 - 1e-8
    print(json.dumps({"scope": "independent verifier synthetic NumPy math self-check only", "status": "PASS", "formal_training_started": False, "author_sources_imported": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    here = Path(__file__).resolve().parent
    parser.add_argument("--model-dir", type=Path, default=here / "development_p20261203_s20261211")
    parser.add_argument("--evaluation-dir", type=Path, default=here / "evaluation_p20261203_s20261211")
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = verify(here, args.model_dir, args.evaluation_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LOCAL_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "LOCAL_CHECK.md").write_text(make_report(result))
    print(json.dumps({key: result[key] for key in ("status", "artifact_check", "checks_passed", "checks_total", "decision", "gates", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
