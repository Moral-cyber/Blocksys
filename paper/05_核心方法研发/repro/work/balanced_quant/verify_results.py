"""Independent saved-artifact checker. No dataset, training or author codec imports.

Inputs: four config.json/result.json/checkpoint.pt and three saved first packets.
Unfinished experiments are RUNNING and cannot receive a GO or STOP decision.
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

METHODS = ("ce_uniform3", "fedlc_uniform3", "balanced_uniform3", "balanced_dense")
N, C, ROUNDS, CLIENTS, CAP = 136874, 10, 20, 20, 54808
SOURCE_SHA = "b94f6a3a985839e6781582c6203e3dd8154fc97cf7bcc8edfe9d4648240625d4"
PLAN_SHA = "6c58ed74e04c8b7477faa7fcaee2e45f11f86f02775bd0642c6b08da2a304397"
CODEC_SHA = "46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863"
HEADER = struct.Struct("<8sIIII40s")
EF_CHECK_TOLERANCE = 5e-6
RULES = {"ce": "ordinary CE", "fedlc": "CE(z-max(n,1)^(-1/4)); inspired explicit variant, tau1", "balanced": "CE(z+log(n+1)); Dirichlet1-smoothed balanced softmax explicit variant"}
STATIC_NOTES = [
    "run.py:27-30,40-41: local counts come from client training ds.labels[ds.indices]; CE shift=0, FedLC-inspired shift=-max(n,1)^(-1/4), balanced shift=+log(n+1), all cast to FP32. Balanced is add-one, not max(n,1).",
    "run.py:42-50,78: batch64 with no dropped tail; one backward/SGD step per actual batch, common seed=20261201+100*r+i; no calibration grids/extra backward.",
    "run.py:63,88; separately read run_followup.py:120-135: evaluation uses raw model(x), CE(logits,y), argmax(logits) and dataset labels for metrics; no inference count/logit shift.",
    "run.py:79-87: quantized delta+old residual is serialized with frozen uniform3, decoded, compared with encoder sent, then aggregated with fixed sample weights. EF=u-decoded; dense sends delta and resets residual. Only first wire is retained.",
    "run.py:89-92: complete metric/client history and residuals stored in checkpoint and result; all downloads and dense uploads are analytical. Resume checks signature only, not independent state-shape validation; no fresh-run blocker was found.",
    "PLAN.md:26-28: all four arms 20 rounds; round16-20 balanced_uniform3 must have lower CE and accuracy >= each CE/FedLC baseline +.005, and accuracy >= balanced_dense -.01. Failure stops this fixed recipe without rescuing its smoothing/temperature/bits/local epochs.",
    "Separately read run_partitions.py:145: each client's train cutoff is round(.75*client_total); exact aggregate train total need not be 37500. This run has 37502 train / 12498 holdout, a disclosed numerical deviation from PLAN, shared by all four arms.",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def grouped_hash(groups):
    data = bytearray()
    for group in groups:
        data.extend(struct.pack("<I", len(group)))
        data.extend(np.asarray(group, dtype="<u4").tobytes())
    return sha(data)


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
    return {"wire_bytes": len(wire), "decoded_coordinates": int(vector.size), "decoded_fp32_sha256": sha(vector.astype("<f4").tobytes()), "decoded_l2": float(np.linalg.norm(vector.astype(np.float64))), "all_finite": bool(np.isfinite(vector).all()), "header_bytes": HEADER.size, "blocks": blocks, "block_metadata_bytes": blocks * 5, "packed_value_bytes": packed_total, "tail_coordinates": N % 256, "uniform_bitwidth": 3, "zero_scale_blocks": zero_blocks, "min_scale": min(scales), "max_scale": max(scales), "code_range": [mincode, maxcode]}


def summarize_last5(rows):
    rows = rows[15:20]
    classes = np.asarray([row["validation"]["class_accuracy"] for row in rows], dtype=float)
    means = classes.mean(axis=0)
    return {"rounds": [row["round"] for row in rows], "ce": float(np.mean([row["validation"]["loss"] for row in rows])), "accuracy": float(np.mean([row["validation"]["accuracy"] for row in rows])), "class_accuracy_mean": means.tolist(), "macro_accuracy_mean": float(means.mean()), "mean_round_worst_class_accuracy": float(classes.min(axis=1).mean()), "minimum_mean_class_accuracy": float(means.min()), "minimum_mean_class_indices": np.flatnonzero(means == means.min()).tolist()}


def verify(root):
    checks, snapshots, configs, arms, histories, warnings = [], {}, {}, {}, {}, []

    def check(name, passed, details=None):
        checks.append({"check": name, "pass": bool(passed), "details": details})

    def read(path):
        content = path.read_bytes()
        snapshots[str(path.resolve())] = {"bytes": len(content), "sha256": sha(content)}
        return content

    for method in METHODS:
        folder = root / method
        files = [folder / name for name in ("config.json", "result.json", "checkpoint.pt")]
        if not all(path.exists() for path in files):
            arms[method] = {"status": "NOT_STARTED_OR_NO_COMPLETE_SNAPSHOT", "rounds": 0}
            continue
        try:
            conf, result = [json.loads(read(path)) for path in files[:2]]
            ck = torch.load(io.BytesIO(read(files[2])), map_location="cpu", weights_only=False)
        except Exception as error:
            arms[method] = {"status": "SNAPSHOT_READ_ERROR", "error": repr(error)}
            continue
        configs[method], histories[method] = conf, result["rows"]
        rows, ckrows = result["rows"], json.loads(json.dumps(ck["rows"]))
        loss, compression = method.split("_", 1)
        dense, prefix = compression == "dense", method + ": "
        manifest = conf["data_manifest"]
        sizes, counts = manifest["client_sizes"], np.asarray(manifest["client_label_counts"], dtype=np.int64)
        weights = np.asarray(sizes, dtype=float) / sum(sizes)
        steps = [math.ceil(n / 64) for n in sizes]
        unsigned = {k: v for k, v in conf.items() if k != "signature"}
        check(prefix + "config/result/checkpoint signatures", sha(json.dumps(unsigned, sort_keys=True).encode()) == conf["signature"] == result["signature"] == ck["signature"])
        check(prefix + "frozen source/plan/codec identities", (conf["source_sha256"], conf["plan_sha256"], conf["codec_sha256"]) == (SOURCE_SHA, PLAN_SHA, CODEC_SHA))
        check(prefix + "fixed setup and loss/raw inference", conf["method"] == method and conf["seed"] == 20261201 and conf["partition_seed"] == manifest["partition_seed"] == manifest["split_seed"] == 20261193 and conf["rounds"] == ROUNDS and conf["parameter_count"] == N and conf["batch"] == 64 and conf["lr"] == .03 and conf["local_epochs"] == 1 and conf["backend"] == "mps" and conf["inference"] == "raw logits" and conf["loss_rule"] == RULES[loss] and manifest["alpha"] == .1 and manifest["num_clients"] == CLIENTS)
        check(prefix + "fixed weights from actual train counts", len(sizes) == CLIENTS and sum(sizes) == manifest["train_samples"] and np.allclose(conf["weights"], weights, rtol=0, atol=1e-15) and abs(sum(conf["weights"]) - 1) < 1e-12)
        if manifest["train_samples"] != 37500 or manifest["holdout_samples"] != 12500:
            warnings.append({"method": method, "warning": "PLAN names train37500/holdout12500; actual per-client rounded split differs", "planned_train": 37500, "actual_train": manifest["train_samples"], "planned_holdout": 12500, "actual_holdout": manifest["holdout_samples"], "action": "disclose actual counts; do not change frozen performance rule or data"})
        check(prefix + "TRAIN count metadata", counts.shape == (CLIENTS, C) and np.all(counts >= 0) and np.array_equal(counts.sum(axis=1), sizes))
        check(prefix + "official test excluded metadata", conf["official_test_loaded"] is False and result["official_test_loaded"] is False and manifest["test_used_for_selection"] is False and manifest["official_split"] == "CIFAR-10 train (50,000 original rows)")
        train, holdout, dev = manifest["train_indices"], manifest["holdout_indices"], manifest["validation_indices"]
        trainids, heldids = [i for ids in train for i in ids], [i for ids in holdout for i in ids]
        check(prefix + "development split disjoint and exhaustive", len(trainids) == len(set(trainids)) == manifest["train_samples"] and len(heldids) == len(set(heldids)) == manifest["holdout_samples"] and set(trainids).isdisjoint(heldids) and set(trainids + heldids) == set(range(50000)) and len(dev) == len(set(dev)) == 2000 and set(dev) <= set(heldids) and manifest["validation_class_counts"] == [200] * C)
        check(prefix + "manifest ID hash and size reconstruction", grouped_hash(train) == manifest["train_indices_sha256"] and grouped_hash(holdout) == manifest["holdout_indices_sha256"] and sha(np.asarray(dev, dtype="<u4").tobytes()) == manifest["validation_ids_sha256"] and [len(ids) for ids in train] == sizes and [len(ids) for ids in holdout] == manifest["client_holdout_sizes"])
        check(prefix + "per-client rounding explains split total", all(ntrain == max(1, min(ntrain + nheld - 1, round(.75 * (ntrain + nheld)))) for ntrain, nheld in zip(sizes, manifest["client_holdout_sizes"])))
        check(prefix + "round sequence/status", [r["round"] for r in rows] == list(range(1, len(rows) + 1)) and len(rows) <= ROUNDS and result["status"] == ("DONE" if len(rows) == ROUNDS else "RUNNING"))
        common = min(len(rows), len(ckrows))
        check(prefix + "checkpoint/result common history", ckrows[:common] == rows[:common], {"result_rounds": len(rows), "checkpoint_rounds": len(ckrows)})
        if result["status"] == "DONE":
            check(prefix + "checkpoint/result final complete history", ckrows == rows)
        check(prefix + "saved values finite", finite_tree(ck) and finite_tree(rows))
        model, residuals = ck["model"], ck["residuals"]
        check(prefix + "checkpoint model and residual layout", sum(t.numel() for t in model.values()) == N and all(t.dtype == torch.float32 and t.device.type == "cpu" for t in model.values()) and len(residuals) == CLIENTS and all(t.shape == (N,) and t.dtype == torch.float32 and t.device.type == "cpu" for t in residuals))
        if dense:
            check(prefix + "dense residual reset", all(torch.count_nonzero(t) == 0 for t in residuals))
        if ckrows:
            check(prefix + "checkpoint final residual norms", all(abs(float(t.norm()) - client["residual_norm"]) <= 5e-6 for t, client in zip(residuals, ckrows[-1]["clients"])))
        expected_shift = np.zeros_like(counts, dtype=float) if loss == "ce" else -np.maximum(counts, 1).astype(float) ** (-.25) if loss == "fedlc" else np.log(counts + 1)
        expected_shift = expected_shift.astype(np.float32)
        nbytes = 64 + 4 * N if dense else 64 + 5 * ((N + 255) // 256) + (N // 256) * 96 + ((N % 256) * 3 + 7) // 8
        upload = download = gradients = 0
        max_ef, max_shift_error, max_residual = 0., 0., 0.
        for row in rows:
            rp, clients = prefix + "round " + str(row["round"]) + ": ", row["clients"]
            observed_counts = np.asarray([cl["train_counts"] for cl in clients])
            observed_shift = np.asarray([cl["logit_shift"] for cl in clients], dtype=np.float32)
            check(rp + "complete clients, actual K and backward count", [cl["client"] for cl in clients] == list(range(CLIENTS)) and [cl["steps"] for cl in clients] == [cl["gradient_evaluations"] for cl in clients] == steps)
            check(rp + "actual TRAIN counts match fixed manifest", np.array_equal(observed_counts, counts) and np.array_equal(observed_counts.sum(axis=1), sizes))
            shift_error = float(np.abs(observed_shift - expected_shift).max())
            max_shift_error = max(max_shift_error, shift_error)
            check(rp + "FP32 exact logit shift and zero counts", np.array_equal(observed_shift, expected_shift), {"max_abs_error": shift_error})
            if loss == "balanced":
                check(rp + "balanced zero-count log shift finite zero", np.all(observed_shift[counts == 0] == 0) and np.isfinite(observed_shift).all())
            check(rp + "exact byte sums and cap", all(cl["upload_bytes"] == nbytes for cl in clients) and row["upload_bytes"] == CLIENTS * nbytes and row["download_bytes_analytical"] == CLIENTS * (64 + 4 * N) and conf["upload_cap_bytes"] == (None if dense else CAP) and (dense or nbytes == 54067 <= CAP))
            check(rp + "EF diagnostic within numerical tolerance", all(0 <= cl["ef_max_error"] <= EF_CHECK_TOLERANCE for cl in clients) and (not dense or all(cl["ef_max_error"] == cl["residual_norm"] == 0 for cl in clients)))
            max_ef = max(max_ef, *(cl["ef_max_error"] for cl in clients))
            max_residual = max(max_residual, *(cl["residual_norm"] for cl in clients))
            val, klass = row["validation"], np.asarray(row["validation"]["class_accuracy"], dtype=float)
            check(rp + "saved metric/class/worst arithmetic", val["samples"] == 2000 and klass.shape == (C,) and np.all((klass >= 0) & (klass <= 1)) and np.allclose(klass * 200, np.rint(klass * 200), rtol=0, atol=1e-8) and abs(klass.mean() - val["accuracy"]) <= 3e-8 and abs(klass.mean() - val["macro_accuracy"]) <= 1e-14 and min(klass) == val["worst_class_accuracy"] and val["loss"] >= 0)
            upload += row["upload_bytes"]
            download += row["download_bytes_analytical"]
            gradients += sum(cl["gradient_evaluations"] for cl in clients)
        packet = None
        if not dense and rows:
            try:
                wire = read(folder / "first_payload.bin")
                packet = decode_independently(wire)
                check(prefix + "independent first packet decode/accounting", len(wire) == rows[0]["clients"][0]["upload_bytes"] == 54067 and packet["header_bytes"] + packet["block_metadata_bytes"] + packet["packed_value_bytes"] == 54067 and len(wire) <= CAP)
            except Exception as error:
                check(prefix + "independent first packet decode/accounting", False, repr(error))
        arms[method] = {"status": result["status"], "rounds": len(rows), "checkpoint_rounds": len(ckrows), "loss": loss, "actual_train_samples": manifest["train_samples"], "actual_holdout_samples": manifest["holdout_samples"], "train_counts_from_manifest": counts.tolist(), "zero_count_class_entries": int((counts == 0).sum()), "expected_fp32_logit_shift": expected_shift.astype(float).tolist(), "recorded_logit_shift_max_error": max_shift_error, "actual_steps_per_client_per_round": steps, "gradient_evaluations_total": gradients, "optimizer_steps_not_separately_logged": True, "upload_bytes_total": upload, "upload_basis": "analytical dense FP32 delta+64B" if dense else "actual serialized uniform3 payload", "upload_bytes_per_client_per_round": nbytes, "download_bytes_analytical_total": download, "total_communication_bytes": upload + download, "ef_max_error_recorded": max_ef, "ef_local_check_tolerance": EF_CHECK_TOLERANCE, "max_recorded_residual_norm": max_residual, "first_packet": packet}
    if configs:
        base = next(iter(configs.values()))
        for method, conf in configs.items():
            check(method + ": common setup/partition/helper identities", all(conf[k] == base[k] for k in ("seed", "partition_seed", "rounds", "backend", "source_sha256", "plan_sha256", "codec_sha256", "runtime", "data_manifest", "weights", "batch", "lr", "local_epochs", "inference", "parameter_count", "official_test_loaded", "evaluation")))
    complete = all(arms[m].get("status") == "DONE" and arms[m].get("rounds") == arms[m].get("checkpoint_rounds") == ROUNDS for m in METHODS)
    decision = {"status": "RUNNING", "go": None, "frozen_rule": "all four DONE; balanced_uniform3 last5 CE lower than CE and FedLC, accuracy >= each+.005, accuracy >= balanced_dense-.01", "accuracy_baseline_margin": .005, "dense_noninferiority_margin": .01}
    if complete:
        for method in METHODS:
            arms[method]["last5"] = summarize_last5(histories[method])
        cand = arms["balanced_uniform3"]["last5"]
        comparisons = {}
        for baseline in ("ce_uniform3", "fedlc_uniform3"):
            ref = arms[baseline]["last5"]
            comparisons[baseline] = {"ce_reduction": ref["ce"] - cand["ce"], "accuracy_gain": cand["accuracy"] - ref["accuracy"], "ce_lower": cand["ce"] < ref["ce"], "accuracy_at_least_plus_0_005": cand["accuracy"] >= ref["accuracy"] + .005}
        dense = arms["balanced_dense"]["last5"]
        dense_cmp = {"quantized_minus_dense_ce": cand["ce"] - dense["ce"], "quantized_minus_dense_accuracy": cand["accuracy"] - dense["accuracy"], "accuracy_at_least_dense_minus_0_01": cand["accuracy"] >= dense["accuracy"] - .01}
        metric_go = all(comp["ce_lower"] and comp["accuracy_at_least_plus_0_005"] for comp in comparisons.values()) and dense_cmp["accuracy_at_least_dense_minus_0_01"]
        decision.update(metric_rule_pass=metric_go, comparisons=comparisons, dense_comparison=dense_cmp)
    unchanged = all(sha(Path(name).read_bytes()) == item["sha256"] for name, item in snapshots.items())
    check("saved inputs unchanged after verification", unchanged)
    failures = [item for item in checks if not item["pass"]]
    if complete:
        metric_go = decision["metric_rule_pass"]
        decision.update(status="GO_FOR_LIMITED_CONFIRMATION" if metric_go and not failures else "STOP_THIS_FIXED_BALANCED_QUANT_RECIPE" if not metric_go else "ARTIFACT_CHECK_FAILED", go=metric_go and not failures)
    return {"experiment_status": "ALL_DONE" if complete else "RUNNING", "artifact_check": "FAIL" if failures else "PASS", "protocol_fidelity": "WARN" if warnings else "PASS", "warnings": warnings, "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures, "review": "REVIEW_UNAVAILABLE", "decision": decision, "arms": arms, "snapshots": snapshots, "saved_inputs_unchanged_after_verification": unchanged, "checks": checks, "separately_read_static_source_notes": STATIC_NOTES, "limits": ["This is one development training seed20261201 and partition20261193 on CIFAR10 official TRAIN only, fixed 20 rounds/SGD.03/epoch1. Rounds are not independent seeds and the new partition is not a new dataset.", "Actual train/holdout totals are 37502/12498 due to each-client round(.75*n); PLAN's 37500/12500 is numerically inaccurate. All arms share actual IDs/counts and sample weights. This is disclosed as a protocol warning, not silently treated as exact PLAN replication.", "Counts are recomputed for consistency from saved TRAIN-count metadata, not from original labels/data. Dataset/official TEST is never loaded by this checker.", "Raw-logit inference, no official TEST access, shared initialization/batch streams and MPS training are supported by metadata/static source. This checker does not independently reproduce the MPS training trajectory or trace filesystem accesses.", "Loss shifts are independently reconstructed and compared bitwise at FP32; +1 balanced loss and FedLC-inspired loss are explicit variants, not official full-method reproductions or new atomic mechanisms.", "Only first saved packet per quantized arm is independently decoded. Historical payloads, local raw deltas and pre-update residuals are absent, so EF bounds are recorded diagnostics, not independently replayed conservation or a contraction guarantee. The 5e-6 local arithmetic tolerance is not a changed performance criterion.", "Checkpoint/result equality checks saved metrics. Last5/class/worst values are arithmetic recomputations; no predictions/CE are reevaluated from data.", "Quantized uploads are actual wire lengths; dense uploads and all downlinks are analytical. No real network/authentication/energy/blockchain performance is tested.", "Failure applies to this fixed add-one balanced recipe and does not invalidate mechanism combination research or the FedLC-inspired low-bit reference. No new-algorithm or all-baselines superiority claim is supported.", "No external cross-model reviewer is available: REVIEW_UNAVAILABLE."]}


def report(out):
    lines = ["# Balanced quant：独立保存产物核验", "", f"实验状态 **{out['experiment_status']}**；保存产物 **{out['artifact_check']}（{out['checks_passed']}/{out['checks_total']}）**；方案数字一致性 **{out['protocol_fidelity']}**；冻结判据 **{out['decision']['status']}**；外部审查 `REVIEW_UNAVAILABLE`。", "", "脚本仅仅读 config/result/checkpoint/首包，不导入训练源文件或原codec，不加载数据、不训练、不读取official TEST。", "", "样本数警告：实际TRAIN37502、holdout12498；PLAN写37500/12500，源于逐client round(.75*n)。四组共享实际分区与样本权，IDs互斥合计50000，不能称精确遵守该PLAN数字。", "", "| 组 | 状态/轮数 | last5 CE | last5 acc% | mean round-worst% | min mean-class% | 梯度次数 | 上行B | 解析下行B | 总通信B |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, arm in out["arms"].items():
        m = arm.get("last5")
        vals = [f"{m['ce']:.10f}", f"{100*m['accuracy']:.6f}", f"{100*m['mean_round_worst_class_accuracy']:.6f}", f"{100*m['minimum_mean_class_accuracy']:.6f}"] if m else ["—"] * 4
        lines.append(f"| {method} | {arm['status']}/{arm.get('rounds', 0)} | " + " | ".join(vals) + f" | {arm.get('gradient_evaluations_total', '—')} | {arm.get('upload_bytes_total', '—')} | {arm.get('download_bytes_analytical_total', '—')} | {arm.get('total_communication_bytes', '—')} |")
    lines += ["", "冻结GO：四组各20轮DONE；round16–20 balanced_uniform3对CE/FedLC分别CE更低且accuracy≥+.005；对balanced_dense accuracy≥−.01。全部条件同时成立。"]
    if out["experiment_status"] == "ALL_DONE":
        for ref, comp in out["decision"]["comparisons"].items():
            lines.append(f"- 相对{ref}：CE降低={comp['ce_reduction']:.10f}；accuracy差={100*comp['accuracy_gain']:.6f}pp；CE更低={comp['ce_lower']}；≥+.5pp={comp['accuracy_at_least_plus_0_005']}。")
        comp = out["decision"]["dense_comparison"]
        lines.append(f"- 相对balanced_dense：量化减dense CE={comp['quantized_minus_dense_ce']:.10f}；accuracy差={100*comp['quantized_minus_dense_accuracy']:.6f}pp；≥−1pp={comp['accuracy_at_least_dense_minus_0_01']}。")
        lines += ["", "逐类last5均值（class0–9，%）：", "", "| 组 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method, arm in out["arms"].items():
            lines.append("| " + method + " | " + " | ".join(f"{100*v:.4f}" for v in arm["last5"]["class_accuracy_mean"]) + " |")
    else:
        lines += ["", "RUNNING：不根据中途数据给GO/STOP。"]
    for method, arm in out["arms"].items():
        if "actual_steps_per_client_per_round" not in arm:
            continue
        lines += ["", f"{method}：实际K={arm['actual_steps_per_client_per_round']}；梯度次数={arm['gradient_evaluations_total']}，optimizer未单独记录（静态每batch一次step）；每client每轮上行={arm['upload_bytes_per_client_per_round']}B（{arm['upload_basis']}）；EF记录max={arm['ef_max_error_recorded']:.3g}，局部核对容差={arm['ef_local_check_tolerance']}。计数零项={arm['zero_count_class_entries']}；独立FP32 shift最大差={arm['recorded_logit_shift_max_error']}。"]
        if arm["first_packet"]:
            p = arm["first_packet"]
            lines.append(f"首包独立解析={p['wire_bytes']}B={p['header_bytes']}+{p['block_metadata_bytes']}+{p['packed_value_bytes']}；{p['blocks']}块，tail{p['tail_coordinates']}，3bit，{p['decoded_coordinates']}个有限FP32坐标。")
    lines += ["", "每client完整TRAIN计数及期望FP32 shift矩阵保存于JSON，逐轮逐client与metadata一致。", "", "只读静态核对（不是脚本运行输入）："] + ["- " + note for note in out["separately_read_static_source_notes"]]
    lines += ["", "证据边界："] + ["- " + note for note in out["limits"]]
    if out["failures"]:
        lines += ["", "失败项："] + [f"- {item['check']}: {item['details']}" for item in out["failures"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=here / "development_p20261193_s20261201")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = verify(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LOCAL_STATS_CHECK.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "LOCAL_RESULTS_CHECK.md").write_text(report(out))
    print(json.dumps({k: out[k] for k in ("experiment_status", "artifact_check", "checks_passed", "checks_total", "decision", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
