"""Read-only, independent arithmetic and saved-artifact checks; never trains.

Inputs are config.json, result.json, checkpoint.pt and the saved first payload.
No training source, original codec, dataset or official test is imported/read.
Incomplete four-arm experiments always produce RUNNING without a GO/STOP gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch


METHODS = ("plain_uniform", "exact_head", "head_control", "head_control_dense")
BODY, HEAD, CAP, ROUNDS, LR = 136224, 650, 54808, 20, .03
PARAMS = BODY + HEAD
WRAPPER = struct.Struct("<8sIIIIII")
QHEADER = struct.Struct("<8sIIII40s")
EXPECTED_SOURCE = "8ee487ea523ee772a99e3555174cc9f41d2c9e1ade300f58b3df778562c63b82"
EXPECTED_PLAN = "31b014425b47eb7af3fe36c51d755246e57127ebed68034f4e02572db277b48e"
EXPECTED_CODEC = "46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863"
STATIC_NOTES = [
    "run.py:42-71: one CE backward per actual batch; correction is added only to the last weight/bias gradients. The telescoping identity c_i'=mean(raw head gradients) is tested inside training at 5e-6, not independently replayed here.",
    "run.py:74-94,132-144: compact arms decode their actual wire before weighted aggregation; head is FP32 exact and has no EF; body alone receives EF for exact_head/control. Wrapper is 32B, head is 2600B, body budget is 52176B.",
    "run.py:142-156: because c=sum(p_i*c_i), full participation and fixed p_i cancel old control terms, so the server reconstruction sum(-p_i*delta_h/(eta*K_i)) matches weighted new c_i. This argument does not establish partial-participation support.",
    "run.py:157-166: checkpoint includes the same saved metric rows as result.json; c/control/client residual states are checkpointed. All download counts and dense uploads are analytical.",
    "run.py:178-181 and PLAN.md:29-32,36-44: shared fixed official-TRAIN development split, no official TEST call on this route; frozen last5 criterion and one exploratory seed/partition.",
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def grouped_hash(groups):
    buf = bytearray()
    for ids in groups:
        buf.extend(struct.pack("<I", len(ids)))
        buf.extend(np.asarray(ids, dtype="<u4").tobytes())
    return sha(buf)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def independent_decode(wire):
    """Independent NumPy little-endian bit extraction, no original decoder."""
    assert len(wire) >= QHEADER.size, "short codec header"
    magic, version, n, block, blocks, reserved = QHEADER.unpack_from(wire)
    assert magic == b"MPQCODEC" and version == 1
    assert n > 0 and block == 256 and blocks == (n + 255) // 256
    assert not any(reserved)
    offset, values, widths, scales, packed_total = QHEADER.size, [], [], [], 0
    for i in range(blocks):
        count = min(256, n - i * 256)
        assert offset + 5 <= len(wire), "short block header"
        width, scale = wire[offset], struct.unpack_from("<f", wire, offset + 1)[0]
        assert width in (2, 3, 4) and math.isfinite(scale) and scale >= 0
        offset += 5
        packed_size = (count * width + 7) // 8
        data = wire[offset:offset + packed_size]
        assert len(data) == packed_size
        used_bits = count * width
        if used_bits % 8:
            assert data[-1] >> (used_bits % 8) == 0
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[:used_bits]
        codes = (bits.reshape(count, width).astype(np.uint16) * (1 << np.arange(width))).sum(axis=1)
        qmax = (1 << (width - 1)) - 1
        assert np.all(codes < (1 << width) - 1), "noncanonical unused code"
        if scale == 0:
            assert np.all(codes == qmax), "noncanonical zero block"
        decoded = (codes.astype(np.int16) - qmax).astype(np.float32) * np.float32(scale)
        assert np.isfinite(decoded).all()
        values.append(decoded)
        widths.append(width)
        scales.append(scale)
        offset += packed_size
        packed_total += packed_size
    assert offset == len(wire), "trailing codec bytes"
    return np.concatenate(values), {
        "numel": n, "nblocks": blocks, "bitwidths": widths, "scales": scales,
        "bitwidth_counts": dict(sorted(Counter(widths).items())),
        "wire_bytes": len(wire), "packed_value_bytes": packed_total,
        "block_metadata_bytes": 5 * blocks, "header_bytes": QHEADER.size,
    }


def verify(root):
    checks, snapshots, arms, configs = [], {}, {}, {}

    def check(name, okay, details=None):
        checks.append({"check": name, "pass": bool(okay), "details": details})

    def read(path):
        b = path.read_bytes()
        snapshots[str(path.relative_to(root))] = {"bytes": len(b), "sha256": sha(b)}
        return b

    for method in METHODS:
        folder = root / method
        required = [folder / name for name in ("config.json", "result.json", "checkpoint.pt")]
        if not all(p.exists() for p in required):
            arms[method] = {"status": "NOT_STARTED_OR_NO_COMPLETE_SNAPSHOT", "rounds": 0}
            continue
        try:
            conf = json.loads(read(required[0]))
            result = json.loads(read(required[1]))
            ck = torch.load(io.BytesIO(read(required[2])), map_location="cpu", weights_only=False)
        except Exception as exc:
            arms[method] = {"status": "SNAPSHOT_READ_ERROR", "error": repr(exc)}
            continue
        configs[method] = conf
        # JSON converts integer map keys (codec width counts) to strings.
        # Normalize the checkpoint's native-Python diagnostics the same way.
        rows = result["rows"]
        ckrows = json.loads(json.dumps(ck["rows"]))
        controlled, dense = method.startswith("head_control"), method.endswith("_dense")
        prefix = method + ": "
        man = conf["manifest"]
        nclients = len(man["client_sizes"])
        expected_steps = [math.ceil(n / 64) for n in man["client_sizes"]]
        weights = np.asarray(man["client_sizes"], dtype=np.float64) / sum(man["client_sizes"])
        conf_without_sig = {k: v for k, v in conf.items() if k != "signature"}
        signature = sha(json.dumps(conf_without_sig, sort_keys=True).encode())
        check(prefix + "config/result/checkpoint signature", signature == conf["signature"] == result["signature"] == ck["signature"])
        check(prefix + "frozen source/plan/codec identities", (conf["source_sha256"], conf["plan_sha256"], conf["codec_sha256"]) == (EXPECTED_SOURCE, EXPECTED_PLAN, EXPECTED_CODEC))
        check(prefix + "fixed setup", conf["method"] == method and conf["seed"] == 20261191 and conf["rounds"] == ROUNDS and conf["backend"] == "mps" and conf["head"] == HEAD and conf["body"] == BODY and conf["batch"] == 64 and conf["local_epochs"] == 1 and conf["lr"] == LR)
        check(prefix + "sample weights", len(conf["weights"]) == nclients and np.allclose(conf["weights"], weights, rtol=0, atol=1e-15))
        check(prefix + "official test excluded in metadata", conf["official_test_loaded"] is False and result["official_test_loaded"] is False and man["test_used_for_selection"] is False and man["official_split"] == "CIFAR-10 train (50,000 original rows)")
        check(prefix + "round identifiers/status", [r["round"] for r in rows] == list(range(1, len(rows) + 1)) and len(rows) <= ROUNDS and result["status"] == ("DONE" if len(rows) == ROUNDS else "RUNNING"))
        common = min(len(rows), len(ckrows))
        check(prefix + "checkpoint/result common row metrics", ckrows[:common] == rows[:common], {"result_rounds": len(rows), "checkpoint_rounds": len(ckrows)})
        if result["status"] == "DONE":
            check(prefix + "final checkpoint/result complete equality", ckrows == rows)
        check(prefix + "finite saved values", finite_tree(rows) and finite_tree(ck))
        trgroups, hogroups = man["train_indices"], man["holdout_indices"]
        trainids = [i for group in trgroups for i in group]
        holdoutids = [i for group in hogroups for i in group]
        devids = man["validation_indices"]
        check(prefix + "37500/12500/2000 disjoint development split", len(trainids) == len(set(trainids)) == 37500 and len(holdoutids) == len(set(holdoutids)) == 12500 and set(trainids).isdisjoint(holdoutids) and set(trainids + holdoutids) == set(range(50000)) and len(devids) == len(set(devids)) == 2000 and set(devids) <= set(holdoutids))
        check(prefix + "manifest ID hashes", grouped_hash(trgroups) == man["train_indices_sha256"] and grouped_hash(hogroups) == man["holdout_indices_sha256"] and sha(np.asarray(devids, dtype="<u4").tobytes()) == man["validation_ids_sha256"])
        check(prefix + "manifest sizes/class counts", [len(x) for x in trgroups] == man["client_sizes"] and [len(x) for x in hogroups] == man["client_holdout_sizes"] and [sum(x) for x in man["client_label_counts"]] == man["client_sizes"] and man["validation_class_counts"] == [200] * 10)
        expected_down = nclients * (64 + 4 * PARAMS + (16 + 4 * HEAD if controlled else 0))
        max_formula = max_invariant = max_ef = 0.
        upload_total = gradient_total = 0
        upload_sizes = []
        for row in rows:
            clients, val = row["clients"], row["validation"]
            rp = prefix + "round " + str(row["round"]) + ": "
            check(rp + "actual steps and gradient evaluations", len(clients) == nclients and [c["client"] for c in clients] == list(range(nclients)) and [c["steps"] for c in clients] == expected_steps and [c["gradient_evaluations"] for c in clients] == expected_steps)
            check(rp + "upload sum and analytical download", row["upload_bytes"] == sum(c["upload_bytes"] for c in clients) and row["download_bytes_analytical"] == expected_down)
            check(rp + "validation raw metric consistency", val["samples"] == 2000 and len(val["class_accuracy"]) == 10 and abs(np.mean(val["class_accuracy"]) - val["accuracy"]) < 3e-8 and abs(np.mean(val["class_accuracy"]) - val["macro_accuracy"]) < 1e-14 and min(val["class_accuracy"]) == val["worst_class_accuracy"] and val["loss"] >= 0 and 0 <= val["accuracy"] <= 1)
            max_invariant = max(max_invariant, row["control_invariant_error"])
            for client in clients:
                max_formula = max(max_formula, client["control_formula_max_error"])
                max_ef = max(max_ef, client["body_ef_max_error"])
                nbytes, diag = client["upload_bytes"], client["codec"]
                upload_sizes.append(nbytes)
                cp = rp + "client " + str(client["client"]) + ": "
                if dense:
                    check(cp + "dense analytical upload", nbytes == 64 + 4 * PARAMS and diag == {} and client["body_ef_max_error"] == 0)
                else:
                    n = PARAMS if method == "plain_uniform" else BODY
                    overhead = 0 if method == "plain_uniform" else WRAPPER.size + 4 * HEAD
                    widths = diag["bitwidths"]
                    blocks = (n + 255) // 256
                    packed = sum((min(256, n - i * 256) * width + 7) // 8 for i, width in enumerate(widths))
                    expected_wire = QHEADER.size + 5 * blocks + packed
                    counts = {str(k): widths.count(k) for k in (2, 3, 4)}
                    check(cp + "serialized codec byte accounting/cap", nbytes <= CAP and conf["upload_cap_bytes"] == CAP and diag["numel"] == n and diag["nblocks"] == blocks and len(widths) == blocks and all(w in (2, 3, 4) for w in widths) and diag["wire_bytes"] == expected_wire and nbytes == expected_wire + overhead and diag["budget_bytes"] == CAP - overhead and diag["header_bytes"] == 64 and diag["block_metadata_bytes"] == 5 * blocks and diag["packed_value_bytes"] == packed and diag["unused_budget_bytes"] == CAP - overhead - expected_wire and diag["bitwidth_counts"] == counts)
                    check(cp + "frozen codec mode", diag["mode"] == ("uniform3" if method == "plain_uniform" else "mixed_mse") and not diag["weighted_allocation"] and (method != "plain_uniform" or set(widths) == {3}))
                gradient_total += client["gradient_evaluations"]
            upload_total += row["upload_bytes"]
        check(prefix + "recorded control/EF bounds", 0 <= max_formula <= 5e-6 and 0 <= max_invariant <= 5e-6 and 0 <= max_ef <= 5e-6)
        model = ck["model"]
        check(prefix + "checkpoint parameter layout", sum(x.numel() for x in model.values()) == PARAMS and list(model)[-2:] == ["classifier.3.weight", "classifier.3.bias"] and sum(model[k].numel() for k in list(model)[-2:]) == HEAD)
        expected_residual = PARAMS if method == "plain_uniform" else BODY
        check(prefix + "checkpoint EF layout", len(ck["residuals"]) == nclients and all(x.numel() == expected_residual for x in ck["residuals"]) and (not dense or all(torch.count_nonzero(x) == 0 for x in ck["residuals"])))
        control_error = None
        if controlled:
            weighted_c = sum((float(w) * ci for w, ci in zip(weights, ck["cis"])), torch.zeros(HEAD))
            control_error = float((weighted_c - ck["c"]).abs().max())
            check(prefix + "checkpoint final weighted control", len(ck["cis"]) == nclients and all(ci.numel() == HEAD for ci in ck["cis"]) and ck["c"].numel() == HEAD and control_error <= 5e-6 and abs(float(ck["c"].norm()) - ckrows[-1]["control_norm"]) <= 5e-6 and conf["control_client_state_bytes"] == 2600)
        else:
            check(prefix + "uncontrolled checkpoint controls zero", torch.count_nonzero(ck["c"]) == 0 and all(torch.count_nonzero(ci) == 0 for ci in ck["cis"]) and all(row["control_norm"] == row["control_invariant_error"] == 0 for row in rows) and conf["control_client_state_bytes"] == 0)
        payload_info = None
        if not dense and rows:
            try:
                wire = read(folder / "first_payload.bin")
                wrapper = None
                if method == "plain_uniform":
                    decoded, desc = independent_decode(wire)
                else:
                    magic, version, p, h, steps, bodylen, reserved = WRAPPER.unpack_from(wire)
                    assert (magic, version, p, h, steps, reserved) == (b"HEADQ001", 1, BODY, HEAD, expected_steps[0], 0)
                    assert len(wire) == WRAPPER.size + bodylen + 4 * HEAD <= CAP
                    qb, desc = independent_decode(wire[WRAPPER.size:WRAPPER.size + bodylen])
                    qh = np.frombuffer(wire, dtype="<f4", offset=WRAPPER.size + bodylen, count=HEAD).copy()
                    assert qb.size == BODY and np.isfinite(qh).all()
                    decoded = np.concatenate((qb, qh))
                    wrapper = {"bytes": WRAPPER.size, "body_payload_bytes": bodylen, "head_bytes": 4 * HEAD, "steps": steps}
                    assert abs(float(torch.from_numpy(qh).norm()) - rows[0]["clients"][0]["head_delta_norm"]) <= 1e-6
                diag0 = rows[0]["clients"][0]["codec"]
                assert len(wire) == rows[0]["clients"][0]["upload_bytes"] and decoded.size == PARAMS
                for key in ("numel", "nblocks", "wire_bytes", "header_bytes", "block_metadata_bytes", "packed_value_bytes", "bitwidths", "scales"):
                    assert desc[key] == diag0[key], key
                payload_info = {"wire_bytes": len(wire), "wrapper": wrapper, "decoded_numel": int(decoded.size), "decoded_fp32_sha256": sha(decoded.astype("<f4").tobytes()), "decoded_l2_norm": float(np.linalg.norm(decoded.astype(np.float64))), "codec": {k: v for k, v in desc.items() if k not in ("bitwidths", "scales")}}
                check(prefix + "independent first-payload parsing", True)
            except Exception as exc:
                check(prefix + "independent first-payload parsing", False, repr(exc))
        arms[method] = {
            "status": result["status"], "rounds": len(rows), "checkpoint_rounds": len(ckrows),
            "steps_per_client_per_round": expected_steps, "gradient_evaluations": gradient_total,
            "upload_bytes_total": upload_total, "upload_bytes_per_client_range": [min(upload_sizes), max(upload_sizes)] if upload_sizes else None,
            "download_bytes_analytical_total": sum(row["download_bytes_analytical"] for row in rows),
            "total_communication_bytes": upload_total + sum(row["download_bytes_analytical"] for row in rows),
            "upload_bytes_basis": "analytical FP32 model delta plus 64B" if dense else "actual serialized wire bytes",
            "download_bytes_basis": "analytical dense model plus 2616B control if controlled",
            "download_bytes_per_client_per_round_analytical": expected_down // nclients,
            "extra_control_download_bytes_per_client_per_round_analytical": 2616 if controlled else 0,
            "control_formula_max_error_recorded": max_formula, "control_invariant_max_error_recorded": max_invariant,
            "body_ef_max_error_recorded": max_ef, "checkpoint_weighted_control_max_error": control_error,
            "first_payload": payload_info,
        }
        if result["status"] == "DONE" and len(rows) == ROUNDS:
            last = rows[15:20]
            arms[method]["last5"] = {"rounds": [r["round"] for r in last], "ce": float(np.mean([r["validation"]["loss"] for r in last])), "accuracy": float(np.mean([r["validation"]["accuracy"] for r in last]))}
    if configs:
        base = next(iter(configs.values()))
        for method, conf in configs.items():
            check(method + ": shared setup/partition", all(conf[k] == base[k] for k in ("seed", "rounds", "backend", "source_sha256", "plan_sha256", "codec_sha256", "runtime", "manifest", "body", "head", "lr", "batch", "local_epochs", "weights", "evaluation", "official_test_loaded")))
    complete = all(arms[m].get("status") == "DONE" and arms[m].get("rounds") == ROUNDS and arms[m].get("checkpoint_rounds") == ROUNDS for m in METHODS)
    failures = [c for c in checks if not c["pass"]]
    decision = {"status": "RUNNING", "go": None, "rule": "head_control last5 CE < min(plain_uniform, exact_head); accuracy >= each baseline + 0.005; all four arms DONE and artifact checks pass"}
    if complete:
        cand = arms["head_control"]["last5"]
        comparisons = {m: {"ce_reduction": arms[m]["last5"]["ce"] - cand["ce"], "accuracy_gain": cand["accuracy"] - arms[m]["last5"]["accuracy"], "ce_lower": cand["ce"] < arms[m]["last5"]["ce"], "accuracy_at_least_plus_0_005": cand["accuracy"] >= arms[m]["last5"]["accuracy"] + .005} for m in ("plain_uniform", "exact_head")}
        metric_go = all(x["ce_lower"] and x["accuracy_at_least_plus_0_005"] for x in comparisons.values())
        decision.update(status="GO_FOR_INDEPENDENT_CONFIRMATION" if metric_go and not failures else "STOP_THIS_FIXED_HEAD_CONTROL_RECIPE" if not metric_go else "ARTIFACT_CHECK_FAILED", go=metric_go and not failures, metric_rule_pass=metric_go, comparisons=comparisons)
        decision["dense_diagnostic"] = {"compressed_minus_dense_ce": cand["ce"] - arms["head_control_dense"]["last5"]["ce"], "compressed_minus_dense_accuracy": cand["accuracy"] - arms["head_control_dense"]["last5"]["accuracy"]}
    return {"scope": "independent local saved-artifact check; no training or dataset evaluation", "review": "REVIEW_UNAVAILABLE", "input_root": str(root.resolve()), "experiment_status": "ALL_DONE" if complete else "RUNNING", "all_four_done": complete, "artifact_check": "PASS" if not failures else "FAIL", "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures, "decision": decision, "arms": arms, "snapshots": snapshots, "checks": checks, "separately_read_static_source_notes": STATIC_NOTES, "limitations": ["Single development seed/partition after repeated development choices; rounds are not independent seeds.", "Training is MPS according to saved metadata; this checker only runs CPU saved-file arithmetic, not an independent MPS trajectory reproduction.", "Checkpoint/result equality checks saved metrics; predictions/CE are not reevaluated because no dataset is read.", "Only first saved quantized payload per arm is independently parsed; historical payloads and local raw gradients are not saved, so EF/control per-step bounds are recorded diagnostics, not independently replayed.", "Dense upload and every download are analytical byte counts; no network, energy, blockchain or distributed deployment is tested.", "Official TEST is not read by this checker. Exclusion in the experiment is supported by saved metadata and separately reviewed static source, not filesystem access tracing.", "No external cross-model reviewer is available: REVIEW_UNAVAILABLE, not a formal external integrity acquittal."]}


def markdown(out):
    d = out["decision"]
    text = ["# Head control 本地独立结果核验", "", f"实验状态：**{out['experiment_status']}**；冻结判据决定：**{d['status']}**。保存产物核验：{out['artifact_check']}（{out['checks_passed']}/{out['checks_total']}）。外部跨模型审查：`REVIEW_UNAVAILABLE`。", "", "输入仅为 config/result/checkpoint 与首个保存 payload；未导入训练脚本或原 codec，未训练、未读取数据集或 official TEST。", "", "| 方法 | 状态/轮数 | last5 CE | last5 accuracy | 累计梯度次数 | 累计实际/解析上行B | 解析下行B | 总通信B |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        a = out["arms"][method]
        l = a.get("last5", {})
        text.append(f"| {method} | {a['status']} / {a.get('rounds', '?')} | {l.get('ce', '—')} | {100*l['accuracy'] if l else '—'}% | {a.get('gradient_evaluations', '—')} | {a.get('upload_bytes_total', '—')} | {a.get('download_bytes_analytical_total', '—')} | {a.get('total_communication_bytes', '—')} |")
    text += ["", "固定判据：round16–20 均值，head_control CE 严格低于 plain_uniform 与 exact_head，accuracy 分别至少提高 0.005（0.5pp）；四组全部 DONE 后才判断。"]
    if out["all_four_done"]:
        for m, x in d["comparisons"].items():
            text.append(f"- 相对 {m}：CE降低 {x['ce_reduction']:.10f}；accuracy 增益 {100*x['accuracy_gain']:.6f}pp；CE条件={x['ce_lower']}，accuracy条件={x['accuracy_at_least_plus_0_005']}。")
        text.append(f"- dense仅诊断：压缩组减dense CE={d['dense_diagnostic']['compressed_minus_dense_ce']:.10f}，accuracy差={100*d['dense_diagnostic']['compressed_minus_dense_accuracy']:.6f}pp。")
    else:
        text += ["", "**RUNNING：不报告 GO/STOP，不用中途轮次或已完成组替代四组固定判据。**"]
    for method, a in out["arms"].items():
        if "gradient_evaluations" not in a:
            continue
        text += ["", f"{method}：实际 batch 步数 {a['steps_per_client_per_round']}；逐 client gradient_evaluations 与 steps 一致。上行范围 {a['upload_bytes_per_client_range']}B/client/round；解析下行 {a['download_bytes_per_client_per_round_analytical']}B/client/round，其中控制额外 {a['extra_control_download_bytes_per_client_per_round_analytical']}B=16+650×4。", f"保存诊断最大 control公式误差={a['control_formula_max_error_recorded']:.3g}，加权control invariant误差={a['control_invariant_max_error_recorded']:.3g}，body EF误差={a['body_ef_max_error_recorded']:.3g}；最终checkpoint独立加权control误差={a['checkpoint_weighted_control_max_error']}。"]
        p = a["first_payload"]
        if p:
            text.append(f"首包独立解码：{p['wire_bytes']}B，{p['decoded_numel']}个有限FP32坐标；wrapper={p['wrapper']}；保存长度/位宽/尺度/头delta范数与首轮metadata一致。")
    text += ["", "核验包括签名、冻结身份、共享配置/分区、ID哈希和互斥、样本权重、逐轮步数/梯度数、codec字节重算、完整checkpoint/result行及指标一致、控制状态及首包独立解析。", "", "单独完成的只读静态核对（下列源文件位置并非该核验脚本的运行输入）："]
    text.extend("- " + x for x in out["separately_read_static_source_notes"])
    text += ["", "证据边界："]
    text.extend("- " + x for x in out["limitations"])
    if out["failures"]:
        text += ["", "失败项："] + [f"- {x['check']}: {x['details']}" for x in out["failures"]]
    return "\n".join(text) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "development_s20261191")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = verify(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "LOCAL_STATS_CHECK.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "LOCAL_RESULTS_CHECK.md").write_text(markdown(out))
    brief = {k: out[k] for k in ("all_four_done", "artifact_check", "checks_passed", "checks_total", "decision")}
    brief["failure_count"] = len(out["failures"])
    brief["first_failures"] = out["failures"][:5]
    print(json.dumps(brief, ensure_ascii=False))


if __name__ == "__main__":
    main()
