"""One fixed CPU development A/B/C probe; no test or validation gradients.

A: saved-delta TopK-EF. B: one extra SGD gradient at w+delta, then TopK-EF.
C: one gradient at w+TopK(delta+r), clipped repair on that same support.
This finite probe is exploratory and cannot establish a trained algorithm.
"""
from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import TensorDataset

import run_followup as runner
from guarded_codec import canonical_pairs, encode


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ALPHA = .03
THETA = .5
BATCH_SIZE = 64
BATCH_SEED_DOMAIN = "sparse_repair_v1_uniform64_fixed_no_search"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(x):
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def batch_seed(source_seed, client):
    raw = f"{BATCH_SEED_DOMAIN}:{source_seed}:{client}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def gradient_at(model, point, batch):
    """One mean-CE gradient on a private model; shared model stays unchanged."""
    before = runner.vectorize(model).clone()
    local = copy.deepcopy(model)
    runner.assign_vector(local, point)
    local.eval()  # frozen SmallCNN has neither BatchNorm nor dropout
    x, y = batch
    loss = nn.functional.cross_entropy(local(x), y, reduction="mean")
    grads = torch.autograd.grad(loss, tuple(local.parameters()))
    gradient = torch.cat([g.detach().flatten() for g in grads])
    if gradient.dtype != torch.float32 or not bool(torch.isfinite(gradient).all()):
        raise ValueError("gradient must be finite FP32")
    if not torch.equal(before, runner.vectorize(model)):
        raise AssertionError("private gradient calculation changed shared model")
    return gradient, {"minibatch_ce": float(loss.detach()), "samples": len(y),
                      "gradient_evaluations": 1, "gradient_norm": float(gradient.double().norm())}


def check_conservation(u, sent, residual):
    eps = torch.finfo(torch.float32).eps
    scale = max(float(u.abs().max()), torch.finfo(torch.float32).tiny)
    error = float((sent + residual - u).abs().max())
    tolerance = 16 * eps * scale
    if error > tolerance:
        raise AssertionError(f"FP32 conservation failed: {error} > {tolerance}")
    return {"conservation_max_absolute_error": error, "conservation_absolute_tolerance": tolerance}


def repaired(u, baseline, gradient, *, alpha=ALPHA, theta=THETA):
    s0, indices = baseline.sent, baseline.indices
    z = torch.zeros_like(u)
    z[indices] = -alpha * gradient[indices]
    original_norm = float(z.double().norm())
    energy = float(s0.double().square().sum())
    radius = math.sqrt(max(0., (1 - theta) * energy))
    clipped = original_norm > radius
    if clipped:
        z.mul_(radius / original_norm)
    sent = s0 + z
    residual = u - sent
    conservation = check_conservation(u, sent, residual)
    u_energy = float(u.double().square().sum())
    residual_energy = float(residual.double().square().sum())
    k, n = indices.numel(), u.numel()
    weak_bound = (1 - theta * k / n) * u_energy
    exact_budget_bound = u_energy - theta * energy
    tolerance = 64 * torch.finfo(torch.float32).eps * max(u_energy, torch.finfo(torch.float32).tiny)
    if residual_energy > exact_budget_bound + tolerance or residual_energy > weak_bound + tolerance:
        raise AssertionError("FP32 weak contraction tolerance exceeded")
    support_mask = torch.zeros(n, dtype=torch.bool)
    support_mask[indices] = True
    if bool((sent[~support_mask] != 0).any()) or bool((z[~support_mask] != 0).any()):
        raise AssertionError("repair changed support")
    return sent, residual, z, {**conservation, "repair_unclipped_norm": original_norm,
        "repair_norm": float(z.double().norm()), "repair_radius": radius, "repair_clipped": clipped,
        "u_energy": u_energy, "topk_retained_energy": energy,
        "new_residual_energy": residual_energy, "weak_contraction_bound": weak_bound,
        "exact_budget_bound": exact_budget_bound, "contraction_tolerance": tolerance}


def packet_bytes(indices, sent):
    # Serialize the actual pairs; explicit zero-valued selected pairs remain.
    return 64 + len(canonical_pairs(indices, sent[indices], sent.numel()))


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(73049)
    model = nn.Linear(4, 3)
    base = runner.vectorize(model).clone()
    delta = torch.randn_like(base) * .1
    old_residual = torch.randn_like(base) * .3
    batch = (torch.randn(7, 4), torch.tensor([0, 1, 2, 1, 0, 2, 1]))
    u = delta + old_residual
    a = encode(delta, old_residual, fraction=.4)
    gb, mb = gradient_at(model, base + delta, batch)
    gc, mc = gradient_at(model, base + a.sent, batch)
    b0 = encode(delta - 0. * gb, old_residual, fraction=.4)
    c0, r0, z0, _ = repaired(u, a, gc, alpha=0.)
    assert torch.equal(a.sent, b0.sent) and torch.equal(a.residual, b0.residual)
    assert torch.equal(a.sent, c0) and torch.equal(a.residual, r0) and not bool(z0.any())
    assert mb["samples"] == mc["samples"] and mb["gradient_evaluations"] == mc["gradient_evaluations"] == 1
    assert torch.equal(base, runner.vectorize(model))
    cases = 0
    for scale in (0., 1e-7, .01, 1., 100.):
        for fraction in (.05, .4, 1.):
            d, r = torch.randn_like(base) * scale, torch.randn_like(base) * scale
            current_u = d + r
            a = encode(d, r, fraction=fraction)
            sent, residual, z, diag = repaired(current_u, a, gc * 1e4)
            assert packet_bytes(a.indices, sent) == 64 + 8 * a.indices.numel()
            assert a.indices.numel() == max(1, int(base.numel() * fraction))
            assert bool(torch.isfinite(residual).all()) and bool(torch.isfinite(z).all())
            assert diag["new_residual_energy"] <= diag["weak_contraction_bound"] + diag["contraction_tolerance"]
            cases += 1
    return {"status": "PASS", "cases": cases, "alpha_zero_B_and_C_equal_A": True,
        "K_and_actual_serialized_pair_bytes": True, "FP32_conservation_and_contraction_tolerance": True,
        "shared_model_unchanged": True, "B_C_same_batch_and_one_gradient_each": True,
        "no_validation_or_test_inputs": True}


def locks(dataset, helper):
    paths = [WORKSPACE / "outputs/method_followup/checks/source_and_dataset_lock.json",
             WORKSPACE / "outputs/method_followup/checks/dataset_membership_current.json"]
    source_lock, membership = [json.loads(p.read_text()) for p in paths]
    if not membership["all_shards_multiset_equals_official_train"] or membership["content_overlap_train_holdout_unique_hashes"] != 0:
        raise ValueError("TRAIN/holdout membership not clean")
    expected = {(WORKSPACE / x["path"]).resolve(): x["sha256"] for x in source_lock["files"]}
    root = runner.PROJECT / "dataset" / dataset
    consumed = [helper, root / "config.json"] + [root / split / f"{i}.npz"
        for split in ("train", "test") for i in range(20)]
    hashes = {}
    for path in consumed:
        path = Path(path).resolve()
        hashes[str(path)] = sha(path)
        if expected.get(path) != hashes[str(path)]:
            raise ValueError(f"consumed source/data lock drift: {path}")
    return {"records": {str(p): sha(p) for p in paths}, "consumed_files": hashes,
        "holdout_policy": "NPZ test shards are official-TRAIN local holdouts, never official test"}


def probe(source, output, fixture_report):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source != HERE / "checkpoint_probes/cpu_topk20":
        raise ValueError("this finite probe accepts only the explicit cpu_topk20 development state")
    if output.exists():
        raise FileExistsError("new output required; no resume or overwrite")
    torch.set_num_threads(1)
    start = time.perf_counter()
    original = json.loads((source / "result.json").read_text())
    run = Path(original["source_run"])
    if run.parent.resolve() != HERE / "development_runs":
        raise ValueError("development source required")
    config = json.loads((run / "config.json").read_text())
    if config["test_final"] is not False or original["official_test_evaluated"] is not False:
        raise ValueError("explicitly test-free source required")
    if config["variant"] != "topk_ef" or config["dataset"] != "Cifar10CleanA0p1" or config.get("training_backend", "cpu") != "cpu":
        raise ValueError("unexpected source setting")
    for key, path in (("source_config_sha256", run/"config.json"),
                      ("source_checkpoint_sha256", run/"checkpoint.pt"),
                      ("shared_state_sha256", source/"shared_state.pt")):
        if original[key] != sha(path):
            raise ValueError(f"source identity drift: {key}")
    for key, name in (("runner_source_sha256", "run_followup.py"), ("codec_source_sha256", "guarded_codec.py")):
        if config[key] != sha(HERE/name):
            raise ValueError(f"frozen source drift: {name}")
    helper = Path(inspect.getsourcefile(runner.SmallCNN)).resolve()
    data_locks = locks(config["dataset"], helper)
    trainsets, validation, manifest = runner.prepare(config["dataset"], config["mixing"], config["split_seed"], config["validation_size"])
    if manifest != config["data_manifest"] or manifest != original["data_manifest"]:
        raise ValueError("source data manifest drift")
    state = torch.load(source/"shared_state.pt", map_location="cpu", weights_only=False)
    base = state["base"]
    weights = [len(ds)/sum(map(len, trainsets)) for ds in trainsets]
    if weights != state["weights"] or len(trainsets) != len(state["deltas"]) or len(trainsets) != len(state["residuals"]):
        raise ValueError("source clients/weights mismatch")
    model = runner.SmallCNN()
    runner.assign_vector(model, base)
    payload = {"base": base, "weights": weights, "old_residuals": state["residuals"],
        "batch_indices": [], "A": {"deltas": [], "residuals": [], "indices": [], "sent_values": []},
        "B": {"deltas": [], "residuals": [], "indices": [], "sent_values": []},
        "C": {"deltas": [], "residuals": [], "indices": [], "sent_values": [], "repair_values": []}}
    aggregate = {name: torch.zeros_like(base) for name in ("A", "B", "C")}
    clients = []
    for i, (dataset, delta, old_r, weight) in enumerate(zip(trainsets, state["deltas"], state["residuals"], weights)):
        seed = batch_seed(config["seed"], i)
        ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:BATCH_SIZE]
        if len(ids) != BATCH_SIZE:
            raise ValueError("each source client must supply 64 uniform distinct TRAIN examples")
        batch = tuple(x[ids] for x in dataset.tensors)
        payload["batch_indices"].append(ids)
        a = encode(delta, old_r, fraction=config["fraction"])
        gb, info_b = gradient_at(model, base + delta, batch)
        gc, info_c = gradient_at(model, base + a.sent, batch)
        # Delta-coordinate SGD has exactly the same mathematical one-step update.
        # It avoids subtracting the base twice and is exactly delta at alpha=0.
        delta_b = delta - ALPHA * gb
        b = encode(delta_b, old_r, fraction=config["fraction"])
        c_sent, c_residual, repair, repair_info = repaired(delta + old_r, a, gc)
        if info_b["samples"] != info_c["samples"] or info_b["gradient_evaluations"] != info_c["gradient_evaluations"]:
            raise AssertionError("B/C compute not matched")
        indices = {"A": a.indices, "B": b.indices, "C": a.indices}
        sent = {"A": a.sent, "B": b.sent, "C": c_sent}
        remainders = {"A": a.residual, "B": b.residual, "C": c_residual}
        deltas = {"A": delta, "B": delta_b, "C": delta}
        row = {"client": i, "samples": len(dataset), "weight": weight, "batch_seed": seed,
            "batch_indices_sha256": tensor_sha(ids), "batch_x_sha256": tensor_sha(batch[0]), "batch_y_sha256": tensor_sha(batch[1]),
            "B_gradient": info_b, "C_gradient": info_c, "C_repair": repair_info,
            "B_increment_sgd_vs_literal_parameter_sgd_norm": float(((base+delta_b).double()-((base+delta)-ALPHA*gb).double()).norm()),
            "variants": {}}
        for name in ("A", "B", "C"):
            current_u = deltas[name] + old_r
            conservation = check_conservation(current_u, sent[name], remainders[name])
            wire = packet_bytes(indices[name], sent[name])
            if indices[name].numel() != a.indices.numel() or wire != 64 + 8*a.indices.numel():
                raise AssertionError("K or actual pair bytes changed")
            aggregate[name].add_(weight * sent[name])
            for key, value in (("deltas", deltas[name]), ("residuals", remainders[name]), ("indices", indices[name]), ("sent_values", sent[name][indices[name]])):
                payload[name][key].append(value)
            row["variants"][name] = {"K": indices[name].numel(), "analytical_and_serialized_bytes": wire,
                "indices_sha256": tensor_sha(indices[name]), "delta_norm": float(deltas[name].double().norm()),
                "new_residual_norm": float(remainders[name].double().norm()), **conservation}
        payload["C"]["repair_values"].append(repair[a.indices])
        clients.append(row)
    if not torch.equal(runner.vectorize(model), base):
        raise AssertionError("shared model modified during private gradients")
    rows = []
    for name in ("A", "B", "C"):
        runner.assign_vector(model, base + aggregate[name])
        metric = runner.metrics(model, validation)
        rows.append({"variant": name, "validation": metric, "aggregate_norm": float(aggregate[name].double().norm()),
            "round_bytes": sum(x["variants"][name]["analytical_and_serialized_bytes"] for x in clients),
            "weighted_new_residual_energy": sum(w*float(v.double().square().sum()) for w,v in zip(weights,payload[name]["residuals"]))})
    a, b, c = rows
    for row in rows:
        row["delta_loss_vs_A"] = row["validation"]["loss"] - a["validation"]["loss"]
        row["delta_accuracy_pp_vs_A"] = 100*(row["validation"]["accuracy"]-a["validation"]["accuracy"])
    go = c["validation"]["loss"] < a["validation"]["loss"] and c["validation"]["loss"] < b["validation"]["loss"] and c["validation"]["accuracy"] >= a["validation"]["accuracy"]
    output.mkdir(parents=True)
    saved = output/"client_states.pt"
    torch.save(payload, saved)
    inputs = [source/"result.json", source/"shared_state.pt", run/"config.json", run/"checkpoint.pt"]
    deps = [Path(__file__).resolve(), HERE/"run_followup.py", HERE/"guarded_codec.py", helper]
    result = {"status": "DONE", "scope": "one fixed saved-development-state A/B/C diagnostic, no closed-loop training or independent repeats",
        "source_probe": str(source), "source_run": str(run), "source_seed": config["seed"], "source_round": original["source_round"],
        "backend": "cpu", "threads": 1, "alpha": ALPHA, "theta": THETA, "minibatch_size": BATCH_SIZE,
        "batch_seed_domain": BATCH_SEED_DOMAIN, "batch_sampling": "uniform without replacement, same indices/tensors for B/C per client",
        "B_numerics": "one SGD gradient at FP32 base+delta; delta_B=delta-alpha*gradient; alpha=0 preserves saved delta exactly; parameter-level rounding discrepancy logged",
        "validation_gradient_used": False, "official_test_accessed": False, "confirmation_results_accessed": False,
        "fraction": config["fraction"], "parameter_count": base.numel(), "data_manifest": manifest, "data_lock_binding": data_locks,
        "input_sha256": {str(x):sha(x) for x in inputs}, "executed_dependency_sha256": {str(x):sha(x) for x in deps},
        "client_states_sha256": sha(saved), "fixture": fixture_report, "clients": clients, "rows": rows,
        "decision_rule": "GO only if C validation CE is strictly below both A and B and C accuracy is at least A; otherwise STOP",
        "decision": "GO_TO_HUMAN_REVIEW_OF_ONE_SMALL_INDEPENDENT_CHECK" if go else "STOP_NO_TUNING_OR_EXPANSION",
        "limitations": ["single exploratory state; no independent training seeds", "client shadow model is not weighted global aggregate",
            "repair increases immediate residual energy; opposite repair is entered into EF debt and may be undone later",
            "contractivity/conservation do not guarantee learning benefit or novelty", "no parameter/batch search or automatic follow-up"],
        "runtime": {"python":sys.version,"executable":sys.executable,"torch":str(torch.__version__),"numpy":str(np.__version__)},
        "elapsed_seconds":time.perf_counter()-start}
    runner.atomic_json(output/"result.json",result)
    lines=["# 一次固定 A/B/C 支持内修复诊断", "", f"判据结果：**{result['decision']}**", "",
        "|组|验证 CE|accuracy|CE 相对 A|accuracy 相对 A，pp|每轮 bytes|", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        v=row["validation"]
        lines.append(f"|{row['variant']}|{v['loss']:.9f}|{v['accuracy']:.9f}|{row['delta_loss_vs_A']:+.9f}|{row['delta_accuracy_pp_vs_A']:+.6f}|{row['round_bytes']}|")
    lines += ["", "A=保存的原 TopK-EF；B=同 batch 额外一步普通 local SGD 后 TopK-EF；C=压缩后原支持内一次受限修复。",
        "alpha=.03，theta=.5，64 个均匀 TRAIN 例，CPU 单线程。B/C 每客户端恰好一次相同 batch 的梯度。",
        "", "仅一个相关开发反事实。未用 validation/test 梯度或读取确认结果。", "修复的相反量进入 EF 残差，后续可能撤销；本地 shadow 模型也不是真实全局聚合模型。",
        "该输出不自动启动训练或参数搜索。"]
    (output/"summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"DONE","decision":result["decision"],"rows":rows,"elapsed_seconds":result["elapsed_seconds"]},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-only",action="store_true")
    parser.add_argument("--source",type=Path,default=HERE/"checkpoint_probes/cpu_topk20")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.fixture_only:
        print(json.dumps(fixture(),indent=2))
        return
    if args.output is None:
        parser.error("--output is required for the one real diagnostic")
    fixture_report=fixture()
    print(json.dumps({"fixture":fixture_report}),flush=True)
    probe(args.source,args.output,fixture_report)


if __name__=="__main__":
    main()
