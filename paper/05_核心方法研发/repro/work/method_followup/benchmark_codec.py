"""Independent CPU codec microbenchmark; default mode is a tiny self-check.

No training, official-test data, or checkpoint discovery occurs here. Proxy
inputs are deterministic synthetic updates, not measured FL updates. A later
run may supply an explicitly captured tensor case with --tensor-file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import guarded_codec
import lagrangian_codec
import run_controls as controls


HERE = Path(__file__).resolve().parent
METHODS = ("topk_ef", "prefix90", "prefix99", "lagrange90", "lagrange99", "regtopk5")
SOURCE_NAMES = ("benchmark_codec.py", "guarded_codec.py", "lagrangian_codec.py",
                "run_controls.py", "run_followup.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def fraction_for_k(n, k):
    if not 1 <= k <= n:
        raise ValueError("require 1 <= K <= N")
    fraction = 1. if k == n else math.nextafter(k/n, math.inf)
    if max(1, min(n, int(n*fraction))) != k:
        raise AssertionError("fraction did not reproduce the explicit K")
    return fraction


@dataclass
class Case:
    delta: torch.Tensor
    residual: torch.Tensor
    reference: torch.Tensor
    previous_u: torch.Tensor
    previous_support: torch.Tensor
    weight: float
    provenance: dict


def validate_case(case, n, k):
    for name in ("delta", "residual", "reference", "previous_u"):
        value = getattr(case, name)
        if value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != (n,):
            raise ValueError(f"{name} must be an N-element CPU FP32 vector")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite {name}")
    support = case.previous_support
    if support.device.type != "cpu" or support.dtype != torch.int64 or support.shape != (k,):
        raise ValueError("previous_support must contain K CPU int64 indices")
    if support.unique().numel() != k or int(support.min()) < 0 or int(support.max()) >= n:
        raise ValueError("invalid previous support")
    if not math.isfinite(case.weight) or not 0 < case.weight <= 1:
        raise ValueError("client weight must be in (0,1]")
    return case


def proxy_case(n, k, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    previous_u = torch.randn(n, generator=generator)
    delta = .25*previous_u + torch.randn(n, generator=generator)
    residual = .1*torch.randn(n, generator=generator)
    support = torch.topk(previous_u.abs(), k=k, sorted=False).indices
    old_sent = torch.zeros(n)
    old_sent[support] = previous_u[support]
    weight = .05
    # This is a constructed correlation/cancellation setting, not a recovered
    # aggregate. Its values are identical across methods within a case.
    reference = weight*old_sent + .03*torch.randn(n, generator=generator)
    return validate_case(Case(delta, residual, reference, previous_u, support, weight,
        {"kind": "deterministic_synthetic_proxy", "seed": seed,
         "description": "correlated FP32 delta/residual and constructed previous aggregate; not training output"}), n, k)


def external_case(path, n, k):
    # Only a caller-selected tensor file is opened. A runner checkpoint alone
    # lacks the current local delta, so it is deliberately not accepted here.
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = ("delta", "residual", "reference", "previous_u", "previous_support", "client_weight")
    if not isinstance(data, dict) or any(name not in data for name in required):
        raise ValueError(f"tensor case requires {required}; a training checkpoint is not a complete codec case")
    provenance = {"kind": "caller_selected_tensor_case", "path": str(Path(path).resolve()),
                  "file_sha256": sha(path), "caller_provenance": data.get("provenance", {}),
                  "verified_real_training_origin": False}
    json.dumps(provenance)
    return validate_case(Case(*(data[name].detach() for name in required[:5]),
                              float(data["client_weight"]), provenance), n, k)


def make_invoker(case, method, k):
    fraction = fraction_for_k(case.delta.numel(), k)
    variant = "regtopk5_ef"

    def reset():
        if method == "regtopk5":
            # Restore the same real/constructed previous state before every
            # call, outside the timer. The codec's new history clone IS timed.
            controls._REG_CONTEXT = {"variant": variant, "weights": [case.weight],
                "call_count": 0, "previous": [(case.previous_u, case.previous_support)]}

    def invoke():
        if method == "topk_ef":
            return guarded_codec.encode(case.delta, case.residual, fraction=fraction,
                                        reference=case.reference, method="topk_ef")
        if method.startswith("prefix"):
            return guarded_codec.encode(case.delta, case.residual, fraction=fraction,
                reference=case.reference, method="guard_alignment_ef", tau=.9 if method == "prefix90" else .99)
        if method.startswith("lagrange"):
            return lagrangian_codec.encode(case.delta, case.residual, fraction=fraction,
                reference=case.reference, tau=.9 if method == "lagrange90" else .99,
                max_expansions=24, bisection_steps=12)
        if method == "regtopk5":
            sent, rem, diagnostics = controls.encode_variant(case.delta, case.residual,
                case.reference, fraction, variant, 0)
            selected = controls._REG_CONTEXT["previous"][0][1]
            return guarded_codec.Encoding(sent, rem, selected, diagnostics)
        raise ValueError(method)

    def retained_state():
        return controls._REG_CONTEXT["previous"][0] if method == "regtopk5" else ()

    return reset, invoke, retained_state


def check_result(case, result, k, method):
    candidate = case.delta + case.residual
    indices = result.indices
    assert indices.numel() == k and indices.unique().numel() == k
    assert torch.equal(result.sent[indices], candidate[indices])
    assert torch.allclose(result.sent+result.residual, candidate, atol=2e-7, rtol=2e-6)
    assert result.diagnostics["encoded_bytes"] == 64+8*k
    assert result.diagnostics["selected_coordinates"] == k
    if method.startswith(("prefix", "lagrange")):
        tau = .9 if method.endswith("90") else .99
        assert result.diagnostics["retained_topk_energy_ratio"] >= tau-1e-12
        assert result.diagnostics["alignment_gain"] >= -1e-10


def quantiles_ns(samples):
    ordered = sorted(samples)
    def percentile(p):
        position = (len(ordered)-1)*p
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low]+(ordered[high]-ordered[low])*(position-low)
    q1, q3 = percentile(.25), percentile(.75)
    return {"calls": len(samples), "median_ms": statistics.median(samples)/1e6,
            "q1_ms": q1/1e6, "q3_ms": q3/1e6, "iqr_ms": (q3-q1)/1e6,
            "minimum_ms": min(samples)/1e6, "maximum_ms": max(samples)/1e6}


def operation_counts(invoke):
    calls = []
    old_topk, old_sort, old_unique = torch.topk, torch.sort, torch.Tensor.unique
    def topk(x, *args, **kwargs):
        k = kwargs.get("k", args[0] if args else None)
        calls.append({"op": "torch.topk", "input_elements": x.numel(), "k": k,
                      "sorted": kwargs.get("sorted", True)})
        return old_topk(x, *args, **kwargs)
    def sort(x, *args, **kwargs):
        calls.append({"op": "torch.sort", "input_elements": x.numel()})
        return old_sort(x, *args, **kwargs)
    def unique(x, *args, **kwargs):
        calls.append({"op": "Tensor.unique", "input_elements": x.numel(),
                      "note": "unique may internally sort; opaque workspace not counted"})
        return old_unique(x, *args, **kwargs)
    torch.topk, torch.sort, torch.Tensor.unique = topk, sort, unique
    try:
        result = invoke()
    finally:
        torch.topk, torch.sort, torch.Tensor.unique = old_topk, old_sort, old_unique
    n = result.sent.numel()
    return result, {"calls": calls, "counts": dict(Counter(x["op"] for x in calls)),
        "full_length_explicit_sort_calls": sum(x["op"] == "torch.sort" and x["input_elements"] == n for x in calls),
        "scope": "one untimed call; explicit Python-visible operations; sorted Top-K sorts selected entries, not an explicit full-vector sort; topk/unique internal algorithms unspecified"}


class VisibleStoragePeak(TorchDispatchMode):
    """Track unique live CPU tensor storages visible at dispatcher boundaries.

    This is NOT RSS or a full allocator peak: C++ internal workspaces, Python
    objects, allocator caching and OS pages are not observable through this API.
    """
    def __init__(self, initial_tensors):
        super().__init__()
        self.tensors, self.storages = {}, {}
        self.live = self.peak = 0
        self.track(initial_tensors)
        self.baseline = self.live

    def release(self, identity, key):
        self.tensors.pop(identity, None)
        record = self.storages[key]
        record[1] -= 1
        if record[1] == 0:
            self.live -= record[0]
            del self.storages[key]

    def track(self, tree):
        for tensor in tree_flatten(tree)[0]:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                continue
            identity = id(tensor)
            if identity in self.tensors:
                continue
            storage = tensor.untyped_storage()
            key = storage._cdata
            size = storage.nbytes()
            if key not in self.storages:
                self.storages[key] = [size, 0]
                self.live += size
                self.peak = max(self.peak, self.live)
            self.storages[key][1] += 1
            self.tensors[identity] = weakref.finalize(tensor, self.release, identity, key)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.track((args, kwargs))
        result = func(*args, **(kwargs or {}))
        self.track(result)
        return result


def storage_bytes(tree):
    storages = {}
    for tensor in tree_flatten(tree)[0]:
        if isinstance(tensor, torch.Tensor):
            storage = tensor.untyped_storage()
            storages[storage._cdata] = storage.nbytes()
    return sum(storages.values())


def rss_highwater():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw*1024)


def memory_observation(case, method, reset, invoke, retained_state):
    inputs = [case.delta, case.residual, case.reference]
    if method == "regtopk5":
        inputs += [case.previous_u, case.previous_support]
    reset()
    tracker = VisibleStoragePeak(inputs)
    before = rss_highwater()
    with tracker:
        result = invoke()
    retained = storage_bytes((result.sent, result.residual, result.indices, retained_state()))
    after = rss_highwater()
    report = {"input_visible_storage_bytes": tracker.baseline,
        "peak_visible_live_tensor_storage_bytes": tracker.peak,
        "incremental_peak_visible_tensor_storage_bytes": tracker.peak-tracker.baseline,
        "retained_output_and_new_history_storage_bytes": retained,
        "process_rss_highwater_before_bytes": before, "process_rss_highwater_after_bytes": after,
        "process_rss_highwater_increase_bytes": max(0,after-before),
        "scope": "untimed instrumented call; incremental peak includes returned outputs/new history as well as temporaries; visible tensor storage is a lower bound on full working memory; RSS high-water change is not an isolated temporary peak"}
    del result
    return report


def state_ledger(n, k, method):
    reg = method == "regtopk5"
    return {"ef_residual_fp32_per_client_bytes": 4*n,
        "history_reference_fp32_bytes": 4*n,
        "reference_storage_scope": "one shared vector in synchronous simulator; per-client caching costs this amount when deployed",
        "regtopk_previous_candidate_fp32_per_client_bytes": 4*n if reg else 0,
        "regtopk_previous_support_int64_per_client_bytes": 8*k if reg else 0,
        "extra_persistent_state_over_topk_ef_per_client_bytes": 4*n+8*k if reg else 0,
        "wire_values_fp32_bytes": 4*k, "wire_indices_uint32_bytes": 4*k,
        "wire_fixed_header_bytes": 64, "wire_total_bytes": 64+8*k,
        "extra_uploaded_fields_bytes_over_same_k_topk": 0,
        "excluded_common_state": "global/local models, optimizer, training activations, source data; reconstructing g may require another cached global model",
        "note": "storage uses actual current int64 support and dense RegTopK candidate history, not minimal theoretical K pairs"}


def environment(threads):
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "requested_threads": threads, "actual_torch_threads": torch.get_num_threads(),
        "actual_torch_interop_threads": torch.get_num_interop_threads(),
        "system_load_1_5_15min": load,
        "source_sha256": {name: sha(HERE/name) for name in SOURCE_NAMES},
        "timer": "time.perf_counter_ns; CPU synchronous calls"}


def self_check():
    original_context = controls._REG_CONTEXT
    reports = []
    try:
        base = torch.zeros(11)
        tracker = VisibleStoragePeak([base])
        with tracker:
            alias = base.view(-1)
            extra = base+1
        assert tracker.baseline == 44 and tracker.peak == 88
        del extra
        assert tracker.live == 44  # aliases must not double-count storage
        del alias
        for n, k in ((31, 1), (127, 17), (31, 31)):
            case = proxy_case(n, k, 991+k)
            input_hashes = [tensor_sha(x) for x in (case.delta, case.residual, case.reference, case.previous_u, case.previous_support)]
            for method in METHODS:
                reset, invoke, retained_state = make_invoker(case, method, k)
                reset()
                result, ops = operation_counts(invoke)
                check_result(case, result, k, method)
                del result
                memory = memory_observation(case, method, reset, invoke, retained_state)
                assert memory["peak_visible_live_tensor_storage_bytes"] >= memory["input_visible_storage_bytes"]
                assert memory["retained_output_and_new_history_storage_bytes"] >= 8*n
                reports.append({"n": n, "k": k, "method": method, "explicit_ops": ops["counts"]})
            assert input_hashes == [tensor_sha(x) for x in (case.delta, case.residual, case.reference, case.previous_u, case.previous_support)]
        samples = quantiles_ns([1_000_000,2_000_000,3_000_000,4_000_000,5_000_000])
        assert samples["median_ms"] == 3 and samples["iqr_ms"] == 2
        assert state_ledger(136874,6843,"regtopk5")["extra_persistent_state_over_topk_ef_per_client_bytes"] == 602240
        return {"status": "PASS", "scope": "tiny functional checks only; no target-size timing or efficiency conclusion",
                "checks": reports, "visible_storage_alias_and_release_check": True,
                "target_sizes_not_timed": {"n":136874,"ks":[6843,27374]},
                "target_arithmetic_only_ledgers": {str(k):{method:state_ledger(136874,k,method)
                    for method in ("topk_ef","regtopk5")} for k in (6843,27374)}}
    finally:
        controls._REG_CONTEXT = original_context


def benchmark(args):
    rows = []
    original_context = controls._REG_CONTEXT
    try:
        for k in args.ks:
            for case_id in range(args.cases if args.tensor_file is None else 1):
                case = (proxy_case(args.n,k,args.seed+case_id) if args.tensor_file is None
                        else external_case(args.tensor_file,args.n,k))
                invokers = {method:make_invoker(case,method,k) for method in args.methods}
                detail, samples = {}, {method:[] for method in args.methods}
                for method,(reset,invoke,state) in invokers.items():
                    reset()
                    result,ops = operation_counts(invoke)
                    check_result(case,result,k,method)
                    diagnostics = copy.deepcopy(result.diagnostics)
                    del result
                    memory = memory_observation(case,method,reset,invoke,state)
                    for _ in range(args.warmup):
                        reset(); result=invoke(); del result
                    detail[method] = {"operations":ops,"memory_observation":memory,"diagnostics":diagnostics}
                orders=[]
                was_gc_enabled=gc.isenabled()
                gc.collect();gc.disable()
                try:
                    for block in range(args.blocks):
                        order=list(args.methods)
                        random.Random(args.seed+case_id*1009+block).shuffle(order)
                        orders.append(order)
                        for method in order:
                            reset,invoke,_=invokers[method]
                            for _ in range(args.repeats):
                                reset()
                                began=time.perf_counter_ns();result=invoke();elapsed=time.perf_counter_ns()-began
                                samples[method].append(elapsed)
                                del result
                finally:
                    if was_gc_enabled:gc.enable()
                for method in args.methods:
                    rows.append({"n":args.n,"k":k,"fraction_passed":fraction_for_k(args.n,k),
                        "case_id":case_id,"method":method,"provenance":case.provenance,
                        "input_sha256":{name:tensor_sha(getattr(case,name)) for name in
                            ("delta","residual","reference","previous_u","previous_support")},
                        "summary":quantiles_ns(samples[method]),"raw_call_ns":samples[method],
                        "block_method_orders":orders,"state_and_wire_ledger":state_ledger(args.n,k,method),
                        **detail[method]})
    finally:
        controls._REG_CONTEXT=original_context
    return {"status":"MICROBENCHMARK_COMPLETE","environment":environment(args.threads),
        "context_note":args.context_note,"configuration":{k:v for k,v in vars(args).items() if k not in ("out","tensor_file")},
        "scope":"single-process CPU codec calls including validation, diagnostics, reconstruction and RegTopK history commit; excludes local training, serialization, network, downlink, authentication and ledger",
        "interpretation":"synthetic cases are proxies; concurrent training/system load makes these descriptive local timings, not rigorous end-to-end efficiency evidence",
        "rows":rows}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("check","benchmark"),default="check")
    parser.add_argument("--n",type=int,default=136874)
    parser.add_argument("--ks",default="6843,27374")
    parser.add_argument("--methods",default=",".join(METHODS))
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--warmup",type=int,default=10)
    parser.add_argument("--repeats",type=int,default=30,help="calls per method per block")
    parser.add_argument("--blocks",type=int,default=5)
    parser.add_argument("--cases",type=int,default=3)
    parser.add_argument("--seed",type=int,default=20261107)
    parser.add_argument("--tensor-file",type=Path)
    parser.add_argument("--context-note",default="Machine idleness and concurrent training were not assessed")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args(argv)
    try:
        args.ks=[int(x) for x in args.ks.split(",")]
        args.methods=args.methods.split(",")
    except ValueError as error:parser.error(str(error))
    if args.n<1 or not args.ks or any(not 1<=k<=args.n for k in args.ks):parser.error("invalid N/K")
    if len(set(args.ks))!=len(args.ks) or len(set(args.methods))!=len(args.methods):parser.error("duplicate K/method")
    if not args.methods or set(args.methods)-set(METHODS):parser.error("unknown method")
    if args.threads<1 or args.warmup<1 or args.repeats<5 or args.blocks<1 or args.cases<1:parser.error("invalid measurement counts")
    if args.mode=="benchmark" and args.out is None:parser.error("benchmark mode requires --out")
    if args.tensor_file and not args.tensor_file.is_file():parser.error("tensor file does not exist")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    result=self_check() if args.mode=="check" else benchmark(args)
    if args.mode=="check":result["environment"]=environment(args.threads)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","checks")},indent=2))
    return result


if __name__=="__main__":
    main()
