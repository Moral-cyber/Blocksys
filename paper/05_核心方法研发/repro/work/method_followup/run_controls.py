"""Additional controls for run_followup.py without modifying that runner.

Explicit --variants is required. Example:
    python run_controls.py --variants prox001_ef,prox01_ef,sign50_ef \
        --rounds 20 --fraction .05 --seeds 20261113 --out control_runs

FedProx controls change the local objective only and use the same Top-K EF
upload. sign50_ef is a sign-guided mechanism control inspired by prior ideas,
not a complete reproduction of an existing named paper. All selected values
remain the original u=delta+residual FP32 values with exactly K uint32 indices.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

import run_followup as runner
from guarded_codec import encode


HERE = Path(__file__).resolve().parent
SHUFFLE_SEED = 73023
CONTROL_VARIANTS = {
    "prox001_ef": {"mu": .01, "selection": "topk_ef"},
    "prox01_ef": {"mu": .1, "selection": "topk_ef"},
    "sign50_ef": {"mu": 0., "selection": "positive_product_magnitude_quota_half"},
    "fixed50_ef": {"mu": 0., "selection": "fixed_half_magnitude_core_then_signed_product"},
    "reverse90_ef": {"mu": 0., "selection": "energy_guard_90_negated_reference"},
    "shuffle90_ef": {"mu": 0., "selection": "energy_guard_90_permuted_reference"},
    "regtopk1_ef": {"mu": 0., "reg_mu": 1., "selection": "RegTopK-on-model-deltas adaptation"},
    "regtopk5_ef": {"mu": 0., "reg_mu": 5., "selection": "RegTopK-on-model-deltas adaptation"},
    "regtopk10_ef": {"mu": 0., "reg_mu": 10., "selection": "RegTopK-on-model-deltas adaptation"},
}

_ORIGINAL_ENCODE = runner.encode_variant
_ORIGINAL_TRAIN = runner.train_local
_ORIGINAL_ONE_RUN = runner.one_run
_ORIGINAL_SNAPSHOT = runner.snapshot_run
_ORIGINAL_ATOMIC_JSON = runner.atomic_json
_ACTIVE_MU = 0.
_PERMUTATIONS = {}
_REG_CONTEXT = None


def control_spec(variant):
    return CONTROL_VARIANTS.get(variant, {"mu": 0., "selection": "original_runner_passthrough"})


def _with_metadata(args, variant=None):
    params = copy.copy(args)
    params.adapter_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    params.adapter_name = "run_controls.py"
    params.adapter_version = 1
    if variant is not None:
        spec = control_spec(variant)
        params.control_variant = variant
        params.fedprox_mu = spec["mu"]
        params.selection_control = spec["selection"]
        params.reference_shuffle_seed = SHUFFLE_SEED if variant == "shuffle90_ef" else None
        params.regtopk_mu = spec.get("reg_mu")
        params.regtopk_adaptation = "RegTopK-on-model-deltas" if variant.startswith("regtopk") else None
        params.stateful_resume_supported = not variant.startswith("regtopk")
    else:
        params.control_specs = {v: control_spec(v) for v in args.variants}
        params.reference_shuffle_seed = SHUFFLE_SEED if "shuffle90_ef" in args.variants else None
    return params


def train_local(model, dataset, seed, lr, epochs):
    """FedProx local objective: F_i(w)+mu/2*||w-w_global||^2.

    The proximal gradient mu*(p-anchor) is added after CE.backward().  For
    mu=0 the original function is called directly, preserving its behavior.
    """
    if _ACTIVE_MU == 0:
        return _ORIGINAL_TRAIN(model, dataset, seed, lr, epochs)
    if not math.isfinite(_ACTIVE_MU) or _ACTIVE_MU < 0:
        raise ValueError("FedProx mu must be finite and nonnegative")
    torch.manual_seed(seed)
    local = copy.deepcopy(model)
    local.train()
    anchors = [p.detach().clone() for p in model.parameters()]
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.SGD(local.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(local(x), y).backward()
            with torch.no_grad():
                for p, anchor in zip(local.parameters(), anchors):
                    if p.grad is not None:
                        p.grad.add_(p - anchor, alpha=_ACTIVE_MU)
            opt.step()
    return runner.vectorize(local) - runner.vectorize(model)


def _support_encoding(candidate, indices, baseline, reference, *, core_size,
                      filler_size, selection):
    """Use the baseline validation/accounting, but change only the support."""
    indices = torch.sort(indices).values
    if indices.numel() != baseline.indices.numel() or indices.unique().numel() != indices.numel():
        raise AssertionError("control must select exactly K unique coordinates")
    sent = torch.zeros_like(candidate)
    sent[indices] = candidate[indices]
    residual = candidate - sent
    diag = dict(baseline.diagnostics)
    retained = float(sent.double().square().sum())
    energy = float(candidate.double().square().sum())
    top_energy = float(baseline.sent.double().square().sum())
    reference_energy = 0. if reference is None else float(reference.double().square().sum())
    alignment = 0. if reference is None else float(torch.dot(sent.double(), reference.double()))
    top_alignment = baseline.diagnostics["alignment"]
    diag.update({
        "core_size": core_size, "filler_size": filler_size,
        "filler_fraction": filler_size / indices.numel(),
        "reference_fallback": False, "candidate_norm": math.sqrt(energy),
        "sent_norm": math.sqrt(retained),
        "residual_norm": float(torch.linalg.vector_norm(residual.double())),
        "retained_energy_fraction": retained / energy if energy else 1.,
        "retained_topk_energy_ratio": retained / top_energy if top_energy else 1.,
        "alignment": alignment, "topk_alignment": top_alignment,
        "alignment_gain": alignment - top_alignment,
        "reference_cosine": alignment / math.sqrt(max(retained * reference_energy, 1e-300)),
        "selection_control": selection,
    })
    return sent.detach(), residual.detach(), diag


def _actual_reference_diagnostics(result, baseline, original_reference, transform):
    """Report alignment to the true previous increment consistently.

    The transformed-reference gain is retained under guidance_alignment_gain;
    mean_alignment_gain in runner logs always refers to the unmodified g.
    """
    diag = dict(result.diagnostics)
    diag["guidance_alignment_gain"] = diag["alignment_gain"]
    diag["reference_transform"] = transform
    if original_reference is None:
        actual_alignment = reference_energy = 0.
    else:
        actual_alignment = float(torch.dot(result.sent.double(), original_reference.double()))
        reference_energy = float(original_reference.double().square().sum())
    top_alignment = baseline.diagnostics["alignment"]
    retained = float(result.sent.double().square().sum())
    diag.update({"alignment": actual_alignment, "topk_alignment": top_alignment,
                 "alignment_gain": actual_alignment - top_alignment,
                 "reference_cosine": actual_alignment / math.sqrt(max(retained * reference_energy, 1e-300))})
    return result.sent, result.residual, diag


def _regtopk_encode(delta, residual, reference, fraction, variant, seed, baseline):
    """Algorithm-2 selection adapted to FedAvg model deltas, C=1.

    For j in the previous selected support, d_j=(gprev_j-w*u_prev_j)/
    (w*u_j), and score_j=|u_j|*tanh(|1+d_j|/mu). Else score_j=|u_j|.
    Zero current values get zero score without evaluating a division by zero.
    The first round is ordinary Top-K.  Per-client histories are not in the
    original runner's checkpoint, so this adapter explicitly forbids resume.
    """
    if _REG_CONTEXT is None or _REG_CONTEXT["variant"] != variant:
        raise RuntimeError("RegTopK requires one_run's per-client state context")
    context = _REG_CONTEXT
    client = context["call_count"] % len(context["weights"])
    previous = context["previous"][client]
    candidate = delta.detach() + residual.detach()
    weight = context["weights"][client]
    mu = CONTROL_VARIANTS[variant]["reg_mu"]
    if previous is None:
        sent, rem, diag = baseline.sent, baseline.residual, dict(baseline.diagnostics)
        selected = baseline.indices
        diag["regtopk_first_round"] = True
    else:
        if reference is None:
            raise ValueError("RegTopK history exists but previous aggregate is missing")
        previous_u, previous_support = previous
        score = candidate.double().abs()
        active = previous_support[candidate[previous_support] != 0]
        distortion = (reference[active].double() - weight * previous_u[active].double()) / (weight * candidate[active].double())
        factors = torch.tanh((1. + distortion).abs() / mu)
        score[active] *= factors
        score[candidate == 0] = 0
        selected = torch.topk(score, k=baseline.indices.numel(), sorted=False).indices
        sent, rem, diag = _support_encoding(candidate, selected, baseline, reference,
            core_size=0, filler_size=0, selection="RegTopK-on-model-deltas adaptation")
        diag["regtopk_first_round"] = False
        diag["regtopk_mean_factor_on_active_previous_support"] = float(factors.mean()) if factors.numel() else 1.
    diag.update({"regtopk_mu": mu, "regtopk_client_weight": weight,
                 "regtopk_client_index": client,
                 "regtopk_adaptation": "RegTopK-on-model-deltas",
                 "selection_control": "RegTopK-on-model-deltas adaptation"})
    context["previous"][client] = (candidate.clone(), selected.clone())
    context["call_count"] += 1
    return sent, rem, diag


def encode_variant(delta, residual, reference, fraction, variant, seed):
    if variant not in CONTROL_VARIANTS:
        return _ORIGINAL_ENCODE(delta, residual, reference, fraction, variant, seed)
    if variant in ("prox001_ef", "prox01_ef"):
        sent, next_residual, diag = _ORIGINAL_ENCODE(delta, residual, reference, fraction, "topk_ef", seed)
        return sent, next_residual, {**diag, "selection_control": "topk_ef"}
    baseline = encode(delta, residual, fraction=fraction, method="topk_ef", reference=reference, seed=seed)
    if variant.startswith("regtopk"):
        return _regtopk_encode(delta, residual, reference, fraction, variant, seed, baseline)
    candidate = delta.detach() + residual.detach()
    k = baseline.indices.numel()
    reference_zero = reference is None or float(reference.double().square().sum()) == 0
    if reference_zero:
        return baseline.sent, baseline.residual, {**baseline.diagnostics,
            "selection_control": CONTROL_VARIANTS[variant]["selection"]}
    if variant in ("reverse90_ef", "shuffle90_ef"):
        if variant == "reverse90_ef":
            guidance = -reference
            transform = "negation"
        else:
            n = reference.numel()
            if n not in _PERMUTATIONS:
                generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
                _PERMUTATIONS[n] = torch.randperm(n, generator=generator)
            guidance = reference[_PERMUTATIONS[n]]
            transform = f"fixed_permutation_seed_{SHUFFLE_SEED}"
        result = encode(delta, residual, fraction=fraction, method="guard_alignment_ef",
                        reference=guidance, tau=.9, seed=seed)
        return _actual_reference_diagnostics(result, baseline, reference, transform)
    allowed = torch.ones(candidate.numel(), dtype=torch.bool)
    if variant == "sign50_ef":
        positives = torch.nonzero(candidate.double() * reference.double() > 0, as_tuple=False).flatten()
        reserved = min(k // 2, positives.numel())
        selected = (positives[torch.topk(candidate[positives].abs(), k=reserved, sorted=False).indices]
                    if reserved else torch.empty(0, dtype=torch.long))
        allowed[selected] = False
        remaining = torch.nonzero(allowed, as_tuple=False).flatten()
        rest = remaining[torch.topk(candidate[remaining].abs(), k=k-reserved, sorted=False).indices]
        indices = torch.cat((selected, rest))
        return _support_encoding(candidate, indices, baseline, reference,
            core_size=k-reserved, filler_size=reserved,
            selection=CONTROL_VARIANTS[variant]["selection"])
    if variant == "fixed50_ef":
        m = k // 2
        core = (torch.topk(candidate.abs(), k=m, sorted=True).indices
                if m else torch.empty(0, dtype=torch.long))
        allowed[core] = False
        remaining = torch.nonzero(allowed, as_tuple=False).flatten()
        score = candidate.double() * reference.double()
        rest = remaining[torch.topk(score[remaining], k=k-m, sorted=False).indices]
        return _support_encoding(candidate, torch.cat((core, rest)), baseline, reference,
            core_size=m, filler_size=k-m,
            selection=CONTROL_VARIANTS[variant]["selection"])
    raise AssertionError(variant)


def one_run(args, trainsets, validation, manifest, variant, seed, out):
    global _ACTIVE_MU, _REG_CONTEXT
    previous_mu = _ACTIVE_MU
    previous_reg_context = _REG_CONTEXT
    _ACTIVE_MU = control_spec(variant)["mu"]
    if variant.startswith("regtopk"):
        result_path = out / "result.json"
        complete = result_path.exists() and json.loads(result_path.read_text()).get("status") == "DONE"
        if not complete and any((out / name).exists() for name in ("result.json", "checkpoint.pt", "config.json")):
            _ACTIVE_MU = previous_mu
            raise RuntimeError("RegTopK per-client histories are not checkpointed; interrupted runs cannot resume. Use a new --out directory.")
        total = sum(len(ds) for ds in trainsets)
        if total <= 0 or any(len(ds) <= 0 for ds in trainsets):
            _ACTIVE_MU = previous_mu
            raise ValueError("RegTopK requires nonempty clients with positive sample weights")
        _REG_CONTEXT = {"variant": variant, "weights": [len(ds)/total for ds in trainsets],
                        "call_count": 0, "previous": [None for _ in trainsets]}
    try:
        # All added fields are visible to original one_run's vars(args), hence
        # included before its run_signature is computed or compared on resume.
        return _ORIGINAL_ONE_RUN(_with_metadata(args, variant), trainsets, validation,
                                 manifest, variant, seed, out)
    finally:
        _ACTIVE_MU = previous_mu
        _REG_CONTEXT = previous_reg_context


def _snapshot_atomic_json(path, obj):
    if path.name == "config.json":
        obj = dict(obj)
        signature_obj = {k: v for k, v in obj.items() if k not in ("out", "run_signature")}
        obj["run_signature"] = hashlib.sha256(json.dumps(signature_obj, sort_keys=True).encode()).hexdigest()
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("run_signature") != obj["run_signature"]:
                raise ValueError(f"incompatible existing snapshot at {path.parent}; use a new output directory")
    return _ORIGINAL_ATOMIC_JSON(path, obj)


def snapshot_run(args, trainsets, validation, manifest, seed, out):
    if any(control_spec(v)["mu"] != 0 for v in args.variants):
        raise ValueError("FedProx controls require --mode train: a shared dense trajectory cannot use multiple local objectives")
    if any(v.startswith("regtopk") for v in args.variants):
        raise ValueError("RegTopK requires --mode train: snapshot skips intermediate selected-support histories")
    previous_atomic = runner.atomic_json
    runner.atomic_json = _snapshot_atomic_json
    try:
        return _ORIGINAL_SNAPSHOT(_with_metadata(args), trainsets, validation, manifest, seed, out)
    finally:
        runner.atomic_json = previous_atomic


def install():
    # This patches only the imported module in the current adapter process;
    # the running runner files and independent baseline processes are untouched.
    for variant in CONTROL_VARIANTS:
        runner.VARIANTS[variant] = ("topk_ef", 1.)
    runner.encode_variant = encode_variant
    runner.train_local = train_local
    runner.one_run = one_run
    runner.snapshot_run = snapshot_run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" not in argv and "-h" not in argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--variants", required=True)
        parser.add_argument("--mode", default="train")
        known, _ = parser.parse_known_args(argv)
        variants = [v.strip() for v in known.variants.split(",") if v.strip()]
        if not variants:
            parser.error("adapter requires at least one explicit --variants entry")
        if known.mode == "snapshot" and any(control_spec(v)["mu"] for v in variants):
            parser.error("FedProx variants require --mode train")
        if known.mode == "snapshot" and any(v.startswith("regtopk") for v in variants):
            parser.error("RegTopK variants require --mode train")
    install()
    original_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        return runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
