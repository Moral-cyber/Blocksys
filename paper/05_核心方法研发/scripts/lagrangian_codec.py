"""Feasible Lagrangian support search for exploratory sparse FL.

This searches Top-K supports of normalized alignment + lambda*energy. It is
not an exact solver for the discrete energy-constrained support problem.
No changes are made to guarded_codec.py or any running experiment source.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from guarded_codec import Encoding, encode as plain_encode


def encode(
    delta: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    fraction: float = .2,
    method: str = "lagrangian_ef",
    reference: Optional[torch.Tensor] = None,
    tau: float = .9,
    seed: int = 0,
    packet_header_bytes: int = 64,
    max_expansions: int = 24,
    bisection_steps: int = 12,
) -> Encoding:
    """Select K coordinates with a checked energy floor and alignment guard.

    Let a=u**2, b=u*g. The searched family is
    TopK(b/max(abs(b)) + lambda*a/max(a)), lambda>=0.
    Lambda=0 is tried first. Otherwise a feasible high endpoint is found by
    doubling from 1, then a fixed 8 or 12 bisections narrow the infeasible-low /
    feasible-high bracket. The known feasible high support is returned.

    An argmax support of b+lambda*a has alignment at least that of magnitude
    Top-K because magnitude Top-K maximizes a. This does not imply that g is a
    valid current descent direction or that FL learning improves. Floating-
    point implementations are checked against the actual Top-K output before
    returning, and fall back when either guard fails.

    Diagnostic core/filler refer to overlap/replacements relative to Top-K,
    not the fixed magnitude-prefix reserve used by guarded_codec.py.
    """
    if method not in ("lagrangian_ef", "lagrange90_ef", "lagrange99_ef"):
        raise ValueError("unsupported Lagrangian method")
    if not isinstance(max_expansions, int) or not 1 <= max_expansions <= 24:
        raise ValueError("max_expansions must be an integer in [1,24]")
    if bisection_steps not in (8, 12):
        raise ValueError("bisection_steps must be 8 or 12")
    baseline = plain_encode(delta, residual, fraction=fraction, method="topk_ef",
                            reference=reference, tau=tau, seed=seed,
                            packet_header_bytes=packet_header_bytes)
    if residual is None:
        residual = torch.zeros_like(delta)
    u = delta.detach() + residual.detach()
    ud = u.double()
    a = ud.square()
    k = baseline.indices.numel()
    top_energy = float(baseline.sent.double().square().sum())
    top_alignment = (0. if reference is None else
                     float(torch.dot(baseline.sent.double(), reference.double())))
    solver_iters = 0
    low = high = None
    a_scale = float(a.max())
    b_scale = 0.

    def build(indices, status, chosen_lambda=None):
        indices = torch.sort(indices).values
        sent = torch.zeros_like(u)
        sent[indices] = u[indices]
        rem = u - sent
        energy = float(sent.double().square().sum())
        alignment = (0. if reference is None else
                     float(torch.dot(sent.double(), reference.double())))
        top_mask = torch.zeros(u.numel(), dtype=torch.bool)
        top_mask[baseline.indices] = True
        replacements = int((~top_mask[indices]).sum())
        reference_energy = 0. if reference is None else float(reference.double().square().sum())
        total_energy = float(a.sum())
        diag = dict(baseline.diagnostics)
        diag.update({
            "core_size": k-replacements, "filler_size": replacements,
            "filler_fraction": replacements/k,
            "replacement_fraction": replacements/k,
            "core_filler_semantics": "Top-K support overlap / replacements; not prefix reserve",
            "candidate_norm": math.sqrt(total_energy), "sent_norm": math.sqrt(energy),
            "residual_norm": float(torch.linalg.vector_norm(rem.double())),
            "retained_energy_fraction": energy/total_energy if total_energy else 1.,
            "retained_topk_energy_ratio": energy/top_energy if top_energy else 1.,
            "alignment": alignment, "topk_alignment": top_alignment,
            "alignment_gain": alignment-top_alignment,
            "reference_cosine": alignment/math.sqrt(max(energy*reference_energy, 1e-300)),
            "reference_fallback": status.startswith("fallback"),
            "solver_status": status, "solver_iters": solver_iters,
            "solver_lambda": chosen_lambda,
            "solver_lambda_effective_raw": (chosen_lambda*b_scale/a_scale
                                             if chosen_lambda is not None and a_scale else None),
            "solver_lambda_low": low, "solver_lambda_high": high,
            "solver_a_scale": a_scale, "solver_b_scale": b_scale,
            "solver_max_expansions": max_expansions,
            "solver_bisection_steps": bisection_steps,
            "energy_constraint_tau": tau,
            "solver_claim": "feasible Lagrangian-family approximation; not exact constrained optimum",
        })
        return Encoding(sent.detach(), rem.detach(), indices, diag)

    if tau == 1:
        return build(baseline.indices, "fallback_tau_one")
    if reference is None:
        return build(baseline.indices, "fallback_missing_reference")
    if float(reference.double().square().sum()) == 0:
        return build(baseline.indices, "fallback_zero_reference")
    if a_scale == 0:
        return build(baseline.indices, "fallback_zero_candidate")
    b = ud * reference.double()
    b_scale = float(b.abs().max())
    if b_scale == 0:
        return build(baseline.indices, "fallback_zero_product")
    normalized_a = a/a_scale
    normalized_b = b/b_scale
    threshold = tau*top_energy

    def support(lam):
        nonlocal solver_iters
        solver_iters += 1
        score = normalized_b + lam*normalized_a
        return torch.topk(score, k=k, largest=True, sorted=False).indices

    def feasible(indices):
        return float(a[indices].sum()) >= threshold

    zero_support = support(0.)
    if feasible(zero_support):
        result = build(zero_support, "alignment_only_feasible", 0.)
    else:
        low, high = 0., 1.
        high_support = None
        for attempt in range(max_expansions):
            proposed = support(high)
            if feasible(proposed):
                high_support = proposed
                break
            low = high
            if attempt + 1 < max_expansions:
                high *= 2.
        if high_support is None:
            return build(baseline.indices, "fallback_no_feasible_high")
        for _ in range(bisection_steps):
            mid = (low+high)/2.
            proposed = support(mid)
            if feasible(proposed):
                high, high_support = mid, proposed
            else:
                low = mid
        result = build(high_support, "bisected_feasible_high", high)
    # Enforce properties using actual reconstructed-vector arithmetic, not only
    # the sparse support's intermediate sum or the scalarized-score argument.
    actual_energy = float(result.sent.double().square().sum())
    actual_alignment = float(torch.dot(result.sent.double(), reference.double()))
    if actual_energy < threshold:
        return build(baseline.indices, "fallback_numeric_energy_guard")
    if actual_alignment < top_alignment:
        return build(baseline.indices, "fallback_numeric_alignment_guard")
    return result
