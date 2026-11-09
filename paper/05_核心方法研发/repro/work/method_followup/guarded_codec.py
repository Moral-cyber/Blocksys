"""Energy-guarded sparse update candidates for exploratory FL experiments.

These are candidate selection rules, not claims of novelty or convergence.
Every method sends exactly K unmodified candidate values and uint32 indices.
The caller must commit the returned candidate residual only when delivery and
aggregation are known to have succeeded.  The benchmark assumes this success.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Optional

import torch


METHODS = (
    "topk_ef",
    "guard_agreement_ef",
    "guard_alignment_ef",
    "guard_innovation_ef",
    "guard_random_ef",
)


@dataclass
class Encoding:
    sent: torch.Tensor
    residual: torch.Tensor
    indices: torch.Tensor
    diagnostics: dict


def _check_vector(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 1 or x.numel() == 0:
        raise ValueError(f"{name} must be a nonempty flattened torch tensor")
    if x.device.type != "cpu" or not x.is_floating_point():
        raise ValueError(f"{name} must be a floating-point CPU tensor")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must contain only finite values")


def encode(
    delta: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    fraction: float = 0.2,
    method: str = "topk_ef",
    reference: Optional[torch.Tensor] = None,
    tau: float = 0.9,
    seed: int = 0,
    packet_header_bytes: int = 64,
) -> Encoding:
    """Return sent update, provisional residual, exact support and diagnostics.

    ``reference`` is the previous global model increment, already available
    from consecutive global models. ``tau`` specifies a fraction of Top-K's
    retained squared norm.  A minimal highest-magnitude core attains this
    fraction; the remaining K-m positions follow the named filler rule.

    ``guard_agreement_ef`` chooses positive u*g first, then fills by |u|.
    ``guard_alignment_ef`` chooses the largest signed u*g, including negative
    values if needed.  Only the latter additionally maximizes alignment among
    supports containing the fixed core, hence its alignment cannot be less
    than Top-K's (up to arithmetic rounding).  Neither implies better learning.

    Zero/missing references, zero candidate energy and tau=1 fall back exactly
    to ordinary Top-K.  A local seeded generator isolates random filler from
    the training random-number stream.
    """
    _check_vector(delta, "delta")
    if delta.dtype != torch.float32:
        raise ValueError("this wire profile requires FP32 candidate values")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be finite and in (0,1]")
    if not math.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("tau must be finite and in (0,1]")
    if not isinstance(packet_header_bytes, int) or packet_header_bytes < 0:
        raise ValueError("packet_header_bytes must be a nonnegative integer")
    if delta.numel() >= 2 ** 32:
        raise ValueError("uint32 indices require fewer than 2**32 elements")
    if residual is None:
        residual = torch.zeros_like(delta)
    _check_vector(residual, "residual")
    if residual.shape != delta.shape or residual.dtype != delta.dtype:
        raise ValueError("delta and residual must have identical shape and dtype")
    if reference is not None:
        _check_vector(reference, "reference")
        if reference.shape != delta.shape or reference.dtype != delta.dtype:
            raise ValueError("reference must match delta's shape and dtype")

    u = (delta.detach() + residual.detach()).clone()
    _check_vector(u, "candidate")
    n = u.numel()
    k = max(1, min(n, int(n * fraction)))
    top_indices = torch.topk(u.abs(), k=k, largest=True, sorted=True).indices
    top_energy = float(torch.sum(u[top_indices].double().square()))
    reference_energy = 0.0 if reference is None else float(reference.double().square().sum())
    fallback = method == "topk_ef" or reference is None or reference_energy == 0 or tau == 1 or top_energy == 0
    core_size = k
    filler_size = 0
    if fallback:
        chosen = top_indices
    else:
        cumulative = torch.cumsum(u[top_indices].double().square(), dim=0)
        threshold = tau * top_energy
        core_size = min(k, int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=torch.float64)).item()) + 1)
        core = top_indices[:core_size]
        filler_size = k - core_size
        allowed = torch.ones(n, dtype=torch.bool)
        allowed[core] = False
        available = torch.nonzero(allowed, as_tuple=False).flatten()
        if filler_size == 0:
            filler = torch.empty(0, dtype=torch.long)
        elif method in ("guard_agreement_ef", "guard_alignment_ef"):
            score = u.double() * reference.double()
            if method == "guard_alignment_ef":
                filler = available[torch.topk(score[available], k=filler_size, sorted=False).indices]
            else:
                positive = available[score[available] > 0]
                count = min(filler_size, positive.numel())
                selected_positive = positive[torch.topk(score[positive], k=count, sorted=False).indices] if count else torch.empty(0, dtype=torch.long)
                left = filler_size - count
                if left:
                    allowed[selected_positive] = False
                    remaining = torch.nonzero(allowed, as_tuple=False).flatten()
                    rest = remaining[torch.topk(u[remaining].abs(), k=left, sorted=False).indices]
                    filler = torch.cat([selected_positive, rest])
                else:
                    filler = selected_positive
        elif method == "guard_innovation_ef":
            ud = u.double()
            gd = reference.double()
            coefficient = torch.dot(ud, gd) / reference_energy
            innovation = (ud - coefficient * gd).abs()
            filler = available[torch.topk(innovation[available], k=filler_size, sorted=False).indices]
        elif method == "guard_random_ef":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            filler = available[torch.randperm(available.numel(), generator=generator)[:filler_size]]
        else:  # guarded by METHODS above
            raise AssertionError(method)
        chosen = torch.cat([core, filler])

    indices = torch.sort(chosen).values
    sent = torch.zeros_like(u)
    sent[indices] = u[indices]
    next_residual = u - sent
    energy = float(torch.sum(u.double().square()))
    retained = float(torch.sum(sent.double().square()))
    if reference is None:
        alignment = top_alignment = 0.0
        cosine = 0.0
    else:
        alignment = float(torch.dot(sent.double(), reference.double()))
        top_alignment = float(torch.dot(u[top_indices].double(), reference[top_indices].double()))
        cosine = alignment / math.sqrt(max(retained * reference_energy, 1e-300))
    diagnostics = {
        "selected_coordinates": k,
        "core_size": core_size,
        "filler_size": filler_size,
        "filler_fraction": filler_size / k,
        "reference_fallback": bool(fallback),
        "candidate_norm": math.sqrt(energy),
        "sent_norm": math.sqrt(retained),
        "residual_norm": float(torch.linalg.vector_norm(next_residual.double())),
        "retained_energy_fraction": retained / energy if energy else 1.0,
        "retained_topk_energy_ratio": retained / top_energy if top_energy else 1.0,
        "alignment": alignment,
        "topk_alignment": top_alignment,
        "alignment_gain": alignment - top_alignment,
        "reference_cosine": cosine,
        "payload_bytes": 4 * k,
        "index_bytes": 4 * k,
        "packet_header_bytes": packet_header_bytes,
        "encoded_bytes": packet_header_bytes + 8 * k,
    }
    return Encoding(sent.detach(), next_residual.detach(), indices, diagnostics)


def canonical_pairs(indices: torch.Tensor, values: torch.Tensor, num_elements: int) -> bytes:
    """Sort complete index/value pairs and serialize uint32-LE / FP32-LE.

    This serializes the pair payload only.  Authentication context/header
    construction remains the caller's responsibility and is not simulated here.
    """
    if indices.ndim != 1 or values.ndim != 1 or indices.numel() != values.numel():
        raise ValueError("indices and values must be equally sized 1-D vectors")
    if indices.device.type != "cpu" or values.device.type != "cpu":
        raise ValueError("canonical pairs require CPU tensors")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must have integer dtype")
    if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
        raise ValueError("wire values must be finite FP32")
    if not 1 <= indices.numel() <= num_elements < 2 ** 32:
        raise ValueError("expected 1 <= K <= N < 2**32")
    sorted_indices, order = torch.sort(indices.long())
    if int(sorted_indices[0]) < 0 or int(sorted_indices[-1]) >= num_elements:
        raise ValueError("index out of range")
    if sorted_indices.numel() > 1 and bool((sorted_indices[1:] == sorted_indices[:-1]).any()):
        raise ValueError("duplicate indices")
    sorted_values = values[order]
    # Using IEEE-754 pack preserves the FP32 value, including signed zero.
    return b"".join(struct.pack("<If", int(j), float(v)) for j, v in zip(sorted_indices, sorted_values))


def pack_encoding(result: Encoding, *, num_elements: Optional[int] = None) -> bytes:
    """Return the result's sorted pair payload; zeros remain explicit pairs."""
    n = result.sent.numel() if num_elements is None else num_elements
    return canonical_pairs(result.indices, result.sent[result.indices], n)
