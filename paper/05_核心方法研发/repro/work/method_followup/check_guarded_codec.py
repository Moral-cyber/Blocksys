"""Deterministic checks of codec invariants; not learning-effect evidence."""

from pathlib import Path
import json
import struct

import torch

from guarded_codec import METHODS, canonical_pairs, encode, pack_encoding


def checks():
    rng = torch.Generator().manual_seed(20261103)
    cases = 0
    min_guard_margin = float("inf")
    min_alignment_gain = float("inf")
    for n in (1, 2, 7, 31, 128, 4097):
        for fraction in (0.01, 0.2, 0.5, 1.0):
            delta = torch.randn(n, generator=rng)
            residual = torch.randn(n, generator=rng) / 5
            reference = torch.randn(n, generator=rng)
            u = delta + residual
            for tau in (0.5, 0.9, 0.99, 1.0):
                plain = encode(delta, residual, fraction=fraction)
                for method in METHODS:
                    result = encode(delta, residual, fraction=fraction, method=method,
                                    reference=reference, tau=tau, seed=51)
                    diag = result.diagnostics
                    k = max(1, int(n * fraction))
                    assert result.indices.numel() == k
                    assert result.indices.unique().numel() == k
                    assert bool((result.indices[1:] >= result.indices[:-1]).all())
                    assert torch.equal(result.sent[result.indices], u[result.indices])
                    assert torch.allclose(result.sent + result.residual, u, atol=1e-6, rtol=1e-6)
                    assert diag["encoded_bytes"] == 64 + 8 * k
                    wire = pack_encoding(result)
                    assert len(wire) == 8 * k
                    decoded = torch.zeros(n)
                    for j, v in struct.iter_unpack("<If", wire):
                        decoded[j] = v
                    assert torch.equal(decoded, result.sent)
                    ratio = diag["retained_topk_energy_ratio"]
                    assert ratio + 1e-12 >= tau
                    assert ratio <= 1 + 1e-12
                    min_guard_margin = min(min_guard_margin, ratio - tau)
                    # Ordinary Top-K has the smallest local squared error.
                    error = float(result.residual.double().square().sum())
                    plain_error = float(plain.residual.double().square().sum())
                    assert error + 1e-10 >= plain_error
                    if method == "guard_alignment_ef":
                        assert diag["alignment_gain"] >= -1e-10
                        min_alignment_gain = min(min_alignment_gain, diag["alignment_gain"])
                    if tau == 1:
                        assert torch.equal(result.sent, plain.sent)
                        assert torch.equal(result.indices, plain.indices)
                    for absent in (None, torch.zeros_like(reference)):
                        fallback = encode(delta, residual, fraction=fraction, method=method,
                                          reference=absent, tau=tau)
                        assert torch.equal(fallback.sent, plain.sent)
                        assert torch.equal(fallback.indices, plain.indices)
                    cases += 1

    # Discriminating cancellation example, with tau low enough to leave budget.
    u = torch.tensor([10., 9., 1., 0.5])
    g = torch.tensor([-1., -1., 1., 1.])
    top = encode(u, fraction=0.5)
    aligned = encode(u, fraction=0.5, method="guard_alignment_ef", reference=g, tau=.5)
    assert aligned.diagnostics["alignment_gain"] > 0
    assert aligned.diagnostics["retained_topk_energy_ratio"] >= .5

    # Random filler is reproducible without touching the global RNG stream.
    u = torch.linspace(-1, 1, 101)
    g = torch.ones_like(u)
    state = torch.random.get_rng_state().clone()
    a = encode(u, method="guard_random_ef", reference=g, seed=71)
    assert torch.equal(torch.random.get_rng_state(), state)
    b = encode(u, method="guard_random_ef", reference=g, seed=71)
    assert torch.equal(a.indices, b.indices)

    # Zero pairs are transmitted explicitly and retain signed-zero encoding.
    z = encode(torch.zeros(20), method="guard_alignment_ef", reference=torch.ones(20), fraction=.2)
    assert len(pack_encoding(z)) == 4 * 8
    signed = canonical_pairs(torch.tensor([1, 0]), torch.tensor([-0., 1.]), 2)
    assert signed == struct.pack("<IfIf", 0, 1., 1, -0.)

    # Invalid inputs must not silently alter budgets or admit malformed pairs.
    failures = 0
    invalid = [
        lambda: encode(torch.ones(3), fraction=0),
        lambda: encode(torch.ones(3), tau=float("nan")),
        lambda: encode(torch.ones(3), tau=0),
        lambda: encode(torch.ones(3), packet_header_bytes=-1),
        lambda: encode(torch.tensor([float("nan")])),
        lambda: encode(torch.ones(3), torch.ones(2)),
        lambda: encode(torch.ones(3), reference=torch.ones(2)),
        lambda: canonical_pairs(torch.tensor([0, 0]), torch.ones(2), 3),
        lambda: canonical_pairs(torch.tensor([-1]), torch.ones(1), 3),
        lambda: canonical_pairs(torch.tensor([3]), torch.ones(1), 3),
    ]
    for f in invalid:
        try:
            f()
        except (ValueError, TypeError):
            failures += 1
    assert failures == len(invalid)
    return {
        "status": "PASS",
        "case_count": cases,
        "invalid_cases_rejected": failures,
        "min_guard_margin": min_guard_margin,
        "min_alignment_gain": min_alignment_gain,
        "meaning": "Deterministic compressor, accounting and serialization invariants only; no learning or novelty claim.",
        "alignment_claim_scope": "guard_alignment_ef signed-product filler only; no guarantee that prior-global direction equals current objective descent.",
    }


if __name__ == "__main__":
    report = checks()
    path = Path(__file__).with_name("codec_check.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
