"""Independent adapter for Lagrangian-family feasible sparse supports.

No existing runner/codec/controls file is modified. Additional variants are
lagrange90_ef and lagrange99_ef. Explicit --variants is required. The inherited
snapshot mode is supported, but no reusable full-state snapshot cache is
written by this adapter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys

import run_controls as controls
import run_followup as runner
from lagrangian_codec import encode


HERE = Path(__file__).resolve().parent
VARIANTS = {"lagrange90_ef": .90, "lagrange99_ef": .99}
_CONTROL_ENCODE = controls.encode_variant
_CONTROL_ONE_RUN = controls.one_run
_CONTROL_SNAPSHOT = controls.snapshot_run


def metadata_args(args, variant=None):
    params = copy.copy(args)
    params.lagrangian_adapter_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    params.lagrangian_codec_source_sha256 = hashlib.sha256((HERE/"lagrangian_codec.py").read_bytes()).hexdigest()
    params.lagrangian_max_expansions = 24
    params.lagrangian_bisection_steps = 12
    params.lagrangian_claim = "feasible Lagrangian-family approximation; not exact constrained optimum"
    params.lagrangian_core_filler_semantics = "Top-K overlap/replacements; not prefix reserve"
    if variant is None:
        params.lagrangian_taus = {v: VARIANTS[v] for v in args.variants if v in VARIANTS}
    else:
        params.lagrangian_tau = VARIANTS.get(variant)
    return params


def encode_variant(delta, residual, reference, fraction, variant, seed):
    if variant not in VARIANTS:
        return _CONTROL_ENCODE(delta, residual, reference, fraction, variant, seed)
    result = encode(delta, residual, fraction=fraction, reference=reference,
                    tau=VARIANTS[variant], seed=seed,
                    max_expansions=24, bisection_steps=12)
    return result.sent, result.residual, result.diagnostics


def one_run(args, trainsets, validation, manifest, variant, seed, out):
    return _CONTROL_ONE_RUN(metadata_args(args, variant), trainsets, validation,
                            manifest, variant, seed, out)


def snapshot_run(args, trainsets, validation, manifest, seed, out):
    return _CONTROL_SNAPSHOT(metadata_args(args), trainsets, validation, manifest, seed, out)


def install():
    for variant, tau in VARIANTS.items():
        controls.CONTROL_VARIANTS[variant] = {
            "mu": 0., "selection": "normalized Lagrangian feasible support search",
            "tau": tau,
        }
    controls.install()
    runner.encode_variant = encode_variant
    runner.one_run = one_run
    runner.snapshot_run = snapshot_run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" not in argv and "-h" not in argv:
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--variants", required=True)
        p.add_argument("--mode", default="train")
        known, _ = p.parse_known_args(argv)
        variants = [v.strip() for v in known.variants.split(",") if v.strip()]
        if not variants:
            p.error("explicit --variants must contain at least one method")
        if known.mode == "snapshot" and any(controls.control_spec(v)["mu"] or v.startswith("regtopk") for v in variants):
            p.error("FedProx/RegTopK controls require --mode train")
    install()
    previous_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *argv]
    try:
        return runner.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
