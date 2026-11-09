"""Explicit prospective TopK context; frozen original entry points stay unchanged."""
from pathlib import Path
import argparse
import types
import torch
import common as c
import train as fixed_train

PLAN_SHA = '703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b'
PARTITION, SEEDS, ROUNDS = 20261217, (20261231, 20261232, 20261233), 20
TOPK_PINS = {
    'PLAN.md': 'b0daad197d50f71c4d18e3ea3d12878192ab9705a3d6b6179479048e8c6d7130',
    'common.py': '37976f773cbaca44ee444d95091cb77cb916e06a9943837dbbee8586425a4a58',
    'wire.py': '6bf9856e73fba6d601bbcd8ce19cb9ec3bdfe09f17e58ca79e0faef000c86693',
    'train.py': '13d8538e671302e1f8ec891fea3f0b87c6cb6a996774222aef4d36856946affe',
    'evaluate.py': '22cec15467ae06629c361a8f510da45cc1637bdf2ef1242db42169c88dbb3e8c'}


def make_context(protocol_plan, seed, *, fixture=False):
    """Return a new recorded context; never mutate original common globals."""
    plan = Path(protocol_plan).resolve()
    if seed not in SEEDS or c.sha(plan) != PLAN_SHA:
        raise ValueError('declared seed and exact prospective protocol PLAN required')
    for name, expected in TOPK_PINS.items():
        if c.sha(c.HERE / name) != expected:
            raise ValueError('frozen original TopK source changed: ' + name)
    extra = {str(plan): PLAN_SHA}
    for name in ('prospective.py', 'prospective_evaluate.py'):
        path = (c.HERE / name).resolve()
        extra[str(path)] = c.sha(path)
    metadata = {'protocol_plan': str(plan), 'protocol_plan_sha256': PLAN_SHA,
        'partition_seed_override': PARTITION, 'training_seed_override': seed,
        'allowed_training_seeds': list(SEEDS), 'formal_rounds': ROUNDS,
        'base_partition_seed': 20261203, 'base_training_seed': 20261211,
        'source_overrides': extra, 'base_source_pins': TOPK_PINS,
        'binding': 'fresh function globals with explicit common/accelerator context; original globals unchanged',
        'scope': 'CPU synthetic fixture' if fixture else 'prospective same-cap compression reference',
        'original_fixed_main_called': False, 'new_candidate_or_go': False}

    def identities():
        c.verify_identities(extra)
        for name, expected in TOPK_PINS.items():
            if c.sha(c.HERE / name) != expected:
                raise ValueError('frozen original TopK source changed')
        return {**c.source_identities(), **extra}

    def runtime(backend, route):
        result = c.accelerator.runtime_metadata(backend, route)
        result['prospective_reference'] = metadata
        return result

    context = types.SimpleNamespace(**vars(c))
    context.source_identities = identities
    context.accelerator = types.SimpleNamespace(**vars(c.accelerator))
    context.accelerator.runtime_metadata = runtime
    return context, metadata


def bind(function, context):
    """Keep the original function code and use a separate globals dictionary."""
    namespace = dict(function.__globals__)
    namespace['c'] = context
    result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def run(args, trainsets, validation, manifest, data_identities, protocol_plan, *, fixture=False):
    if args.partition_seed != PARTITION or args.seed not in SEEDS or (not fixture and args.rounds != ROUNDS):
        raise ValueError('prospective partition/seed/rounds mismatch')
    if manifest.get('partition_seed') != PARTITION:
        raise ValueError('prospective partition manifest mismatch')
    context, metadata = make_context(protocol_plan, args.seed, fixture=fixture)
    bound = bind(fixed_train.run, context)
    for method in c.METHODS:
        bound(args, method, trainsets, validation, manifest, data_identities)
    context.verify_identities(context.source_identities())
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol-plan', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--partition-seed', type=int, choices=(PARTITION,), default=PARTITION)
    parser.add_argument('--rounds', type=int, choices=(ROUNDS,), default=ROUNDS)
    parser.add_argument('--backend', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    context, _ = make_context(args.protocol_plan, args.seed)
    attempts = context.install_test_guard()
    torch.set_num_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    datasets, validation, manifest, identities, _, _ = context.prepare(PARTITION)
    if len(datasets) != 20:
        raise ValueError('prospective reference requires20 clients')
    run(args, datasets, validation, manifest, identities, args.protocol_plan)
    if attempts:
        raise ValueError('official TEST byte access attempted')


if __name__ == '__main__':
    main()
