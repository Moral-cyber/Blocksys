"""Thin, explicit prospective CIFAR wrapper; default DESIGN_ONLY, no raw reads."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN_RELATIVE = 'work/predictive_pfl_confirm/PLAN.md'
PLAN_SHA256 = '703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b'
PARTITION_SEED = 20261217
TRAINING_SEEDS = (20261231, 20261232, 20261233)
ROUNDS = 20
FROZEN_SOURCES = {
    'PLAN.md': 'c5817d080e4ac55ba9a3ea8a4739b1885ba33a84a26be5056e97aed51eacd895',
    'reference.py': '3649edec016ac3b06005a624bbf2dcea22d23965d53637b8ec36875b5be74dfc',
    'train.py': '7ff112e2e3f5575835d1c0358e501a4815231fb6572be0ef33b148dc05eaac70'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    for name, expected in FROZEN_SOURCES.items():
        if sha(HERE / name) != expected:
            raise ValueError('frozen FedRoD source changed: ' + name)
    if sha(ROOT / PLAN_RELATIVE) != PLAN_SHA256:
        raise ValueError('prospective PLAN does not match its predeclared hash')


def load_frozen_entry():
    verify_sources()
    name = '_fedrod_prospective_frozen_entry'
    spec = importlib.util.spec_from_file_location(name, HERE / 'train.py')
    entry = importlib.util.module_from_spec(spec)
    sys.modules[name] = entry
    spec.loader.exec_module(entry)
    return entry


def overrides(seed, partition_seed):
    if seed not in TRAINING_SEEDS or partition_seed != PARTITION_SEED:
        raise ValueError('prospective CIFAR requires p20261217 and seed20261231/32/33')
    return {'SEED': seed, 'PARTITION_SEED': partition_seed, 'ROUNDS': ROUNDS}


@contextmanager
def module_overrides(entry, seed, partition_seed):
    """Scope in-memory overrides; the three frozen source files never change."""
    chosen = overrides(seed, partition_seed)
    previous = {name: getattr(entry, name) for name in chosen}
    try:
        for name, value in chosen.items():
            setattr(entry, name, value)
        yield chosen
    finally:
        for name, value in previous.items():
            setattr(entry, name, value)


def base_config(entry, backend, seed, partition_seed):
    verify_sources()
    chosen = overrides(seed, partition_seed)
    if any(getattr(entry, name) != value for name, value in chosen.items()):
        raise ValueError('base config must be built inside the explicit override scope')
    base = entry.base_config(backend)
    if (base['seed'], base['partition_seed'], base['rounds']) != (seed, partition_seed, ROUNDS):
        raise ValueError('frozen entry did not apply prospective overrides')
    relative_wrapper = str(Path(__file__).resolve().relative_to(ROOT))
    wrapper_hash = sha(__file__)
    base['source_identities'].update({relative_wrapper: wrapper_hash,
        PLAN_RELATIVE: PLAN_SHA256})
    identity = {'entry': relative_wrapper, 'wrapper_sha256': wrapper_hash,
        'plan': PLAN_RELATIVE, 'plan_sha256': PLAN_SHA256,
        'frozen_entry_sha256': FROZEN_SOURCES['train.py'],
        'module': entry.__name__, 'module_global_overrides': chosen,
        'allowed_training_seeds': list(TRAINING_SEEDS),
        'local_seed_rule': 'SEED + 100 * round_zero_based + client_id',
        'model_initialization_rule': 'unchanged frozen SmallCNN initialized with overridden SEED'}
    base['prospective'] = identity
    base['runtime']['prospective_entry'] = identity.copy()
    base['prospective_primary_endpoint'] = 'client_macro_ce is client-macro NLL, natural logarithm'
    base['prospective_reference_scope'] = (
        'same new CIFAR partition, initialization/batch seed policy and shared uniform3 cap; '
        'plain FedRoD BSM+ERM changes learning objective, not codec; old accuracy STOP retained')
    base['prospective_metric_note'] = (
        'frozen entry emits accuracy and CE/NLL; Brier/ECE require the separate locked final evaluator')
    return base


def execute(args, entry=None):
    entry = load_frozen_entry() if entry is None else entry
    entry.torch.set_num_threads(1)
    with module_overrides(entry, args.seed, args.partition_seed):
        base = base_config(entry, args.backend, args.seed, args.partition_seed)
        saved_config, checkpoint = entry.preflight(args.out, base)
        if not args.execute_formal:
            result = {'status': 'DESIGN_ONLY', 'formal_training_executed': False,
                'raw_data_opened': False, 'train_adapter_imported': False,
                'resume_preflight': 'VALIDATED' if saved_config else 'NO_SAVED_STATE',
                'base_config_signature': entry.r.signature(base), 'base_config': base}
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return result
        # All existing training, gradients, shared-wire accounting and resume code is reused.
        entry.formal(args, base, saved_config, checkpoint)
    verify_sources()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--partition-seed', type=int, choices=(PARTITION_SEED,), required=True)
    parser.add_argument('--seed', type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument('--backend', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute-formal', action='store_true',
        help='explicitly launch the fixed prospective 20-round TRAIN run')
    args = parser.parse_args()
    if args.out is None:
        args.out = HERE / f'prospective_p{args.partition_seed}_s{args.seed}'
    execute(args)


if __name__ == '__main__':
    main()
