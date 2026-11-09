"""Isolated multi-partition extension using unchanged main training and solver."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import sys
import time
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / 'baselines/adapter.py'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

b = load('_multipartition_baseline_dependencies', BASE_PATH)
PARTITIONS = (20261105, 20261106)
SEEDS = {'cifar': (20261231, 20261232, 20261233),
         'fashion': (20261241, 20261242, 20261243)}

def source_ids(c):
    paths = [Path(__file__), HERE / 'PLAN.md', BASE_PATH,
             b.WORK / 'prior_vector/solver.py', b.WORK / 'calibration_baselines/solver.py']
    return {**c.source_identities(), **{str(p): b.sha(p) for p in paths}}

def run_root(task, partition, seed):
    return HERE / 'runs' / task / f'p{partition}_s{seed}'

def training_module(c):
    old = sys.modules.get('common')
    sys.modules['common'] = c
    try:
        return load('_multipartition_frozen_training', b.WORK / 'calibrated_lowbit_confirm/train.py')
    finally:
        if old is None: sys.modules.pop('common', None)
        else: sys.modules['common'] = old

def preflight():
    attempts = b.test_guard(); torch.set_num_threads(1)
    records = []; sources = {}
    for task in SEEDS:
        c, prepare = b.dependencies(task)
        _, _, original, _, _, _ = prepare(b.TASKS[task][0])
        expected, _ = b.original_manifest(task, SEEDS[task][0])
        for key in ('train_indices', 'holdout_indices', 'validation_indices', 'client_label_counts', 'client_sizes'):
            if original[key] != expected[key]: raise ValueError('original protocol membership mismatch: '+key)
        for partition in PARTITIONS:
            sets, validation, manifest, ids, _, labels = prepare(partition)
            train = np.concatenate(manifest['train_indices'])
            holdout = np.concatenate(manifest['holdout_indices'])
            if len(sets) != 20 or np.intersect1d(train, holdout).size: raise ValueError('invalid new membership')
            if not np.array_equal(np.sort(np.concatenate([train, holdout])), np.arange(len(labels))):
                raise ValueError('new membership fails exact coverage')
            if len(validation) != 2000: raise ValueError('validation protocol mismatch')
            records.append({'task': task, 'partition_seed': partition, 'clients': 20,
                            'train_samples': len(train), 'holdout_samples': len(holdout),
                            'validation_samples': len(validation), 'seeds': SEEDS[task],
                            'original_protocol_membership_match': True,
                            'manifest_sha256': b.signed(manifest), 'source_and_data': {**source_ids(c), **ids}})
            b.verify(ids)
        sources.update(source_ids(c))
    solver = load('_multipartition_solver_fixture', b.WORK / 'prior_vector/solver.py')
    if solver.base.RIDGE != 10: raise ValueError('frozen lambda differs')
    rng = np.random.default_rng(20261105); z = rng.normal(size=(51, 10)); y = np.arange(51) % 10
    state, info = solver.fit(z, y, solver.train_prior_anchor(y))
    if state.dtype != np.float32 or state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError('finite FP32 fixture failed')
    b.write(HERE / 'PREFLIGHT.json', {'status': 'PASS', 'records': records, 'lambda': 10,
                                    'fp32_solver_fixture': True, 'official_test_reads': 0,
                                    'forbidden_open_attempts': attempts})
    b.write(HERE / 'SOURCE_LOCK.json', {'status': 'FROZEN_BEFORE_MULTIPARTITION_TRAINING',
                                      'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': sources})
    print(json.dumps({'status': 'PREFLIGHT_PASS', 'new_runs': 12, 'records': len(records)}), flush=True)

def execute(task, partition, seed):
    if partition not in PARTITIONS or seed not in SEEDS[task]: raise ValueError('only fixed matrix allowed')
    attempts = b.test_guard(); torch.set_num_threads(1)
    lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text()); b.verify(lock['inputs'])
    if not torch.backends.mps.is_available(): raise RuntimeError('original MPS backend unavailable; no fallback')
    c, prepare = b.dependencies(task)
    sets, validation, manifest, ids, images, labels = prepare(partition)
    source = source_ids(c); b.verify(source)
    out = run_root(task, partition, seed)
    args = argparse.Namespace(out=out, seed=seed, partition_seed=partition, rounds=20, backend='mps')
    train = training_module(c)
    train.run(args, 'fedlc_uniform3', sets, validation, manifest, ids)
    folder = out / 'fedlc_uniform3'
    config = json.loads((folder / 'config.json').read_text())
    ck = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model = c.validate_checkpoint(ck, config); model.eval()
    if len(ck['rows']) != 20: raise ValueError('only complete final20 checkpoint can be scored')
    result_path = out / 'evaluation.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get('status') != 'DONE' or old['checkpoint_sha256'] != b.sha(folder/'checkpoint.pt'):
            raise ValueError('existing evaluation mismatch')
        print(json.dumps({'status': 'ALREADY_EVALUATED', 'task': task, 'partition': partition, 'seed': seed}), flush=True)
        return
    solver = load('_multipartition_vector_solver', b.WORK / 'prior_vector/solver.py')
    if solver.base.RIDGE != 10: raise ValueError('fixed coefficient changed')
    holdouts = [c.partitions.IndexedCifar10(images, labels, idx) for idx in manifest['holdout_indices']]
    clients = []; before = b.sha(folder/'checkpoint.pt'); began = time.perf_counter()
    states = out / 'states'; states.mkdir(exist_ok=True)
    for i, (ds, hs) in enumerate(zip(sets, holdouts)):
        z, y = b.forward(model, ds)
        zero, zi = solver.fit(z, y, np.zeros(10))
        prior, pi = solver.fit(z, y, solver.train_prior_anchor(y))
        hz, hy = b.forward(model, hs)
        scores = {'zero_anchor_vector': solver.transform(hz.astype(np.float64), zero.astype(np.float64)),
                  'prioranchored_vector': solver.transform(hz.astype(np.float64), prior.astype(np.float64))}
        metrics = {key: solver.stats(value, hy) for key, value in scores.items()}
        np.savez_compressed(states/f'client{i}.npz', zero_anchor_vector=zero, prioranchored_vector=prior,
                            train_ids=np.asarray(ds.indices, dtype=np.uint32),
                            holdout_ids=np.asarray(hs.indices, dtype=np.uint32))
        clients.append({'client': i, 'train_samples': len(ds), 'holdout_samples': len(hs),
                        'metrics': metrics, 'zero_fit': zi, 'prior_fit': pi})
    b.verify({**source, **ids}); b.verify(lock['inputs'])
    if before != b.sha(folder/'checkpoint.pt') or attempts: raise ValueError('mutated model or TEST read')
    summaries = solver.summarize(clients)
    delta = summaries['prioranchored_vector']['macro_loss'] - summaries['zero_anchor_vector']['macro_loss']
    b.write(result_path, {'status': 'DONE', 'task': task, 'partition_seed': partition, 'seed': seed,
                         'scope': 'development TRAIN-fit/holdout-score, no official TEST',
                         'method': 'fedlc_uniform3', 'rounds': 20, 'lambda': 10, 'backend': 'mps',
                         'clients': clients, 'summaries': summaries, 'prior_minus_zero_macro_nll': delta,
                         'checkpoint_sha256': before, 'source_and_data_identities': {**source, **ids},
                         'identities_unchanged': True, 'official_test_reads': 0,
                         'evaluation_seconds': time.perf_counter()-began})
    print(json.dumps({'status': 'RUN_DONE', 'task': task, 'partition': partition, 'seed': seed,
                      'summaries': summaries, 'delta_nll': delta}), flush=True)

def summarize():
    rows = []
    for task in SEEDS:
        for partition in PARTITIONS:
            values = [json.loads((run_root(task, partition, seed)/'evaluation.json').read_text()) for seed in SEEDS[task]]
            if any(v['status'] != 'DONE' for v in values): raise ValueError('all runs required')
            metrics = {}
            for variant in ('zero_anchor_vector', 'prioranchored_vector'):
                metrics[variant] = {key: {'mean': float(np.mean([v['summaries'][variant][key] for v in values])),
                                         'sample_sd': float(np.std([v['summaries'][variant][key] for v in values], ddof=1))}
                                    for key in ('macro_loss', 'macro_accuracy', 'weighted_loss', 'weighted_accuracy')}
            ds = [v['prior_minus_zero_macro_nll'] for v in values]
            rows.append({'task': task, 'partition_seed': partition, 'seeds': SEEDS[task], 'metrics': metrics,
                         'delta_nll_mean': float(np.mean(ds)), 'delta_nll_sample_sd': float(np.std(ds, ddof=1)),
                         'individual_seed_deltas': ds, 'all_three_seeds_improve': all(x < 0 for x in ds)})
    b.write(HERE/'result.json', {'status': 'DONE', 'scope': 'all twelve prespecified new development runs',
                                'partitions': PARTITIONS, 'rows': rows, 'official_test_reads': 0})
    print(json.dumps({'status': 'ALL_DONE', 'rows': rows}), flush=True)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['preflight','run','summarize'], required=True)
    p.add_argument('--task', choices=SEEDS); p.add_argument('--partition',type=int); p.add_argument('--seed',type=int)
    a = p.parse_args()
    if a.stage == 'preflight': preflight()
    elif a.stage == 'summarize': summarize()
    elif a.task and a.partition and a.seed: execute(a.task, a.partition, a.seed)
    else: p.error('run requires task/partition/seed')

if __name__ == '__main__': main()
