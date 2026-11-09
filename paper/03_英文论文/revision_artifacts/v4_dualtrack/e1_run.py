"""Fixed v4 alpha sweep. New outputs only; never opens official TEST bytes."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'route_a_20261105/baselines/adapter.py'
ALPHAS = (0.1, 0.5, 1.0)
SEEDS = {'cifar': (20261231, 20261232, 20261233),
         'fashion': (20261241, 20261242, 20261243)}
PARTITIONS = {'cifar': 20261217, 'fashion': 20261219}
VARIANTS = ('raw', 'zero_anchor_vector', 'prioranchored_vector')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b = load('_v4_baseline_helpers', BASE)


def now():
    return datetime.now(timezone.utc).isoformat()


def run_root(task, alpha, seed):
    return HERE / 'e1_runs' / task / ('alpha_' + str(alpha)) / ('s' + str(seed))


def sources(c):
    paths = [Path(__file__), HERE / 'e1_aggregate.py', HERE / 'PLAN.md', BASE,
             b.WORK / 'prior_vector/solver.py', b.WORK / 'calibration_baselines/solver.py']
    for task, seeds in SEEDS.items():
        paths.extend(b.original_manifest(task, seed)[1] for seed in seeds)
    return {**c.source_identities(), **{str(p): b.sha(p) for p in paths}}


def prepare(task, alpha):
    """Read TRAIN-only via frozen loader; alpha is the sole partition change."""
    c, _ = b.dependencies(task)
    if task == 'cifar':
        raw = b.PAPER / '02_实验工程/dataset/Cifar10/rawdata'
        images, labels, provenance, identities = c.load_train_only(raw)
        partitioner, dataset = c.partitions.partition_indices, c.partitions.IndexedCifar10
    else:
        wrapper = sys.modules['_baseline_fashion_adapter']
        raw = b.METHOD / 'scripts/datasets/FashionMNIST/raw'
        images, labels, provenance, identities = wrapper.load_train_only(raw)
        partitioner, dataset = wrapper.fashion.partition_indices, wrapper.fashion.FashionIndexDataset
    train_ids, holdout_ids, validation_ids, manifest = partitioner(
        labels, alpha=alpha, split_seed=PARTITIONS[task], mixing=0., validation_size=2000)
    manifest.update({'source': 'v4_E1_official_TRAIN_only', 'task': task, **provenance,
                     'official_test_policy': 'no official TEST bytes, inference, fitting, or reassignment',
                     'v4_design': 'one fixed partition seed per task/alpha; three model seeds; final20'})
    if task == 'fashion':
        manifest['preprocessing'] = 'uint8 gray28 -> pad2 -> repeat3 -> normalize[-1,1]'
    combined = np.concatenate(train_ids + holdout_ids)
    if (len(train_ids) != 20 or len(holdout_ids) != 20 or
            not np.array_equal(np.sort(combined), np.arange(len(labels))) or
            np.intersect1d(np.concatenate(train_ids), np.concatenate(holdout_ids)).size or
            len(validation_ids) != 2000 or
            np.isin(validation_ids, np.concatenate(train_ids)).any()):
        raise ValueError('invalid TRAIN/holdout/monitor membership')
    if alpha == .1:
        for seed in SEEDS[task]:
            expected, expected_path = b.original_manifest(task, seed)
            for key in ('train_indices', 'holdout_indices', 'validation_indices',
                        'client_label_counts', 'client_sizes'):
                if manifest[key] != expected[key]:
                    raise ValueError('alpha=.1 differs from original membership: ' + key)
            identities[str(expected_path)] = b.sha(expected_path)
    trainsets = [dataset(images, labels, ids) for ids in train_ids]
    # Monitoring rows stay in the original manifest but are excluded from E1 scoring.
    evaluation_ids = [np.setdiff1d(ids, validation_ids) for ids in holdout_ids]
    if any(len(ids) == 0 for ids in evaluation_ids):
        raise ValueError('client has no development holdout after monitor exclusion')
    evaluated = np.concatenate(evaluation_ids)
    if (np.intersect1d(evaluated, validation_ids).size or
            not np.array_equal(np.sort(np.concatenate([evaluated, validation_ids])),
                               np.sort(np.concatenate(holdout_ids)))):
        raise ValueError('scoring holdouts plus monitor must exactly cover original holdouts')
    holdouts = [dataset(images, labels, ids) for ids in evaluation_ids]
    validation = dataset(images, labels, validation_ids)
    return c, trainsets, holdouts, validation, manifest, identities


def skew(counts):
    counts = np.asarray(counts, dtype=np.float64)
    p = counts / counts.sum(axis=1, keepdims=True)
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    ent = -(p * logp).sum(axis=1) / np.log(10.)
    return {'maximum_class_fraction': p.max(axis=1).tolist(),
            'normalized_entropy': ent.tolist(),
            'mean_maximum_class_fraction': float(p.max(axis=1).mean()),
            'mean_normalized_entropy': float(ent.mean())}


def preflight():
    attempts = b.test_guard()
    torch.set_num_threads(1)
    if not (HERE / 'PLAN.md').is_file():
        raise ValueError('PLAN.md must be written before preflight')
    records, identities = [], {}
    for task in SEEDS:
        for alpha in ALPHAS:
            c, sets, holdouts, monitor, manifest, data_ids = prepare(task, alpha)
            identities.update(sources(c))
            identities.update(data_ids)
            records.append({'task': task, 'alpha': alpha, 'partition_seed': PARTITIONS[task],
                            'model_seeds': SEEDS[task], 'clients': len(sets),
                            'train_samples': sum(map(len, sets)),
                            'original_holdout_samples': sum(manifest['client_holdout_sizes']),
                            'evaluation_holdout_samples': sum(map(len, holdouts)),
                            'monitor_samples': len(monitor),
                            'min_client_train_samples': min(map(len, sets)),
                            'min_client_evaluation_holdout_samples': min(map(len, holdouts)),
                            'manifest_sha256': b.signed(manifest),
                            'original_membership_matched': True if alpha == .1 else None,
                            'skew': skew(manifest['client_label_counts'])})
    solver = load('_v4_preflight_solver', b.WORK / 'prior_vector/solver.py')
    if solver.base.RIDGE != 10:
        raise ValueError('fixed lambda=10 differs')
    rng = np.random.default_rng(20261106)
    z, y = rng.normal(size=(51, 10)), np.arange(51) % 10
    for anchor in (np.zeros(10), solver.train_prior_anchor(y)):
        state, _ = solver.fit(z, y, anchor)
        if state.shape != (20,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError('finite 20-FP32 state fixture failed')
    if attempts:
        raise ValueError('forbidden official TEST read attempted')
    b.verify(identities)
    lock_path = HERE / 'E1_SOURCE_LOCK.json'
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        if previous['inputs'] != identities:
            raise ValueError('existing E1 lock differs; inspect instead of overwriting')
    else:
        b.write(lock_path, {'status': 'FROZEN_BEFORE_E1_TRAINING', 'created_at': now(),
                            'inputs': identities, 'matrix_runs': 18})
    b.write(HERE / 'E1_PREFLIGHT.json', {'status': 'PASS', 'created_at': now(),
            'records': records, 'lambda': 10, 'state_fp32_bytes': 80,
            'official_test_reads': 0, 'forbidden_open_attempts': attempts,
            'mps_available_in_preflight_process': bool(torch.backends.mps.is_available()),
            'monitor_note': '2000 monitor rows are excluded from scored development holdouts; original partition manifest retained; final20 fixed, no selection'})
    print(json.dumps({'status': 'E1_PREFLIGHT_PASS', 'matrix_runs': 18, 'records': records}), flush=True)


def training_module(c):
    previous = sys.modules.get('common')
    sys.modules['common'] = c
    try:
        return load('_v4_frozen_training', b.WORK / 'calibrated_lowbit_confirm/train.py')
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous


def e2_gate():
    path = HERE / 'e2_skew_gain.json'
    if not path.exists() or json.loads(path.read_text()).get('status') != 'COMPLETE':
        raise ValueError('E2 must complete before E1 training')
    return {str(path): b.sha(path)}


def execute(task, alpha, seed):
    if alpha not in ALPHAS or seed not in SEEDS[task]:
        raise ValueError('only the prespecified 18-run matrix is allowed')
    attempts = b.test_guard()
    torch.set_num_threads(1)
    lock = json.loads((HERE / 'E1_SOURCE_LOCK.json').read_text())
    if lock.get('status') != 'FROZEN_BEFORE_E1_TRAINING':
        raise ValueError('preflight source lock required')
    b.verify(lock['inputs'])
    gating = e2_gate()
    if not torch.backends.mps.is_available():
        raise RuntimeError('original MPS backend unavailable; no silent fallback')
    c, sets, holdouts, monitor, manifest, data_ids = prepare(task, alpha)
    own_sources = sources(c)
    b.verify({**own_sources, **data_ids})
    out = run_root(task, alpha, seed)
    out.mkdir(parents=True, exist_ok=True)
    start_path = out / 'V4_DESIGN_LOCK.json'
    design = {'task': task, 'alpha': alpha, 'partition_seed': PARTITIONS[task], 'seed': seed,
              'rounds': 20, 'clients': 20, 'method': 'fedlc_uniform3', 'lambda': 10,
              'plan_sha256': b.sha(HERE / 'PLAN.md'), 'inputs': {**own_sources, **data_ids, **gating},
              'scope': 'TRAIN-fit / development holdout score excluding 2000 monitor rows; final20 fixed',
              'inherited_training_metadata_note': 'Frozen trainer selection/runtime labels describe its historical origin; this V4 design and alpha manifest specify the current experiment.',
              'no_selection': 'all 18 prespecified runs and both fitted variants retained; no TEST access'}
    if start_path.exists():
        if json.loads(start_path.read_text()) != design:
            raise ValueError('existing V4 design differs')
    else:
        b.write(start_path, design)
    started = time.perf_counter()
    b.write(out / 'STATUS.json', {'status': 'TRAINING', 'updated_at': now(),
                                 'task': task, 'alpha': alpha, 'seed': seed})
    args = argparse.Namespace(out=out, seed=seed, partition_seed=PARTITIONS[task], rounds=20, backend='mps')
    training_module(c).run(args, 'fedlc_uniform3', sets, monitor, manifest, data_ids)
    folder = out / 'fedlc_uniform3'
    config = json.loads((folder / 'config.json').read_text())
    ck = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model = c.validate_checkpoint(ck, config)
    model.eval()
    if len(ck['rows']) != 20 or config['data_manifest']['alpha'] != alpha:
        raise ValueError('only exact final20 checkpoint for the intended alpha may be scored')
    checkpoint_sha = b.sha(folder / 'checkpoint.pt')
    result_path = out / 'evaluation.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get('status') != 'COMPLETE' or old['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('existing evaluation is incompatible')
        b.verify(old['outputs'])
        b.verify(design['inputs'])
        b.write(out / 'STATUS.json', {'status': 'COMPLETE', 'updated_at': now(),
                                     'task': task, 'alpha': alpha, 'seed': seed,
                                     'delta_nll': old['prior_minus_zero_macro_nll']})
        print(json.dumps({'status': 'E1_RUN_ALREADY_COMPLETE', 'task': task, 'alpha': alpha, 'seed': seed}), flush=True)
        return
    b.write(out / 'STATUS.json', {'status': 'TRAIN_ONLY_FIT_AND_DEVELOPMENT_SCORE', 'updated_at': now(),
                                 'task': task, 'alpha': alpha, 'seed': seed})
    solver = load('_v4_vector_solver', b.WORK / 'prior_vector/solver.py')
    if solver.base.RIDGE != 10:
        raise ValueError('fixed lambda differs')
    states = out / 'client_arrays'
    states.mkdir(exist_ok=True)
    clients, outputs = [], {}
    eval_started = time.perf_counter()
    for client, (trainset, holdout) in enumerate(zip(sets, holdouts)):
        z, y = b.forward(model, trainset)
        if not np.array_equal(y, trainset.labels[trainset.indices]):
            raise ValueError('TRAIN row order changed')
        counts = np.bincount(y, minlength=10)
        if counts.tolist() != manifest['client_label_counts'][client]:
            raise ValueError('TRAIN counts differ')
        zero, zi = solver.fit(z, y, np.zeros(10))
        anchor = solver.train_prior_anchor(y)
        prior, pi = solver.fit(z, y, anchor)
        hz, hy = b.forward(model, holdout)
        if not np.array_equal(hy, holdout.labels[holdout.indices]):
            raise ValueError('holdout row order changed')
        if zero.nbytes != 80 or prior.nbytes != 80:
            raise ValueError('unexpected deployment state bytes')
        scores = {'raw': hz.astype(np.float64),
                  'zero_anchor_vector': solver.transform(hz.astype(np.float64), zero.astype(np.float64)),
                  'prioranchored_vector': solver.transform(hz.astype(np.float64), prior.astype(np.float64))}
        metrics = {variant: solver.stats(value, hy) for variant, value in scores.items()}
        for value in metrics.values():
            value['nll'] = value['loss']
        path = states / ('client' + str(client) + '.npz')
        np.savez_compressed(path, train_logits=z, train_labels=y,
                            train_ids=np.asarray(trainset.indices, dtype=np.uint32),
                            train_counts=counts, holdout_logits=hz, holdout_labels=hy,
                            holdout_ids=np.asarray(holdout.indices, dtype=np.uint32),
                            original_holdout_ids=np.asarray(manifest['holdout_indices'][client], dtype=np.uint32),
                            zero_anchor_vector=zero, prioranchored_vector=prior, prior_anchor=anchor)
        outputs[str(path)] = b.sha(path)
        delta = metrics['prioranchored_vector']['nll'] - metrics['zero_anchor_vector']['nll']
        clients.append({'client': client, 'train_samples': len(trainset),
                        'holdout_samples': len(holdout), 'evaluation_holdout_count': len(holdout),
                        'original_holdout_count': len(manifest['holdout_indices'][client]), 'metrics': metrics,
                        'prior_minus_zero_nll': delta, 'zero_minus_prior_nll_gain': -delta,
                        'train_counts': counts.tolist(), 'zero_fit': zi, 'prior_fit': pi})
    summaries = solver.summarize(clients)
    delta = summaries['prioranchored_vector']['macro_loss'] - summaries['zero_anchor_vector']['macro_loss']
    b.verify(design['inputs'])
    b.verify(lock['inputs'])
    if checkpoint_sha != b.sha(folder / 'checkpoint.pt') or attempts:
        raise ValueError('checkpoint changed or official TEST access attempted')
    result = {'status': 'COMPLETE', 'created_at': now(), 'task': task, 'alpha': alpha,
              'partition_seed': PARTITIONS[task], 'seed': seed, 'rounds': 20, 'clients': clients,
              'lambda': 10, 'method': 'FedLC-inspired tau1 + actual uniform3 / error feedback',
              'scope': design['scope'], 'summaries': summaries,
              'prior_minus_zero_macro_nll': delta, 'zero_minus_prior_macro_nll_gain': -delta,
              'skew': skew(manifest['client_label_counts']), 'outputs': outputs,
              'checkpoint_sha256': checkpoint_sha, 'inputs': design['inputs'],
              'design_lock_sha256': b.sha(start_path), 'official_test_reads': 0,
              'official_test_reassignments': 0, 'forbidden_open_attempts': attempts,
              'training_and_evaluation_seconds': time.perf_counter() - started,
              'evaluation_seconds': time.perf_counter() - eval_started,
              'original_holdout_count': sum(manifest['client_holdout_sizes']),
              'evaluation_holdout_count': sum(map(len, holdouts)),
              'monitor_count': len(monitor),
              'development_note': '2000 global monitoring rows excluded from scoring. Original TRAIN/holdout membership manifest retained. Neither holdout nor monitor is used by vector fitting or final-round selection.'}
    b.write(result_path, result)
    b.write(out / 'STATUS.json', {'status': 'COMPLETE', 'updated_at': now(), 'task': task,
                                 'alpha': alpha, 'seed': seed, 'delta_nll': delta})
    print(json.dumps({'status': 'E1_RUN_COMPLETE', 'task': task, 'alpha': alpha,
                      'seed': seed, 'delta_nll': delta, 'gain_nll': -delta,
                      'evaluation_seconds': result['evaluation_seconds']}), flush=True)


def matrix():
    e2_gate()
    completed = []
    status_path = HERE / 'E1_STATUS.json'
    for task in SEEDS:
        for alpha in ALPHAS:
            for seed in SEEDS[task]:
                current = {'task': task, 'alpha': alpha, 'seed': seed}
                b.write(status_path, {'status': 'RUNNING', 'updated_at': now(), 'completed_runs': completed,
                                      'current_run': current, 'total_runs': 18})
                cmd = [sys.executable, '-B', str(Path(__file__)), '--stage', 'run', '--task', task,
                       '--alpha', str(alpha), '--seed', str(seed)]
                code = subprocess.run(cmd, check=False).returncode
                if code:
                    b.write(status_path, {'status': 'FAILED', 'updated_at': now(), 'completed_runs': completed,
                                          'current_run': current, 'exit_code': code, 'total_runs': 18})
                    raise RuntimeError('E1 run failed; earlier checkpoints retained: ' + str(current))
                completed.append(current)
    subprocess.run([sys.executable, '-B', str(HERE / 'e1_aggregate.py')], check=True)
    b.write(status_path, {'status': 'COMPLETE', 'updated_at': now(), 'completed_runs': completed, 'total_runs': 18})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--stage', choices=('preflight', 'run', 'matrix'), required=True)
    parser.add_argument('--task', choices=SEEDS)
    parser.add_argument('--alpha', type=float, choices=ALPHAS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.stage == 'preflight':
        preflight()
    elif args.stage == 'matrix':
        matrix()
    elif args.task is not None and args.alpha is not None and args.seed is not None:
        execute(args.task, args.alpha, args.seed)
    else:
        parser.error('run requires --task, --alpha and --seed')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
