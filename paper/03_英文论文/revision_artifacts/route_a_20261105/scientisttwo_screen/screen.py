"""Predeclared fail-fast count-geometry screen on saved development arrays."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
import tracemalloc

import numpy as np
import solver

HERE = Path(__file__).resolve().parent
METHOD = Path('/Users/wind/Desktop/Codex/paper/05_核心方法研发')
TASKS = {'cifar': (20261217, (20261231, 20261232, 20261233)),
         'fashion': (20261219, (20261241, 20261242, 20261243))}
ARMS = ('prior10', 'zero10', 'geometry', 'permuted_geometry', 'count_ridge', 'isotropic1')
EPSILON = 1e-7
attempts = []


def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = str(args[0])
        if any(v in p for v in ('predictive_final_test', 'test_batch', 't10k', '/dataset/', '/datasets/', 'checkpoint.pt', 'multipartition')):
            attempts.append(p); raise ValueError('saved original-development arrays only')
sys.addaudithook(audit)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def verify(values):
    if any(sha(p) != h for p, h in values.items()): raise ValueError('locked source/development input changed')


def roots(task, seed):
    p = TASKS[task][0]; base = METHOD / 'results/predictive_pfl_confirm'
    return base / f'evaluation_{task}_p{p}_s{seed}', base / f'adaptation_{task}_p{p}_s{seed}'


def metrics(logits, labels):
    z = np.asarray(logits, dtype=np.float64); y = np.asarray(labels, dtype=np.int64)
    maxima = z.max(axis=1); ex = np.exp(z - maxima[:, None]); prob = ex / ex.sum(axis=1, keepdims=True)
    nll = maxima + np.log(ex.sum(axis=1)) - z[np.arange(len(y)), y]
    correct = prob.argmax(axis=1) == y; conf = prob.max(axis=1)
    error = prob.copy(); error[np.arange(len(y)), y] -= 1
    bins = np.minimum((conf * 15).astype(np.int64), 14)
    conf_sum = np.bincount(bins, weights=conf, minlength=15)
    correct_sum = np.bincount(bins, weights=correct.astype(np.float64), minlength=15)
    return {'accuracy': float(correct.mean()), 'nll': float(nll.mean()), 'brier': float(np.mean(np.sum(error**2, axis=1))),
            'ece15': float(np.abs(correct_sum - conf_sum).sum() / len(y)),
            'conf_sum': conf_sum.tolist(), 'correct_sum': correct_sum.tolist()}


def transform(z, state):
    p = state.astype(np.float64); return np.asarray(z, dtype=np.float64) * p[:10] + p[10:]


def summarize(clients):
    weights = np.asarray([r['holdout_samples'] for r in clients], dtype=np.float64); total = weights.sum(); weights /= total
    result = {}
    for arm in ARMS:
        row = {}
        for metric in ('nll', 'accuracy', 'brier', 'ece15'):
            values = np.asarray([r['metrics'][arm][metric] for r in clients])
            row['macro_' + metric] = float(values.mean()); row['weighted_' + metric] = float(values @ weights)
        row['client_accuracy_q10'] = float(np.quantile([r['metrics'][arm]['accuracy'] for r in clients], .1))
        conf = np.sum([r['metrics'][arm]['conf_sum'] for r in clients], axis=0)
        correct = np.sum([r['metrics'][arm]['correct_sum'] for r in clients], axis=0)
        row['pooled_ece15'] = float(np.abs(conf - correct).sum() / total); result[arm] = row
    return result


def freeze():
    path = HERE / 'SOURCE_LOCK.json'
    if path.exists(): raise ValueError('never overwrite pre-screen source lock')
    values = {}
    sources = [HERE / name for name in ('PLAN.md', 'solver.py', 'screen.py', 'fixtures.py', 'FIXTURES.json')]
    sources += [solver.SOURCE, METHOD / 'scripts/calibration_baselines/solver.py']
    for source in sources: values[str(source)] = sha(source)
    # Hash all declared old files before scores; no new-partition or TEST access.
    for task, (_, seeds) in TASKS.items():
        for seed in seeds:
            ef, af = roots(task, seed)
            values[str(af / 'result.json')] = sha(af / 'result.json')
            for i in range(20):
                for folder in (ef, af):
                    source = folder / f'fedlc_uniform3_client{i}.npz'; values[str(source)] = sha(source)
    fixture = json.loads((HERE / 'FIXTURES.json').read_text())
    if fixture['status'] != 'PASS_SYNTHETIC_FIXTURES': raise ValueError('fixtures must pass')
    write(path, {'status': 'FROZEN_BEFORE_CANDIDATE_FITS', 'created_at': datetime.now(timezone.utc).isoformat(),
                 'inputs': values, 'official_test_access': False, 'new_partition_outputs_read': False,
                 'epsilon_nll': EPSILON, 'subset_first_seeds': {task: seeds[0] for task, (_, seeds) in TASKS.items()}})
    print(json.dumps({'status': 'SCREEN_FROZEN', 'inputs': len(values)}), flush=True)


def run_one(task, seed, locked):
    out = HERE / 'runs' / task / f's{seed}'
    if out.exists(): raise ValueError('fresh candidate run required')
    ef, af = roots(task, seed); old_result = json.loads((af / 'result.json').read_text())
    out.mkdir(parents=True); clients = []
    for client in range(20):
        with np.load(ef / f'fedlc_uniform3_client{client}.npz', allow_pickle=False) as saved:
            data = {name: saved[name].copy() for name in ('train_logits', 'train_labels', 'train_counts', 'raw', 'labels', 'train_ids', 'holdout_ids')}
        with np.load(af / f'fedlc_uniform3_client{client}.npz', allow_pickle=False) as saved:
            old = {name: saved[name].copy() for name in ('prioranchored_vector', 'zero_anchor_vector', 'train_bias_anchor', 'train_ids', 'holdout_ids')}
        if not np.array_equal(data['train_ids'], old['train_ids']) or not np.array_equal(data['holdout_ids'], old['holdout_ids']):
            raise ValueError('paired development IDs differ')
        labels = data['train_labels']; counts = np.bincount(labels, minlength=10)
        if not np.array_equal(counts, data['train_counts']): raise ValueError('saved TRAIN counts differ')
        anchor = solver.frozen.train_prior_anchor(labels)
        if not np.array_equal(anchor, old['train_bias_anchor']): raise ValueError('prior anchor differs')
        # Real-array exact regression of both unchanged reference families.
        prior, prior_info = solver.frozen.fit(data['train_logits'], labels, anchor)
        zero, zero_info = solver.frozen.fit(data['train_logits'], labels, np.zeros(10))
        if not np.array_equal(prior, old['prioranchored_vector']) or not np.array_equal(zero, old['zero_anchor_vector']):
            raise ValueError('original state replay differs')
        iso, iso_info = solver.fit(data['train_logits'], labels, anchor, solver.H)
        if not np.array_equal(iso, prior): raise ValueError('real-data isotropic FP32 limit differs')
        states = {'prior10': prior, 'zero10': zero}; fits = {'prior10': prior_info, 'zero10': zero_info}
        matrices, covariance = solver.geometries(counts)
        for arm, matrix in matrices.items():
            tracemalloc.start(); began = time.perf_counter()
            state, info = solver.fit(data['train_logits'], labels, anchor, matrix)
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            states[arm] = state; fits[arm] = {**info, 'python_traced_peak_bytes_entire_fit': peak, 'wall_seconds': time.perf_counter() - began}
        state, info = solver.fit(data['train_logits'], labels, anchor, solver.H, 1., 1.)
        states['isotropic1'] = state; fits['isotropic1'] = info
        measured = {arm: metrics(transform(data['raw'], states[arm]), data['labels']) for arm in ARMS}
        for arm, old_name in (('prior10', 'prioranchored_vector'), ('zero10', 'zero_anchor_vector')):
            expected = old_result['results']['fedlc_uniform3']['clients'][client]['metrics'][old_name]
            if any(abs(measured[arm][key] - expected[key]) > 1e-12 for key in ('nll', 'accuracy', 'brier', 'ece15')):
                raise ValueError('original development score replay differs')
        pi = (counts + 1.) / (counts.sum() + 10.)
        np.savez_compressed(out / f'client{client}.npz', **states, **{'precision_' + k: v for k, v in matrices.items()},
                            covariance=covariance, train_counts=counts, train_ids=data['train_ids'], holdout_ids=data['holdout_ids'])
        clients.append({'client': client, 'train_samples': len(labels), 'holdout_samples': len(data['labels']),
                        'effective_smoothed_class_support': float(np.exp(-(pi * np.log(pi)).sum())),
                        'counts': counts.tolist(), 'precision_normalized_trace': {k: float(np.trace(v)) for k, v in matrices.items()},
                        'geometry_count_max_matrix_difference': float(np.max(np.abs(matrices['geometry'] - matrices['count_ridge']))),
                        'original_fp32_replay_exact': True, 'metrics': measured, 'fits': fits})
        print(json.dumps({'task': task, 'seed': seed, 'client': client, 'status': 'CLIENT_DONE'}), flush=True)
    verify(locked['inputs'])
    report = {'status': 'DONE', 'task': task, 'seed': seed, 'clients': clients, 'summaries': summarize(clients),
              'scope': 'SAVED_DEVELOPMENT_ONLY', 'official_test_reads': 0, 'new_partition_output_reads': 0,
              'input_hashes_unchanged': True, 'all_original_states_and_metrics_replayed': True}
    write(out / 'result.json', report)
    return report


def gate(reports):
    checks = {}; contrasts = {}
    for task, report in reports.items():
        nll = {a: report['summaries'][a]['macro_nll'] for a in ARMS}
        contrasts[task] = {a + '_minus_prior10': nll[a] - nll['prior10'] for a in ARMS if a != 'prior10'}
        contrasts[task].update(geometry_minus_permuted=nll['geometry'] - nll['permuted_geometry'],
                               geometry_minus_count=nll['geometry'] - nll['count_ridge'],
                               geometry_minus_isotropic1=nll['geometry'] - nll['isotropic1'])
        checks[task + '_beats_prior10'] = nll['geometry'] < nll['prior10'] - EPSILON
        checks[task + '_beats_permuted'] = nll['geometry'] < nll['permuted_geometry'] - EPSILON
    for rival in ('count_ridge', 'isotropic1'):
        dominated = all(r['summaries'][rival]['macro_nll'] <= r['summaries']['geometry']['macro_nll'] + EPSILON for r in reports.values())
        checks['not_dominated_by_' + rival] = not dominated
    equivalent = all(abs(r['summaries']['geometry']['macro_nll'] - r['summaries']['count_ridge']['macro_nll']) <= EPSILON for r in reports.values())
    checks['not_equivalent_to_count_ridge'] = not equivalent
    return {'status': 'PASS_TO_THREE_SEED_DEVELOPMENT' if all(checks.values()) else 'STOP_CANDIDATE',
            'checks': checks, 'contrasts': contrasts, 'epsilon_nll': EPSILON,
            'scope': 'first-seed fail-fast; no novelty/acceptance claim'}


def run(stage):
    locked = json.loads((HERE / 'SOURCE_LOCK.json').read_text()); verify(locked['inputs'])
    if stage == 'subset':
        reports = {task: run_one(task, seeds[0], locked) for task, (_, seeds) in TASKS.items()}
        result = gate(reports)
        result.update(tasks={task: r['summaries'] for task, r in reports.items()}, forbidden_open_attempts=attempts,
                      actual_new_fits=160, fitted_clients=40, old_official_test_unchanged=True)
        write(HERE / 'SUBSET_RESULT.json', result)
    else:
        previous = json.loads((HERE / 'SUBSET_RESULT.json').read_text())
        if previous['status'] != 'PASS_TO_THREE_SEED_DEVELOPMENT': raise ValueError('candidate stopped; no extension')
        extra = [run_one(task, seed, locked) for task, (_, seeds) in TASKS.items() for seed in seeds[1:]]
        write(HERE / 'EXTENSION_RESULT.json', {'status': 'DONE_DEVELOPMENT_CONFIRMATION',
                                             'runs': [{k: r[k] for k in ('task', 'seed', 'summaries')} for r in extra],
                                             'scope': 'development only; all declared seeds; no automatic promotion'})
        result = {'status': 'DONE_DEVELOPMENT_CONFIRMATION'}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage', choices=('freeze', 'subset', 'extend'), required=True)
    args = parser.parse_args()
    if args.stage == 'freeze': freeze()
    else: run(args.stage)
