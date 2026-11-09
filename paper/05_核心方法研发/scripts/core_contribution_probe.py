"""One fixed class-permutation control on saved development logits only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
SAVED = BASE / 'results/predictive_pfl_confirm'
PLAN = BASE / 'design/core_contribution/PLAN.md'
OUT = BASE / 'results/core_contribution_probe/result.json'
TASKS = {'cifar': (20261217, (20261231, 20261232, 20261233)),
         'fashion': (20261219, (20261241, 20261242, 20261243))}
VARIANTS = ('correct', 'zero', 'permuted')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(z, state):
    state = np.asarray(state, dtype=np.float64)
    return np.asarray(z, dtype=np.float64) * state[:10] + state[10:]


def metrics(z, y, absent):
    z = np.asarray(z, dtype=np.float64)
    maximum = z.max(axis=1)
    exps = np.exp(z - maximum[:, None])
    probabilities = exps / exps.sum(axis=1, keepdims=True)
    nll = maximum + np.log(exps.sum(axis=1)) - z[np.arange(len(y)), y]
    return {'nll': float(nll.mean()), 'accuracy': float(np.mean(z.argmax(1) == y)),
            'mass_on_train_absent_classes': float(probabilities[:, absent].sum(1).mean())}


def main():
    if OUT.exists():
        raise ValueError('Never overwrite a completed diagnostic')
    forbidden = []
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = str(arguments[0])
            if any(part in path for part in ('/dataset/', '/datasets/', 'test_batch', 't10k', '/predictive_final_test/')):
                forbidden.append(path)
                raise ValueError('Saved development arrays only; original data and official TEST forbidden')
    sys.addaudithook(audit)
    source = BASE / 'scripts/prior_vector/solver.py'
    spec = importlib.util.spec_from_file_location('_core_frozen_vector', source)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    identities = {str(p.relative_to(BASE)): sha(p) for p in (
        Path(__file__), PLAN, source, BASE/'scripts/calibration_baselines/solver.py')}
    started = time.perf_counter()
    result = {'status': 'RUNNING', 'analysis_type': 'exploratory development mechanism diagnostic',
              'declared_plan_sha256': sha(PLAN), 'permutation': 'np.roll(mu, 1), exactly one fixed permutation',
              'tasks': {}, 'new_vector_fits': 0, 'new_cnn_forwards': 0, 'new_cnn_backwards': 0,
              'new_training_seeds': 0, 'official_test_reads': 0}
    for task, (partition, seeds) in TASKS.items():
        runs = []
        for seed in seeds:
            evaluation = SAVED / f'evaluation_{task}_p{partition}_s{seed}'
            adaptation = SAVED / f'adaptation_{task}_p{partition}_s{seed}'
            reports = {}
            for kind, folder in [('evaluation', evaluation), ('adaptation', adaptation)]:
                path = folder / 'result.json'
                identities[str(path.relative_to(BASE))] = sha(path)
                reports[kind] = json.loads(path.read_text())
                assert reports[kind]['status'] == 'DONE'
            clients = []
            for client in range(20):
                paths = [folder/f'fedlc_uniform3_client{client}.npz' for folder in (evaluation, adaptation)]
                for path in paths:
                    identities[str(path.relative_to(BASE))] = sha(path)
                with np.load(paths[0], allow_pickle=False) as d:
                    data = {key: d[key].copy() for key in d.files}
                with np.load(paths[1], allow_pickle=False) as d:
                    original = {key: d[key].copy() for key in d.files}
                assert np.intersect1d(data['train_ids'], data['holdout_ids']).size == 0
                assert np.array_equal(data['train_ids'], original['train_ids'])
                assert np.array_equal(data['holdout_ids'], original['holdout_ids'])
                counts = np.bincount(data['train_labels'], minlength=10)
                assert np.array_equal(counts, data['train_counts'])
                mu = solver.train_prior_anchor(data['train_labels'])
                assert np.array_equal(mu, original['train_bias_anchor'])
                bad = np.roll(mu, 1)
                assert np.array_equal(np.sort(mu), np.sort(bad))
                state, diagnostic = solver.fit(data['train_logits'], data['train_labels'], bad)
                result['new_vector_fits'] += 1
                states = {'correct': original['prioranchored_vector'], 'zero': original['zero_anchor_vector'], 'permuted': state}
                absent = counts == 0
                measured = {split: {kind: metrics(scores(data[z], value), data[y], absent)
                            for kind, value in states.items()}
                            for split, z, y in [('train', 'train_logits', 'train_labels'), ('holdout', 'raw', 'labels')]}
                for kind, old in [('correct', 'prioranchored_vector'), ('zero', 'zero_anchor_vector')]:
                    expected = reports['adaptation']['results']['fedlc_uniform3']['clients'][client]['metrics'][old]
                    assert abs(measured['holdout'][kind]['nll'] - expected['nll']) < 1e-12
                    assert abs(measured['holdout'][kind]['accuracy'] - expected['accuracy']) < 1e-12
                    assert np.array_equal(scores(data['raw'], states[kind]), original[old+'_holdout_logits'])
                clients.append({'client': client, 'train_samples': len(data['train_labels']),
                    'holdout_samples': len(data['labels']), 'train_absent_classes': int(absent.sum()),
                    'holdout_label_absent_from_train_fraction': float(np.mean(absent[data['labels']])),
                    'metrics': measured, 'permuted_deployed_fp32_state': state.tolist(),
                    'permuted_anchor': bad.tolist(), 'fit': diagnostic})
            summaries = {split: {kind: {metric: float(np.mean([c['metrics'][split][kind][metric] for c in clients]))
                         for metric in ('nll', 'accuracy', 'mass_on_train_absent_classes')}
                         for kind in VARIANTS} for split in ('train', 'holdout')}
            wins = {kind: sum(c['metrics']['holdout']['correct']['nll'] < c['metrics']['holdout'][kind]['nll'] for c in clients)
                    for kind in ('zero', 'permuted')}
            runs.append({'training_seed': seed, 'partition_seed': partition, 'summaries': summaries,
                         'client_nll_improvements_correct_vs': wins, 'clients': clients})
            print(json.dumps({'task': task, 'seed': seed, 'holdout': summaries['holdout'], 'client_wins': wins}), flush=True)
        aggregated = {kind: {metric: {'mean': float(np.mean([r['summaries']['holdout'][kind][metric] for r in runs])),
                            'sample_sd': float(np.std([r['summaries']['holdout'][kind][metric] for r in runs], ddof=1))}
                            for metric in ('nll', 'accuracy', 'mass_on_train_absent_classes')} for kind in VARIANTS}
        result['tasks'][task] = {'runs': runs, 'holdout_three_seed_summary': aggregated,
            'nll_reduction_correct_vs': {kind: float(np.mean([r['summaries']['holdout'][kind]['nll'] - r['summaries']['holdout']['correct']['nll'] for r in runs])) for kind in ('zero', 'permuted')}}
    assert all(sha(BASE/relative) == expected for relative, expected in identities.items())
    result.update(status='DONE', completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  inputs_and_sources_sha256=identities, input_bytes_unchanged=True,
                  forbidden_open_attempts=forbidden, elapsed_seconds=time.perf_counter()-started,
                  interpretation_scope='The fixed class-alignment contrast on existing development data; no new final TEST, new model choice or independent client-replicate claim.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'DONE', 'new_fits': result['new_vector_fits'], 'elapsed_seconds': result['elapsed_seconds'],
                      'summary': {task: r['nll_reduction_correct_vs'] for task, r in result['tasks'].items()}}))


if __name__ == '__main__':
    main()
