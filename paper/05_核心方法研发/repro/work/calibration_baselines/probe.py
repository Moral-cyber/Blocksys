"""Fit fixed calibration baselines on saved TRAIN arrays; score holdout last."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import solver

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_INPUT = HERE.parent / 'prior_bias/diagnostic_v1'
OLD_MODELS = ('ce_uniform3', 'fedlc_uniform3', 'balanced_uniform3', 'balanced_dense')
CONFIRM_MODELS = ('ce_uniform3', 'fedlc_uniform3', 'fedlc_dense')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_saved(path):
    with np.load(path, allow_pickle=False) as source:
        saved = {name: source[name].copy() for name in source.files}
    old = 'holdout_raw' in saved
    result = {'train_logits': saved['train_logits'], 'train_labels': saved['train_labels'],
        'train_ids': saved['train_ids'], 'raw': saved['holdout_raw'] if old else saved['raw'],
        'labels': saved['holdout_labels'] if old else saved['labels'],
        'holdout_ids': saved['holdout_ids'], 'prior': saved['prior_shift'],
        'prioranchor_bias': saved['fitted_bias'],
        'learned': saved.get('holdout_learned') if old else saved.get('learned')}
    if (result['train_logits'].shape != (len(result['train_labels']), 10)
            or result['raw'].shape != (len(result['labels']), 10)
            or len(result['train_ids']) != len(result['train_labels'])
            or len(result['holdout_ids']) != len(result['labels'])
            or len(np.unique(result['train_ids'])) != len(result['train_ids'])
            or len(np.unique(result['holdout_ids'])) != len(result['holdout_ids'])
            or np.intersect1d(result['train_ids'], result['holdout_ids']).size):
        raise ValueError('saved TRAIN/holdout array shape or membership mismatch')
    counts = np.bincount(result['train_labels'], minlength=10)
    expected_prior = np.log((counts + 1.) / (len(result['train_labels']) + 10)).astype(np.float32)
    if not np.array_equal(expected_prior, result['prior']):
        raise ValueError('saved prior not generated from saved TRAIN counts')
    return result, sum(value.nbytes for value in saved.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--out', type=Path, default=HERE / 'diagnostic_v1')
    args = parser.parse_args()
    input_root, out = args.input_root.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError('fresh calibration output required')
    forbidden_opens = []
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = str(arguments[0])
            if '/dataset/' in path or 'test_batch' in path:
                forbidden_opens.append(path)
                raise ValueError('calibration may read saved arrays only; no dataset bytes')
    sys.addaudithook(audit)
    sources = {str(HERE / name): sha(HERE / name) for name in ('PLAN.md', 'solver.py', 'probe.py')}
    inputs = {str(input_root / 'result.json'): sha(input_root / 'result.json')}
    existing = json.loads((input_root / 'result.json').read_text())
    if existing['status'] != 'DONE':
        raise ValueError('saved source diagnostics not complete')
    keys = set(existing['results'])
    models = OLD_MODELS if keys == set(OLD_MODELS) else CONFIRM_MODELS if keys == set(CONFIRM_MODELS) else None
    if models is None:
        raise ValueError('source must contain the complete fixed old-four or confirm-three model set')
    all_saved = {}
    for method in models:
        for client in range(20):
            path = input_root / f'{method}_client{client}.npz'
            inputs[str(path)] = sha(path)
            saved, bytes_cached = read_saved(path)
            all_saved[(method, client)] = saved, bytes_cached
    out.mkdir(parents=True)
    atomic_json(out / 'inputs_before.json', {**sources, **inputs})
    results, totals = {}, {'train_rows_used': 0, 'input_arrays_bytes': 0,
        'accepted_newton_steps': 0, 'fit_seconds': 0., 'peak_single_design_cache_bytes': 0}
    started = time.perf_counter()
    for method in models:
        clients = []
        for client in range(20):
            saved, bytes_cached = all_saved[(method, client)]
            states, optimization = {}, {}
            # No holdout arrays are passed to any optimizer.
            for kind in solver.KINDS:
                state, diagnostic = solver.fit(saved['train_logits'], saved['train_labels'], kind)
                states[kind], optimization[kind] = state, diagnostic
                totals['accepted_newton_steps'] += diagnostic['accepted_steps']
                totals['fit_seconds'] += diagnostic['seconds']
                totals['peak_single_design_cache_bytes'] = max(totals['peak_single_design_cache_bytes'], diagnostic['design_cache_bytes'])
            raw, labels = saved['raw'].astype(np.float64), saved['labels']
            scores = {'raw': raw, 'analytic_prior': raw + saved['prior'],
                'prioranchor_bias': raw + saved['prioranchor_bias'],
                **{kind: solver.transform(raw, kind, state) for kind, state in states.items()}}
            if saved['learned'] is not None:
                learned = saved['learned'].astype(np.float64)
                scores.update(learned_head=learned, learned_head_plus_prior=learned + saved['prior'])
            metrics = {kind: solver.stats(logits, labels) for kind, logits in scores.items()}
            source_metrics = existing['results'][method]['clients'][client]['metrics']
            for variant, source_variant in [('raw', 'raw'), ('analytic_prior', 'analytic_prior'),
                ('prioranchor_bias', 'fitted_bias' if 'fitted_bias' in source_metrics else 'prioranchor_bias')]:
                if (abs(metrics[variant]['accuracy'] - source_metrics[source_variant]['accuracy']) > 1e-12
                        or abs(metrics[variant]['loss'] - source_metrics[source_variant]['loss']) > 1e-9):
                    raise ValueError('saved fixed baseline does not reproduce')
            np.savez_compressed(out / f'{method}_client{client}.npz',
                train_ids=saved['train_ids'], holdout_ids=saved['holdout_ids'],
                prioranchor_bias=saved['prioranchor_bias'], **states,
                **{kind + '_holdout_logits': logits for kind, logits in scores.items()})
            clients.append({'client': client, 'train_samples': len(saved['train_labels']),
                'holdout_samples': len(labels), 'metrics': metrics, 'optimization': optimization,
                'input_cache_bytes': bytes_cached, 'new_cnn_forwards': 0,
                'new_cnn_backwards': 0, 'extra_uplink_bytes': 0})
            totals['train_rows_used'] += len(saved['train_labels'])
            totals['input_arrays_bytes'] += bytes_cached
        results[method] = {'clients': clients, 'summaries': solver.summarize(clients)}
        print(json.dumps({'model': method, 'summaries': results[method]['summaries']}), flush=True)
    frozen = {**sources, **inputs}
    if any(sha(path) != expected for path, expected in frozen.items()):
        raise ValueError('saved input or source changed')
    comparisons = {}
    for method, result in results.items():
        anchor = result['summaries']['prioranchor_bias']
        comparisons[method] = {kind: {
            'delta_accuracy_pp_vs_prioranchor': (result['summaries'][kind]['macro_accuracy'] - anchor['macro_accuracy']) * 100,
            'delta_ce_vs_prioranchor': result['summaries'][kind]['macro_loss'] - anchor['macro_loss']}
            for kind in solver.KINDS}
    report = {'status': 'DONE', 'purpose': 'strong known calibration baselines and prior-anchor deletion; no new candidate GO/STOP judgment',
        'results': results, 'comparisons': comparisons, 'source_identities': sources,
        'input_sha256_before_after_equal': frozen, 'input_root': str(input_root),
        'original_candidate_decision_unchanged': existing.get('decision'),
        'new_confirm_decision': 'not inferred from old saved-array baselines',
        'regularization': 'sum TRAIN CE + 10/2||theta-anchor||²; same coefficient, differing dimensions disclosed',
        'constraints': 'TS and BCTS scalar a>=0.001; vector A unrestricted; centered biases',
        'prioranchor_source': 'existing frozen lambda10 fitted state reused byte-for-byte; solver unchanged',
        'deployment_state_bytes': {'zero_anchor_bias': 40, 'temperature_scaling': 4, 'bcts': 44, 'vector_scaling': 80, 'prioranchor_bias': 40},
        'label_usage': 'only saved TRAIN GT passed to fit; holdout only final metrics; no coefficient/grid selection',
        'known_method_scope': 'local independent calibration only; no server aggregation or official FedCal reproduction claim',
        'new_cnn_forwards': 0, 'new_cnn_backwards': 0, 'extra_uplink_bytes': 0,
        'forbidden_dataset_open_attempts': forbidden_opens,
        'totals': totals, 'elapsed_seconds': time.perf_counter() - started,
        'numpy': np.__version__, 'python': sys.version}
    atomic_json(out / 'result.json', report)
    print(json.dumps({'status': 'DONE', 'models': list(results), 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
