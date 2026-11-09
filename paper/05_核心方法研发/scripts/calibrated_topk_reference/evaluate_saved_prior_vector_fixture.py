"""Fabricated two-TopK/20-client saved-array adapter integration; never ML evidence."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import json
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WRAPPER = HERE / 'evaluate_saved_prior_vector.py'
PLAN = HERE.parent / 'predictive_pfl_confirm/PLAN.md'
REGISTRATION = HERE.parent / 'predictive_pfl_confirm/PREREGISTRATION.json'
VARIANTS = ('raw', 'analytic_prior', 'prioranchor_bias', 'zero_anchor_vector',
            'prioranchored_vector', 'learned_head', 'learned_head_plus_prior')
METRICS = ('accuracy', 'nll', 'brier', 'ece15')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent(logits, labels):
    """Scalar sample loop and explicit half-open bins, independent of evaluator."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    probabilities, losses, correct, confidence, squared = [], [], [], [], []
    for row, label in zip(z, y):
        maximum = float(max(row))
        ex = np.exp(row - maximum)
        p = ex / sum(ex)
        probabilities.append(p)
        losses.append(maximum + np.log(sum(ex)) - row[label])
        correct.append(int(np.argmax(row) == label))
        confidence.append(float(max(p)))
        squared.append(sum((p[k] - int(k == label)) ** 2 for k in range(10)))
    bins = []
    for k in range(15):
        selected = [i for i, value in enumerate(confidence)
                    if k / 15 <= value and (value < (k + 1) / 15 if k < 14 else value <= 1)]
        bins.append({'index': k, 'lower': k / 15, 'upper': (k + 1) / 15,
                     'count': len(selected), 'correct_sum': sum(correct[i] for i in selected),
                     'confidence_sum': sum(confidence[i] for i in selected)})
    return {'accuracy': float(np.mean(correct)), 'nll': float(np.mean(losses)),
            'loss': float(np.mean(losses)), 'brier': float(np.mean(squared)),
            'ece15': float(sum(abs(b['correct_sum'] - b['confidence_sum']) for b in bins) / len(y)),
            'samples': len(y), 'ece15_bins': bins}


def main():
    spec = importlib.util.spec_from_file_location('_topk_saved_fixture_adapter', WRAPPER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    protected = dict(adapter.PINS)
    protected.update({PLAN: adapter.PLAN_SHA, REGISTRATION: adapter.REGISTRATION_SHA})
    for name in ('PLAN.md', 'common.py', 'wire.py', 'train.py', 'evaluate.py',
                 'prospective.py', 'prospective_evaluate.py'):
        protected[HERE / name] = sha(HERE / name)
    before = {str(path.resolve()): sha(path) for path in protected}
    assert all(before[str(path.resolve())] == expected for path, expected in protected.items())

    scratch = Path(tempfile.mkdtemp(prefix='topk_two_prior_vector_', dir=ROOT / 'work'))
    source, output = scratch / 'fabricated_saved', scratch / 'evaluation'
    source.mkdir()
    arrays = {}
    source_report = {'status': 'DONE', 'fixture_only': True, 'decision': 'REFERENCE_ONLY',
                     'task': 'fabricated saved arrays; no CIFAR/Fashion images',
                     'training_seed': 20261231, 'partition_seed': 20261217, 'results': {}}
    for method_index, method in enumerate(adapter.TOPK_MODELS):
        clients = []
        for client in range(20):
            ntrain, nholdout = 11 + client % 5, 5 + client % 4
            label_rng = np.random.default_rng(20261900 + client)
            train_labels = label_rng.integers(0, 4 + client % 7, ntrain, dtype=np.int64)
            labels = label_rng.integers(0, 10, nholdout, dtype=np.int64)
            rng = np.random.default_rng(20262000 + method_index * 100 + client)
            train = rng.normal(size=(ntrain, 10)).astype(np.float32)
            raw = (rng.normal(size=(nholdout, 10)) * 1.4).astype(np.float32)
            if client == 0:
                raw[0] = 0
                raw[1] = -1000
                raw[1, labels[1]] = 1000
                raw[2] = -1000
                raw[2, (labels[2] + 1) % 10] = 1000
            learned = (raw + rng.normal(size=raw.shape) * .2).astype(np.float32)
            counts = np.bincount(train_labels, minlength=10)
            prior = np.log((counts + 1.) / (ntrain + 10)).astype(np.float32)
            bias = np.linspace(-.3, .3, 10, dtype=np.float32)
            ids = np.arange(client * 100, client * 100 + ntrain + nholdout, dtype=np.uint32)
            data = {'train_logits': train, 'train_labels': train_labels, 'train_ids': ids[:ntrain],
                    'raw': raw, 'learned': learned, 'labels': labels, 'holdout_ids': ids[ntrain:],
                    'train_counts': counts, 'prior_shift': prior, 'fitted_bias': bias}
            arrays[method, client] = data
            np.savez_compressed(source / f'{method}_client{client}.npz', **data)
            scores = {'raw': raw.astype(np.float64), 'analytic_prior': raw.astype(np.float64) + prior,
                      'fitted_bias': raw.astype(np.float64) + bias}
            clients.append({'client': client, 'metrics': {key: independent(z, labels) for key, z in scores.items()},
                            'fixture_only': True, 'synthetic_fitted_bias_not_optimization_evidence': True})
        source_report['results'][method] = {'clients': clients}
    (source / 'result.json').write_text(json.dumps(source_report, indent=2) + '\n')

    negative = []
    def reject(name, input_root, out, plan=PLAN, registration=REGISTRATION):
        try:
            adapter.run(input_root, out, plan, registration)
        except (ValueError, FileNotFoundError) as error:
            negative.append({'case': name, 'error': str(error)})
        else:
            raise AssertionError('accepted invalid case: ' + name)
        assert not out.exists()

    bad_plan = scratch / 'bad_plan.md'
    bad_plan.write_text('fabricated wrong protocol\n')
    reject('wrong explicit PLAN hash', source, scratch / 'bad_plan_output', plan=bad_plan)
    bad_registration = scratch / 'bad_registration.json'
    bad_registration.write_text('{}\n')
    reject('wrong explicit registration hash', source, scratch / 'bad_registration_output', registration=bad_registration)
    for name, change in (
            ('incomplete_methods', lambda r: r['results'].pop('fedlc_topk')),
            ('nineteen_clients', lambda r: r['results']['ce_topk']['clients'].pop()),
            ('reordered_clients', lambda r: r['results']['fedlc_topk']['clients'].reverse()),
            ('not_DONE', lambda r: r.update(status='RUNNING'))):
        broken = scratch / name
        broken.mkdir()
        report = json.loads(json.dumps(source_report))
        change(report)
        (broken / 'result.json').write_text(json.dumps(report) + '\n')
        reject(name, broken, scratch / (name + '_output'))

    fitted = {'prior_vector': 0, 'zero_vector': 0}
    fit_paths = {str(adapter.SOLVER): 'prior_vector',
                 str(HERE.parent / 'calibration_baselines/solver.py'): 'zero_vector'}
    def profile(frame, event, arg):
        key = fit_paths.get(frame.f_code.co_filename)
        if event == 'call' and frame.f_code.co_name == 'fit' and key:
            z, y = frame.f_locals['logits'], frame.f_locals['labels']
            assert any(np.array_equal(z, a['train_logits']) and np.array_equal(y, a['train_labels'])
                       for a in arrays.values()), 'optimizer received something other than fabricated TRAIN arrays'
            fitted[key] += 1

    forbidden = []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if any(part in path for part in ('/dataset/', '/datasets/', 'test_batch', 't10k')):
                forbidden.append(path)
                raise AssertionError('fixture may never read dataset bytes')
    sys.addaudithook(audit)
    previous_profile = sys.getprofile()
    try:
        sys.setprofile(profile)
        with (scratch / 'evaluation_stdout.jsonl').open('w') as log, contextlib.redirect_stdout(log):
            report = adapter.run(source, output, PLAN, REGISTRATION)
    finally:
        sys.setprofile(previous_profile)
    assert fitted == {'prior_vector': 40, 'zero_vector': 40}
    assert 'torch' not in sys.modules
    disk_report = json.loads((output / 'result.json').read_text())
    assert report == disk_report
    assert report['input_schema'] == 'topk_two' and report['status'] == 'DONE'
    assert 'gates' not in report and 'decision' not in report
    assert report['training_seed'] == 20261231 and report['partition_seed'] == 20261217
    assert report['totals']['prior_vector_fits'] == report['totals']['zero_vector_fits'] == 40
    assert report['totals']['train_rows_used_per_recipe'] == sum(len(a['train_labels']) for a in arrays.values())
    assert not forbidden and not report['forbidden_dataset_open_attempts']
    for key in ('new_cnn_forwards', 'new_cnn_backwards', 'new_dataset_bytes', 'extra_uplink_bytes'):
        assert report[key] == 0

    counts = {'client_variant_metrics': 0, 'summary_values': 0, 'ece_bin_values': 0,
              'deployed_vector_reconstructions': 0}
    max_error = {'metric': 0., 'summary': 0., 'ece_bin': 0.}
    for method in adapter.TOPK_MODELS:
        result = report['results'][method]
        recomputed, scores_by_client = [], []
        assert len(result['clients']) == 20 and set(result['summaries']) == set(VARIANTS)
        for client, row in enumerate(result['clients']):
            data = arrays[method, client]
            raw, learned = data['raw'].astype(np.float64), data['learned'].astype(np.float64)
            scores = {'raw': raw, 'analytic_prior': raw + data['prior_shift'],
                      'prioranchor_bias': raw + data['fitted_bias'], 'learned_head': learned,
                      'learned_head_plus_prior': learned + data['prior_shift']}
            with np.load(output / f'{method}_client{client}.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['train_ids'], data['train_ids'])
                assert np.array_equal(saved['holdout_ids'], data['holdout_ids'])
                expected_anchor = np.log(data['train_counts'] + 1.)
                expected_anchor -= expected_anchor.mean()
                assert np.array_equal(saved['train_bias_anchor'], expected_anchor)
                assert np.array_equal(saved['prioranchor_bias'], data['fitted_bias'])
                for variant in ('zero_anchor_vector', 'prioranchored_vector'):
                    state = saved[variant]
                    assert state.shape == (20,) and state.dtype == np.float32 and state.nbytes == 80
                    scores[variant] = raw * state[:10].astype(np.float64) + state[10:].astype(np.float64)
                    assert np.array_equal(scores[variant], saved[variant + '_holdout_logits'])
                    counts['deployed_vector_reconstructions'] += 1
            assert set(row['metrics']) == set(VARIANTS)
            metrics = {variant: independent(z, data['labels']) for variant, z in scores.items()}
            recomputed.append(metrics)
            scores_by_client.append(scores)
            for variant in VARIANTS:
                for key in (*METRICS, 'loss'):
                    max_error['metric'] = max(max_error['metric'], abs(metrics[variant][key] - row['metrics'][variant][key]))
                    counts['client_variant_metrics'] += 1
                assert row['metrics'][variant]['loss'] == row['metrics'][variant]['nll']
                for expected_bin, actual_bin in zip(metrics[variant]['ece15_bins'], row['metrics'][variant]['ece15_bins']):
                    for key in ('count', 'correct_sum', 'confidence_sum', 'lower', 'upper', 'index'):
                        max_error['ece_bin'] = max(max_error['ece_bin'], abs(expected_bin[key] - actual_bin[key]))
                        counts['ece_bin_values'] += 1
        weights = np.array([len(arrays[method, c]['labels']) for c in range(20)], dtype=np.float64)
        weights /= weights.sum()
        for variant in VARIANTS:
            expected = {}
            for metric in METRICS:
                values = np.array([row[variant][metric] for row in recomputed])
                expected['macro_' + metric] = float(np.mean(values))
                expected['weighted_' + metric] = float(sum(v * w for v, w in zip(values, weights)))
            expected['macro_loss'] = expected['macro_nll']
            expected['weighted_loss'] = expected['weighted_nll']
            expected['client_accuracy_q10'] = float(np.quantile([row[variant]['accuracy'] for row in recomputed], .1))
            pooled = independent(np.concatenate([scores[variant] for scores in scores_by_client]),
                                 np.concatenate([arrays[method, c]['labels'] for c in range(20)]))
            expected['pooled_ece15'] = pooled['ece15']
            assert set(expected) == set(result['summaries'][variant])
            for key, value in expected.items():
                max_error['summary'] = max(max_error['summary'], abs(value - result['summaries'][variant][key]))
                counts['summary_values'] += 1
    assert max(max_error.values()) < 1e-12, max_error

    original = adapter.load_original()
    known = [(np.zeros((10, 10)), np.arange(10), (.1, np.log(10), .9, 0.)),
             (np.eye(10) * 1000, np.arange(10), (1., 0., 0., 0.)),
             (np.array([[-1000., 1000.] + [0.] * 8]), np.array([0]), (0., 2000., 2., 1.))]
    for logits, labels, expected in known:
        values = original.predictive_metrics(logits, labels)
        for key, value in zip(METRICS, expected):
            assert abs(values[key] - value) < 1e-14
    try:
        adapter.run(source, output, PLAN, REGISTRATION)
    except ValueError as error:
        assert 'fresh' in str(error)
        negative.append({'case': 'existing_output', 'error': str(error)})
    else:
        raise AssertionError('existing output accepted')

    for path, expected in before.items():
        assert sha(path) == expected
    for collection in (report['source_identities'], report['input_sha256_before_after_equal'],
                       json.loads((output / 'inputs_before.json').read_text())):
        for path in (*adapter.PINS, PLAN, REGISTRATION, WRAPPER):
            assert collection[str(path.resolve())] == sha(path)
    evidence = {'status': 'PASS', 'fixture_only': True,
                'scope': 'fabricated two-method by twenty-client saved arrays only; no scientific result',
                'fixture_root': str(scratch), 'method_tuple': list(adapter.TOPK_MODELS), 'clients_per_method': 20,
                'variants': list(VARIANTS), 'checks': counts, 'max_absolute_errors': max_error,
                'synthetic_fit_calls_observed': fitted, 'all_fits_received_saved_TRAIN_only': True,
                'known_metric_cases': ['balanced uniform: Brier 0.9', 'saturated perfect', 'saturated wrong: Brier 2, ECE1'],
                'negative_cases': negative, 'input_schema': report['input_schema'],
                'contains_gates_or_decision': False, 'torch_imported': False,
                'new_cnn_forwards': 0, 'new_cnn_backwards': 0, 'real_dataset_bytes': 0,
                'real_saved_arrays_evaluated': False, 'formal_training_started': False,
                'forbidden_dataset_open_attempts': forbidden,
                'protected_source_hashes_before_after_equal': before,
                'wrapper_sha256': sha(WRAPPER), 'fixture_harness_sha256': sha(__file__),
                'main_plan_sha256': sha(PLAN), 'main_registration_sha256': sha(REGISTRATION),
                'external_cross_family_review': 'REVIEW_UNAVAILABLE'}
    (HERE / 'SAVED_PRIOR_VECTOR_FIXTURES.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (HERE / 'SAVED_PRIOR_VECTOR_FIXTURES.md').write_text(
        '# TopK two-method saved-array adapter fixture\n\n'
        'PASS: fabricated 2 methods × 20 clients, seven variants. This is synthetic integration evidence only.\n\n'
        f'- Wrapper SHA256: `{sha(WRAPPER)}`\n'
        f'- Harness SHA256: `{sha(__file__)}`\n'
        f'- Scratch arrays and full output: `{scratch}`\n'
        f'- Independent maximum absolute errors: `{max_error}`\n'
        f'- Checks: `{counts}`\n'
        '- 40 prior-vector + 40 zero-vector numerical fits, profiler-verified saved TRAIN inputs only.\n'
        '- FP32 deployed states independently reconstructed with FP64 score application.\n'
        '- NLL/loss alias, class-summed Brier, ECE15 bins, macro/weighted summaries, pooled ECE, q10 checked.\n'
        '- Exact external PLAN and registration mandatory; wrong identities, incomplete/reordered clients, incomplete methods, non-DONE, existing output rejected.\n'
        '- No gates/decision, CNN import/forward/backward, real image bytes, actual saved arrays or formal training.\n'
        '- Frozen evaluator/solver/reader, main protocol, TopK original/prospective sources unchanged.\n'
        '- External separate-family review: REVIEW_UNAVAILABLE.\n\n'
        'Real-array invocation for root (not executed by this fixture):\n\n'
        '```sh\n/opt/anaconda3/envs/Tinyproto/bin/python work/calibrated_topk_reference/evaluate_saved_prior_vector.py \\\n'
        '  --input-root <complete_topk_saved_arrays> --out <fresh_prior_vector_output> \\\n'
        '  --protocol-plan work/predictive_pfl_confirm/PLAN.md \\\n'
        '  --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json\n```\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
