"""Independent saved-array audit. No training, torch, dataset or model loader."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'diagnostic_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main():
    reference = json.loads((RUN / 'result.json').read_text())
    protected = [ROOT / 'probe.py', ROOT / 'PLAN.md', RUN / 'result.json']
    protected += [RUN / (name + '.npz') for name in reference['results']]
    protected += [Path(p) for p in reference['input_sha256']]
    before = {str(p): digest(p) for p in protected}
    checks = []

    def check(name, passed, detail=None):
        row = {'check': name, 'passed': bool(passed)}
        if detail is not None:
            row['detail'] = detail
        checks.append(row)

    for path, expected in reference['input_sha256'].items():
        check('recorded_input_hash:' + path, before[path] == expected)
    check('probe_source_hash', before[str(ROOT / 'probe.py')] == reference['source_sha256'])
    check('plan_hash', before[str(ROOT / 'PLAN.md')] == reference['plan_sha256'])

    results, first_ids, first_labels = {}, None, None
    max_ce_error, max_metric_error = 0.0, 0.0
    for name, expected in reference['results'].items():
        config = json.loads(Path(expected['config']).read_text())
        manifest = config.get('manifest', config.get('data_manifest'))
        counts = np.asarray(manifest['client_label_counts'], dtype=np.float64)
        sizes = np.asarray(manifest['client_sizes'], dtype=np.float64)
        check(name + ':counts_equal_sizes', np.array_equal(counts.sum(axis=1), sizes))
        check(name + ':positive_sizes', np.all(sizes > 0))
        check(name + ':epoch_one', config['local_epochs'] == 1)
        weights = sizes / sizes.sum()
        steps = np.ceil(sizes / 64.0).astype(np.int64)
        pooled = counts.sum(axis=0) / sizes.sum()
        effective_weights = weights * steps / np.dot(weights, steps)
        effective = (effective_weights[:, None] * counts / sizes[:, None]).sum(axis=0)
        check(name + ':positive_priors', np.all(pooled > 0) and np.all(effective > 0))
        check(name + ':normalized_priors', abs(pooled.sum() - 1) < 1e-12 and abs(effective.sum() - 1) < 1e-12)
        difference = np.log(pooled) - np.log(effective)
        shifts = {'raw': np.zeros(10), 'pooled_to_uniform': -np.log(pooled),
                  'effective_to_pooled': difference, 'wrong_direction': -difference}
        ids = np.asarray(manifest['validation_indices'], dtype='<u4')
        id_hash = hashlib.sha256(ids.tobytes()).hexdigest()
        check(name + ':id_encoding', manifest['index_hash_encoding'] ==
              'uint32 little endian; grouped hashes prefix each client list with uint32 length')
        check(name + ':dev_ids_count_unique', len(ids) == 2000 and len(np.unique(ids)) == 2000)
        check(name + ':dev_ids_hash', id_hash == manifest['validation_ids_sha256'] == expected['development_id_sha256'])
        if first_ids is None:
            first_ids = ids.copy()
        check(name + ':dev_ids_same_order', np.array_equal(ids, first_ids))

        with np.load(RUN / (name + '.npz'), allow_pickle=False) as saved:
            logits, labels = saved['logits'].copy(), saved['labels'].copy()
            check(name + ':logit_shape_finite', logits.shape == (2000, 10) and np.isfinite(logits).all())
            check(name + ':label_shape_range', labels.shape == (2000,) and np.all((labels >= 0) & (labels < 10)))
            for field, rebuilt in [('pooled', pooled), ('effective', effective), ('shift', difference)]:
                check(name + ':npz_' + field, np.allclose(saved[field], rebuilt, rtol=0, atol=1e-12))
                check(name + ':record_' + field, np.allclose(expected[field], rebuilt, rtol=0, atol=1e-12))
        if first_labels is None:
            first_labels = labels.copy()
        check(name + ':labels_same_order', np.array_equal(labels, first_labels))
        totals = np.bincount(labels, minlength=10)
        check(name + ':gt_counts', np.array_equal(totals, manifest['validation_class_counts']))
        check(name + ':balanced_gt_counts', np.array_equal(totals, np.full(10, 200)))
        metrics = {}
        for method, shift in shifts.items():
            # Preserve the probe's FP32 shift/addition, then independently
            # evaluate stable log-sum-exp in FP64 with NumPy.
            z = (logits + shift.astype(logits.dtype)).astype(np.float64)
            maxima = z.max(axis=1)
            ce = float(np.mean(maxima + np.log(np.exp(z - maxima[:, None]).sum(axis=1)) - z[np.arange(len(labels)), labels]))
            prediction = z.argmax(axis=1)
            hits = np.bincount(labels[prediction == labels], minlength=10)
            class_acc = hits / totals
            row = {'loss': ce, 'accuracy': float(np.mean(prediction == labels)),
                   'class_accuracy': class_acc.tolist(), 'macro_accuracy': float(class_acc.mean()),
                   'worst_class_accuracy': float(class_acc.min()), 'samples': len(labels),
                   'class_correct': hits.tolist(), 'class_samples': totals.tolist()}
            metrics[method] = row
            wanted = expected['metrics'][method]
            max_ce_error = max(max_ce_error, abs(row['loss'] - wanted['loss']))
            for field in ['accuracy', 'class_accuracy', 'worst_class_accuracy']:
                error = float(np.max(np.abs(np.asarray(row[field]) - np.asarray(wanted[field]))))
                max_metric_error = max(max_metric_error, error)
                check(name + ':' + method + ':' + field, error <= 1e-12)
            check(name + ':' + method + ':ce', abs(ce - wanted['loss']) <= 2e-6)
            check(name + ':' + method + ':samples', row['samples'] == wanted['samples'])
        for field in ['accuracy', 'class_accuracy', 'worst_class_accuracy']:
            check(name + ':raw_saved_final:' + field,
                  np.allclose(metrics['raw'][field], expected['raw_saved_final'][field], rtol=0, atol=2e-8))
        check(name + ':raw_saved_final:ce', abs(metrics['raw']['loss'] - expected['raw_saved_final']['loss']) <= 2e-6)
        check(name + ':twenty_round_config', config['rounds'] == 20)
        results[name] = {'pooled': pooled.tolist(), 'effective': effective.tolist(),
                         'effective_client_weights': effective_weights.tolist(), 'steps_inferred_from_epoch_one_counts': steps.tolist(),
                         'shifts': {k: v.tolist() for k, v in shifts.items()},
                         'development_id_sha256': id_hash, 'metrics': metrics}

    gates = {}
    for name in ['hetero_dense', 'hetero_topk']:
        raw, proposed, reverse = (results[name]['metrics'][key] for key in ['raw', 'effective_to_pooled', 'wrong_direction'])
        gates[name] = {'ce': proposed['loss'] < raw['loss'],
                       'accuracy': proposed['accuracy'] >= raw['accuracy'] + .005,
                       'direction': proposed['loss'] < reverse['loss'] and proposed['accuracy'] > reverse['accuracy']}
        check(name + ':gates_match', gates[name] == reference['gates'][name])
    decision = 'FOLLOW_UP_ONLY' if all(all(g.values()) for g in gates.values()) else 'STOP_THIS_FIXED_PRIOR_RULE'
    check('decision_matches', decision == reference['decision'])
    after = {str(p): digest(p) for p in protected}
    check('all_protected_inputs_unchanged', before == after)
    output = {'verification': 'PASS' if all(x['passed'] for x in checks) else 'FAIL',
              'independent_checker_sha256': digest(__file__),
              'decision': decision, 'evaluation_scope': 'official TRAIN-holdout development; frozen final round only; not last5',
              'official_test_loaded': False, 'official_dataset_loaded': False, 'checkpoint_model_loaded': False,
              'training_started': False, 'calibrator_fitted': False, 'used_dev_statistics_to_estimate_prior': False,
              'will_extend_or_flip_sign_to_rescue_rule': False,
              'pi_effective_is_proven_predictive_prior': False,
              'input_file_count': len(protected), 'recorded_model_config_hashes_matched':
                  sum(before[path] == expected for path, expected in reference['input_sha256'].items()),
              'inputs_unchanged': before == after, 'before': before, 'after': after,
              'development_ids_same_order_across_configs': all(x['passed'] for x in checks if x['check'].endswith(':dev_ids_same_order')),
              'saved_labels_same_order_across_npz': all(x['passed'] for x in checks if x['check'].endswith(':labels_same_order')),
              'development_binding_limit': 'NPZ has no ID array. Config IDs and saved GT agree; original image-to-GT binding was not reloaded or revalidated.',
              'metric_arithmetic': 'FP32 stored logits plus FP32 shift; stable FP64 NumPy CE; no torch/probe/data loader imports',
              'ce_comparison_tolerance': 2e-6, 'max_ce_difference_from_saved_fp32': max_ce_error,
              'max_accuracy_or_class_accuracy_difference': max_metric_error,
              'gates': gates, 'results': results, 'checks': checks,
              'checks_run': len(checks), 'checks_passed': sum(x['passed'] for x in checks)}
    (ROOT / 'LOCAL_CHECK.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({k: output[k] for k in ['verification', 'decision', 'checks_run', 'checks_passed',
                     'input_file_count', 'inputs_unchanged', 'max_ce_difference_from_saved_fp32',
                     'max_accuracy_or_class_accuracy_difference', 'gates']}, ensure_ascii=False, indent=2))
    if output['verification'] != 'PASS':
        print(json.dumps([x for x in checks if not x['passed']], ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
