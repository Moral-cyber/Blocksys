"""Descriptive probability-quality metrics on completed saved holdout arrays.

No calibrator fitting, model loading, original image access, or checkpoint choice.
ECE uses 15 fixed equal-width confidence bins; multiclass Brier is the sum
over classes, in [0,2]. Clients are not independent training repetitions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if (logits.shape != (len(labels), 10) or not len(labels)
            or not np.isfinite(logits).all()
            or np.any((labels < 0) | (labels >= 10))):
        raise ValueError('finite nonempty ten-class scores and dataset GT required')
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    p = np.exp(shifted - log_normalizer[:, None])
    predicted = logits.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    confidence = p.max(axis=1)
    brier = (p * p).sum(axis=1) - 2 * p[np.arange(len(labels)), labels] + 1
    nll = log_normalizer - shifted[np.arange(len(labels)), labels]
    index = np.minimum((confidence * 15).astype(np.int64), 14)
    count = np.bincount(index, minlength=15)
    conf_sum = np.bincount(index, weights=confidence, minlength=15)
    acc_sum = np.bincount(index, weights=correct, minlength=15)
    ece = np.abs(acc_sum - conf_sum).sum() / len(labels)
    bins = [{'index': b, 'lower': b / 15, 'upper': (b + 1) / 15,
             'right_closed': b == 14, 'count': int(count[b]),
             'confidence_sum': float(conf_sum[b]), 'correct_sum': float(acc_sum[b])}
            for b in range(15)]
    return {'samples': len(labels), 'accuracy': float(correct.mean()),
            'nll': float(nll.mean()), 'brier_sum_classes': float(brier.mean()),
            'ece_15_equal_width': float(ece), 'bins': bins}


def summarize(clients):
    result = {}
    for variant in clients[0]['metrics']:
        rows = [client['metrics'][variant] for client in clients]
        n = np.asarray([row['samples'] for row in rows], dtype=np.float64)
        weights = n / n.sum()
        metrics = {}
        for name in ('accuracy', 'nll', 'brier_sum_classes', 'ece_15_equal_width'):
            values = np.asarray([row[name] for row in rows])
            metrics['macro_' + name] = float(values.mean())
            metrics['weighted_' + name] = float(weights @ values)
        pooled_conf = np.asarray([[item['confidence_sum'] for item in row['bins']]
                                  for row in rows]).sum(axis=0)
        pooled_correct = np.asarray([[item['correct_sum'] for item in row['bins']]
                                     for row in rows]).sum(axis=0)
        metrics['pooled_ece_15_equal_width'] = float(np.abs(pooled_correct - pooled_conf).sum() / n.sum())
        result[variant] = metrics
    return result


def fixtures():
    z = np.zeros((4, 10))
    row = score(z, np.array([0, 1, 2, 3]))
    assert abs(row['nll'] - np.log(10)) < 1e-14
    assert abs(row['brier_sum_classes'] - .9) < 1e-14
    assert abs(row['ece_15_equal_width'] - .15) < 1e-14
    assert sum(item['count'] for item in row['bins']) == 4
    # Confidence exactly one belongs to the last bin.
    z = np.full((2, 10), -1000.)
    z[np.arange(2), [1, 2]] = 1000.
    row = score(z, np.array([1, 3]))
    assert row['accuracy'] == .5 and row['brier_sum_classes'] == 1.
    assert row['ece_15_equal_width'] == .5 and row['bins'][-1]['count'] == 2
    return {'uniform_and_onehot_boundary': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--calibration-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.calibration_root.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError('fresh output required')
    def guard(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = str(arguments[0])
            if 'test_batch' in path or 't10k' in path or '/datasets/' in path:
                raise ValueError('saved-array metrics only; no dataset access')
    sys.addaudithook(guard)
    report_path = root / 'result.json'
    source = json.loads(report_path.read_text())
    if source['status'] != 'DONE':
        raise ValueError('complete calibration source required')
    originals = Path(source['input_root'])
    inputs = {str(report_path): sha(report_path), str(Path(__file__).resolve()): sha(__file__)}
    results = {}
    fixture_status = fixtures()
    for method in source['results']:
        clients = []
        for client in range(20):
            cal_path = root / f'{method}_client{client}.npz'
            data_path = originals / f'{method}_client{client}.npz'
            inputs[str(cal_path)], inputs[str(data_path)] = sha(cal_path), sha(data_path)
            with np.load(cal_path, allow_pickle=False) as cal, np.load(data_path, allow_pickle=False) as data:
                labels = data['labels'] if 'labels' in data else data['holdout_labels']
                if not np.array_equal(cal['holdout_ids'], data['holdout_ids']):
                    raise ValueError('holdout ID order mismatch')
                scores = {key.removesuffix('_holdout_logits'): cal[key]
                          for key in cal.files if key.endswith('_holdout_logits')}
                # A fixed TRAIN-count prediction exposes class-prior-only performance.
                scores['prior_only'] = np.broadcast_to(data['prior_shift'], (len(labels), 10))
                metrics = {name: score(z, labels) for name, z in scores.items()}
            for variant in source['results'][method]['summaries']:
                expected = source['results'][method]['clients'][client]['metrics'][variant]
                if (abs(metrics[variant]['accuracy'] - expected['accuracy']) > 1e-14
                        or abs(metrics[variant]['nll'] - expected['loss']) > 1e-12):
                    raise ValueError('accuracy/NLL changed from completed source')
            clients.append({'client': client, 'metrics': metrics})
        results[method] = {'clients': clients, 'summaries': summarize(clients)}
    if any(sha(path) != expected for path, expected in inputs.items()):
        raise ValueError('source changed')
    report = {'status': 'DONE', 'purpose': 'descriptive probability-quality metrics; no fitting or new GO/STOP criterion',
              'results': results, 'source_identities': inputs, 'fixtures': fixture_status,
              'ece': '15 pre-fixed equal-width bins, [lower,upper), last includes1; macro per-client and pooled differ',
              'brier': 'sum_c (p_c - onehot_c)^2, unnormalized ten-class score in [0,2]',
              'scope': 'official TRAIN internal client holdout development, not final TEST or independent client seeds',
              'calibration_root': str(root), 'new_training': False,
              'original_candidate_decision_unchanged': source['original_candidate_decision_unchanged']}
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for method, values in results.items():
        print(json.dumps({'method': method, 'summaries': values['summaries']}), flush=True)


if __name__ == '__main__':
    main()
