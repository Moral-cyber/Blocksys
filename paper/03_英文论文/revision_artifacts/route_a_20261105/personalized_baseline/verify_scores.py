"""Independent arithmetic replay of exported pFedMe logits; no inference or fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = {'cifar': [20261231, 20261232, 20261233],
         'fashion': [20261241, 20261242, 20261243]}
METRICS = ('accuracy', 'nll', 'brier', 'ece15')
TOLERANCE = 1e-10


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def independent_metrics(logits, labels):
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.shape != (len(labels), 10) or labels.ndim != 1 or not len(labels):
        raise ValueError('Invalid/nonempty ten-class array shape')
    if labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels > 9):
        raise ValueError('Invalid integer class labels')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite raw theta logits')
    normalizer = np.logaddexp.reduce(scores, axis=1)
    probability = np.exp(scores - normalizer[:, None])
    prediction = scores.argmax(axis=1)
    confidence = probability[np.arange(len(labels)), prediction]
    correct = prediction == labels
    target = np.eye(10)[labels]
    loss = normalizer - scores[np.arange(len(labels)), labels]
    edges = np.linspace(0, 1, 16)
    bucket = np.clip(np.searchsorted(edges, confidence, side='right') - 1, 0, 14)
    counts, confidence_mass, correct_mass = [], [], []
    for index in range(15):
        selected = bucket == index
        counts.append(int(selected.sum()))
        confidence_mass.append(float(confidence[selected].sum()))
        correct_mass.append(float(correct[selected].sum()))
    return {'accuracy': float(correct.mean()), 'nll': float(loss.mean()),
            'brier': float(((probability - target) ** 2).sum(axis=1).mean()),
            'ece15': float(np.abs(np.asarray(confidence_mass) - correct_mass).sum() / len(labels)),
            'samples': len(labels), 'counts': counts,
            'confidence_mass': confidence_mass, 'correct_mass': correct_mass}


def verify():
    differences, evidence = [], {}
    model_lock = read(HERE / 'MODEL_LOCK.json')
    if model_lock['status'] != 'ALL_SIX_PERSONAL_MODELS_FROZEN':
        raise ValueError('All six model checkpoints must be locked before score replay')
    for filename, expected in model_lock['inputs'].items():
        if sha(filename) != expected:
            raise ValueError(f'Model-lock source changed: {filename}')
    evidence[str(HERE / 'MODEL_LOCK.json')] = sha(HERE / 'MODEL_LOCK.json')

    def compare(label, actual, expected):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f'Invalid comparison: {label}')
        difference = float(np.max(np.abs(actual - expected)))
        differences.append({'quantity': label, 'maximum_absolute_error': difference})
        if difference > TOLERANCE:
            raise ValueError(f'Numerical replay mismatch: {label}: {difference}')

    task_summaries = {}
    for task, seeds in TASKS.items():
        root = HERE / 'official_test' / task
        report = read(root / 'result.json')
        if report['status'] != 'DONE' or report['seeds'] != seeds or report['TEST_fits'] != 0:
            raise ValueError(f'Incomplete/incorrect task score report: {task}')
        assignment = read(root / 'assignment.json')
        if assignment['assignment_hash'] != report['assignment_hash']:
            raise ValueError(f'Assignment hash mismatch: {task}')
        original_assignment_path = Path('/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/predictive_final_test') / f'official_{task}/assignment.json'
        original_assignment = read(original_assignment_path)
        if assignment['assignment_hash'] != original_assignment['assignment_hash']:
            raise ValueError(f'Assignment differs from protected original task: {task}')
        evidence[str(original_assignment_path)] = sha(original_assignment_path)
        root_lock = read(root / 'ROOT_LOCK.json')
        if root_lock['personalized_model_lock_sha256'] != sha(HERE / 'MODEL_LOCK.json'):
            raise ValueError(f'TEST root is not bound to current model lock: {task}')
        baseline_ids = None
        expected_labels = None
        summaries = []
        for seed in seeds:
            directory = root / f'pfedme_dense_s{seed}'
            seed_report = read(directory / 'result.json')
            if (seed_report['status'] != 'DONE' or seed_report['seed'] != seed
                    or seed_report['task'] != task or seed_report['TEST_fits'] != 0
                    or seed_report['TEST_backwards'] != 0
                    or seed_report['predictor'] != 'raw final theta'
                    or seed_report['assignment_hash'] != report['assignment_hash']):
                raise ValueError(f'Invalid seed report: {task}/{seed}')
            if [row['client'] for row in seed_report['clients']] != list(range(20)):
                raise ValueError('Exactly 20 ordered clients required')
            clients, all_ids, all_labels, individual_ids = [], [], [], []
            for client, row in enumerate(seed_report['clients']):
                filename = directory / f'client{client}.npz'
                with np.load(filename, allow_pickle=False) as archive:
                    if set(archive.files) != {'row_ids', 'labels', 'raw_theta_logits'}:
                        raise ValueError(f'Unexpected exported keys: {filename}')
                    ids, labels = archive['row_ids'], archive['labels']
                    logits = archive['raw_theta_logits']
                    if ids.dtype.kind not in 'iu' or ids.ndim != 1 or len(ids) != len(labels):
                        raise ValueError('Invalid row membership')
                    if ids.tolist() != assignment['client_ids'][client]:
                        raise ValueError('Exported membership differs from frozen assignment')
                    calculated = independent_metrics(logits, labels)
                    all_ids.append(ids.copy())
                    all_labels.append(labels.copy())
                    individual_ids.append(ids.tolist())
                evidence[str(filename)] = sha(filename)
                if row['test_rows'] != len(ids):
                    raise ValueError('Reported client sample size differs from exported membership')
                for name in METRICS:
                    compare(f'{task}/{seed}/client{client}/{name}', calculated[name], row['metrics']['raw'][name])
                compare(f'{task}/{seed}/client{client}/loss_alias', calculated['nll'], row['metrics']['raw']['loss'])
                bins = row['metrics']['raw']['ece_bins']
                for own, original in (('counts', 'counts'), ('confidence_mass', 'confidence_sum'),
                                      ('correct_mass', 'correct_sum')):
                    compare(f'{task}/{seed}/client{client}/{own}', calculated[own], bins[original])
                clients.append(calculated)
            ids = np.concatenate(all_ids)
            labels = np.concatenate(all_labels)
            if not np.array_equal(np.sort(ids), np.arange(10000)):
                raise ValueError(f'TEST rows duplicated/missing/out of range: {task}/{seed}')
            ordered_labels = labels[np.argsort(ids)]
            label_hash = hashlib.sha256(ordered_labels.astype('<i8').tobytes()).hexdigest()
            if label_hash != assignment['labels_int64le_sha256']:
                raise ValueError('Exported labels do not match frozen assignment label hash')
            if baseline_ids is None:
                baseline_ids, expected_labels = individual_ids, ordered_labels
            elif individual_ids != baseline_ids or not np.array_equal(ordered_labels, expected_labels):
                raise ValueError('Client memberships/labels differ across fixed training seeds')
            summary = {}
            sizes = np.asarray([client['samples'] for client in clients])
            for name in METRICS:
                values = np.asarray([client[name] for client in clients])
                summary['macro_' + name] = float(values.mean())
                summary['weighted_' + name] = float(np.average(values, weights=sizes))
            summary['client_accuracy_q10'] = float(np.quantile([client['accuracy'] for client in clients], .1))
            confidence = np.asarray([client['confidence_mass'] for client in clients]).sum(axis=0)
            correct = np.asarray([client['correct_mass'] for client in clients]).sum(axis=0)
            summary['pooled_ece15'] = float(np.abs(confidence - correct).sum() / 10000)
            for name, value in summary.items():
                compare(f'{task}/{seed}/{name}', value, seed_report['summaries']['raw'][name])
            evidence[str(directory / 'result.json')] = sha(directory / 'result.json')
            summaries.append(summary)
        aggregate = {}
        for name in summaries[0]:
            values = [summary[name] for summary in summaries]
            aggregate[name] = {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}
            for statistic, value in aggregate[name].items():
                compare(f'{task}/three_seed/{name}/{statistic}', value, report['summaries']['raw'][name][statistic])
        task_summaries[task] = aggregate
        for filename in (root / 'result.json', root / 'assignment.json', root / 'ROOT_LOCK.json'):
            evidence[str(filename)] = sha(filename)
    # Repeat lock verification after numerical replay to catch concurrent mutation.
    for filename, expected in model_lock['inputs'].items():
        if sha(filename) != expected:
            raise ValueError(f'Model-lock source changed during verification: {filename}')
    return {'status': 'PASS', 'created_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'Independent exported-logit arithmetic/artifact replay; no CNN, training, or raw TEST reads',
            'independent_scorer_sha256': sha(__file__), 'tolerance': TOLERANCE,
            'checks': len(differences),
            'maximum_absolute_error': max(row['maximum_absolute_error'] for row in differences),
            'task_summaries': task_summaries, 'differences': differences,
            'evidence_sha256': evidence, 'new_fits': 0, 'new_inference': 0,
            'original_locked_inputs_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', action='store_true', help='Run synthetic uniform-logit arithmetic only')
    args = parser.parse_args()
    if args.fixture:
        result = independent_metrics(np.zeros((10, 10)), np.arange(10))
        expected = {'accuracy': .1, 'nll': float(np.log(10)), 'brier': .9, 'ece15': 0.0}
        for name, value in expected.items():
            if abs(result[name] - value) > TOLERANCE:
                raise ValueError(f'Uniform-logit fixture failed: {name}')
        print(json.dumps({'status': 'SYNTHETIC_ARITHMETIC_PASS', 'metrics': result,
                          'scope': 'Numerical fixture only, no FL performance evidence'}, indent=2))
        return 0
    try:
        report = verify()
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'status': 'NOT_READY_OR_FAILED', 'error': str(error)}, indent=2))
        return 2
    target = HERE / 'INDEPENDENT_SCORE_VERIFICATION.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'report': str(target),
                      'maximum_absolute_error': report['maximum_absolute_error']}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
