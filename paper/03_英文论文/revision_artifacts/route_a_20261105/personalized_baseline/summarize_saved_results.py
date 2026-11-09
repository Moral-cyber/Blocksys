"""Summarize frozen pFedMe outputs without loading a model or raw TEST data."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = read(HERE / 'MODEL_LOCK.json')
    assert lock['status'] == 'ALL_SIX_PERSONAL_MODELS_FROZEN'
    assert len(lock['inputs']) == 43
    for path, expected in lock['inputs'].items():
        assert sha(Path(path)) == expected, path
    verification = read(HERE / 'INDEPENDENT_SCORE_VERIFICATION.json')
    assert verification['status'] == 'PASS'
    assert verification['original_locked_inputs_unchanged']
    tasks, csv_rows, source_hashes, runtime_rows = {}, [], {}, []
    max_summary_error = 0.
    for task in ('cifar', 'fashion'):
        aggregate_path = HERE / 'official_test' / task / 'result.json'
        aggregate = read(aggregate_path)
        assert aggregate['status'] == 'DONE' and aggregate['TEST_fits'] == 0
        assert aggregate['predictor'] == 'raw final theta'
        assert aggregate['model_inputs_unchanged']
        source_hashes[str(aggregate_path)] = sha(aggregate_path)
        values = []
        metrics = list(aggregate['summaries']['raw'])
        for seed in aggregate['seeds']:
            seed_path = aggregate_path.parent / f'pfedme_dense_s{seed}' / 'result.json'
            score = read(seed_path)
            assert score['status'] == 'DONE'
            assert score['TEST_fits'] == 0 and score['TEST_backwards'] == 0
            assert score['assignment_hash'] == aggregate['assignment_hash']
            source_hashes[str(seed_path)] = sha(seed_path)
            values.append(score['summaries']['raw'])
            csv_rows.append({'task': task, 'statistic': 'seed', 'seed': seed,
                             **score['summaries']['raw']})
            train_path = HERE / 'runs' / task / f's{seed}' / 'pfedme_dense' / 'result.json'
            train = read(train_path)
            assert train['status'] == 'DONE' and len(train['rows']) == 20
            assert train['official_test_reads'] == 0
            assert all(len(row['clients']) == 20 for row in train['rows'])
            assert all(row['gradient_evaluations'] == 5 * row['outer_steps'] for row in train['rows'])
            runtime_rows.append({'task': task, 'seed': seed,
                'seconds': sum(row['seconds'] for row in train['rows']),
                'gradient_evaluations': sum(row['gradient_evaluations'] for row in train['rows']),
                'outer_steps': sum(row['outer_steps'] for row in train['rows'])})
        result = {}
        for metric in metrics:
            arr = np.asarray([row[metric] for row in values])
            stats = {'mean': float(arr.mean()), 'sample_sd': float(arr.std(ddof=1))}
            for name, value in stats.items():
                error = abs(value - aggregate['summaries']['raw'][metric][name])
                max_summary_error = max(max_summary_error, error)
                assert error < 1e-12, (task, metric, name)
                assert abs(value - verification['task_summaries'][task][metric][name]) < 1e-12
            result[metric] = stats
        for name in ('mean', 'sample_sd'):
            csv_rows.append({'task': task, 'statistic': name, 'seed': '',
                             **{key: result[key][name] for key in metrics}})
        tasks[task] = result
    source_hashes[str(HERE / 'MODEL_LOCK.json')] = sha(HERE / 'MODEL_LOCK.json')
    source_hashes[str(HERE / 'INDEPENDENT_SCORE_VERIFICATION.json')] = sha(HERE / 'INDEPENDENT_SCORE_VERIFICATION.json')
    summary = {'status': 'PASS_SIX_RUN_AGGREGATION',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'method': 'pFedMe author optimizer with matched protocol adapter',
        'predictor': 'raw final personalized theta; no post-hoc calibrator',
        'model_lock_inputs_checked': 43, 'seed_runs': 6,
        'metrics_per_task': len(metrics), 'summary_max_absolute_error': max_summary_error,
        'independent_score_replay_max_absolute_error': verification['maximum_absolute_error'],
        'metric_units': 'accuracy, q10 and ECE are proportions; Brier is multiclass sum; NLL is natural-log units',
        'tasks': tasks, 'runtime_rows': runtime_rows,
        'training_seconds_total': sum(row['seconds'] for row in runtime_rows),
        'source_sha256': source_hashes, 'new_training': 0, 'new_fits': 0,
        'new_inference': 0, 'raw_test_reads': 0}
    (HERE / 'compact_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    bridge_result = {key: summary[key] for key in ('status', 'created_at', 'method', 'predictor',
        'seed_runs', 'model_lock_inputs_checked', 'metric_units', 'tasks', 'runtime_rows',
        'training_seconds_total', 'summary_max_absolute_error',
        'independent_score_replay_max_absolute_error', 'source_sha256')}
    bridge_result.update({'evidence_scope': 'final',
        'execution': 'MPS model training; frozen raw-personalized-theta CPU official-TEST scoring',
        'adaptation': {'author_outer_lr_default': .005, 'matched_outer_lr': .03,
                       'K_inner_gradients': 5, 'lambda': 15, 'beta': 1,
                       'personal_lr': .09, 'optimizer_mu': .001},
        'dense_upload_bytes': 547560, 'dense_downlink_bytes': 547560,
        'test_fits': 0, 'test_backward_steps': 0,
        'interpretation': 'Matched protocol comparator, not full author-code reproduction or state of the art',
        'new_training_by_aggregation': 0, 'new_inference_by_aggregation': 0})
    (HERE / 'result_pfedme.json').write_text(json.dumps(bridge_result, indent=2) + '\n')
    with (HERE / 'result_all_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task', 'statistic', 'seed', *metrics])
        writer.writeheader()
        writer.writerows(csv_rows)
    c, f = tasks['cifar'], tasks['fashion']
    def pair(table, metric, scale=1, digits=3):
        return f"{scale*table[metric]['mean']:.{digits}f} $\\pm$ {scale*table[metric]['sample_sd']:.{digits}f}"
    row = 'pFedMe & Raw theta & ' + ' & '.join([
        pair(c, 'macro_nll', digits=4), pair(c, 'macro_accuracy', 100, 2),
        pair(f, 'macro_nll', digits=4), pair(f, 'macro_accuracy', 100, 2)]) + ' ' + chr(92)*2 + '\n'
    (HERE / 'table_row_suggestion.tex').write_text('% Author optimizer with matched protocol adapter; not a full author-code reproduction.\n' + row)
    print(json.dumps({key: summary[key] for key in ('status', 'seed_runs', 'model_lock_inputs_checked',
        'summary_max_absolute_error', 'independent_score_replay_max_absolute_error', 'training_seconds_total')}, indent=2))


if __name__ == '__main__':
    main()
