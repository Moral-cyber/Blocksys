"""Independent saved-score replay; never performs CNN training or fitting."""
from pathlib import Path
import importlib.util
import json

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_baseline_saved_replay', HERE / 'adapter.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
lock = json.loads((HERE / 'FIT_LOCK.json').read_text())
a.verify(lock['inputs'])
_, s = a.final_modules()
max_error = 0.
checked_files = 0
details = []
for task, (_, seeds) in a.TASKS.items():
    original = json.loads((a.METHOD / f'results/predictive_final_test/official_{task}/assignment.json').read_text())
    assignment = json.loads((HERE / 'official_test' / task / 'assignment.json').read_text())
    if assignment['assignment_hash'] != original['assignment_hash'] or assignment['client_ids'] != original['client_ids']:
        raise ValueError('baseline clients differ from main fixed TEST protocol')
    for seed in seeds:
        for arm in a.ARMS:
            out = HERE / 'official_test' / task / f'{arm}_s{seed}'
            report = json.loads((out / 'result.json').read_text())
            rows = []
            for client in range(20):
                with np.load(out / f'client{client}.npz', allow_pickle=False) as saved:
                    if saved['row_ids'].tolist() != assignment['client_ids'][client]:
                        raise ValueError('saved original-row IDs do not match client membership')
                    scores = {name: saved['score_' + name].copy() for name in a.VARIANTS}
                    labels = saved['labels'].copy()
                    for name in a.VARIANTS[1:]:
                        state = saved[name]
                        expected = scores['raw'] * state[:10].astype(np.float64) + state[10:].astype(np.float64)
                        if not np.array_equal(expected, scores[name]):
                            raise ValueError('saved FP32 deployment-state arithmetic differs')
                    metrics = {name: s.metrics(value, labels) for name, value in scores.items()}
                for name in a.VARIANTS:
                    for metric in a.METRICS:
                        error = abs(metrics[name][metric] - report['clients'][client]['metrics'][name][metric])
                        max_error = max(error, max_error)
                        if error > 1e-13:
                            raise ValueError('saved-score metric replay differs')
                rows.append({'client': client, 'test_rows': len(labels), 'metrics': metrics})
                checked_files += 1
            reconstructed = a.summary(rows)
            for name in a.VARIANTS:
                for key, value in reconstructed[name].items():
                    error = abs(value - report['summaries'][name][key])
                    max_error = max(error, max_error)
                    if error > 1e-13:
                        raise ValueError('saved-score aggregate replay differs')
            details.append({'task': task, 'seed': seed, 'arm': arm, 'clients': 20,
                            'predictors': list(a.VARIANTS), 'metric_replay': 'PASS'})
a.verify(lock['inputs'])
result = {'status': 'PASS_SAVED_SCORE_REPLAY', 'checked_client_exports': checked_files,
          'metric_max_absolute_error': max_error, 'all_original_test_assignments_equal': True,
          'fp32_deployment_arithmetic_exact': True, 'all_frozen_model_state_inputs_unchanged': True,
          'new_training_or_fits': 0, 'scope': 'local deterministic replay; not a cross-family audit',
          'runs': details}
a.write(HERE / 'VERIFICATION.json', result)
print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
