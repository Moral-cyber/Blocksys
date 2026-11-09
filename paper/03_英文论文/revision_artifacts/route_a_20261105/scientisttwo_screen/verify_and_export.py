"""Readback of saved candidate states; independent NumPy metric replay."""
from pathlib import Path
import csv
import importlib.util
import json

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_candidate_screen_replay', HERE / 'screen.py')
screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text()); screen.verify(lock['inputs'])


def independent_metrics(logits, labels):
    z = np.asarray(logits, dtype=np.float64); y = np.asarray(labels, dtype=np.int64)
    shifted = z - z.max(axis=1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(axis=1, keepdims=True)
    correct = np.argmax(z, axis=1) == y
    error = probability - np.eye(10)[y]
    confidence = probability.max(axis=1)
    ece = 0.
    for i in range(15):
        mask = (confidence >= i / 15) & ((confidence <= 1.) if i == 14 else (confidence < (i + 1) / 15))
        if mask.any(): ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {'accuracy': float(correct.mean()),
            'nll': float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y])),
            'brier': float(np.mean(np.sum(error * error, axis=1))), 'ece15': ece}


rows = []; max_error = 0.; max_kkt = 0.; traced_peaks = []; files = 0
for task, (_, seeds) in screen.TASKS.items():
    seed = seeds[0]; ef, _ = screen.roots(task, seed)
    root = HERE / 'runs' / task / f's{seed}'
    report = json.loads((root / 'result.json').read_text())
    for client in range(20):
        with np.load(ef / f'fedlc_uniform3_client{client}.npz', allow_pickle=False) as old:
            logits = old['raw'].astype(np.float64); labels = old['labels'].copy()
        with np.load(root / f'client{client}.npz', allow_pickle=False) as saved:
            states = {arm: saved[arm].copy() for arm in screen.ARMS}
        for arm, state in states.items():
            if state.dtype != np.float32 or state.shape != (20,) or not np.isfinite(state).all():
                raise ValueError('finite twenty-FP32 state required')
            p = state.astype(np.float64); scores = logits * p[:10] + p[10:]
            metrics = independent_metrics(scores, labels)
            original = report['clients'][client]['metrics'][arm]
            for key in metrics:
                max_error = max(max_error, abs(metrics[key] - original[key]))
                if abs(metrics[key] - original[key]) > 1e-12: raise ValueError('independent metrics differ')
            rows.append({'task': task, 'seed': seed, 'client': client, 'arm': arm,
                         'train_samples': report['clients'][client]['train_samples'], 'holdout_samples': len(labels), **metrics})
            info = report['clients'][client]['fits'][arm]
            max_kkt = max(max_kkt, info['projected_gradient_inf'])
            if 'python_traced_peak_bytes_entire_fit' in info: traced_peaks.append(info['python_traced_peak_bytes_entire_fit'])
        files += 1
screen.verify(lock['inputs'])
with (HERE / 'client_metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
result = {'status': 'PASS_INDEPENDENT_SAVED_STATE_REPLAY', 'client_exports': files, 'arm_client_rows': len(rows),
          'independent_metric_max_error': max_error, 'all_locked_inputs_unchanged': True,
          'maximum_reported_projected_gradient_inf': max_kkt,
          'python_traced_peak_entire_new_geometry_fit_bytes': {'minimum': min(traced_peaks), 'maximum': max(traced_peaks)},
          'memory_scope': 'tracemalloc entire fit includes baseline design/workspace and Python/NumPy allocations; excludes untracked native allocator/RSS, not incremental geometry-only cost',
          'new_training_or_fits': 0, 'official_test_reads': 0, 'new_partition_output_reads': 0,
          'scope': 'local independent formula replay; not external or cross-family audit'}
(HERE / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
