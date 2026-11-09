"""Full readback of fixed dense training matrix; official TEST forbidden."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_baseline_training_readback', HERE / 'adapter.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
attempts = a.test_guard()
a.verify(json.loads((HERE / 'SOURCE_LOCK.json').read_text())['inputs'])
rows = []
for task, (_, seeds) in a.TASKS.items():
    c, _ = a.dependencies(task)
    for seed in seeds:
        for arm in a.ARMS:
            model, config = a.load_model(c, task, seed, arm)
            report = json.loads((a.folder(task, seed, arm) / 'result.json').read_text())
            if [v['round'] for v in report['rows']] != list(range(1, 21)):
                raise ValueError('fixed 20 consecutive rounds required')
            for row in report['rows']:
                if not row['global_anchor_unchanged_between_clients'] or len(row['clients']) != 20:
                    raise ValueError('incomplete clients or changed anchor')
                for client, saved in enumerate(row['clients']):
                    if saved['client'] != client or saved['seed'] != seed + 100 * (row['round'] - 1) + client:
                        raise ValueError('training seed or client order differs')
                    if saved['train_counts'] != config['data_manifest']['client_label_counts'][client]:
                        raise ValueError('actual local training counts differ from original partition')
                    if saved['gradient_evaluations'] != (config['data_manifest']['client_sizes'][client] + 63) // 64:
                        raise ValueError('local epoch/batch count differs')
            if report['official_test_reads'] != 0:
                raise ValueError('training accessed TEST')
            rows.append({'task': task, 'seed': seed, 'method': arm, 'status': 'PASS_COMPLETE20',
                         'rounds': 20, 'full_client_participation': True, 'actual_counts_match_original': True,
                         'local_seed_and_batch_count_exact': True,
                         'seconds': sum(v['seconds'] for v in report['rows'])})
a.write(HERE / 'TRAINING_ACCEPTANCE.json', {'status': 'PASS_EXECUTION_INTEGRITY', 'runs': rows,
                                         'source_and_data_hashes_unchanged': True,
                                         'official_test_reads': 0, 'forbidden_open_attempts': attempts,
                                         'scope': 'execution integrity; no model-quality gate'})
print(json.dumps({'status': 'TRAINING_ACCEPTANCE_PASS', 'runs': len(rows)}, indent=2))
