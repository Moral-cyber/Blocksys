"""Tables of completed prospective runs; never pools clients as training seeds."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/predictive_pfl_confirm'
TASKS = {'cifar': (20261217, (20261231, 20261232, 20261233)),
         'fashion': (20261219, (20261241, 20261242, 20261243))}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows, inputs = [], {}
    for task, (partition, seeds) in TASKS.items():
        for seed in seeds:
            path = BASE / f'adaptation_{task}_p{partition}_s{seed}/result.json'
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if data['status'] != 'DONE':
                continue
            inputs[str(path)] = sha(path)
            for method, result in data['results'].items():
                for variant, metrics in result['summaries'].items():
                    rows.append({'task': task, 'partition_seed': partition, 'training_seed': seed,
                                 'method': method, 'variant': variant,
                                 'accuracy_percent': 100 * metrics['macro_accuracy'],
                                 'nll': metrics['macro_nll'], 'brier': metrics['macro_brier'],
                                 'ece15_percent': 100 * metrics['macro_ece15'],
                                 'q10_accuracy_percent': 100 * metrics['client_accuracy_q10']})
    summaries = []
    groups = sorted({(row['task'], row['method'], row['variant']) for row in rows})
    for task, method, variant in groups:
        selected = [row for row in rows if (row['task'], row['method'], row['variant']) == (task, method, variant)]
        item = {'task': task, 'method': method, 'variant': variant,
                'independent_training_seeds': len(selected), 'seeds': [row['training_seed'] for row in selected],
                'status': 'THREE_SEEDS_COMPLETE' if len(selected) == 3 else 'INCOMPLETE'}
        for metric in ('accuracy_percent', 'nll', 'brier', 'ece15_percent', 'q10_accuracy_percent'):
            values = np.asarray([row[metric] for row in selected])
            item[metric + '_mean'] = float(values.mean())
            item[metric + '_sample_sd'] = None if len(values) < 2 else float(values.std(ddof=1))
        summaries.append(item)
    out = BASE / 'tables'
    out.mkdir(exist_ok=True)
    for name, values in (('development_per_seed.csv', rows), ('development_summary.csv', summaries)):
        with (out / name).open('w') as stream:
            if values:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    (out / 'development_summary.json').write_text(json.dumps({
        'status': 'DESCRIPTIVE_TABLES', 'rows': rows, 'summaries': summaries,
        'source_identities': inputs, 'scope': 'TRAIN-internal clientholdout development; no official TEST',
        'seed_policy': 'exactly three declared training seeds per task; incomplete groups explicitly labeled',
        'review': 'source numbers independently checked separately; this formatter is not external review'}, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'groups': len(summaries), 'complete_groups': sum(item['status'] == 'THREE_SEEDS_COMPLETE' for item in summaries)}))


if __name__ == '__main__':
    main()
