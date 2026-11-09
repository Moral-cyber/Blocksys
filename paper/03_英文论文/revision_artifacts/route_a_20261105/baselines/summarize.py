"""Build a compact appendix table only after all fixed baseline scores exist."""
from pathlib import Path
import csv
import importlib.util
import json

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_baseline_paper_summary', HERE / 'adapter.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.verify(json.loads((HERE / 'SOURCE_LOCK.json').read_text())['inputs'])
tasks = {task: json.loads((HERE / 'official_test' / task / 'result.json').read_text()) for task in a.TASKS}
if any(value['status'] != 'DONE' or tuple(value['seeds']) != a.TASKS[task][1] for task, value in tasks.items()):
    raise ValueError('all fixed three-seed task scores required')

columns = ['task', 'seed', 'method', 'predictor', 'client', 'test_rows', 'accuracy', 'nll', 'brier', 'ece15']
with (HERE / 'per_client_seed_metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for task, (_, seeds) in a.TASKS.items():
        for seed in seeds:
            for arm in a.ARMS:
                path = HERE / 'official_test' / task / f'{arm}_s{seed}' / 'result.json'
                result = json.loads(path.read_text())
                for client in result['clients']:
                    for variant in a.VARIANTS:
                        writer.writerow({'task': task, 'seed': seed, 'method': arm, 'predictor': variant,
                                         'client': client['client'], 'test_rows': client['test_rows'],
                                         **{name: client['metrics'][variant][name] for name in a.METRICS}})

def cell(task, arm, variant, metric):
    value = tasks[task]['summaries'][arm][variant][metric]
    factor = 100 if metric.endswith('accuracy') else 1
    digits = 2 if factor == 100 else 4
    return f"{value['mean'] * factor:.{digits}f} $\\pm$ {value['sample_sd'] * factor:.{digits}f}"

names = {'fedavg_dense': 'FedAvg', 'fedprox01_dense': 'FedProx ($\\mu=0.1$)'}
predictors = {'raw': 'Raw', 'zero_anchor_vector': '+ zero center', 'prioranchored_vector': '+ prior center'}
rows = []
for arm in a.ARMS:
    for variant in a.VARIANTS:
        rows.append(names[arm] + ' & ' + predictors[variant] + ' & ' +
                    ' & '.join(cell(task, arm, variant, metric) for task in a.TASKS for metric in ('macro_nll', 'macro_accuracy')) + r' \\')
tex = '\n'.join([
    r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3.2pt}',
    r'\caption{Dense basic federated baselines on the same partitions, backbone, and three training seeds. Raw is the original baseline predictor; zero/prior center are separately fitted post-hoc variants. Values are client-macro mean $\pm$ sample SD; accuracy is in percent. These are matched in-house implementations.}',
    r'\label{tab:dense_baselines}', r'\begin{tabular}{llrrrr}', r'\toprule',
    r'& & \multicolumn{2}{c}{CIFAR-10} & \multicolumn{2}{c}{FashionMNIST} \\',
    r'Base training & Predictor & NLL & Acc. & NLL & Acc. \\', r'\midrule',
    *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
(HERE / 'table_compact.tex').write_text(tex)

# A full-metric appendix alternative, and unrounded machine-readable values.
full = []
for task in a.TASKS:
    for arm in a.ARMS:
        for variant in a.VARIANTS:
            full.append(task + ' & ' + names[arm] + ' & ' + predictors[variant] + ' & ' +
                        ' & '.join(cell(task, arm, variant, metric) for metric in ('macro_nll', 'macro_accuracy', 'macro_brier', 'macro_ece15')) + r' \\')
(HERE / 'table_all_macro_metrics.tex').write_text('\n'.join([
    r'\begin{table}[t]', r'\centering\scriptsize', r'\setlength{\tabcolsep}{3pt}',
    r'\caption{Matched dense FedAvg and FedProx: client-macro mean $\pm$ sample SD over the three fixed training seeds. Raw denotes the basic algorithm; zero/prior center add post-hoc vector calibration.}',
    r'\label{tab:dense_baselines_full}', r'\begin{tabular}{lllrrrr}', r'\toprule',
    r'Task & Base training & Predictor & NLL & Acc. (\%) & Brier & ECE \\', r'\midrule',
    *full, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']))

paired = {}
for task, (_, seeds) in a.TASKS.items():
    paired[task] = {}
    for arm in a.ARMS:
        delta = []
        for seed in seeds:
            run = json.loads((HERE / 'official_test' / task / f'{arm}_s{seed}' / 'result.json').read_text())
            delta.append(run['summaries']['prioranchored_vector']['macro_nll'] - run['summaries']['zero_anchor_vector']['macro_nll'])
        paired[task][arm] = {'prior_minus_zero_macro_nll': delta, 'mean': float(np.mean(delta)),
                             'sample_sd': float(np.std(delta, ddof=1)), 'all_three_negative': bool(np.all(np.asarray(delta) < 0))}
a.write(HERE / 'summary.json', {'status': 'DONE', 'tasks': tasks, 'paired_nll': paired,
                              'definitions': {'raw': 'base algorithm predictor', 'zero/prior': 'our added post-hoc calibrators'},
                              'no_test_selection': True})
print(json.dumps({'status': 'SUMMARY_DONE', 'paired': paired}, indent=2))
