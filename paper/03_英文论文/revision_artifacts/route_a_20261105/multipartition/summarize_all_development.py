"""Summarize three partitions per task without reading official TEST or models.

Requires the entire prespecified new matrix. The original partition is read from
the saved development probe, never substituted with the original TEST table.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import csv
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/core_contribution_probe/result.json')
PARTITIONS = {'cifar': (20261217, 20261105, 20261106),
              'fashion': (20261219, 20261105, 20261106)}
SEEDS = {'cifar': (20261231, 20261232, 20261233),
         'fashion': (20261241, 20261242, 20261243)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_summary(clients, getter):
    weights = [int(c['holdout_samples']) for c in clients]
    if len(clients) != 20 or min(weights) <= 0:
        raise ValueError('expected twenty nonempty client development holdouts')
    result = {}
    for variant in ('zero', 'prior'):
        values = [getter(c, variant) for c in clients]
        result[variant] = {
            'macro_nll': statistics.mean(float(v['nll']) for v in values),
            'macro_accuracy': statistics.mean(float(v['accuracy']) for v in values),
            'weighted_nll': sum(n*float(v['nll']) for n,v in zip(weights, values))/sum(weights),
            'weighted_accuracy': sum(n*float(v['accuracy']) for n,v in zip(weights, values))/sum(weights),
        }
    diffs = [float(getter(c,'prior')['nll'])-float(getter(c,'zero')['nll']) for c in clients]
    result['delta_nll'] = result['prior']['macro_nll']-result['zero']['macro_nll']
    result['delta_accuracy_pp'] = 100*(result['prior']['macro_accuracy']-result['zero']['macro_accuracy'])
    result['client_nll_improved'] = sum(v < 0 for v in diffs)
    result['client_nll_tied'] = sum(v == 0 for v in diffs)
    return result


def main():
    old = json.loads(ORIGINAL.read_text())
    if old['status'] != 'DONE' or old['official_test_reads'] != 0:
        raise ValueError('original development result scope is invalid')
    identities = {str(ORIGINAL): sha(ORIGINAL)}
    rows = []
    for task, partitions in PARTITIONS.items():
        old_runs = {r['training_seed']: r for r in old['tasks'][task]['runs']}
        if set(old_runs) != set(SEEDS[task]):
            raise ValueError('original development seeds mismatch')
        for partition in partitions:
            for seed in SEEDS[task]:
                if partition == partitions[0]:
                    run = old_runs[seed]
                    if run['partition_seed'] != partition:
                        raise ValueError('original partition mismatch')
                    def original_metric(client, variant):
                        return client['metrics']['holdout']['correct' if variant == 'prior' else 'zero']
                    metrics = pair_summary(run['clients'], original_metric)
                    for variant, key in (('prior','correct'), ('zero','zero')):
                        expected = run['summaries']['holdout'][key]
                        if not math.isclose(metrics[variant]['macro_nll'], expected['nll'], abs_tol=1e-12):
                            raise ValueError('original client aggregation mismatch')
                    source = str(ORIGINAL)
                else:
                    path = HERE/'runs'/task/f'p{partition}_s{seed}'/'evaluation.json'
                    run = json.loads(path.read_text())
                    if (run['status'] != 'DONE' or run['official_test_reads'] != 0 or
                        not run['identities_unchanged'] or run['task'] != task or
                        run['partition_seed'] != partition or run['seed'] != seed):
                        raise ValueError('new evaluation identity/scope mismatch')
                    def new_metric(client, variant):
                        value = client['metrics']['prioranchored_vector' if variant == 'prior' else 'zero_anchor_vector']
                        return {'nll': value['loss'], 'accuracy': value['accuracy']}
                    metrics = pair_summary(run['clients'], new_metric)
                    for variant, key in (('prior','prioranchored_vector'), ('zero','zero_anchor_vector')):
                        expected = run['summaries'][key]
                        for m, source_m in (('macro_nll','macro_loss'), ('weighted_nll','weighted_loss'),
                                            ('macro_accuracy','macro_accuracy'), ('weighted_accuracy','weighted_accuracy')):
                            if not math.isclose(metrics[variant][m], expected[source_m], abs_tol=1e-12):
                                raise ValueError('new client aggregation mismatch')
                    source = str(path)
                    identities[source] = sha(path)
                rows.append({'task': task, 'partition_seed': partition, 'training_seed': seed,
                             'source': source, **metrics})
    if len(rows) != 18:
        raise ValueError('complete eighteen-configuration table required')
    grouped = []
    for task, partitions in PARTITIONS.items():
        for partition in partitions:
            rs = [r for r in rows if r['task'] == task and r['partition_seed'] == partition]
            metrics = {v: {m: {'mean': statistics.mean(r[v][m] for r in rs),
                               'sample_sd': statistics.stdev(r[v][m] for r in rs)}
                           for m in rs[0][v]} for v in ('zero','prior')}
            grouped.append({'task':task, 'partition_seed':partition, 'metrics':metrics,
                            'delta_nll_mean':statistics.mean(r['delta_nll'] for r in rs),
                            'delta_nll_sample_sd':statistics.stdev(r['delta_nll'] for r in rs),
                            'delta_accuracy_pp_mean':statistics.mean(r['delta_accuracy_pp'] for r in rs),
                            'seed_nll_improved':sum(r['delta_nll'] < 0 for r in rs),
                            'client_seed_nll_improved':sum(r['client_nll_improved'] for r in rs),
                            'individual_seed_deltas':[r['delta_nll'] for r in rs]})
    before = dict(identities)
    if any(sha(path) != value for path,value in before.items()):
        raise ValueError('source changed during summary')
    payload = {'status':'DONE', 'created_at':datetime.now(timezone.utc).isoformat(),
               'scope':'TRAIN-fit/designated-development-holdout score only; three partitions x three model seeds per task',
               'primary_metric':'client-macro NLL; negative prior-minus-zero delta favors prior',
               'official_test_reads':0, 'new_training_or_fits_in_this_summary':0,
               'all_prespecified_configurations_reported':True, 'rows':rows, 'partitions':grouped,
               'source_sha256':before, 'summary_script_sha256':sha(__file__),
               'independence_note':'Nine configurations per task reuse three model-seed numbers across three partitions; do not call them nine independent training seeds. Client/seed cells are dependent.'}
    (HERE/'combined_development.json').write_text(json.dumps(payload, indent=2)+'\n')
    fields = ['task','partition_seed','training_seed','zero_macro_nll','prior_macro_nll',
              'delta_nll','zero_macro_accuracy','prior_macro_accuracy','delta_accuracy_pp',
              'zero_weighted_nll','prior_weighted_nll','zero_weighted_accuracy',
              'prior_weighted_accuracy','client_nll_improved','client_nll_tied','source']
    with (HERE/'combined_development.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            item = {k:r[k] for k in ('task','partition_seed','training_seed','delta_nll',
                                     'delta_accuracy_pp','client_nll_improved','client_nll_tied','source')}
            for variant in ('zero','prior'):
                item.update({variant+'_'+key:value for key,value in r[variant].items()})
            writer.writerow(item)
    tex = [r'\begin{table}[t]', r'\centering', r'\scriptsize',
           r'\caption{Partition check on development holdouts: client-macro NLL, mean $\pm$ sample SD over three paired model seeds per partition. $\Delta$ is prior minus zero center; fitting uses TRAIN only. Original TEST scores are separate.}',
           r'\label{tab:partitiondev}', r'\begin{tabular}{llrrr}',
           r'\toprule', r'Task & Partition & Zero center & Prior center & $\Delta$ \\', r'\midrule']
    for r in grouped:
        z,p = r['metrics']['zero']['macro_nll'], r['metrics']['prior']['macro_nll']
        task_label = 'CIFAR-10' if r['task']=='cifar' else 'Fashion'
        tex.append(f"{task_label} & {r['partition_seed']} & ${z['mean']:.4f}\\pm{z['sample_sd']:.4f}$ & "
                   f"${p['mean']:.4f}\\pm{p['sample_sd']:.4f}$ & ${r['delta_nll_mean']:+.4f}$ \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (HERE/'table_development_compact.tex').write_text('\n'.join(tex))
    md = ['# Three-partition development comparison', '',
          'Frozen original post-hoc protocol: fixed lambda 10, TRAIN fitting and designated client holdout scoring. '
          'These are development results; the protected original TEST values are not mixed into this table. '
          'Each partition uses the same three model-seed numbers. All eighteen configurations are retained.', '',
          '| Task | Partition | Zero NLL mean ± SD | Prior NLL mean ± SD | Δ NLL | Δ macro accuracy (pp) | NLL improved seeds | Improved client-seed cells |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in grouped:
        z,p = r['metrics']['zero']['macro_nll'], r['metrics']['prior']['macro_nll']
        md.append(f"| {r['task']} | {r['partition_seed']} | {z['mean']:.6f} ± {z['sample_sd']:.6f} | "
                  f"{p['mean']:.6f} ± {p['sample_sd']:.6f} | {r['delta_nll_mean']:+.6f} | "
                  f"{r['delta_accuracy_pp_mean']:+.4f} | {r['seed_nll_improved']}/3 | {r['client_seed_nll_improved']}/60 |")
    md.extend(['', 'The SD above describes the three model runs conditional on a partition. It is not a partition-level confidence interval. '
               'Negative Δ NLL favors the prior center. No significance or universal generalization claim is inferred from these cells.', '',
               '## Every development configuration', '',
               '| Task | Partition | Model seed | Zero macro NLL | Prior macro NLL | Δ NLL | Zero macro acc (%) | Prior macro acc (%) | Zero weighted NLL | Prior weighted NLL | Zero weighted acc (%) | Prior weighted acc (%) | Improved clients |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in rows:
        z,p = r['zero'],r['prior']
        md.append(f"| {r['task']} | {r['partition_seed']} | {r['training_seed']} | {z['macro_nll']:.6f} | "
                  f"{p['macro_nll']:.6f} | {r['delta_nll']:+.6f} | {100*z['macro_accuracy']:.4f} | "
                  f"{100*p['macro_accuracy']:.4f} | {z['weighted_nll']:.6f} | {p['weighted_nll']:.6f} | "
                  f"{100*z['weighted_accuracy']:.4f} | {100*p['weighted_accuracy']:.4f} | {r['client_nll_improved']}/20 |")
    md.extend(['', 'Source paths and SHA-256 identities for the original probe and all twelve new evaluations '
               'are in `combined_development.json`. This script reads saved development scores only; '
               'it does not fit predictors, forward images, modify checkpoints, or read official TEST.', ''])
    (HERE/'combined_development.md').write_text('\n'.join(md))
    print(json.dumps({'status':'DONE','configurations':len(rows),'partition_rows':len(grouped),
                      'client_seed_cells':len(rows)*20,
                      'all_partition_means_prior_better':all(r['delta_nll_mean'] < 0 for r in grouped),
                      'all_run_means_prior_better':all(r['delta_nll'] < 0 for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
