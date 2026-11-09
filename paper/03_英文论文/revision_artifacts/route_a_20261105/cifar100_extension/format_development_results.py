"""Format every predeclared CIFAR-100 development score after verification."""
from pathlib import Path
import csv
import json
import statistics

HERE = Path(__file__).resolve().parent
SEEDS = (20261231,20261232,20261233)
VARIANTS = ('raw','fixed_prior','zero','prior')


def main():
    verification = json.loads((HERE/'INDEPENDENT_RESULTS_VERIFICATION.json').read_text())
    if verification['status']!='PASS_INDEPENDENT_SAVED_ARRAY_REPLAY':
        raise ValueError('independent saved-array verification required')
    runs = [json.loads((HERE/'runs'/f's{s}'/'evaluation.json').read_text()) for s in SEEDS]
    if any(r['status']!='DONE' or r['official_test_reads'] for r in runs):
        raise ValueError('three complete development-only runs required')
    fields = ['seed','client','variant','train_samples','holdout_samples','accuracy','nll','brier','ece15']
    with (HERE/'client_metrics.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for client in run['clients']:
                for variant in VARIANTS:
                    m = client['metrics'][variant]
                    writer.writerow({'seed':run['seed'],'client':client['client'],'variant':variant,
                        'train_samples':client['train_samples'],'holdout_samples':client['holdout_samples'],
                        'accuracy':m['accuracy'],'nll':m['loss'],'brier':m['brier'],'ece15':m['ece15']})
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {key:{'mean':statistics.mean(r['summaries'][variant][key] for r in runs),
                                 'sample_sd':statistics.stdev(r['summaries'][variant][key] for r in runs)}
                            for key in runs[0]['summaries'][variant]}
    deltas = [r['summaries']['prior']['macro_loss']-r['summaries']['zero']['macro_loss'] for r in runs]
    improved = sum(c['metrics']['prior']['loss']<c['metrics']['zero']['loss'] for r in runs for c in r['clients'])
    report = {'status':'COMPLETE_VERIFIED_DEVELOPMENT','classes':100,'partition_seed':20261107,
              'model_seeds':SEEDS,'summary':summary,'prior_minus_zero_seed_nll':deltas,
              'prior_minus_zero_mean_nll':statistics.mean(deltas),'nll_improved_runs':sum(x<0 for x in deltas),
              'nll_improved_client_seed_cells':improved,'predictor_fp32_bytes':800,'upload_bytes':56376,
              'official_test_reads':0,'new_fits_or_cnn_runs':0,
              'scope':'One CIFAR100 label-skew partition; three fixed model seeds; TRAIN fit / development holdout only'}
    (HERE/'formatted_development.json').write_text(json.dumps(report,indent=2)+'\n')
    tex = [r'\begin{table}[t]\centering\small',
           r'\caption{CIFAR-100 development check on one label-skew partition, three paired model seeds. Both affine predictors store 200 FP32 values (800 B); shared three-bit updates occupy 56,376 B. Values are client-macro mean $\pm$ sample SD; accuracy is in percent.}',
           r'\label{tab:cifar100dev}',r'\begin{tabular}{lrr}\toprule',r'Predictor & NLL & Accuracy \\\midrule']
    names = {'raw':'Shared logits','fixed_prior':'Fixed prior addition','zero':'Zero-center affine','prior':'Prior-center affine'}
    for variant in VARIANTS:
        n,a = summary[variant]['macro_loss'],summary[variant]['macro_accuracy']
        tex.append(f"{names[variant]} & ${n['mean']:.4f}\\pm{n['sample_sd']:.4f}$ & "
                   f"${100*a['mean']:.2f}\\pm{100*a['sample_sd']:.2f}$ \\\\")
    tex.extend([r'\bottomrule\end{tabular}\end{table}',''])
    (HERE/'table_development_compact.tex').write_text('\n'.join(tex))
    lines = ['# Complete CIFAR-100 development results','',
             'One partition (20261107), three fixed model seeds, final round20; no TEST scoring or choice. '
             'This is an added-dataset check of the same fixed method, not a new mechanism or state-of-the-art comparison.','',
             '| Model seed | Zero macro NLL | Prior macro NLL | Prior-minus-zero NLL | Zero macro acc (%) | Prior macro acc (%) |',
             '|---:|---:|---:|---:|---:|---:|']
    for r,d in zip(runs,deltas):
        z,p = r['summaries']['zero'],r['summaries']['prior']
        lines.append(f"| {r['seed']} | {z['macro_loss']:.6f} | {p['macro_loss']:.6f} | {d:+.6f} | "
                     f"{100*z['macro_accuracy']:.4f} | {100*p['macro_accuracy']:.4f} |")
    lines.extend(['',f"NLL improves in {sum(x<0 for x in deltas)}/3 model runs and {improved}/60 client-seed cells. "
                  f"The mean prior-minus-zero NLL is {statistics.mean(deltas):+.6f}. "
                  'Every arm/client, Brier/ECE and macro/weighted metric remains in JSON/CSV. '
                  'The 800 B predictor excludes local fit memory and the FP32 EF residual.',''])
    (HERE/'DEVELOPMENT_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k!='summary'},indent=2))


if __name__=='__main__':
    main()
