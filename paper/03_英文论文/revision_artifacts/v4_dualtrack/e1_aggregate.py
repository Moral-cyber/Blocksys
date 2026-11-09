"""Aggregate all prespecified E1 development runs and draw seed-level gains."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ALPHAS = (0.1, 0.5, 1.0)
SEEDS = {'cifar': (20261231, 20261232, 20261233),
         'fashion': (20261241, 20261242, 20261243)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def describe(values):
    return {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)),
            'individual_model_seed_values': [float(v) for v in values]}


def main():
    inputs, tasks = {}, {}
    for task, seeds in SEEDS.items():
        rows = []
        gain_matrix = []
        for alpha in ALPHAS:
            runs = []
            for seed in seeds:
                path = HERE / 'e1_runs' / task / ('alpha_' + str(alpha)) / ('s' + str(seed)) / 'evaluation.json'
                value = json.loads(path.read_text())
                if (value['status'] != 'COMPLETE' or value['task'] != task or value['seed'] != seed or
                        value['alpha'] != alpha or value['rounds'] != 20 or value['official_test_reads'] != 0):
                    raise ValueError('incomplete or incompatible E1 run: ' + str(path))
                for name, digest in {**value['inputs'], **value['outputs']}.items():
                    if sha(name) != digest:
                        raise ValueError('E1 input/output changed: ' + name)
                checkpoint = path.parent / 'fedlc_uniform3/checkpoint.pt'
                if sha(checkpoint) != value['checkpoint_sha256']:
                    raise ValueError('E1 checkpoint changed')
                inputs[str(path)] = sha(path)
                runs.append(value)
            gains = [r['zero_minus_prior_macro_nll_gain'] for r in runs]
            deltas = [r['prior_minus_zero_macro_nll'] for r in runs]
            gain_matrix.append(gains)
            metrics = {variant: {metric: describe([r['summaries'][variant][metric] for r in runs])
                                for metric in runs[0]['summaries'][variant]}
                       for variant in runs[0]['summaries']}
            if any(r['skew'] != runs[0]['skew'] for r in runs):
                raise ValueError('model seeds unexpectedly changed the partition')
            client_deltas = [[c['prior_minus_zero_nll'] for c in r['clients']] for r in runs]
            rows.append({'alpha': alpha, 'seeds': seeds, 'partition_seed': runs[0]['partition_seed'],
                         'metrics': metrics, 'prior_minus_zero_macro_nll': describe(deltas),
                         'zero_minus_prior_macro_nll_gain': describe(gains),
                         'all_three_model_seeds_improve': bool(np.all(np.asarray(gains) > 0)),
                         'improved_client_seed_pairs': int(np.count_nonzero(np.asarray(client_deltas) < 0)),
                         'client_seed_pairs': 60, 'client_seed_delta_matrix': client_deltas,
                         'skew': runs[0]['skew']})
        gain_matrix = np.asarray(gain_matrix)
        means = gain_matrix.mean(axis=1)
        mean_monotonic = bool(np.all(np.diff(means) < 0))
        realized_order = bool(np.all(np.diff([r['skew']['mean_maximum_class_fraction'] for r in rows]) < 0)
                              and np.all(np.diff([r['skew']['mean_normalized_entropy'] for r in rows]) > 0))
        per_seed_monotonic = np.all(np.diff(gain_matrix, axis=0) < 0, axis=0)
        tasks[task] = {'rows': rows,
                       'realized_heterogeneity_follows_alpha_order': realized_order,
                       'greater_heterogeneity_greater_gain_mean_order_observed': mean_monotonic,
                       'monotonic_order_by_model_seed': per_seed_monotonic.tolist(),
                       'interpretation': ('Mean gain ordering matches the prespecified hypothesis in this fixed partition family.'
                                          if mean_monotonic else 'Mean gain ordering does not match the prespecified monotonic hypothesis.'),
                       'uncertainty': 'Only one partition realization per alpha; three model seeds share each partition. Alpha also changes client sizes, memberships and learned model quality. This is descriptive evidence, not a general or causal heterogeneity law.'}
    supported = all(t['greater_heterogeneity_greater_gain_mean_order_observed'] and
                    t['realized_heterogeneity_follows_alpha_order'] for t in tasks.values())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    for ax, (task, data) in zip(axes, tasks.items()):
        rows = data['rows']
        mean = [r['zero_minus_prior_macro_nll_gain']['mean'] for r in rows]
        sd = [r['zero_minus_prior_macro_nll_gain']['sample_sd'] for r in rows]
        vals = np.array([r['zero_minus_prior_macro_nll_gain']['individual_model_seed_values'] for r in rows])
        x = np.arange(3)
        ax.errorbar(x, mean, yerr=sd, marker='o', color='#0072B2', capsize=4,
                    linewidth=1.5, label='Mean ± sample SD')
        for seed_index, marker in enumerate(('s', '^', 'D')):
            ax.scatter(x + (seed_index - 1) * .035, vals[:, seed_index], marker=marker,
                       s=25, color='#D55E00', edgecolor='white', linewidth=.4,
                       label='Individual model seeds' if seed_index == 0 else None, zorder=4)
        ax.axhline(0., color='0.5', linestyle='--', linewidth=.8)
        ax.set_xticks(x, [str(a) for a in ALPHAS])
        ax.set_xlabel('Dirichlet α (smaller = stronger nominal skew)')
        ax.set_ylabel('Development NLL gain (zero − prior)')
        ax.set_title('CIFAR-10' if task == 'cifar' else 'FashionMNIST')
        ax.grid(axis='y', alpha=.18)
        ax.legend(fontsize=7, loc='best', frameon=False)
    figure = HERE / 'figures/alpha_vs_gain.pdf'
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure, metadata={'Title': 'E1 alpha sweep: development NLL gain',
                                'Subject': 'Positive gain means lower NLL with the prior anchor; 3 model seeds, one partition per alpha'})
    fig.savefig(figure.with_suffix('.png'), dpi=180)
    plt.close(fig)
    result = {'status': 'COMPLETE', 'created_at': datetime.now(timezone.utc).isoformat(),
              'design': {'tasks': list(SEEDS), 'alphas': ALPHAS, 'model_seeds': SEEDS,
                         'runs': 18, 'clients': 20, 'rounds': 20, 'lambda': 10,
                         'primary_unit': 'paired model-seed macro client NLL difference; 3 per task/alpha',
                         'delta_convention': 'prior minus zero; negative is beneficial',
                         'gain_convention': 'zero minus prior; positive is beneficial',
                         'data_scope': 'official TRAIN for model training and local fitting; disjoint TRAIN-derived development holdouts for scoring; no official TEST',
                         'hypothesis': 'Greater heterogeneity produces greater gain is a prespecified hypothesis, never an assumed conclusion.',
                         'decision_rule': 'Descriptive mean ordering G(alpha=.1)>G(.5)>G(1) assessed separately on both tasks; all raw seed/client outcomes retained, no tuning to force ordering.'},
              'tasks': tasks, 'mean_order_observed_on_both_tasks': supported,
              'overall_interpretation': ('Prespecified mean ordering observed on both tasks within this design; replication across partitions remains required for a general claim.'
                                         if supported else 'Prespecified monotonic claim is not supported on both tasks; retain the observed alpha-specific effects.'),
              'inputs': inputs, 'figure': str(figure), 'figure_sha256': sha(figure),
              'official_test_reads': 0, 'official_test_reassignments': 0,
              'software': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                           'analysis_script_sha256': sha(Path(__file__))}}
    write(HERE / 'e1_alpha_sweep.json', result)
    print(json.dumps({'status': 'E1_COMPLETE', 'runs': 18,
                      'mean_order_observed_on_both_tasks': supported,
                      'gains': {task: [{ 'alpha': row['alpha'], **row['zero_minus_prior_macro_nll_gain']}
                                       for row in value['rows']] for task, value in tasks.items()}}), flush=True)


if __name__ == '__main__':
    main()
