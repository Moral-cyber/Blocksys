"""Read-only reanalysis of saved metrics, joined to original TRAIN manifests."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
METHOD = PAPER / '05_核心方法研发'
SOURCE = HERE.parent / 'route_a_20261105/stats/result.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def associations(x, y, matrix):
    x, y = np.asarray(x), np.asarray(y)
    loo = [float(spearmanr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(20)]
    return {'pearson_r': float(pearsonr(x, y).statistic),
            'spearman_rho': float(spearmanr(x, y).statistic),
            'per_seed_spearman_rho': [float(spearmanr(x, row).statistic) for row in matrix],
            'per_seed_pearson_r': [float(pearsonr(x, row).statistic) for row in matrix],
            'leave_one_client_out_spearman_min': min(loo),
            'leave_one_client_out_spearman_max': max(loo),
            'leave_one_client_out_spearman': loo,
            'ols_slope_gain_per_unit_x': float(np.polyfit(x, y, 1)[0]),
            'scope': 'descriptive, conditional on one partition and three shared seed-models; no independent-pair p-value'}


def run():
    source = json.loads(SOURCE.read_text())
    hashes = {str(SOURCE): sha(SOURCE), str(Path(__file__)): sha(__file__),
              str(HERE / 'PLAN.md'): sha(HERE / 'PLAN.md')}
    for p, expected in source['source_and_input_sha256'].items():
        assert sha(p) == expected, ('source identity mismatch', p)
        hashes[p] = expected
    result = {'status': 'COMPLETE', 'created_at': dt.datetime.now().astimezone().isoformat(),
        'design': {'datasets': ['CIFAR-10', 'FashionMNIST'], 'clients_per_task': 20,
            'model_seeds_per_task': 3, 'new_training_runs': 0, 'new_fits': 0,
            'new_test_inference': 0, 'official_test_bytes_opened': 0,
            'outcome_source': 'previously saved official-TEST metrics; no new TEST inference or repartition',
            'skew_source': 'unsmoothed original client TRAIN label counts',
            'delta': 'NLL(prior) - NLL(zero); negative favors prior',
            'gain': 'NLL(zero) - NLL(prior); positive favors prior',
            'max_class_fraction': 'max_k n_k / sum_k n_k',
            'normalized_entropy': '-sum_k p_k ln(p_k) / ln(10); 0 ln 0 = 0',
            'aggregation': 'unweighted mean over three seeds for each client, then equal-client mean',
            'hypothesis': 'Greater heterogeneity produces greater gain is unverified; associations are descriptive',
            'limitations': ['Clients share a trained model within seed; 60 pairs are dependent.',
                'One fixed partition per dataset, not replicated partition sampling.',
                'Skew correlates with client sample count; association is not causation.',
                'E2 uses historical saved TEST outcomes; E1 uses a separate development assessment.'],
            'exclusions': 'none'}, 'tasks': {}}
    rows = []
    for task, old in source['tasks'].items():
        seeds, partition = old['seeds'], old['partition_seed']
        manifests = []
        for seed in seeds:
            p = METHOD / f'results/predictive_pfl_confirm/development_{task}_p{partition}_s{seed}/fedlc_uniform3/config.json'
            hashes[str(p)] = sha(p)
            manifests.append(json.loads(p.read_text())['data_manifest'])
        for key in ['train_indices', 'holdout_indices', 'validation_indices', 'client_label_counts', 'client_sizes']:
            assert all(m[key] == manifests[0][key] for m in manifests), (task, key)
        counts = np.asarray(manifests[0]['client_label_counts'], dtype=np.int64)
        sizes = np.asarray(manifests[0]['client_sizes'], dtype=np.int64)
        assert counts.shape == (20, 10) and np.all(counts >= 0)
        assert np.array_equal(counts.sum(axis=1), sizes) and np.all(sizes > 0)
        p = counts / sizes[:, None]
        maximum = p.max(axis=1)
        logp = np.zeros_like(p); np.log(p, out=logp, where=p > 0)
        entropy = -(p * logp).sum(axis=1) / np.log(10)
        delta = np.asarray(old['paired_nll']['client_seed_delta_matrix'], dtype=float)
        assert delta.shape == (3, 20) and np.isfinite(delta).all()
        reconstructed = []
        for seed in seeds:
            path = METHOD / f'results/predictive_final_test/official_{task}/fedlc_uniform3_s{seed}/result.json'
            saved = sorted(json.loads(path.read_text())['clients'], key=lambda r: r['client'])
            assert [c['client'] for c in saved] == list(range(20))
            reconstructed.append([c['metrics']['prioranchored_vector']['nll'] - c['metrics']['zero_anchor_vector']['nll'] for c in saved])
        error = float(np.max(np.abs(delta - reconstructed)))
        assert error < 1e-12
        gain = -delta; means = gain.mean(axis=0)
        assert np.allclose(-means, old['paired_nll']['client_mean_deltas'], rtol=0, atol=1e-12)
        correlations = {name: associations(x, means, gain) for name, x in
            [('max_class_fraction', maximum), ('normalized_entropy', entropy), ('train_rows', sizes)]}
        observations = []
        for client in range(20):
            observations.append({'client': client, 'train_rows': int(sizes[client]),
                'train_class_counts': counts[client].tolist(), 'train_class_probabilities': p[client].tolist(),
                'max_class_fraction': float(maximum[client]), 'normalized_entropy': float(entropy[client]),
                'delta_nll_prior_minus_zero_by_seed': delta[:, client].tolist(),
                'gain_nll_zero_minus_prior_by_seed': gain[:, client].tolist(),
                'mean_delta_nll': float(-means[client]), 'mean_gain_nll': float(means[client]),
                'gain_seed_sample_sd': float(gain[:, client].std(ddof=1))})
            for j, seed in enumerate(seeds):
                rows.append({'task': task, 'client': client, 'seed': seed, 'train_rows': int(sizes[client]),
                    'max_class_fraction': maximum[client], 'normalized_entropy': entropy[client],
                    'delta_nll_prior_minus_zero': delta[j, client], 'gain_nll_zero_minus_prior': gain[j, client],
                    **{f'train_count_class_{k}': int(counts[client, k]) for k in range(10)}})
        result['tasks'][task] = {'partition_seed': partition, 'alpha': manifests[0]['alpha'], 'seeds': seeds,
            'mean_gain_nll': float(means.mean()), 'seed_macro_gains': gain.mean(axis=1).tolist(),
            'mean_gain_seed_sample_sd': float(gain.mean(axis=1).std(ddof=1)),
            'improved_client_seed_pairs': int((gain > 0).sum()), 'total_client_seed_pairs': 60,
            'mean_max_class_fraction': float(maximum.mean()), 'mean_normalized_entropy': float(entropy.mean()),
            'correlations_with_gain': correlations,
            'skew_vs_train_rows_spearman': {'max_class_fraction': float(spearmanr(maximum, sizes).statistic),
                'normalized_entropy': float(spearmanr(entropy, sizes).statistic)},
            'source_reconstruction_max_abs_error': error, 'clients': observations}
    result['source_and_input_sha256'] = hashes
    result['software'] = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__}
    with (HERE / 'e2_client_data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(result)
    result['output_sha256'] = {str(p): sha(p) for p in [HERE / 'e2_client_data.csv', HERE / 'figures/skew_vs_gain.pdf', HERE / 'figures/skew_vs_gain.png']}
    write(HERE / 'e2_skew_gain.json', result)
    for task, value in result['tasks'].items():
        print(json.dumps({'task': task, 'gain': value['mean_gain_nll'], 'correlations': value['correlations_with_gain']}, ensure_ascii=False))


def plot(result):
    (HERE / 'figures').mkdir(exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.1), layout='constrained', sharey='row')
        for i, (task, value) in enumerate(result['tasks'].items()):
            clients = value['clients']; color = ['#0072B2', '#9E4F00'][i]
            for j, (measure, label) in enumerate([('max_class_fraction', 'Maximum TRAIN class fraction (higher = more skew)'),
                                                  ('normalized_entropy', 'Normalized TRAIN entropy (lower = more skew)')]):
                ax = axes[i, j]
                x = np.array([c[measure] for c in clients]); y = np.array([c['mean_gain_nll'] for c in clients])
                seed_y = np.array([c['gain_nll_zero_minus_prior_by_seed'] for c in clients])
                ax.scatter(np.repeat(x, 3), seed_y.ravel(), s=12, marker='x', color='#777777', linewidth=.6, label='Seed observations')
                ax.scatter(x, y, s=25, color=color, marker='o' if i == 0 else 's', label='Client mean (3 seeds)', zorder=3)
                ax.axhline(0, color='#444444', lw=.7)
                rho = value['correlations_with_gain'][measure]['spearman_rho']
                ax.set_title(f"{'CIFAR-10' if task == 'cifar' else 'FashionMNIST'} | Spearman rho = {rho:+.3f}", fontsize=10)
                ax.set_xlabel(label, fontsize=8)
                ax.set_ylim(bottom=-.005)
                if j == 0: ax.set_ylabel('NLL gain = zero - prior (nats)')
                ax.grid(alpha=.15)
                if i == 0 and j == 1: ax.legend(fontsize=7, loc='upper right')
        fig.suptitle('Existing saved outcomes: 20 clients per task, one fixed partition', fontsize=11)
        fig.savefig(HERE / 'figures/skew_vs_gain.pdf', facecolor='white')
        fig.savefig(HERE / 'figures/skew_vs_gain.png', dpi=200, facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    run()
