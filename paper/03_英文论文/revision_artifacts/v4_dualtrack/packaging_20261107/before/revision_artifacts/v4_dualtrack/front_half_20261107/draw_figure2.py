"""Two-panel heterogeneity/size figure from frozen E1 and E2 outcomes."""
from pathlib import Path
import datetime
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / 'revision_artifacts/v4_dualtrack'
OUT = ROOT / 'figures'
META = Path(__file__).resolve().parent
SOURCES = [ARTIFACTS / 'e1_alpha_sweep.json', ARTIFACTS / 'e2_skew_gain.json']
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
hashes = {str(path): sha(path) for path in SOURCES}
e1, e2 = [json.loads(path.read_text()) for path in SOURCES]
series = [('cifar', 'CIFAR-10', '#0072B2', 'o', '-'),
          ('fashion', 'FashionMNIST', '#A64B00', 's', '--')]
source_data = {'E1': {}, 'E2': {}}
style = {'font.family': 'DejaVu Sans', 'font.size': 11,
         'axes.labelsize': 11, 'axes.titlesize': 11,
         'xtick.labelsize': 11, 'ytick.labelsize': 11,
         'legend.fontsize': 11, 'pdf.fonttype': 42,
         'axes.spines.top': False, 'axes.spines.right': False}
with plt.rc_context(style):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), layout='constrained')
    for task, label, color, marker, line in series:
        rows = e1['tasks'][task]['rows']
        x = np.array([r['alpha'] for r in rows])
        values = np.array([r['zero_minus_prior_macro_nll_gain']['individual_model_seed_values'] for r in rows])
        assert values.shape == (3, 3)
        axes[0].errorbar(x, values.mean(axis=1), yerr=values.std(axis=1, ddof=1),
                         marker=marker, color=color, linestyle=line, markersize=5,
                         capsize=3, linewidth=1.2, label=label)
        axes[0].scatter(np.repeat(x, 3), values.ravel(), color=color,
                        s=14, marker='x', linewidths=.8, zorder=4)
        source_data['E1'][task] = {'alpha': x.tolist(), 'seed_gains': values.tolist(),
                                  'mean': values.mean(axis=1).tolist(),
                                  'sample_sd': values.std(axis=1, ddof=1).tolist()}
        clients = e2['tasks'][task]['clients']
        n = np.array([c['train_rows'] for c in clients])
        gains = np.array([c['gain_nll_zero_minus_prior_by_seed'] for c in clients])
        assert gains.shape == (20, 3) and np.all(n > 0)
        means = gains.mean(axis=1)
        rho = float(spearmanr(n, means).statistic)
        expected = {'cifar': -.7714285714285714, 'fashion': -.7293233082706766}[task]
        assert abs(rho - expected) < 1e-12
        axes[1].scatter(np.repeat(n, 3), gains.ravel(), marker=marker,
                        s=12, linewidths=.55, facecolors='none', edgecolors=color,
                        alpha=.28, zorder=2)
        axes[1].scatter(n, means, marker=marker, color=color, s=22,
                        linewidths=.4, edgecolors='white', zorder=3,
                        label=f'{label}: ρ={rho:.3f}')
        source_data['E2'][task] = {'n_i': n.tolist(), 'seed_gains': gains.tolist(),
                                  'client_means': means.tolist(), 'spearman_rho': rho}
    axes[0].set_title('(a) Heterogeneity sweep', loc='left', pad=6)
    axes[0].set_xlabel('Dirichlet α')
    axes[0].set_ylabel('ΔNLL')
    axes[0].set_xticks([.1, .5, 1.0], ['0.1', '0.5', '1.0'])
    axes[0].set_xlim(.045, 1.055)
    axes[0].set_ylim(-.002, .047)
    axes[0].set_yticks([0, .02, .04], ['0', '0.02', '0.04'])
    axes[0].legend(loc='upper right', frameon=False, handlelength=1.1,
                   handletextpad=.4, borderaxespad=.15, labelspacing=.35)
    axes[1].set_title('(b) Client sample count', loc='left', pad=6)
    axes[1].set_xlabel('TRAIN samples nᵢ (log₁₀ scale)')
    axes[1].set_ylabel('ΔNLL')
    axes[1].set_xscale('log', base=10)
    axes[1].xaxis.set_major_locator(FixedLocator([200, 1000, 5000]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:,.0f}'))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_ylim(-.008, .213)
    axes[1].set_yticks([0, .1, .2], ['0', '0.10', '0.20'])
    axes[1].legend(loc='upper right', frameon=False, handletextpad=.1,
                   borderaxespad=.05, labelspacing=.35)
    for ax in axes:
        ax.axhline(0, color='.4', linewidth=.65, linestyle=':', zorder=1)
        ax.grid(axis='y', color='.9', linewidth=.6)
        ax.tick_params(length=3)
    OUT.mkdir(exist_ok=True)
    paths = [OUT / 'figure2_heterogeneity_dual.pdf', OUT / 'figure2_heterogeneity_dual.png']
    for path in paths:
        fig.savefig(path, dpi=300, facecolor='white', transparent=False)
    plt.close(fig)

for path, expected in hashes.items():
    assert sha(path) == expected, 'Source changed during figure generation'
provenance = {
    'created_at': datetime.datetime.now().astimezone().isoformat(),
    'source_sha256': hashes,
    'script_sha256': sha(__file__),
    'dimensions_inches': [6.8, 2.65],
    'panels': 2,
    'delta_definition': 'NLL(zero) - NLL(prior); positive is beneficial',
    'E1': {'x': 'alpha = 0.1, 0.5, 1.0', 'y': 'mean over three model-seed macro NLL gains',
           'uncertainty': 'sample SD over three model seeds; crosses show each seed',
           'assessment': 'TRAIN-derived development holdout',
           'scope': 'descriptive three-point sweep; one partition realization per alpha'},
    'E2': {'x': 'positive TRAIN count n_i; base-10 logarithmic axis',
           'y': 'three-seed mean gain for each of twenty clients per task',
           'observations': 'filled markers are client means; hollow markers are three seed observations',
           'correlation': 'Spearman over twenty client means, separately by task',
           'assessment': 'saved original TEST outcome metrics; no new inference',
           'scope': 'association, not causal inference'},
    'filters': 'none; all source task/alpha/seed/client observations shown',
    'smoothing': 'none; E1 lines join the three observed means',
    'font_points_at_export': 11,
    'minimum_final_width_inches_for_7pt': 6.8 * 7 / 11,
    'outputs_sha256': {str(path): sha(path) for path in paths},
    'software': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
                 'numpy': np.__version__},
    'data': source_data,
}
(META / 'figure2_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
(META / 'figure2_alt.md').write_text('Two panels compare prior-centered prediction gains. In the left three-point sweep, CIFAR-10 and FashionMNIST mean gain is greatest at alpha0.1 and falls at alpha0.5 and1.0; errors show seed sample SD. The right panel plots all twenty client means per task against training sample count on a base-10 logarithmic axis, with hollow seed observations; Spearman rho is -0.771 for CIFAR-10 and -0.729 for FashionMNIST. Positive delta means lower NLL under prior centering.\n')
print(json.dumps({'outputs': [str(p) for p in paths], 'dimensions_inches': [6.8, 2.65],
                  'minimum_final_width_for_7pt_inches': 6.8 * 7 / 11}))
