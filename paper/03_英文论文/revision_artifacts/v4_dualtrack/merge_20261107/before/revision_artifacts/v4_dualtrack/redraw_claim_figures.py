from pathlib import Path
import json, hashlib, csv
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
P=Path(__file__).resolve().parent
A=P/'revision_20261107'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
e1=json.loads((P/'e1_alpha_sweep.json').read_text());e2=json.loads((P/'e2_skew_gain.json').read_text())
style={'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}
figdata={'delta_definition':'NLL(zero)-NLL(prior); positive is beneficial','E1_claim':'In this three-point sweep, stronger heterogeneity accompanies larger mean gain.','E2_claim':'TRAIN sample count is inversely associated with gain; descriptive association.','sources':{str(P/n):sha(P/n) for n in ['e1_alpha_sweep.json','e2_skew_gain.json']},'E1':{},'E2':{}}
with plt.rc_context(style):
 fig,ax=plt.subplots(figsize=(6.7,3.4),layout='constrained')
 for j,(task,t) in enumerate(e1['tasks'].items()):
  x=np.array([r['alpha'] for r in t['rows']]);v=np.array([r['zero_minus_prior_macro_nll_gain']['individual_model_seed_values'] for r in t['rows']]);color=['#0072B2','#A64B00'][j];marker=['o','s'][j]
  ax.errorbar(x,v.mean(1),yerr=v.std(1,ddof=1),marker=marker,color=color,capsize=4,label=['CIFAR-10','FashionMNIST'][j]);ax.scatter(np.repeat(x,3),v.ravel(),s=15,marker='x',color=color,zorder=4)
  figdata['E1'][task]={'alpha':x.tolist(),'delta_by_alpha_and_seed':v.tolist(),'improved_pairs_at_alpha0.1':t['rows'][0]['improved_client_seed_pairs']}
 ax.axhline(0,color='.5',ls='--',lw=.7);ax.set_xticks([.1,.5,1]);ax.set(xlabel='Dirichlet α (smaller = stronger nominal heterogeneity)',ylabel='ΔNLL = NLL(zero) − NLL(prior)',title='Larger mean gain under stronger heterogeneity in this three-point sweep');ax.legend();ax.grid(axis='y',alpha=.15)
 fig.savefig(P/'figures/figure1_delta_nll_vs_alpha.pdf');fig.savefig(P/'figures/figure1_delta_nll_vs_alpha.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.3),layout='constrained')
 for j,((task,t),ax) in enumerate(zip(e2['tasks'].items(),axes)):
  cs=t['clients'];x=np.array([c['train_rows'] for c in cs]);v=np.array([c['gain_nll_zero_minus_prior_by_seed'] for c in cs]);y=v.mean(1);rho=float(spearmanr(x,y).statistic);assert abs(rho-[-.7714285714285714,-.7293233082706766][j])<1e-12
  ax.scatter(np.repeat(x,3),v.ravel(),marker='x',color='.6',s=15,lw=.6,label='Seed observations');ax.scatter(x,y,color=['#0072B2','#A64B00'][j],marker=['o','s'][j],s=28,label='Client mean (3 seeds)');ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator([200,500,1000,2000,5000]));ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:,.0f}'));ax.xaxis.set_minor_formatter(NullFormatter());ax.axhline(0,color='.5',ls='--',lw=.7);ax.set(xlabel='Client TRAIN sample count nᵢ (log scale)',ylabel='ΔNLL = NLL(zero) − NLL(prior)',title=f"{['CIFAR-10','FashionMNIST'][j]} | Spearman ρ = {rho:+.3f}");ax.grid(alpha=.12);ax.legend(fontsize=7)
  figdata['E2'][task]={'rho':rho,'n_i':x.tolist(),'delta_by_client_and_seed':v.tolist(),'client_means':y.tolist(),'unit':'20 client means, one fixed partition','causal_inference':False}
 for j,ax in enumerate(axes):
  rho=figdata['E2'][['cifar','fashion'][j]]['rho'];ax.set_title(f"{['CIFAR-10','FashionMNIST'][j]}\nSpearman ρ = {rho:+.3f}",fontsize=10)
 fig.savefig(P/'figures/figure2_delta_nll_vs_samples.pdf');fig.savefig(P/'figures/figure2_delta_nll_vs_samples.png',dpi=200);plt.close(fig)
figdata['output_sha256']={str(p):sha(p) for p in (P/'figures').glob('figure[12]*')}
write(A/'claim_figures.json',figdata);write(P/'e2_sample_gain.json',{'status':'COMPLETE','primary_result':'TRAIN sample count versus gain','delta_definition':figdata['delta_definition'],'tasks':figdata['E2'],'sources':figdata['sources'],'skew_correlations_in_manuscript':False})
print(json.dumps({'E1_alpha01_counts':{t:v['improved_pairs_at_alpha0.1'] for t,v in figdata['E1'].items()},'E2_rho':{t:v['rho'] for t,v in figdata['E2'].items()}}))
