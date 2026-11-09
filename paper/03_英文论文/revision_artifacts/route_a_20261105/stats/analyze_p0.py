from pathlib import Path
import csv,hashlib,json,datetime
import numpy as np
from scipy.stats import wilcoxon
ROOT=Path('/Users/wind/Desktop/Codex/paper')
OUT=Path('/private/tmp/pr_lfl_route_a/stats')
CSV=ROOT/'03_英文论文/evidence/individual_seed_metrics.csv'
ALL=list(csv.DictReader(CSV.open()))
SOURCES={str(CSV):hashlib.sha256(CSV.read_bytes()).hexdigest()}
SEEDS={'cifar':[20261231,20261232,20261233],'fashion':[20261241,20261242,20261243]}
VARIANTS=['zero_anchor_vector','prioranchored_vector']
METRICS=['macro_accuracy','weighted_accuracy','macro_nll','weighted_nll','client_accuracy_q10']
result={'status':'PASS','created_at':datetime.datetime.now().astimezone().isoformat(),'source_and_input_sha256':SOURCES,
 'design':{'new_training':0,'new_fits':0,'new_test_inference':0,'training_replicates_per_task':3,'client_count_per_task':20,'paired_descriptions_per_task':60,'bootstrap_repetitions':1000,'bootstrap_rng':'numpy.random.Generator(PCG64(20261104)) independently reset per task','delta_convention':'prior center minus zero center; negative NLL difference favors prior','bootstrap_estimand':'unweighted average client NLL difference, averaged over the existing three seeds; client clusters resampled with all three seeds retained','inference_boundary':'One fixed partition and shared fitted model per seed. Client-seed pairs are repeated, dependent observations, not sixty independent training replicates. Pooled Wilcoxon p-values are nominal descriptive calculations, not valid population-level evidence of sixty independent replications. Client-cluster percentile intervals describe variation over observed client clusters conditional on these three seeds and the fixed partition; they omit new partition and training-run uncertainty.'},'tasks':{}}
for task,seeds in SEEDS.items():
 ds=[];counts=[];checks=[]
 for seed in seeds:
  p=ROOT/f'05_核心方法研发/results/predictive_final_test/official_{task}/fedlc_uniform3_s{seed}/result.json'
  d=json.loads(p.read_text());SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  assert d['status']=='DONE' and d['seed']==seed and len(d['clients'])==20
  clients=sorted(d['clients'],key=lambda z:z['client']);assert [x['client'] for x in clients]==list(range(20))
  w=np.asarray([x['test_rows'] for x in clients]);assert w.sum()==10000
  counts.append(w)
  for v in VARIANTS:
   a=np.asarray([x['metrics'][v]['accuracy'] for x in clients]);n=np.asarray([x['metrics'][v]['nll'] for x in clients]);calculated={'macro_accuracy':float(a.mean()),'weighted_accuracy':float(np.average(a,weights=w)),'macro_nll':float(n.mean()),'weighted_nll':float(np.average(n,weights=w)),'client_accuracy_q10':float(np.quantile(a,.1))}
   matches=[x for x in ALL if x['task']==task and int(x['seed'])==seed and x['method']=='fedlc_uniform3' and x['variant']==v];assert len(matches)==1
   diffs={key:abs(value-float(matches[0][key])) for key,value in calculated.items()};assert max(diffs.values())<1e-12
   checks.append({'seed':seed,'variant':v,'maximum_csv_reconstruction_error':max(diffs.values()),'metrics':calculated})
  ds.append([x['metrics']['prioranchored_vector']['nll']-x['metrics']['zero_anchor_vector']['nll'] for x in clients])
 delta=np.array(ds,dtype=np.float64);assert delta.shape==(3,20);assert all(np.array_equal(counts[0],x) for x in counts[1:])
 aggregate={}
 for v in VARIANTS:
  rows=[x['metrics'] for x in checks if x['variant']==v]
  aggregate[v]={key:{'mean':float(np.mean([x[key] for x in rows])),'sample_sd':float(np.std([x[key] for x in rows],ddof=1))} for key in METRICS}
 pooled=delta.ravel();wil=wilcoxon(pooled,zero_method='wilcox',correction=False,alternative='two-sided',method='exact')
 means=delta.mean(axis=0);rng=np.random.default_rng(20261104);indices=rng.integers(0,20,size=(1000,20));boot=means[indices].mean(axis=1);ci=np.quantile(boot,[.025,.975])
 seedmeans=delta.mean(axis=1)
 # A client-average comparison avoids labeling its three repeated observations as three independent clients.
 wil_clients=wilcoxon(means,zero_method='wilcox',correction=False,alternative='two-sided',method='exact')
 t={'seeds':seeds,'partition_seed':d['partition_seed'],'summary_by_variant':aggregate,'csv_reconstruction_checks':checks,
 'paired_nll':{'client_seed_pairs':60,'improved_pairs':int((pooled<0).sum()),'worse_pairs':int((pooled>0).sum()),'tied_pairs':int((pooled==0).sum()),'mean_delta':float(pooled.mean()),'median_delta':float(np.median(pooled)),'minimum_delta':float(pooled.min()),'maximum_delta':float(pooled.max()),'wilcoxon_two_sided':{'statistic':float(wil.statistic),'nominal_p':float(wil.pvalue),'method':'SciPy exact signed-rank distribution; no zero differences','interpretation':'Nominal pooled-pair calculation only; repeated clients and shared seed-model dependence violate independent-pair sampling. Do not present as n=60 independent training runs.'},'cluster_bootstrap_95_percentile_interval':ci.tolist(),'cluster_bootstrap_repetitions':1000,'client_mean_negative_count':int((means<0).sum()),'client_mean_positive_count':int((means>0).sum()),'all_clients_mean_improve':bool(np.all(means<0)),'client_seed_delta_matrix':delta.tolist(),'client_mean_deltas':means.tolist(),'seed_mean_deltas':seedmeans.tolist(),'independent_training_seed_improvements':int((seedmeans<0).sum()),'seed_regressions':int((seedmeans>0).sum()),'client_average_wilcoxon_sensitivity':{'statistic':float(wil_clients.statistic),'nominal_p':float(wil_clients.pvalue),'scope':'20 client averages across the same three seeds; conditional descriptive sensitivity, not new training replications'}},'test_client_sample_counts':counts[0].tolist()}
 t['deltas']={key:aggregate[VARIANTS[1]][key]['mean']-aggregate[VARIANTS[0]][key]['mean'] for key in METRICS}
 result['tasks'][task]=t
result['source_and_input_sha256']=SOURCES
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lines=['# P0 existing-data results','',result['design']['inference_boundary'],'']
for task,t in result['tasks'].items():
 n=t['paired_nll'];lines += [f'## {task}', '',f"NLL paired differences: {n['improved_pairs']}/60 improve, {n['worse_pairs']}/60 worsen, median prior-minus-zero = {n['median_delta']:.8f}; nominal pooled Wilcoxon two-sided p = {n['wilcoxon_two_sided']['nominal_p']:.8g}. All {n['independent_training_seed_improvements']}/3 seed-level means improve. Client-cluster bootstrap (1000 draws; three seeds kept within each client): mean delta {n['mean_delta']:.8f}, 95% percentile interval [{n['cluster_bootstrap_95_percentile_interval'][0]:.8f}, {n['cluster_bootstrap_95_percentile_interval'][1]:.8f}]. {n['client_mean_negative_count']}/20 client means improve.", '', '| Endpoint | Zero center | Prior center | Difference |','|---|---:|---:|---:|']
 for key in METRICS:
  z=t['summary_by_variant'][VARIANTS[0]][key]['mean'];a=t['summary_by_variant'][VARIANTS[1]][key]['mean'];lines.append(f'| {key} | {z:.10f} | {a:.10f} | {a-z:+.10f} |')
 lines+=['']
lines+=['Sample-weighted accuracy equals global pooled test accuracy for each seed, because every original TEST example is assigned once. Macro averages weight clients equally, whereas sample weighting uses their assigned TEST sizes. This difference in client weighting changes the measured contribution of each client; it should not be described as proof that minority-class scarcity alone caused the difference.','', 'Caption safety: every client improves is false for these main-control results if any mean or seed-level pair is positive. Plot actual prior-minus-zero differences around a zero reference line and label the observed count.','', 'No P1/P2 experiment was performed by this script.']
(OUT/'prose.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({task:{k:t['paired_nll'][k] for k in ['improved_pairs','worse_pairs','median_delta','mean_delta','client_mean_negative_count','all_clients_mean_improve','seed_mean_deltas','cluster_bootstrap_95_percentile_interval']} for task,t in result['tasks'].items()},indent=2))
