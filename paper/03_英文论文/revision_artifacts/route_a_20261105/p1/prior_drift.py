from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
METHOD=Path('/Users/wind/Desktop/Codex/paper/05_核心方法研发')
TASKS={'cifar':(20261217,(20261231,20261232,20261233)),'fashion':(20261219,(20261241,20261242,20261243))}
RHOS=(0.,.25,.5,.75,1.)
OUT=HERE/'drift_result.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(z,s):
 s=np.asarray(s,dtype=np.float64);return np.asarray(z,dtype=np.float64)*s[:10]+s[10:]
def stats(z,y):
 maximum=z.max(1);loss=maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(len(y)),y]
 return {'nll':float(loss.mean()),'accuracy':float(np.mean(z.argmax(1)==y)),'samples':len(y)}
def quotas(prob,n):
 raw=prob*n;v=np.floor(raw).astype(int);order=np.lexsort((np.arange(20),-(raw-v)));v[order[:n-v.sum()]]+=1;assert v.sum()==n;return v

def main():
 if OUT.exists():raise ValueError('Never overwrite a completed drift result')
 started=time.perf_counter();hashes={};forbidden=[]
 def pin(p):hashes[str(p)]=sha(p)
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   p=str(args[0])
   if any(x in p for x in ['/predictive_final_test/','/dataset/','/datasets/','test_batch','t10k']):
    forbidden.append(p);raise ValueError('saved development arrays only')
 sys.addaudithook(audit);pin(Path(__file__));pin(HERE/'DRIFT_PLAN.md')
 result={'status':'RUNNING','scope':'Frozen local predictors under controlled development holdout reallocation; no prior update','rhos':list(RHOS),'prng':'numpy PCG64 seed20261105, independent spawned task streams','dirichlet_alpha':.1,'tasks':{},'new_training_seeds':0,'new_predictor_fits':0,'cnn_forwards':0,'cnn_backwards':0,'official_test_reads':0,'rho0_max_metric_difference':0.}
 for taskidx,(task,(partition,seeds)) in enumerate(TASKS.items()):
  models=[]
  for seed in seeds:
   ef=METHOD/f'results/predictive_pfl_confirm/evaluation_{task}_p{partition}_s{seed}';af=METHOD/f'results/predictive_pfl_confirm/adaptation_{task}_p{partition}_s{seed}'
   pin(af/'result.json');report=json.loads((af/'result.json').read_text());pieces=[];states={}
   for client in range(20):
    a=ef/f'fedlc_uniform3_client{client}.npz';b=af/f'fedlc_uniform3_client{client}.npz';pin(a);pin(b)
    with np.load(a,allow_pickle=False) as d:
     pieces.append({'ids':d['holdout_ids'].copy(),'labels':d['labels'].copy(),'raw':d['raw'].copy(),'original_clients':np.full(len(d['labels']),client,dtype=np.int64)})
    with np.load(b,allow_pickle=False) as d:
     states[client]={'zero':d['zero_anchor_vector'].copy(),'prior':d['prioranchored_vector'].copy()}
   pooled={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]};order=np.argsort(pooled['ids']);pooled={k:x[order] for k,x in pooled.items()};assert np.unique(pooled['ids']).size==len(pooled['ids'])
   if models:
    for key in ['ids','labels','original_clients']:assert np.array_equal(pooled[key],models[0]['pooled'][key])
   models.append({'seed':seed,'pooled':pooled,'states':states,'report':report})
  original=models[0]['pooled'];labels=original['labels'];origin=original['original_clients'];ids=original['ids'];rng=np.random.default_rng(np.random.SeedSequence([20261105,taskidx]))
  classrows=[np.flatnonzero(labels==c) for c in range(10)];movement=[rng.permutation(rows) for rows in classrows];attempts=[]
  for attempt in range(1,10001):
   q=rng.dirichlet(np.full(20,.1),size=10);assignments={}
   for rho in RHOS:
    assigned=origin.copy()
    for c,rows in enumerate(movement):
     moved=rows[:int(np.floor(rho*len(rows)))];quota=quotas(q[c],len(moved));assigned[moved]=np.repeat(np.arange(20),quota)
    assert np.bincount(assigned,minlength=20).sum()==len(labels)
    assignments[rho]=assigned
   empty={str(rho):np.flatnonzero(np.bincount(a,minlength=20)==0).tolist() for rho,a in assignments.items()};empty={k:v for k,v in empty.items() if v}
   attempts.append({'attempt':attempt,'empty_clients':empty})
   if not empty:break
  else:raise RuntimeError('No nonempty target within label-only cap')
  assert np.array_equal(assignments[0.],origin)
  baseprobs=np.array([np.bincount(labels[origin==i],minlength=10)/np.sum(origin==i) for i in range(20)])
  scenarios=[]
  for rho,assigned in assignments.items():
   sizes=np.bincount(assigned,minlength=20);newprobs=np.array([np.bincount(labels[assigned==i],minlength=10)/sizes[i] for i in range(20)]);tv=.5*np.abs(newprobs-baseprobs).sum(1)
   runs=[]
   for model in models:
    clients=[]
    for i in range(20):
     rows=np.flatnonzero(assigned==i);metrics={variant:stats(score(model['pooled']['raw'][rows],state),labels[rows]) for variant,state in model['states'][i].items()}
     if rho==0.:
      for variant,oldkey in [('zero','zero_anchor_vector'),('prior','prioranchored_vector')]:
       expected=model['report']['results']['fedlc_uniform3']['clients'][i]['metrics'][oldkey]
       err=max(abs(metrics[variant][k]-expected[k]) for k in ['nll','accuracy']);assert err<1e-12;result['rho0_max_metric_difference']=max(result['rho0_max_metric_difference'],err)
     clients.append({'client':i,'samples':int(sizes[i]),'prior_TV':float(tv[i]),'metrics':metrics})
    summaries={variant:{agg+'_'+key:float(np.mean([c['metrics'][variant][key] for c in clients]) if agg=='macro' else np.average([c['metrics'][variant][key] for c in clients],weights=sizes)) for agg in ['macro','weighted'] for key in ['nll','accuracy']} for variant in ['zero','prior']}
    runs.append({'seed':model['seed'],'clients':clients,'summaries':summaries})
   summary={variant:{key:{'mean':float(np.mean([r['summaries'][variant][key] for r in runs])),'sample_sd':float(np.std([r['summaries'][variant][key] for r in runs],ddof=1))} for key in ['macro_nll','weighted_nll','macro_accuracy','weighted_accuracy']} for variant in ['zero','prior']}
   scenarios.append({'rho':rho,'assignment_sha256':hashlib.sha256(assigned.astype('<i8').tobytes()).hexdigest(),'sample_count':len(labels),'client_sizes':sizes.tolist(),'client_class_counts':[np.bincount(labels[assigned==i],minlength=10).tolist() for i in range(20)],'mean_client_prior_TV':float(tv.mean()),'runs':runs,'summary':summary,'delta_prior_minus_zero_macro_nll':float(np.mean([r['summaries']['prior']['macro_nll']-r['summaries']['zero']['macro_nll'] for r in runs]))})
   print(json.dumps({'task':task,'rho':rho,'TV':float(tv.mean()),'prior_nll':summary['prior']['macro_nll']['mean'],'zero_nll':summary['zero']['macro_nll']['mean'],'difference':scenarios[-1]['delta_prior_minus_zero_macro_nll']}),flush=True)
  result['tasks'][task]={'partition':partition,'holdout_ids_sha256':hashlib.sha256(ids.astype('<u4').tobytes()).hexdigest(),'target_q':q.tolist(),'label_only_target_attempts':attempts,'original_class_counts':np.bincount(labels,minlength=10).tolist(),'scenarios':scenarios}
 assert all(sha(Path(p))==v for p,v in hashes.items())
 result.update(status='DONE',completed_at=datetime.now(timezone.utc).isoformat(),source_and_input_sha256=hashes,inputs_unchanged=True,forbidden_open_attempts=forbidden,elapsed_seconds=time.perf_counter()-started)
 OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'DONE','seconds':result['elapsed_seconds'],'result':str(OUT)}))
if __name__=='__main__':main()
