from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,hashlib,sys,time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
METHOD=Path('/Users/wind/Desktop/Codex/paper/05_核心方法研发')
OUT=HERE/'lambda_result.json'
TASKS={'cifar':(20261217,(20261231,20261232,20261233)),'fashion':(20261219,(20261241,20261242,20261243))}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(z,y):
 z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.int64);v=z.max(1);nll=v+np.log(np.exp(z-v[:,None]).sum(1))-z[np.arange(len(y)),y]
 return {'nll':float(nll.mean()),'accuracy':float(np.mean(z.argmax(1)==y))}
def score(z,s):
 s=np.asarray(s,dtype=np.float64);return np.asarray(z,dtype=np.float64)*s[:10]+s[10:]
def main():
 if OUT.exists():raise ValueError('Never overwrite completed sensitivity')
 started=time.perf_counter();identities={};forbidden=[]
 def pin(p):identities[str(p)]=sha(p)
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   p=str(args[0])
   if any(x in p for x in ['/predictive_final_test/','/dataset/','/datasets/','test_batch','t10k']):
    forbidden.append(p);raise ValueError('saved development arrays only')
 sys.addaudithook(audit)
 source=METHOD/'scripts/prior_vector/solver.py';pin(source);pin(METHOD/'scripts/calibration_baselines/solver.py');pin(Path(__file__));pin(HERE/'PLAN.md')
 spec=importlib.util.spec_from_file_location('_sensitivity_frozen_solver',source);solver=importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
 result={'status':'RUNNING','scope':'fixed-grid development sensitivity, no selection','coefficient_grid':['0','0.1','1','10','100','inf'],'tasks':{},'new_fits':0,'new_training_seeds':0,'cnn_forwards':0,'cnn_backwards':0,'official_test_reads':0,'lambda10_states_bitwise_equal':True,'lambda10_max_metric_error':0.}
 for task,(partition,seeds) in TASKS.items():
  runs=[]
  for seed in seeds:
   ef=METHOD/f'results/predictive_pfl_confirm/evaluation_{task}_p{partition}_s{seed}'
   af=METHOD/f'results/predictive_pfl_confirm/adaptation_{task}_p{partition}_s{seed}'
   pin(af/'result.json');report=json.loads((af/'result.json').read_text());clients=[]
   for i in range(20):
    a=ef/f'fedlc_uniform3_client{i}.npz';b=af/f'fedlc_uniform3_client{i}.npz';pin(a);pin(b)
    with np.load(a,allow_pickle=False) as d: data={k:d[k].copy() for k in d.files}
    with np.load(b,allow_pickle=False) as d: old={k:d[k].copy() for k in d.files}
    labels=data['train_labels'];counts=np.bincount(labels,minlength=10);assert (counts==0).any()
    mu=solver.train_prior_anchor(labels);assert np.array_equal(mu,old['train_bias_anchor'])
    row={'client':i,'train_samples':len(labels),'holdout_samples':len(data['labels']),'absent_train_classes':np.flatnonzero(counts==0).tolist(),'coefficients':{'0':{'status':'NO_FINITE_MINIMIZER','reason':'At least one TRAIN-absent class; its logits can decrease indefinitely under centered bias, strictly reducing cross-entropy.'}}}
    for lam in (.1,1.,10.,100.):
     solver.base.RIDGE=lam
     state,fit=solver.fit(data['train_logits'],labels,mu);result['new_fits']+=1
     measured=metrics(score(data['raw'],state),data['labels'])
     if lam==10.:
      assert np.array_equal(state,old['prioranchored_vector'])
      expected=report['results']['fedlc_uniform3']['clients'][i]['metrics']['prioranchored_vector']
      err=max(abs(measured[k]-expected[k]) for k in ['nll','accuracy']);assert err<1e-12;result['lambda10_max_metric_error']=max(result['lambda10_max_metric_error'],err)
     row['coefficients'][f'{lam:g}']={'status':'FITTED','state_fp32':state.tolist(),'holdout':measured,'fit':{k:fit[k] for k in ['accepted_steps','stop_reason','projected_gradient_inf','hessian_min_eigenvalue','final_objective']}}
    state=np.concatenate([np.ones(10),mu]).astype(np.float32)
    row['coefficients']['inf']={'status':'ANALYTIC_FIXED_PRIOR','state_fp32':state.tolist(),'holdout':metrics(score(data['raw'],state),data['labels'])}
    clients.append(row)
   summaries={l:{k:float(np.mean([c['coefficients'][l]['holdout'][k] for c in clients])) for k in ['nll','accuracy']} for l in ['0.1','1','10','100','inf']}
   runs.append({'seed':seed,'partition':partition,'clients':clients,'client_macro_holdout':summaries})
   print(json.dumps({'task':task,'seed':seed,'macro':summaries}),flush=True)
  result['tasks'][task]={'runs':runs,'summary':{l:{k:{'mean':float(np.mean([r['client_macro_holdout'][l][k] for r in runs])),'sample_sd':float(np.std([r['client_macro_holdout'][l][k] for r in runs],ddof=1))} for k in ['nll','accuracy']} for l in ['0.1','1','10','100','inf']},'lambda0_clients_without_finite_minimizer':60}
 assert all(sha(Path(p))==v for p,v in identities.items())
 result.update(status='DONE',completed_at=datetime.now(timezone.utc).isoformat(),source_and_input_sha256=identities,input_hashes_unchanged=True,forbidden_open_attempts=forbidden,elapsed_seconds=time.perf_counter()-started)
 OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'DONE','fits':result['new_fits'],'seconds':result['elapsed_seconds'],'result':str(OUT)}))
if __name__=='__main__':main()
