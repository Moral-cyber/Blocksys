"""Independent saved-result arithmetic/counter checker; no training/test loading."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;folder=HERE/'development_s20261161'
methods=['standard','samebatch','shadow'];report={'status':'PARTIAL','runs':{},'external_cross_model_audit':'REVIEW_UNAVAILABLE'}
manifest0=None
for method in methods:
 d=folder/method
 if not (d/'result.json').exists():
  report['runs'][method]={'status':'NOT_STARTED'};continue
 cfg=json.loads((d/'config.json').read_text());result=json.loads((d/'result.json').read_text())
 c=cfg.copy();sig=c.pop('signature')
 assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==sig==result['signature']
 assert cfg['source_sha256']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
 assert cfg['plan_sha256']==hashlib.sha256((HERE/'PLAN.md').read_bytes()).hexdigest()
 assert cfg['method']==method and cfg['seed']==20261161 and cfg['rounds']==20
 assert cfg['selected_coordinates']==6843 and cfg['parameter_count']==136874
 assert cfg['lr_per_step']==(.03 if method=='standard' else .015)
 assert cfg['gradient_evaluations_per_batch']==(1 if method=='standard' else 2)
 m=cfg['data_manifest']
 if manifest0 is None:manifest0=m
 else:assert m==manifest0
 tau=[math.ceil(n/64) for n in m['client_sizes']]
 rows=result['rows'];assert all(row['round']==j+1 for j,row in enumerate(rows))
 factor=1 if method=='standard' else 2
 for row in rows:
  v=row['validation'];assert v['samples']==2000 and np.isfinite(v['loss']) and 0<=v['accuracy']<=1
  classacc=np.asarray(v['class_accuracy']);assert len(classacc)==10 and np.isfinite(classacc).all()
  assert abs(float(classacc.mean())-v['macro_accuracy'])<1e-12
  assert abs(float(classacc.mean())-v['accuracy'])<6e-8
  assert row['analytical_upload_bytes']==1096160
  assert len(row['clients'])==20 and sorted(x['client'] for x in row['clients'])==list(range(20))
  for client in row['clients']:
   i=client['client'];assert client['batches']==tau[i]
   assert client['optimizer_steps']==client['gradient_evaluations']==factor*tau[i]
   assert np.isfinite([client['residual_norm'],client['dense_delta_norm']]).all()
   assert min(client['residual_norm'],client['dense_delta_norm'])>=0
 entry={'status':result['status'],'rounds':len(rows),'evaluation':'global fixed balanced dev2000, not client macro',
  'analytical_upload_bytes_total':sum(row['analytical_upload_bytes'] for row in rows),
  'gradient_evaluations_total':sum(x['gradient_evaluations'] for row in rows for x in row['clients']),
  'seconds_round_iterations_including_evaluation':sum(row['seconds'] for row in rows)}
 ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False)
 if len(ck['rows'])==len(rows):
  assert ck['signature']==sig and len(ck['residuals'])==20
  assert all(torch.isfinite(t).all() for t in ck['model'].values())
  for i,rr in enumerate(ck['residuals']):
   assert rr.numel()==136874 and torch.isfinite(rr).all()
   recorded=next(x['residual_norm'] for x in rows[-1]['clients'] if x['client']==i)
   assert float(rr.norm())==recorded
  entry['final_checkpoint']='MATCH'
 else:entry['final_checkpoint']='ADVANCED_DURING_READ'
 if result['status']=='DONE' and len(rows)==20:
  entry['last5']={'rounds':[16,17,18,19,20],
   'accuracy':sum(row['validation']['accuracy'] for row in rows[-5:])/5,
   'ce':sum(row['validation']['loss'] for row in rows[-5:])/5}
  entry['round20']={'accuracy':rows[-1]['validation']['accuracy'],'ce':rows[-1]['validation']['loss']}
 report['runs'][method]=entry
if all(report['runs'][method]['status']=='DONE' and report['runs'][method]['rounds']==20 for method in methods):
 report['status']='ALL_DONE';means={method:report['runs'][method]['last5'] for method in methods}
 A,B,C=[means[method] for method in methods]
 ce_pass=C['ce']<min(A['ce'],B['ce']);acc_pass=C['accuracy']>=max(A['accuracy'],B['accuracy'])+.01
 report['frozen_go']={'definition':'CE(C)<min(CE(A),CE(B)) AND acc(C)>=max(acc(A),acc(B))+.01',
  'ce_pass':bool(ce_pass),'accuracy_pass':bool(acc_pass),
  'decision':'GO_TO_SEED_AND_LONGER_VALIDATION' if ce_pass and acc_pass else 'STOP_THIS_SHADOW_RECIPE'}
 effects={}
 for target,control,label in [('samebatch','standard','extra_samebatch_B_minus_A'),('shadow','standard','shadow_C_minus_A'),('shadow','samebatch','shadow_C_minus_B')]:
  effects[label]={'ce_delta':means[target]['ce']-means[control]['ce'],
    'accuracy_delta_pp':100*(means[target]['accuracy']-means[control]['accuracy'])}
 report['descriptive_differences']=effects
(HERE/'LOCAL_STATS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
