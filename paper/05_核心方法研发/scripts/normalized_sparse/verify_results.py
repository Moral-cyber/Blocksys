"""Independent bounded arithmetic/checkpoint check, no training/evaluation."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
folder=HERE/'development_s20261151'
methods=['topk','nova_topk','rms','nova_rms']
report={'status':'PARTIAL','runs':{},'external_cross_model_audit':'REVIEW_UNAVAILABLE'}
base_manifest=None
for method in methods:
 d=folder/method
 if not (d/'result.json').exists():
  report['runs'][method]={'status':'NOT_STARTED'};continue
 cfg=json.loads((d/'config.json').read_text());result=json.loads((d/'result.json').read_text())
 conf=cfg.copy();sig=conf.pop('signature')
 assert hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest()==sig==result['signature']
 assert cfg['source_sha256']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
 assert cfg['plan_sha256']==hashlib.sha256((HERE/'PLAN.md').read_bytes()).hexdigest()
 assert cfg['seed']==20261151 and cfg['rounds']==20 and cfg['method']==method
 assert cfg['lr']==.03 and cfg['batch']==64 and cfg['local_epoch']==1
 assert cfg['parameter_count']==136874 and cfg['selected_coordinates']==6843
 m=cfg['data_manifest'];sizes=m['client_sizes']
 if base_manifest is None:base_manifest=m
 else:assert m==base_manifest
 tau=np.array([math.ceil(n/64) for n in sizes]);ww=np.array(sizes,dtype=float)/sum(sizes)
 assert cfg['local_steps']==tau.tolist()
 expected_multiplier=float(ww@tau) if method.startswith('nova') else 1.
 assert cfg['server_multiplier']==expected_multiplier
 assert cfg['ef_units']==('delta/local_steps' if method.startswith('nova') else 'delta')
 assert cfg['normalization']==('RMS per parameter tensor' if method.endswith('rms') else 'none')
 rows=result['rows'];assert all(x['round']==i+1 for i,x in enumerate(rows))
 for row in rows:
  v=row['validation'];assert v['samples']==2000 and np.isfinite(v['loss']) and 0<=v['accuracy']<=1
  # Fixed balanced validation: overall and class-macro accuracy agree modulo FP32 report rounding.
  class_acc=np.asarray(v['class_accuracy'])
  assert len(class_acc)==10 and np.isfinite(class_acc).all()
  assert abs(v['macro_accuracy']-float(class_acc.mean()))<1e-12
  assert abs(v['accuracy']-float(class_acc.mean()))<6e-8
  assert row['analytical_upload_bytes']==20*(64+8*6843)
  assert len(row['clients'])==20 and sorted(x['client'] for x in row['clients'])==list(range(20))
  for c in row['clients']:
   assert len(c['selected_by_tensor'])==8 and sum(c['selected_by_tensor'])==6843
   assert all(isinstance(n,int) and n>=0 for n in c['selected_by_tensor'])
   cn,rn,sn=[c[k] for k in ['candidate_norm','residual_norm','sent_norm']]
   assert np.isfinite([cn,rn,sn]).all() and min(cn,rn,sn)>=0
   # FP32 norm logs are coarse diagnostics, not exact squared-energy records.
   # Coordinate-level EF is tested separately against the production compressor.
   assert abs(cn*cn-rn*rn-sn*sn)<=1e-4*max(1.,cn*cn)
 entry={'status':result['status'],'rounds':len(rows),'per_round_analytical_upload_bytes':1096160,
  'completed_rounds_analytical_upload_bytes':len(rows)*1096160,'server_multiplier':expected_multiplier,
  'evaluation':'global balanced 2000-row development subset, not client macro'}
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
  last=rows[-5:]
  # Explicit summation independent of any parent summarizer.
  entry['last5']={'rounds':[x['round'] for x in last],
    'accuracy':sum(float(x['validation']['accuracy']) for x in last)/5,
    'ce':sum(float(x['validation']['loss']) for x in last)/5}
  entry['round20']={'accuracy':rows[-1]['validation']['accuracy'],'ce':rows[-1]['validation']['loss']}
  tensor_sizes={name:value.numel() for name,value in ck['model'].items()}
  last_clients=[client for row in rows[-5:] for client in row['clients']]
  selections={}
  for i,name in enumerate(rows[-1]['tensor_names']):
   values=[client['selected_by_tensor'][i] for client in last_clients]
   selections[name]={'tensor_parameters':tensor_sizes[name],
    'mean_selected_coordinates':sum(values)/len(values),
    'mean_selected_fraction':sum(values)/len(values)/tensor_sizes[name],
    'zero_selection_client_rounds_of_100':sum(x==0 for x in values)}
  entry['last5_tensor_selection']=selections
 report['runs'][method]=entry
if all(report['runs'][x]['status']=='DONE' and report['runs'][x]['rounds']==20 for x in methods):
 report['status']='ALL_DONE'
 means={x:report['runs'][x]['last5'] for x in methods}
 B,C,D=[means[x] for x in ['nova_topk','rms','nova_rms']]
 ce_pass=D['ce']<min(B['ce'],C['ce']);acc_pass=D['accuracy']>=max(B['accuracy'],C['accuracy'])+.005
 report['frozen_go']={'definition':'CE(D)<min(CE(B),CE(C)) AND acc(D)>=max(acc(B),acc(C))+.005',
  'ce_pass':bool(ce_pass),'accuracy_pass':bool(acc_pass),'decision':'GO_TO_PAIRED_REPLICATIONS' if ce_pass and acc_pass else 'STOP_THIS_NORMALIZATION_COMBINATION'}
 effects={}
 for target,control,label in [('nova_topk','topk','known_nova_component_B_minus_A'),('rms','topk','rms_component_C_minus_A'),('nova_rms','nova_topk','added_rms_under_nova_D_minus_B'),('nova_rms','rms','added_nova_under_rms_D_minus_C')]:
  effects[label]={'ce_delta':means[target]['ce']-means[control]['ce'],'accuracy_delta_pp':100*(means[target]['accuracy']-means[control]['accuracy'])}
 effects['interaction_D_minus_B_minus_C_plus_A']={
  'ce':means['nova_rms']['ce']-means['nova_topk']['ce']-means['rms']['ce']+means['topk']['ce'],
  'accuracy_pp':100*(means['nova_rms']['accuracy']-means['nova_topk']['accuracy']-means['rms']['accuracy']+means['topk']['accuracy'])}
 report['descriptive_2x2_differences']=effects
(HERE/'LOCAL_STATS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
