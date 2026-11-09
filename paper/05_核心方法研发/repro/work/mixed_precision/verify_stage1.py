"""Read-only Stage1 saved statistics/counter checker; never touches quantizer."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;folder=HERE/'development_s20261181'
methods=['ce_dense','ce_topk','cal_dense','cal_topk'];report={'status':'PENDING','runs':{},'external_cross_model_audit':'REVIEW_UNAVAILABLE'}
manifest0=None
for method in methods:
 d=folder/method
 if not (d/'result.json').exists():report['runs'][method]={'status':'NOT_STARTED'};continue
 cfg=json.loads((d/'config.json').read_text());result=json.loads((d/'result.json').read_text())
 c=cfg.copy();sig=c.pop('signature');assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==sig==result['signature']
 assert cfg['source_sha256']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
 assert cfg['plan_sha256']==hashlib.sha256((HERE/'EXPERIMENT_PLAN.md').read_bytes()).hexdigest()
 assert cfg['method']==method and cfg['seed']==20261181 and cfg['rounds']==20
 assert cfg['topk']==6843 and cfg['parameter_count']==136874 and cfg['application_budget_bytes']==54808
 assert cfg['quantizer_sha256'] is None and cfg['official_test_loaded'] is False and result['test_evaluated'] is False
 m=cfg['data_manifest'];cal=method.startswith('cal_');dense=method.endswith('dense')
 if manifest0 is None:manifest0=m
 else:assert m==manifest0
 train=np.concatenate(m['train_indices']);holdout=np.concatenate(m['holdout_indices']);dev=np.array(m['validation_indices'])
 assert len(train)==37500 and len(holdout)==12500 and len(dev)==2000
 assert not np.intersect1d(train,holdout).size and np.isin(dev,holdout).all()
 assert np.array_equal(np.sort(np.r_[train,holdout]),np.arange(50000))
 expected_counts=m['client_label_counts'];sizes=m['client_sizes'];tau=[math.ceil(n/64) for n in sizes]
 rows=result['rows'];assert all(row['round']==i+1 for i,row in enumerate(rows))
 for row in rows:
  val=row['validation'];assert val['samples']==2000 and np.isfinite(val['loss']) and 0<=val['accuracy']<=1
  assert row['byte_accounting']=='analytical fixed layout'
  expect=547560 if dense else 54808
  assert row['upload_bytes']==20*expect and len(row['clients'])==20
  assert sorted(x['client'] for x in row['clients'])==list(range(20))
  for client in row['clients']:
   i=client['client'];assert client['train_label_counts']==expected_counts[i]
   assert sum(client['train_label_counts'])==sizes[i]
   assert client['optimizer_steps']==client['gradient_evaluations']==tau[i]
   assert client['payload_bytes']==expect
   assert np.isfinite([client['residual_norm'],client['delta_norm'],client['gradient_second_moment_mean']]).all()
   assert min(client['residual_norm'],client['delta_norm'],client['gradient_second_moment_mean'])>=0
   if dense:assert client['residual_norm']==0
   if cal:
    shift=(np.maximum(np.array(expected_counts[i]),1)**(-.25)).astype(np.float32)
    np.testing.assert_array_equal(np.array(client['calibration_shift'],dtype=np.float32),shift)
   else:assert client['calibration_shift'] is None
 entry={'status':result['status'],'rounds':len(rows),'evaluation':'raw-logit global fixed dev2000',
  'recorded_train_label_counts_match_manifest':True,'calibration_shift_matches_rule':True,
  'gradients_total':sum(x['gradient_evaluations'] for row in rows for x in row['clients']),
  'upload_bytes_analytical_total':sum(row['upload_bytes'] for row in rows),
  'seconds_iterations_including_evaluation':sum(row['seconds'] for row in rows)}
 ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False)
 if len(ck['rows'])==len(rows):
  assert ck['signature']==sig and len(ck['residuals'])==20
  assert all(torch.isfinite(t).all() for t in ck['model'].values())
  for i,rr in enumerate(ck['residuals']):
   assert torch.isfinite(rr).all() and rr.numel()==136874
   recorded=next(x['residual_norm'] for x in rows[-1]['clients'] if x['client']==i)
   assert recorded==float(rr.norm())
   if dense:assert not rr.any()
  entry['checkpoint']='MATCH'
 else:entry['checkpoint']='ADVANCED_DURING_READ'
 if result['status']=='DONE' and len(rows)==20:
  entry['last5']={'rounds':[16,17,18,19,20],
   'accuracy':sum(row['validation']['accuracy'] for row in rows[-5:])/5,
   'ce':sum(row['validation']['loss'] for row in rows[-5:])/5}
 report['runs'][method]=entry
if all(report['runs'][m]['status']=='DONE' and report['runs'][m]['rounds']==20 for m in methods):
 report['status']='ALL_DONE';effects={}
 for compression in ['dense','topk']:
  cal=report['runs']['cal_'+compression]['last5'];ce=report['runs']['ce_'+compression]['last5']
  effects[compression]={'ce_cal_minus_ce':cal['ce']-ce['ce'],'accuracy_cal_minus_ce_pp':100*(cal['accuracy']-ce['accuracy']),
   'both_ce_and_accuracy_improved':bool(cal['ce']<ce['ce'] and cal['accuracy']>ce['accuracy']),
   'attribution':'known FedLC-inspired calibration component, not a new communication contribution'}
 report['calibration_component_comparisons']=effects
(HERE/'TRAINING_STATS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
